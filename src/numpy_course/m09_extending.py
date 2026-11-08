# %% [markdown]
# # Module 9: Extending and interoperating
# Run:  `uv run python -m numpy_course.m09_extending`

# %%
import numbers
import os

import numpy as np

from numpy_course.utils import bench, section, show

rng = np.random.default_rng(0)

# %% [markdown]
# ## 9.1 Structured / record arrays (recap + field views)

# %%
section("9.1 Structured arrays")
trade_dt = np.dtype([("ts", "datetime64[s]"), ("sym", "U4"), ("qty", "i8"), ("px", "f8")])
trades = np.array(
    [("2026-11-08T09:30:00", "SAP", 100, 150.25), ("2026-11-08T09:31:10", "SIE", -50, 180.10),
     ("2026-11-08T09:32:05", "SAP", 25, 150.40)], dtype=trade_dt)
notional = trades["qty"] * trades["px"]
assert np.isclose(notional.sum(), 100 * 150.25 - 50 * 180.10 + 25 * 150.40)
sap = trades[trades["sym"] == "SAP"]
assert len(sap) == 2 and sap["qty"].sum() == 125
rec = trades.view(np.recarray)  # attribute access sugar
assert rec.px[0] == 150.25

# %% [markdown]
# ## 9.2 `np.vectorize` is a loop in disguise; `frompyfunc` makes object ufuncs

# %%
section("9.2 vectorize vs real vectorisation")


def py_clip_sq(v):
    return v * v if v < 5 else 25.0


vec = np.vectorize(py_clip_sq, otypes=[float])
x = rng.random(100_000) * 10
real = np.where(x < 5, x * x, 25.0)
assert np.allclose(vec(x), real)
t_vec, t_real = bench(lambda: vec(x)), bench(lambda: np.where(x < 5, x * x, 25.0))
show("np.vectorize (s)", f"{t_vec:.2e}")
show("true vectorisation (s)", f"{t_real:.2e}")
assert t_vec > 5 * t_real  # np.vectorize is convenience, NOT speed

fp = np.frompyfunc(lambda a, b: a + b, 2, 1)  # returns dtype=object arrays; supports .reduce/.accumulate
assert fp.reduce(np.array(["a", "b", "c"], dtype=object)) == "abc"
assert fp.accumulate(np.array([1, 2, 3], dtype=object)).tolist() == [1, 3, 6]

# %% [markdown]
# ## 9.3 Array protocols: `__array_ufunc__` (a units-aware Quantity)
# NumPy calls `__array_ufunc__` on any operand that defines it, so `np.add(q1, q2)` and `q1 + q2`
# are routed to YOUR logic. `NDArrayOperatorsMixin` maps +,-,*,/ etc. onto ufuncs for free.

# %%
section("9.3 __array_ufunc__")


def _merge(u, v, sign=1):
    out = dict(u or {})
    for k, e in (v or {}).items():
        out[k] = out.get(k, 0) + sign * e
        if out[k] == 0:
            del out[k]
    return out


class Quantity(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value, unit=None):
        self.value = np.asarray(value)
        self.unit = dict(unit or {})

    def __repr__(self):
        u = " ".join(f"{k}^{e:g}" if e != 1 else k for k, e in sorted(self.unit.items())) or "dimensionless"
        return f"Quantity({self.value!r}, {u})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or "out" in kwargs:
            return NotImplemented
        if not all(isinstance(x, (Quantity, np.ndarray, numbers.Number)) for x in inputs):
            return NotImplemented
        units = [x.unit if isinstance(x, Quantity) else {} for x in inputs]
        vals = [x.value if isinstance(x, Quantity) else x for x in inputs]
        if ufunc in (np.add, np.subtract):
            if units[0] != units[1]:
                raise ValueError(f"unit mismatch: {units[0]} vs {units[1]}")
            unit = units[0]
        elif ufunc is np.multiply:
            unit = _merge(units[0], units[1])
        elif ufunc is np.true_divide:
            unit = _merge(units[0], units[1], sign=-1)
        elif ufunc is np.square:
            unit = {k: 2 * e for k, e in units[0].items()}
        elif ufunc is np.sqrt:
            unit = {k: e / 2 for k, e in units[0].items()}
        elif ufunc in (np.negative, np.absolute):
            unit = units[0]
        else:
            return NotImplemented  # numpy raises TypeError: unsupported op for Quantity
        return Quantity(ufunc(*vals, **kwargs), unit)


d = Quantity([100.0, 200.0, 300.0], {"m": 1})
t = Quantity(10.0, {"s": 1})
v = d / t
assert np.allclose(v.value, [10, 20, 30]) and v.unit == {"m": 1, "s": -1}
assert (d + d).unit == {"m": 1} and np.allclose((2 * d).value, [200, 400, 600])
assert np.sqrt(d * d).unit == {"m": 1} and np.square(v).unit == {"m": 2, "s": -2}
for bad in (lambda: d + t, lambda: np.sin(d)):
    try:
        bad()
        raise AssertionError("should have failed")
    except (ValueError, TypeError):
        pass
show("d / t", v)

# %% [markdown]
# ## 9.4 `__array_function__`: intercept np.mean, np.concatenate, ...

# %%
section("9.4 __array_function__")


class Logged:
    calls: list = []

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array_function__(self, func, types, args, kwargs):
        Logged.calls.append(func.__name__)
        unwrap = lambda a: a.data if isinstance(a, Logged) else a  # noqa: E731
        return func(*[unwrap(a) for a in args], **kwargs)


lg = Logged([1.0, 2.0, 3.0])
assert np.mean(lg) == 2.0 and np.sum(lg) == 6.0
assert Logged.calls == ["mean", "sum"]

# %% [markdown]
# ## 9.5 Interoperability: zero-copy where possible

# %%
section("9.5 Interop")
a = np.arange(6, dtype=np.float64)
b = np.from_dlpack(a)  # DLPack: the cross-library zero-copy standard (PyTorch/JAX/CuPy speak it)
assert np.shares_memory(a, b)

try:
    import pandas as pd

    df = pd.DataFrame({"x": rng.random(5), "y": rng.random(5)})
    arr2 = df.to_numpy()  # may or may not copy: never ASSUME sharing
    assert arr2.shape == (5, 2)
    s = pd.Series(a)
    assert np.array_equal(np.asarray(s), a)
    show("pandas round-trip", "ok")
except ImportError:
    show("pandas", "not installed (dev group)")

try:
    import torch  # optional, heavy

    tt = torch.from_numpy(a)  # shares memory
    a[0] = 99
    assert tt[0].item() == 99
    show("torch zero-copy", "ok")
except ImportError:
    show("torch", "not installed (optional)")

import scipy.sparse as sp

dense = np.diag([1.0, 2.0, 3.0])
assert np.array_equal(sp.csr_array(dense).toarray(), dense)  # SciPy sparse <-> NumPy

# %% [markdown]
# ## 9.6 Testing numerical code

# %%
section("9.6 Testing numerics")
np.testing.assert_allclose(0.1 + 0.2, 0.3, rtol=1e-12)  # tolerance-aware (relative)
np.testing.assert_array_equal(np.arange(3), [0, 1, 2])  # exact
try:
    np.testing.assert_allclose([1.0, 2.0], [1.0, 2.1], rtol=1e-6)
    raise RuntimeError("should have failed")
except AssertionError as e:
    assert "Mismatched elements" in str(e)  # the failure message pinpoints the offending entries
np.testing.assert_allclose(np.nan, np.nan)  # NaNs compare equal here (equal_nan=True by default)
np.testing.assert_allclose(1e-14, 0.0, atol=1e-12)  # relative tol is useless at 0: always give atol
print("  Property-based tests with hypothesis live in tests/test_properties.py")

# %% [markdown]
# ## 9.7 Read real source

# %%
section("9.7 Where the code lives")
core = os.path.join(os.path.dirname(np.__file__), "_core")
show("numpy version", np.__version__)
show("numpy package", os.path.dirname(np.__file__))
if os.path.isdir(core):
    show("_core has", ", ".join(sorted(f for f in os.listdir(core) if f in {"numeric.py", "fromnumeric.py", "multiarray.py", "umath.py", "_methods.py"})))
print("  Start with numpy/_core/_methods.py (how .mean/.std are built) and fromnumeric.py (thin wrappers).")

# %% [markdown]
# ### Your turn
# 1. Extend `Quantity` with `np.power`, comparison ufuncs, and `__array_function__` support for `np.sum`.
# 2. Write a hypothesis test: for random arrays, `np.sort(x)` is non-decreasing and a permutation of `x`.
# 3. Read `_methods.py` and explain how `np.std` is composed from other reductions.

print("\nModule 9 complete: all assertions passed.")
