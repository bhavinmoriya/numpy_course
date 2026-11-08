# %% [markdown]
# # Module 6: Memory model and strides
# Run:  `uv run python -m numpy_course.m06_memory`

# %%
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from scipy.signal import correlate2d

from numpy_course.utils import bench, peak_mem, section, show

rng = np.random.default_rng(0)

# %% [markdown]
# ## 6.1 C vs Fortran order, contiguity, and speed
# An array is (buffer, dtype, shape, strides). Walking the buffer in memory order is fast (cache!).

# %%
section("6.1 Memory order")
a_c = np.arange(12, dtype=np.int64).reshape(3, 4)  # C order: last axis varies fastest
a_f = np.asfortranarray(a_c)  # F order: first axis varies fastest
assert a_c.strides == (32, 8) and a_f.strides == (8, 24)
assert a_c.flags.c_contiguous and a_f.flags.f_contiguous
assert np.array_equal(a_c, a_f)  # same values, different layout
assert np.array_equal(a_c.ravel(order="K"), np.arange(12))
assert not a_c[:, ::2].flags.c_contiguous  # strided slice breaks contiguity

big_c = rng.random((4000, 4000))
big_f = np.asfortranarray(big_c)
t_c_rows = bench(lambda: big_c.sum(axis=1))
t_c_cols = bench(lambda: big_c.sum(axis=0))
t_f_cols = bench(lambda: big_f.sum(axis=0))
show("C array, sum over last axis", f"{t_c_rows:.2e}")
show("C array, sum over first axis", f"{t_c_cols:.2e}")
show("F array, sum over first axis", f"{t_f_cols:.2e}")
# Honest takeaway: NumPy's reductions reorder their inner loops, so plain sums barely care about layout.
# Layout bites when YOU control the traversal: strided copies, transposes, and column-by-column Python loops.
t_copy_c = bench(lambda: big_c.T.copy(order="C"))   # strided read -> contiguous write
t_copy_same = bench(lambda: big_c.copy(order="C"))  # contiguous read -> contiguous write
show("copy of C array (s)", f"{t_copy_same:.2e}")
show("copy of its transpose (s)", f"{t_copy_c:.2e}")
assert t_copy_c > t_copy_same  # cache-unfriendly traversal costs real time

# %% [markdown]
# ## 6.2 Views, copies, in-place operations, `out=`

# %%
section("6.2 In-place semantics")
a = np.arange(5)
b = a  # alias, NOT a copy
b += 1
assert a[0] == 1 and a is b
c = a + 1  # new array
assert c is not a

x = np.arange(5)
try:
    x += 1.5  # in-place cannot change dtype: float result -> int array refused
    raise AssertionError
except TypeError:
    pass
x = x.astype(float)
x += 1.5  # fine

big = rng.random(2_000_000)
tmp = np.empty_like(big)


def with_temporaries():
    return big * 2.0 + 1.0  # allocates 2 temporaries


def with_out():
    np.multiply(big, 2.0, out=tmp)
    tmp.__iadd__(1.0)
    return tmp


assert np.allclose(with_temporaries(), with_out())
m_tmp, m_out = peak_mem(with_temporaries), peak_mem(with_out)
show("peak bytes, with temporaries", f"{m_tmp:,}")
show("peak bytes, with out=", f"{m_out:,}")
assert m_out < m_tmp / 2

# %% [markdown]
# ## 6.3 `as_strided` and `sliding_window_view`: windows with zero copying
# `as_strided` is unchecked: a wrong shape/strides reads arbitrary memory. Prefer `sliding_window_view`.

# %%
section("6.3 Stride tricks")
x = np.arange(10, dtype=np.float64)
w = 4
manual = as_strided(x, shape=(len(x) - w + 1, w), strides=(x.strides[0], x.strides[0]), writeable=False)
safe = sliding_window_view(x, w)
assert np.array_equal(manual, safe) and np.shares_memory(x, safe)
assert safe.shape == (7, 4) and safe.base is not None
assert safe.nbytes == 7 * 4 * 8 and x.nbytes == 80  # "virtual" size >> real memory

# Rolling mean three ways
y = rng.normal(size=100_000)
roll_view = sliding_window_view(y, 50).mean(axis=-1)
cs = np.cumsum(np.insert(y, 0, 0.0))
roll_cumsum = (cs[50:] - cs[:-50]) / 50
assert np.allclose(roll_view, roll_cumsum)
show("rolling mean via view (s)", f"{bench(lambda: sliding_window_view(y, 50).mean(axis=-1)):.2e}")
def rolling_cumsum(y, w=50):
    c = np.cumsum(np.insert(y, 0, 0.0))
    return (c[w:] - c[:-w]) / w


show("rolling mean via cumsum (s)", f"{bench(lambda: rolling_cumsum(y)):.2e}")

# Rolling std: windows make any statistic trivial
assert np.allclose(sliding_window_view(y, 50).std(axis=-1)[:3], [y[i : i + 50].std() for i in range(3)])

# %% [markdown]
# ## Mini-project: 2-D convolution and blur using strides only

# %%
section("6.4 Convolution via strides")


def correlate2d_valid(img, kernel):
    kh, kw = kernel.shape
    windows = sliding_window_view(img, (kh, kw))  # (H-kh+1, W-kw+1, kh, kw), zero-copy
    return np.einsum("ijkl,kl->ij", windows, kernel)


def convolve2d_valid(img, kernel):
    return correlate2d_valid(img, kernel[::-1, ::-1])  # convolution = correlation with flipped kernel


img = rng.random((256, 256))
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
assert np.allclose(correlate2d_valid(img, sobel_x), correlate2d(img, sobel_x, mode="valid"))

k = 7
box = np.full((k, k), 1.0 / k**2)
blur = correlate2d_valid(img, box)
assert blur.shape == (250, 250) and blur.std() < img.std() / 3  # blur reduces variance ~ 1/k^2

mem_view = peak_mem(lambda: sliding_window_view(img, (k, k)))
mem_materialised = peak_mem(lambda: np.ascontiguousarray(sliding_window_view(img, (k, k))))
show("peak bytes: view only", f"{mem_view:,}")
show("peak bytes: materialised windows", f"{mem_materialised:,}")
assert mem_view < mem_materialised / 20

# Edge detection on a synthetic image: a bright square must light up its border only
square = np.zeros((40, 40))
square[10:30, 10:30] = 1.0
edges = np.hypot(correlate2d_valid(square, sobel_x), correlate2d_valid(square, sobel_x.T))
assert edges.max() > 0 and edges[15:25, 15:25].max() == 0  # interior has no gradient

# %% [markdown]
# ## 6.5 Structured arrays, memmap, buffer protocol

# %%
section("6.5 Structured arrays, memmap, buffers")
dt = np.dtype([("name", "U10"), ("age", "i4"), ("weight", "f8")])
people = np.array([("Ada", 36, 60.5), ("Alan", 41, 72.0), ("Grace", 85, 58.1)], dtype=dt)
assert people["age"].tolist() == [36, 41, 85] and people.itemsize == 40 + 4 + 8
assert people[np.argsort(people["weight"])]["name"].tolist() == ["Grace", "Ada", "Alan"]
assert people[people["age"] > 40]["name"].tolist() == ["Alan", "Grace"]
people["age"] += 1  # field access is a view
assert people["age"][0] == 37

with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "big.dat"
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(10_000, 100))
    mm[:] = np.arange(10_000, dtype=np.float32)[:, None]
    mm.flush()
    del mm
    ro = np.memmap(path, dtype=np.float32, mode="r", shape=(10_000, 100))  # data stays on disk
    assert ro[1234, 5] == 1234 and path.stat().st_size == 10_000 * 100 * 4
    chunk_sum = sum(float(ro[i : i + 1000, 0].sum()) for i in range(0, 10_000, 1000))  # out-of-core style
    assert chunk_sum == sum(range(10_000))
    del ro

# Buffer protocol: share memory with bytes-like objects, zero-copy
buf = bytearray(8 * 4)
arr = np.frombuffer(buf, dtype=np.float64)
arr[0] = 3.5
assert np.frombuffer(buf, dtype=np.float64)[0] == 3.5 and np.shares_memory(arr, np.frombuffer(buf, dtype=np.float64))
mv = memoryview(np.arange(6, dtype=np.int32).reshape(2, 3))
assert mv.shape == (2, 3) and mv.format == "i" and mv.strides == (12, 4)
assert np.frombuffer(b"\x01\x00\x00\x00\x02\x00\x00\x00", dtype="<i4").tolist() == [1, 2]  # bytes -> read-only

# %% [markdown]
# ### Your turn
# 1. Implement max-pooling (2x2, stride 2) with `reshape` only (no `as_strided`).
# 2. Use `as_strided` to build a Toeplitz matrix view from a vector; why must it be read-only?
# 3. Profile `a.T.copy()` vs `np.ascontiguousarray(a.T)` for a 8000x8000 array and explain cache behaviour.

print("\nModule 6 complete: all assertions passed.")
