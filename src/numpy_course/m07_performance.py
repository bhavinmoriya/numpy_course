# %% [markdown]
# # Module 7: Profiling and optimisation
# Run:  `uv run python -m numpy_course.m07_performance`
# Rule #1: measure first. Rule #2: change one thing at a time. Rule #3: keep a correctness check.

# %%
import cProfile
import io
import math
import pstats
import time

import numpy as np

from numpy_course.utils import bench, peak_mem, section, show

rng = np.random.default_rng(0)

# %% [markdown]
# ## 7.1 Measuring: timeit, cProfile (and line_profiler)

# %%
section("7.1 Profiling")


def slow_pipeline(x):
    total = 0.0
    for v in x:  # <- the hotspot
        total += math.sqrt(v) * math.sin(v)
    return total


data = rng.random(200_000)
prof = cProfile.Profile()
prof.enable()
slow_pipeline(data)
prof.disable()
stream = io.StringIO()
pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(4)
assert "slow_pipeline" in stream.getvalue()
show("profile says hotspot is", "slow_pipeline (Python loop)")
print("  (line-by-line:  uv run kernprof -l -v your_script.py   with @profile on the function)")


def fast_pipeline(x):
    return float((np.sqrt(x) * np.sin(x)).sum())


assert np.isclose(slow_pipeline(data), fast_pipeline(data))  # same answer first, THEN compare speed
t_slow, t_fast = bench(lambda: slow_pipeline(data)), bench(lambda: fast_pipeline(data))
show("python loop (s)", f"{t_slow:.2e}")
show("vectorised (s)", f"{t_fast:.2e}")
# Transcendental functions dominate here, so the gap is modest; cheap arithmetic gives 50-100x.
assert t_slow > 2 * t_fast

# %% [markdown]
# ## 7.2 Temporaries, `out=`, dtype, and chunking

# %%
section("7.2 Cheap wins")
x = rng.random(3_000_000)


def expr_naive():
    return np.sqrt(x * 2.0 + 1.0) * 3.0 - 1.0  # every step allocates a fresh full-size array


def expr_inplace():
    out = x * 2.0  # one allocation...
    out += 1.0  # ...then everything reuses it
    np.sqrt(out, out=out)
    out *= 3.0
    out -= 1.0
    return out


assert np.allclose(expr_naive(), expr_inplace())
show("naive temporaries (s)", f"{bench(expr_naive):.2e}")
show("in-place / out= (s)", f"{bench(expr_inplace):.2e}")
show("peak bytes naive", f"{peak_mem(expr_naive):,}")
show("peak bytes in-place", f"{peak_mem(expr_inplace):,}")
assert peak_mem(expr_inplace) < 0.6 * peak_mem(expr_naive)
# (NumPy also elides some temporaries automatically in simple expressions; never assume, measure.)

# float32 halves memory and (for memory-bound work) roughly halves time. Mind the precision!
x64, x32 = rng.random(5_000_000), None
x32 = x64.astype(np.float32)
assert x32.nbytes == x64.nbytes // 2
show("float64 exp (s)", f"{bench(lambda: np.exp(x64)):.2e}")
show("float32 exp (s)", f"{bench(lambda: np.exp(x32)):.2e}")
assert abs(float(np.exp(x32).sum(dtype=np.float64)) - float(np.exp(x64).sum())) / float(np.exp(x64).sum()) < 1e-6

# Chunking: bound peak memory on huge inputs without changing the result
big = rng.random(4_000_000)


def chunked_sqrt_sum(arr, chunk=250_000):
    s = 0.0
    for i in range(0, len(arr), chunk):
        s += float(np.sqrt(arr[i : i + chunk]).sum())
    return s


assert np.isclose(chunked_sqrt_sum(big), np.sqrt(big).sum())
assert peak_mem(lambda: chunked_sqrt_sum(big)) < peak_mem(lambda: np.sqrt(big).sum()) / 5

# %% [markdown]
# ## 7.3 BLAS threads: why your benchmarks may lie
# `@`, `solve`, `svd`... run multithreaded BLAS/LAPACK. Timings change with core count and
# competing processes. Control with env vars (OPENBLAS_NUM_THREADS / MKL_NUM_THREADS) or threadpoolctl.

# %%
section("7.3 BLAS threading")
try:
    from threadpoolctl import threadpool_info

    info = threadpool_info()
    show("BLAS pools", [(p["internal_api"], p["num_threads"]) for p in info])
except ImportError:  # optional: uv add threadpoolctl
    show("threadpoolctl", "not installed (optional)")
M = rng.normal(size=(1500, 1500))
show("1500x1500 matmul (s)", f"{bench(lambda: M @ M):.2e}")

# %% [markdown]
# ## 7.4 Project: arithmetic Asian option, four stages of optimisation
# Same random numbers in every stage, so correctness is checked before speed is compared.
# Z has shape (steps, paths): each time-step row is contiguous.

# %%
section("7.4 Four-stage optimisation")
S0, K, r, sigma, T, STEPS = 100.0, 100.0, 0.03, 0.2, 1.0, 50


def asian_python(Z, S0=S0, K=K, r=r, sigma=sigma, T=T):
    """Stage 1: pure Python."""
    steps, paths = len(Z), len(Z[0])
    dt = T / steps
    drift, vol = (r - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
    total = 0.0
    for i in range(paths):
        s, acc = S0, 0.0
        for t in range(steps):
            s *= math.exp(drift + vol * Z[t][i])
            acc += s
        total += max(acc / steps - K, 0.0)
    return math.exp(-r * T) * total / paths


def asian_numpy(Z, S0=S0, K=K, r=r, sigma=sigma, T=T):
    """Stage 2: vectorised, but materialises the full (steps, paths) price matrix."""
    steps = Z.shape[0]
    dt = T / steps
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=0))
    return float(np.exp(-r * T) * np.maximum(S.mean(axis=0) - K, 0).mean())


def asian_lowmem(seed, paths, steps=STEPS, S0=S0, K=K, r=r, sigma=sigma, T=T):
    """Stage 3: O(paths) memory. Generate one time-step at a time, keep running sums only."""
    g = np.random.default_rng(seed)
    dt = T / steps
    drift, vol = (r - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    s = np.full(paths, S0)
    acc = np.zeros(paths)
    for _ in range(steps):
        s *= np.exp(drift + vol * g.standard_normal(paths))
        acc += s
    return float(np.exp(-r * T) * np.maximum(acc / steps - K, 0).mean())


try:  # Stage 4: Numba (optional extra:  uv sync --extra numba)
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _asian_nb(Z, S0, K, r, sigma, T):
        steps, paths = Z.shape
        dt = T / steps
        drift, vol = (r - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
        total = 0.0
        for i in range(paths):
            s, acc = S0, 0.0
            for t in range(steps):
                s *= math.exp(drift + vol * Z[t, i])
                acc += s
            total += max(acc / steps - K, 0.0)
        return math.exp(-r * T) * total / paths

    def asian_numba(Z):
        return _asian_nb(Z, S0, K, r, sigma, T)

    HAVE_NUMBA = True
except ImportError:
    asian_numba = None
    HAVE_NUMBA = False

# --- correctness: all stages agree on identical randoms
paths_small, SEED = 2_000, 99
Z_small = np.random.default_rng(SEED).standard_normal((STEPS, paths_small))
p1 = asian_python(Z_small.tolist())
p2 = asian_numpy(Z_small)
p3 = asian_lowmem(SEED, paths_small)
assert np.isclose(p1, p2, rtol=1e-9) and np.isclose(p2, p3, rtol=1e-9)
if HAVE_NUMBA:
    assert np.isclose(asian_numba(Z_small), p2, rtol=1e-9)

# --- speed (stage 1 timed on few paths, others on many; compare per-path cost)
t1 = bench(lambda: asian_python(Z_small.tolist()), repeat=1) / paths_small
PATHS = 200_000
Z_big = np.random.default_rng(SEED).standard_normal((STEPS, PATHS))
t2 = bench(lambda: asian_numpy(Z_big), repeat=2) / PATHS
t3 = bench(lambda: asian_lowmem(SEED, PATHS), repeat=2) / PATHS
rows = [("1 pure Python", t1), ("2 vectorised NumPy", t2), ("3 low-memory NumPy", t3)]
if HAVE_NUMBA:
    asian_numba(Z_small)  # warm up / compile
    rows.append(("4 Numba @njit", bench(lambda: asian_numba(Z_big), repeat=2) / PATHS))
print(f"  {'stage':<22}{'sec/path':>12}{'speedup vs 1':>16}")
for name, t in rows:
    print(f"  {name:<22}{t:>12.2e}{t1 / t:>15.0f}x")
show("peak MB: stage 2", f"{peak_mem(lambda: asian_numpy(Z_big)) / 1e6:.0f}")
show("peak MB: stage 3", f"{peak_mem(lambda: asian_lowmem(SEED, PATHS)) / 1e6:.0f}")
# Pure-Python scalar loops are not THAT slow for 50 steps; the real gaps appear with Numba or larger problems.
assert t2 < t1 / 2
assert peak_mem(lambda: asian_lowmem(SEED, PATHS)) < peak_mem(lambda: asian_numpy(Z_big)) / 10
if not HAVE_NUMBA:
    print("  (Numba not installed: run `uv sync --extra numba` to see stage 4)")

# %% [markdown]
# ### Your turn
# 1. Add `@profile` and run `kernprof -l -v` on `asian_python`; confirm the inner loop dominates.
# 2. Try `scipy.sparse` for a 100k x 100k tridiagonal matrix-vector product; compare with dense memory needs.
# 3. Pre-allocate with `np.empty` + fill vs `np.append` in a loop; measure the O(n^2) vs O(n) difference.

print("\nModule 7 complete: all assertions passed.")
