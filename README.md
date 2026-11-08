# NumPy: Zero to Expert (runnable course)

Ten modules, one self-checking Python file each. Every concept is **demonstrated, then asserted**,
so if a module finishes with `all assertions passed`, each claim in it was verified on your machine
with your NumPy version.

## Quick start (uv)

```bash
uv sync                                   # core + dev tools (pytest, hypothesis, jupyterlab, jupytext, ...)
uv run python -m numpy_course.m01_ndarray # run any module
uv run pytest                             # run all modules + unit + property tests (~1 min)
uv sync --extra numba                     # optional: unlocks stage 4 of Module 7
```

Prefer notebooks? Each module uses `# %%` cell markers, so:

```bash
uv run jupytext --to ipynb src/numpy_course/m01_ndarray.py
uv run jupyter lab
```

(VS Code and PyCharm also run `# %%` files as interactive cells directly.)

## Course map

| Week | Module | Core ideas | Capstone-style project inside |
|---|---|---|---|
| 1 | `m01_ndarray` | memory layout, dtypes, promotion, **views vs copies**, masks, reshaping | 4 worked indexing exercises |
| 2 | `m02_broadcasting` | ufuncs, broadcasting rule (re-implemented), reductions, `ufunc.at`, sorting/sets | loop-free pairwise distances (3 methods vs SciPy) |
| 3 | `m03_linalg` | products, `solve` vs `inv`, LU/QR/Cholesky/SVD/eig, floating-point pitfalls | Gram-Schmidt vs Householder; PCA via SVD |
| 4 | `m04_random` | `Generator`, `SeedSequence.spawn`, Monte Carlo, variance reduction | European, Asian (control variate, ~1000x variance cut), correlated basket |
| 5 | `m05_einsum` | einsum, `einsum_path`, advanced indexing, stacked linalg | batched multi-head attention; batched Kalman filter |
| 6 | `m06_memory` | strides, C/F order, `out=`, `sliding_window_view`, memmap, buffers | 2-D convolution / blur / edge detection via strides |
| 7 | `m07_performance` | cProfile, temporaries, float32, chunking, BLAS threads | 4-stage optimisation (Python → NumPy → low-memory → Numba) |
| 8 | `m08_numerics` | `numpy.polynomial`, FFT, quadrature orders, SDE solvers, masked arrays | heat equation: finite differences vs spectral; EM vs Milstein orders |
| 9 | `m09_extending` | structured arrays, `vectorize` myths, `__array_ufunc__`, `__array_function__`, DLPack | units-aware `Quantity` class |
| 10 | `m10_capstone` + `miniquant.py` | everything combined | Black-Scholes + Greeks, implied vol, MC engine, mean-variance optimiser |

## How to study each module (the weekly rhythm)

1. **Predict**: before running a block, write down what you expect each `assert` to say.
2. **Run and break it**: change a number, remove a `keepdims`, flip a dtype, and see which assert fires.
3. **Do "Your turn"**: three extension exercises at the bottom of every module.
4. **Test it yourself**: add your solutions under `tests/` with `np.testing.assert_allclose`.
5. Write one paragraph: *what surprised me?*

## Final exam

Solve the 100 problems of Nicolas Rougier's *100 NumPy exercises* (GitHub: `rougier/numpy-100`) in one
week, then extend `miniquant` with Heston and Longstaff-Schwartz American options, with tests.

## Notes on honesty of the numbers

Timing assertions use deliberately loose thresholds (a few x, not "100x") because speedups depend on
your CPU, BLAS build, and thread count. Where results differ from the textbook (for example a loop that
is only ~5x slower than vectorised code because `sin` dominates), the module says so rather than
hiding it.
