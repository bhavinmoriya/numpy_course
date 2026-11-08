# %% [markdown]
# # Module 2: Vectorization and Broadcasting
# Run:  `uv run python -m numpy_course.m02_broadcasting`

# %%
import numpy as np
from scipy.spatial.distance import cdist

from numpy_course.utils import bench, section, show

# %% [markdown]
# ## 2.1 Universal functions (ufuncs)
# Element-wise, compiled, and they accept `out=`, `where=`, `dtype=`.

# %%
section("2.1 ufuncs")
x = np.linspace(0, np.pi, 5)
assert np.allclose(np.sin(x) ** 2 + np.cos(x) ** 2, 1.0)
assert np.allclose(np.exp(np.log([1.0, 2.0, 3.0])), [1, 2, 3])
assert np.array_equal(np.maximum([1, 5, 3], [4, 2, 6]), [4, 5, 6])  # element-wise max (not reduction!)
# where= computes only where True; leave the rest untouched via out=
out = np.full(4, -1.0)
np.sqrt(np.array([4.0, -1.0, 9.0, 16.0]), out=out, where=np.array([True, False, True, True]))
assert np.array_equal(out, [2.0, -1.0, 3.0, 4.0])
# Comparisons are ufuncs too; NaN is never equal to anything
assert np.isnan(np.array([1.0, np.nan])).tolist() == [False, True]
assert (np.nan == np.nan) is False

# %% [markdown]
# ## 2.2 Broadcasting: the rule
# Align shapes on the RIGHT. Two dims are compatible if equal or one of them is 1.
# Missing leading dims are treated as 1. Size-1 dims are stretched (no copy is made).

# %%
section("2.2 Broadcasting rule")


def broadcast_shape(*shapes):
    """Re-implementation of np.broadcast_shapes, to prove you know the rule."""
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    result = []
    for dims in zip(*padded):
        non_one = {d for d in dims if d != 1}
        if len(non_one) > 1:
            raise ValueError(f"incompatible dims {dims}")
        result.append(non_one.pop() if non_one else 1)
    return tuple(result)


cases = [((3,), (3,)), ((5, 1), (1, 4)), ((8, 1, 6, 1), (7, 1, 5)), ((2, 1, 3), (4, 1)), ((1,), (9, 9))]
for s1, s2 in cases:
    assert broadcast_shape(s1, s2) == np.broadcast_shapes(s1, s2)
    show(f"{s1} + {s2}", broadcast_shape(s1, s2))

for bad in [((3,), (4,)), ((2, 3), (3, 2))]:
    for fn in (broadcast_shape, np.broadcast_shapes):
        try:
            fn(*bad)
            raise AssertionError("should have failed")
        except ValueError:
            pass

# Classic idioms
col = np.arange(3)[:, None]  # (3,1)
row = np.arange(4)[None, :]  # (1,4)
assert (col + row).shape == (3, 4)  # outer sum
table = col * row  # multiplication table
assert table[2, 3] == 6

# Stretching makes no copy: the stretched axis has stride 0
b = np.broadcast_to(np.arange(4), (1000, 4))
assert b.strides[0] == 0 and b.size == 4000

# Normalise columns / rows: keepdims makes the shapes line up
X = np.random.default_rng(0).normal(size=(6, 4))
Xc = X - X.mean(axis=0)  # (6,4) - (4,)   : broadcasts
Xr = X - X.mean(axis=1, keepdims=True)  # (6,4) - (6,1) : keepdims REQUIRED
assert np.allclose(Xc.mean(axis=0), 0) and np.allclose(Xr.mean(axis=1), 0)
try:
    X - X.mean(axis=1)  # (6,4) - (6,) -> error
    raise AssertionError
except ValueError:
    pass

# %% [markdown]
# ## 2.3 Reductions and accumulations

# %%
section("2.3 Reductions")
a = np.arange(1, 13).reshape(3, 4)
assert a.sum() == 78
assert np.array_equal(a.sum(axis=0), [15, 18, 21, 24])  # collapse axis 0 (rows) -> one value per column
assert np.array_equal(a.sum(axis=1), [10, 26, 42])
assert a.sum(axis=(0, 1)) == 78
assert np.array_equal(a.cumsum(axis=1)[:, -1], a.sum(axis=1))
assert a.argmax() == 11 and np.unravel_index(a.argmax(), a.shape) == (2, 3)
assert np.array_equal(a.argmax(axis=0), [2, 2, 2, 2])
assert a.mean() == 6.5 and np.isclose(a.std(), np.sqrt(((a - 6.5) ** 2).mean()))  # ddof=0 by default!
assert np.isclose(a.std(ddof=1), np.sqrt(((a - 6.5) ** 2).sum() / 11))
assert a.sum(axis=1, keepdims=True).shape == (3, 1)
assert np.diff([1, 4, 9, 16]).tolist() == [3, 5, 7]

# %% [markdown]
# ## 2.4 ufunc methods: reduce, accumulate, outer, at

# %%
section("2.4 ufunc methods")
v = np.array([1, 2, 3, 4])
assert np.add.reduce(v) == v.sum() == 10
assert np.multiply.reduce(v) == 24
assert np.array_equal(np.add.accumulate(v), np.cumsum(v))
assert np.array_equal(np.maximum.accumulate([1, 3, 2, 5, 4]), [1, 3, 3, 5, 5])  # running max
assert np.array_equal(np.multiply.outer([1, 2], [10, 20, 30]), [[10, 20, 30], [20, 40, 60]])
assert np.logical_and.reduce([True, True, False]) == False  # noqa: E712

# Pitfall: repeated indices with `+=` are NOT accumulated (buffered fancy-index read-modify-write).
idx = np.array([0, 0, 1])
wrong = np.zeros(3)
wrong[idx] += 1
right = np.zeros(3)
np.add.at(right, idx, 1)
assert np.array_equal(wrong, [1, 1, 0]) and np.array_equal(right, [2, 1, 0])
assert np.array_equal(np.bincount(idx, minlength=3), [2, 1, 0])  # much faster for counting

# %% [markdown]
# ## 2.5 Sorting, searching, sets

# %%
section("2.5 Sorting and searching")
x = np.array([30, 10, 50, 20, 40])
order = np.argsort(x)
assert np.array_equal(x[order], np.sort(x))
assert np.array_equal(np.argsort(order), [2, 0, 4, 1, 3])  # argsort of argsort = ranks
s = np.sort(x)
assert np.searchsorted(s, 35) == 3  # insertion point in a SORTED array
assert np.array_equal(np.searchsorted(s, [5, 10, 100]), [0, 0, 5])
vals, counts = np.unique([3, 1, 3, 2, 3, 1], return_counts=True)
assert vals.tolist() == [1, 2, 3] and counts.tolist() == [2, 1, 3]
k = 2
assert np.partition(x, k)[k] == np.sort(x)[k]  # O(n) selection of k-th smallest
assert np.array_equal(np.sort(np.partition(x, 2)[:3]), [10, 20, 30])  # 3 smallest, unordered then sorted
assert np.isin([1, 5, 3], [3, 4, 5]).tolist() == [False, True, True]
assert np.intersect1d([1, 2, 3, 4], [3, 4, 5]).tolist() == [3, 4]
assert np.union1d([1, 2], [2, 3]).tolist() == [1, 2, 3]
assert np.setdiff1d([1, 2, 3, 4], [2, 4]).tolist() == [1, 3]
# lexsort: sort by last key first
names_age = np.lexsort((np.array([30, 25, 30, 25]), np.array([1, 1, 0, 0])))
assert names_age.tolist() == [3, 2, 1, 0]

# %% [markdown]
# ## Mini-project: pairwise distances with ZERO Python loops

# %%
section("Mini-project: pairwise distance matrix")


def pdist_loop(X):
    n = len(X)
    D = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())
    return D


def pdist_broadcast(X):
    """(n,1,d) - (1,n,d) -> (n,n,d): simple but allocates n*n*d floats."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def pdist_gram(X):
    """||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y : O(n^2) memory, one BLAS matmul."""
    sq = (X**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)  # round-off can produce tiny negatives
    return np.sqrt(d2)


X = np.random.default_rng(42).normal(size=(200, 5))
ref = cdist(X, X)
for f in (pdist_loop, pdist_broadcast, pdist_gram):
    assert np.allclose(f(X), ref, atol=1e-7), f.__name__

for f in (pdist_loop, pdist_broadcast, pdist_gram):
    show(f.__name__, f"{bench(lambda f=f: f(X)):.2e} s")
show("scipy cdist", f"{bench(lambda: cdist(X, X)):.2e} s")
assert bench(lambda: pdist_broadcast(X)) < bench(lambda: pdist_loop(X))

# %% [markdown]
# ### Your turn
# 1. Predict the shape of `np.ones((5,1,4)) + np.ones((3,1))` before running it.
# 2. Make `pdist_broadcast` memory-safe for n=50_000 by processing row-blocks.
# 3. Implement one-hot encoding of integer labels with broadcasting: `labels[:, None] == np.arange(k)`.

print("\nModule 2 complete: all assertions passed.")
