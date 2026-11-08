# %% [markdown]
# # Module 1: The ndarray
# Run:  `uv run python -m numpy_course.m01_ndarray`
# Every concept is demonstrated, then *asserted*, so a silent pass means you understood it.

# %%
import sys

import numpy as np

from numpy_course.utils import bench, section, show

# %% [markdown]
# ## 1.1 Why NumPy is fast
# A Python list stores pointers to boxed objects scattered in memory.
# An ndarray stores raw, typed values in ONE contiguous block, and loops run in C.

# %%
section("1.1 Why NumPy is fast")
n = 1_000_000
py_list = list(range(n))
arr = np.arange(n)

show("list object overhead (bytes)", sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:1000]) * (n // 1000))
show("ndarray data (bytes)", arr.nbytes)

t_list = bench(lambda: [x * 2 for x in py_list])
t_arr = bench(lambda: arr * 2)
show("list comprehension x*2 (s)", f"{t_list:.2e}")
show("ndarray arr*2 (s)", f"{t_arr:.2e}")
show("speedup", f"{t_list / t_arr:.0f}x")
assert t_arr < t_list

# %% [markdown]
# ## 1.2 Creating arrays

# %%
section("1.2 Creating arrays")
a = np.array([[1, 2, 3], [4, 5, 6]])
assert np.array_equal(np.zeros((2, 3)), np.full((2, 3), 0.0))
assert np.array_equal(np.arange(0, 10, 2), [0, 2, 4, 6, 8])
# linspace includes the endpoint (arange does not): prefer linspace for floats.
assert np.linspace(0, 1, 5)[-1] == 1.0
assert len(np.arange(0, 1, 0.1)) == 10
assert np.array_equal(np.eye(3), np.diag([1.0, 1.0, 1.0]))
e = np.empty(3)  # uninitialised memory: contents are garbage, never rely on them
assert e.shape == (3,)
show("arange(0,10,2)", np.arange(0, 10, 2))
show("linspace(0,1,5)", np.linspace(0, 1, 5))

# %% [markdown]
# ## 1.3 Attributes: shape, ndim, size, dtype, strides, flags
# `strides` = bytes to step along each axis. It is the key to views, transposes, and Module 6.

# %%
section("1.3 Attributes")
a = np.arange(12, dtype=np.int64).reshape(3, 4)
assert (a.shape, a.ndim, a.size) == ((3, 4), 2, 12)
assert a.dtype == np.int64 and a.itemsize == 8
assert a.strides == (32, 8)  # next row = 4*8 bytes, next column = 8 bytes
assert a.nbytes == a.size * a.itemsize
assert a.flags["C_CONTIGUOUS"] and not a.flags["F_CONTIGUOUS"]
# Transposing changes only shape+strides, not the data:
assert a.T.strides == (8, 32) and np.shares_memory(a, a.T)
show("a.strides", a.strides)
show("a.T.strides", a.T.strides)

# %% [markdown]
# ## 1.4 Data types: casting, overflow, promotion

# %%
section("1.4 dtypes")
x = np.array([1.7, -1.7, 2.5])
assert np.array_equal(x.astype(np.int64), [1, -1, 2])  # truncates toward zero, not rounds
assert np.array_equal(np.round(x), [2.0, -2.0, 2.0])  # banker's rounding: 2.5 -> 2

# Integer overflow wraps SILENTLY in arrays:
u = np.array([250], dtype=np.uint8) + 10
assert u[0] == 4
# ...so choose dtypes deliberately:
assert (np.array([250], dtype=np.int64) + 10)[0] == 260

# Promotion (NumPy 2, NEP 50): Python scalars are "weak", array dtypes win.
f32 = np.array([1, 2], dtype=np.float32)
assert (f32 + 3.0).dtype == np.float32
assert (f32 + np.float64(3.0)).dtype == np.float64  # NumPy scalars are NOT weak
assert np.result_type(np.int8, np.uint8) == np.int16
assert np.result_type(np.float32, np.int64) == np.float64
assert np.result_type(np.complex64, np.float64) == np.complex128

# Floating point basics
assert 0.1 + 0.2 != 0.3 and np.isclose(0.1 + 0.2, 0.3)

# datetime64 does calendar arithmetic
d = np.datetime64("2026-11-08") + np.timedelta64(30, "D")  # always give a unit
assert str(d) == "2026-12-08"
assert np.datetime64("2026-11-08") - np.datetime64("2026-01-01") == np.timedelta64(311, "D")
show("uint8 250+10", u[0])
show("result_type(float32,int64)", np.result_type(np.float32, np.int64))

# %% [markdown]
# ## 1.5 Indexing: views vs copies  (THE most important rule in NumPy)
# * basic slicing (`a[1:4]`, `a[::2]`, `a[..., 0]`)  -> **view** (shares memory)
# * fancy / boolean indexing (`a[[1,2]]`, `a[a>3]`)  -> **copy**

# %%
section("1.5 Views vs copies")


def is_view(base: np.ndarray, result: np.ndarray) -> bool:
    """Checkpoint helper: does `result` share memory with `base`?"""
    return np.shares_memory(base, result)


a = np.arange(10)
v = a[2:5]
assert is_view(a, v) and v.base is a
v[0] = 99
assert a[2] == 99  # writing through the view changed the original!

b = np.arange(10)
assert not is_view(b, b[[2, 3, 4]])  # fancy -> copy
assert not is_view(b, b[b > 5])  # boolean -> copy
assert is_view(b, b[::2])  # strided slice -> view
assert is_view(b, b[...])  # Ellipsis -> view
assert is_view(b, b.reshape(2, 5))  # reshape -> view when possible
assert is_view(b, b.T) and is_view(b, b.ravel())
assert not is_view(b, b.flatten())  # flatten ALWAYS copies
assert not is_view(b, b.copy())

# Assignment through fancy indexing writes into the original (it's a setitem, not a get):
b[[0, 1]] = -1
assert np.array_equal(b[:2], [-1, -1])

# reshape cannot always return a view: non-contiguous data must be copied
m = np.arange(6).reshape(2, 3)
assert is_view(m, m.reshape(3, 2))
assert not is_view(m, m.T.reshape(6))  # transposed layout can't be flattened without copying
show("view then write changes base", True)

# %% [markdown]
# ## 1.6 Boolean masks, where, nonzero

# %%
section("1.6 Masks")
x = np.array([3, -1, 4, -1, 5, -9, 2, 6])
mask = x > 0
assert np.array_equal(x[mask], [3, 4, 5, 2, 6])
assert np.array_equal(np.where(x > 0, x, 0), [3, 0, 4, 0, 5, 0, 2, 6])  # vectorised if/else
assert np.array_equal(np.nonzero(mask)[0], [0, 2, 4, 6, 7])
assert np.array_equal(np.flatnonzero(x < 0), [1, 3, 5])
assert ((x > 0) & (x < 5)).sum() == 3  # use & | ~  (NOT and/or/not), and parenthesise
assert np.any(x > 5) and not np.all(x > 0)
x_clip = x.copy()
x_clip[x_clip < 0] = 0  # masked assignment
assert x_clip.min() == 0

# %% [markdown]
# ## 1.7 Reshaping toolkit

# %%
section("1.7 Reshaping")
a = np.arange(24)
b = a.reshape(2, 3, 4)
assert a.reshape(4, -1).shape == (4, 6)  # -1 = infer
assert b.transpose(2, 0, 1).shape == (4, 2, 3)
assert b.swapaxes(0, 2).shape == (4, 3, 2)
assert a[:, None].shape == (24, 1) and a[None, :].shape == (1, 24)  # newaxis
assert np.squeeze(np.zeros((1, 3, 1))).shape == (3,)
assert np.moveaxis(b, 0, -1).shape == (3, 4, 2)
assert np.concatenate([b, b], axis=0).shape == (4, 3, 4)
assert np.stack([a, a]).shape == (2, 24) and np.vstack([a, a]).shape == (2, 24)
assert [p.shape for p in np.split(a, 4)] == [(6,)] * 4

# %% [markdown]
# ## Exercises with worked solutions

# %%
section("Exercises")
# E1: checkerboard of an 8x8 array, no loops
cb = np.zeros((8, 8), dtype=int)
cb[1::2, ::2] = 1
cb[::2, 1::2] = 1
assert np.array_equal(cb, np.indices((8, 8)).sum(axis=0) % 2)

# E2: flip / rotate without loops
m = np.arange(9).reshape(3, 3)
assert np.array_equal(m[::-1], np.flipud(m)) and np.array_equal(m[:, ::-1], np.fliplr(m))
assert np.array_equal(np.rot90(m), m.T[::-1])

# E3: every second row, columns 1..2, only where value is even
big = np.arange(30).reshape(6, 5)
sub = big[::2, 1:3]
assert np.array_equal(sub[sub % 2 == 0], [2, 12, 22])

# E4: second-largest value per row
r = np.array([[5, 1, 9], [4, 8, 2]])
assert np.array_equal(np.sort(r, axis=1)[:, -2], [5, 4])

# %% [markdown]
# ### Your turn
# 1. Write 30 more indexing puzzles; for each, predict "view or copy?" before checking `np.shares_memory`.
# 2. Using only `strides` arithmetic, compute the byte offset of `a[i, j, k]` and verify with `a.ctypes.data`.
# 3. Find an expression where `a.reshape(...)` silently copies. Why?

print("\nModule 1 complete: all assertions passed.")
