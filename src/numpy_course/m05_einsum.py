# %% [markdown]
# # Module 5: einsum and advanced indexing
# Run:  `uv run python -m numpy_course.m05_einsum`

# %%
import numpy as np

from numpy_course.utils import bench, section, show

rng = np.random.default_rng(0)

# %% [markdown]
# ## 5.1 Einstein summation
# Rule: letters repeated across inputs are multiplied; letters absent from the output are summed.

# %%
section("5.1 einsum basics")
A, B = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
v, w = rng.normal(size=5), rng.normal(size=5)
S = rng.normal(size=(4, 4))

assert np.allclose(np.einsum("ij,jk->ik", A, B), A @ B)  # matmul
assert np.isclose(np.einsum("i,i->", v, w), v @ w)  # dot
assert np.allclose(np.einsum("i,j->ij", v, w), np.outer(v, w))  # outer
assert np.isclose(np.einsum("ii->", S), np.trace(S))  # trace
assert np.allclose(np.einsum("ii->i", S), np.diag(S))  # diagonal
assert np.allclose(np.einsum("ij->ji", A), A.T)  # transpose
assert np.allclose(np.einsum("ij->i", A), A.sum(axis=1))  # row sums
assert np.allclose(np.einsum("ij,ij->", A, A), (A * A).sum())  # Frobenius^2
assert np.allclose(np.einsum("ij,ij->i", A, A), (A * A).sum(axis=1))  # row-wise dot products

# Batched ops: a leading batch letter is just carried along
Ab, Bb = rng.normal(size=(7, 4, 5)), rng.normal(size=(7, 5, 3))
assert np.allclose(np.einsum("bij,bjk->bik", Ab, Bb), Ab @ Bb)

# Bilinear form x^T M x for a batch of vectors, in one shot
M = rng.normal(size=(5, 5))
X = rng.normal(size=(100, 5))
assert np.allclose(np.einsum("bi,ij,bj->b", X, M, X), np.array([x @ M @ x for x in X]))

# Ellipsis: work on the last axes regardless of leading dims
assert np.allclose(np.einsum("...ij,...jk->...ik", Ab, Bb), Ab @ Bb)

# %% [markdown]
# ## 5.2 `optimize=True` and `einsum_path`
# Multi-operand contractions can be astronomically cheaper when ordered well.

# %%
section("5.2 Contraction order")
a, b, c = rng.normal(size=(200, 30)), rng.normal(size=(30, 200)), rng.normal(size=(200, 30))
path, info = np.einsum_path("ij,jk,kl->il", a, b, c, optimize="optimal")
t_naive = bench(lambda: np.einsum("ij,jk,kl->il", a, b, c, optimize=False))
t_opt = bench(lambda: np.einsum("ij,jk,kl->il", a, b, c, optimize=True))
assert np.allclose(np.einsum("ij,jk,kl->il", a, b, c, optimize=False), (a @ b) @ c)
show("naive einsum (s)", f"{t_naive:.2e}")
show("optimized einsum (s)", f"{t_opt:.2e}")
show("chosen path", path)
assert t_opt < t_naive

# %% [markdown]
# ## 5.3 Batched attention: softmax(Q K^T / sqrt(d)) V, using only einsum

# %%
section("5.3 Multi-head attention")


def attention(Q, K, V):
    """Shapes: Q,K,V = (batch, heads, seq, d_head)."""
    d = Q.shape[-1]
    scores = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)  # stable softmax
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, V), weights


Bsz, H, Tn, D = 2, 4, 6, 8
Q, K, V = (rng.normal(size=(Bsz, H, Tn, D)) for _ in range(3))
out, wts = attention(Q, K, V)
assert out.shape == (Bsz, H, Tn, D) and np.allclose(wts.sum(-1), 1.0)

# Verify one (batch, head) against a plain 2-D implementation
q, k, vv = Q[1, 2], K[1, 2], V[1, 2]
s = q @ k.T / np.sqrt(D)
p = np.exp(s - s.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
assert np.allclose(out[1, 2], p @ vv)

# Causal mask: position t may only attend to positions <= t
mask = np.triu(np.ones((Tn, Tn), dtype=bool), k=1)
scores = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(D)
scores = np.where(mask, -np.inf, scores)
cw = np.exp(scores - scores.max(-1, keepdims=True))
cw /= cw.sum(-1, keepdims=True)
assert np.allclose(np.triu(cw[0, 0], k=1), 0) and np.allclose(cw.sum(-1), 1)

# %% [markdown]
# ## 5.4 Advanced indexing toolbox

# %%
section("5.4 Advanced indexing")
a = np.array([[30, 10, 20], [3, 1, 2], [9, 7, 8]])

# take / put
assert np.array_equal(np.take(a, [0, 2], axis=1), a[:, [0, 2]])
t = np.zeros(5)
np.put(t, [1, 3], [10, 30])
assert np.array_equal(t, [0, 10, 0, 30, 0])

# take_along_axis: apply per-row argsort
idx = np.argsort(a, axis=1)
assert np.array_equal(np.take_along_axis(a, idx, axis=1), np.sort(a, axis=1))
top1 = np.take_along_axis(a, a.argmax(axis=1)[:, None], axis=1)
assert np.array_equal(top1[:, 0], a.max(axis=1))

# Two index arrays pair up element-wise; ix_ gives the outer (cross) product of indices
assert np.array_equal(a[[0, 2], [1, 2]], [10, 8])  # elements (0,1) and (2,2)
assert np.array_equal(a[np.ix_([0, 2], [1, 2])], [[10, 20], [7, 8]])  # 2x2 sub-matrix

# Mixed basic + fancy: where the fancy axes land can surprise you
T = np.zeros((2, 3, 4, 5))
# fancy indices separated by a slice: the broadcast fancy dim goes FIRST, then the sliced axes
assert T[:, [0, 1], :, [0, 1]].shape == (2, 2, 4)
assert T[[0, 1], :, :, [0, 1]].shape == (2, 3, 4)
assert T[0, :, :, 0].shape == (3, 4)

# choose / select (vectorised multi-way if/else)
x = np.array([-5, 0, 3, 12])
assert np.array_equal(np.choose([0, 1, 1, 0], [[10, 11, 12, 13], [20, 21, 22, 23]]), [10, 21, 22, 13])
cat = np.select([x < 0, x == 0, x < 10], ["neg", "zero", "small"], default="big")
assert cat.tolist() == ["neg", "zero", "small", "big"]
assert np.array_equal(np.clip(x, 0, 5), [0, 0, 3, 5])

# Boolean-mask assignment with broadcasting
m = np.arange(12).reshape(3, 4)
m[m % 2 == 0] = -1
assert (m[m < 0] == -1).all()

# %% [markdown]
# ## 5.5 Batched linear algebra on stacks of matrices

# %%
section("5.5 Stacked linalg")
stack = rng.normal(size=(100, 3, 3)) + 3 * np.eye(3)  # (..., m, n)
rhs = rng.normal(size=(100, 3, 1))  # column vectors: (..., m, 1)
inv = np.linalg.inv(stack)
assert np.allclose(inv @ stack, np.eye(3))
sol = np.linalg.solve(stack, rhs)
assert np.allclose(stack @ sol, rhs)
dets = np.linalg.det(stack)
assert dets.shape == (100,) and np.isclose(dets[0], np.linalg.det(stack[0]))
sym = stack + stack.transpose(0, 2, 1)
ev = np.linalg.eigvalsh(sym)
assert ev.shape == (100, 3) and np.all(np.diff(ev, axis=1) >= 0)

# %% [markdown]
# ## Exercise: a Kalman filter step over many particles at once

# %%
section("5.6 Batched Kalman filter")


def kalman_step(x, P, z, F, Q, H, R):
    """x:(p,n) state, P:(p,n,n) cov, z:(p,m) measurement. All particles in one call."""
    x_pred = x @ F.T
    P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T + R  # (p,m,m)
    K = P_pred @ H.T @ np.linalg.inv(S)  # (p,n,m)
    innov = z - x_pred @ H.T
    x_new = x_pred + np.einsum("pnm,pm->pn", K, innov)
    P_new = P_pred - K @ H @ P_pred
    return x_new, P_new


def kalman_step_single(x, P, z, F, Q, H, R):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    return x_pred + K @ (z - H @ x_pred), P_pred - K @ H @ P_pred


n_, m_, p_ = 4, 2, 500
F = np.eye(n_) + 0.1 * np.eye(n_, k=2)  # constant-velocity style
Qn, Rn = 0.01 * np.eye(n_), 0.1 * np.eye(m_)
H = np.eye(m_, n_)
x0 = rng.normal(size=(p_, n_))
P0 = np.broadcast_to(np.eye(n_), (p_, n_, n_)).copy()
z = rng.normal(size=(p_, m_))
xb, Pb = kalman_step(x0, P0, z, F, Qn, H, Rn)
for i in (0, 123, 499):
    xs, Ps = kalman_step_single(x0[i], P0[i], z[i], F, Qn, H, Rn)
    assert np.allclose(xb[i], xs) and np.allclose(Pb[i], Ps)
t_batch = bench(lambda: kalman_step(x0, P0, z, F, Qn, H, Rn))
t_loop = bench(lambda: [kalman_step_single(x0[i], P0[i], z[i], F, Qn, H, Rn) for i in range(p_)])
show("batched (s)", f"{t_batch:.2e}")
show("python loop (s)", f"{t_loop:.2e}")
assert t_batch < t_loop

# %% [markdown]
# ### Your turn
# 1. Express a 2-D convolution of a batch of images as one einsum over a `sliding_window_view` (see Module 6).
# 2. Write `einsum` strings for: Mahalanobis distance of each row, covariance of a (n, d) data matrix, tensor train contraction.
# 3. Implement top-k per row using `argpartition` + `take_along_axis`.

print("\nModule 5 complete: all assertions passed.")
