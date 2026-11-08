# %% [markdown]
# # Module 3: Linear algebra with np.linalg
# Run:  `uv run python -m numpy_course.m03_linalg`

# %%
import numpy as np
import scipy.linalg as sla

from numpy_course.utils import section, show

rng = np.random.default_rng(0)

# %% [markdown]
# ## 3.1 Products: `@`, dot, inner, outer, tensordot, kron

# %%
section("3.1 Products")
A = rng.normal(size=(3, 4))
B = rng.normal(size=(4, 2))
u, v = rng.normal(size=3), rng.normal(size=3)
assert np.allclose(A @ B, np.matmul(A, B)) and np.allclose(A @ B, np.dot(A, B))
assert np.allclose(A @ B, np.einsum("ij,jk->ik", A, B))
assert np.isclose(u @ v, np.inner(u, v)) and np.isclose(u @ v, (u * v).sum())
assert np.allclose(np.outer(u, v), u[:, None] * v[None, :])
# tensordot contracts chosen axes
T = rng.normal(size=(2, 3, 4))
assert np.allclose(np.tensordot(T, B, axes=([2], [0])), T @ B)  # matmul broadcasts over leading axes
# Kronecker mixed-product property: (A (x) B)(C (x) D) = AC (x) BD
C, D = rng.normal(size=(3, 3)), rng.normal(size=(2, 2))
E, F = rng.normal(size=(3, 3)), rng.normal(size=(2, 2))
assert np.allclose(np.kron(C, D) @ np.kron(E, F), np.kron(C @ E, D @ F))
show("kron(C,D).shape", np.kron(C, D).shape)

# %% [markdown]
# ## 3.2 Solving systems: `solve`, `lstsq`, and why NOT `inv`

# %%
section("3.2 Solving systems")
n = 10
H = sla.hilbert(n)  # famously ill-conditioned
x_true = np.ones(n)
b = H @ x_true
x_solve = np.linalg.solve(H, b)
x_inv = np.linalg.inv(H) @ b
err_solve = np.linalg.norm(x_solve - x_true)
err_inv = np.linalg.norm(x_inv - x_true)
show("cond(H10)", f"{np.linalg.cond(H):.2e}")
show("error with solve", f"{err_solve:.2e}")
show("error with inv(H)@b", f"{err_inv:.2e}")
# Rule of thumb: you lose log10(cond) digits of accuracy whatever you do; inv adds more work + error.
assert np.linalg.cond(H) > 1e12

# Least squares: overdetermined system, fit y = a + b t
t = np.linspace(0, 1, 50)
y = 2.0 + 3.0 * t + 0.01 * rng.normal(size=t.size)
X = np.column_stack([np.ones_like(t), t])
coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
assert np.allclose(coef, [2.0, 3.0], atol=0.02) and rank == 2
# normal equations square the condition number -> avoid for ill-conditioned X
assert np.allclose(np.linalg.solve(X.T @ X, X.T @ y), coef)

# %% [markdown]
# ## 3.3 Decompositions: LU, QR, Cholesky, SVD, eig/eigh

# %%
section("3.3 Decompositions")
M = rng.normal(size=(5, 5))
S = M @ M.T + 5 * np.eye(5)  # symmetric positive definite

P, L, U = sla.lu(M)
assert np.allclose(P @ L @ U, M)

Q, R = np.linalg.qr(M)
assert np.allclose(Q @ R, M) and np.allclose(Q.T @ Q, np.eye(5)) and np.allclose(R, np.triu(R))

Lc = np.linalg.cholesky(S)  # S = Lc Lc^T, lower-triangular; fails if S is not SPD
assert np.allclose(Lc @ Lc.T, S) and np.allclose(Lc, np.tril(Lc))

Uu, s, Vt = np.linalg.svd(M)
assert np.allclose(Uu @ np.diag(s) @ Vt, M) and np.all(np.diff(s) <= 0)  # sorted descending
assert np.isclose(np.linalg.norm(M, 2), s[0])  # spectral norm = largest singular value
assert np.isclose(np.linalg.cond(M), s[0] / s[-1])

w, V = np.linalg.eigh(S)  # symmetric/Hermitian: real eigenvalues, orthonormal V, ascending order
assert np.allclose(V @ np.diag(w) @ V.T, S) and np.all(np.diff(w) >= 0)
w_g, V_g = np.linalg.eig(M)  # general: may be complex; V need not be orthogonal
assert np.allclose(M @ V_g, V_g * w_g)
assert np.isclose(np.trace(M), w_g.sum().real) and np.isclose(np.linalg.det(M), np.prod(w_g).real)

# Cholesky is also the right way to test positive definiteness
try:
    np.linalg.cholesky(-np.eye(3))
    raise AssertionError
except np.linalg.LinAlgError:
    pass

# %% [markdown]
# ## 3.4 Norms, rank, pseudo-inverse

# %%
section("3.4 Norms, rank, pinv")
x = np.array([3.0, 4.0])
assert np.linalg.norm(x) == 5.0 and np.linalg.norm(x, 1) == 7.0 and np.linalg.norm(x, np.inf) == 4.0
R1 = np.outer(rng.normal(size=4), rng.normal(size=6))  # rank-1 by construction
assert np.linalg.matrix_rank(R1) == 1
Ap = np.linalg.pinv(R1)  # Moore-Penrose
assert np.allclose(R1 @ Ap @ R1, R1) and np.allclose(Ap @ R1 @ Ap, Ap)

# %% [markdown]
# ## 3.5 Numerical stability

# %%
section("3.5 Floating point")
eps = np.finfo(np.float64).eps
assert eps == 2.0**-52 and np.finfo(np.float32).eps > 1e-8
assert 1.0 + eps / 2 == 1.0 and 1.0 + eps > 1.0
assert np.isclose(1e10 + 1e-10, 1e10)  # tolerances: rtol AND atol
assert not np.isclose(0.0, 1e-12, atol=0)  # relative tolerance alone is useless near 0

# Catastrophic cancellation: (1 - cos x)/x^2 -> 1/2 as x -> 0
xs = 1e-8
naive = (1 - np.cos(xs)) / xs**2
stable = 2 * np.sin(xs / 2) ** 2 / xs**2
show("naive (1-cos x)/x^2", naive)
show("stable 2sin^2(x/2)/x^2", stable)
assert naive == 0.0 and np.isclose(stable, 0.5)

# Summation error: pairwise (NumPy default) vs naive accumulation
vals = np.full(10**7, 0.1, dtype=np.float32)
naive_sum = np.float32(0)
for chunk in vals[:200000]:  # small naive loop for illustration
    naive_sum += chunk
assert abs(float(vals[:200000].sum()) - 20000.0) < abs(float(naive_sum) - 20000.0)

# log-sum-exp: never exponentiate large numbers directly
z = np.array([1000.0, 1000.5])
with np.errstate(over="ignore"):
    assert np.isinf(np.exp(z).sum())
lse = z.max() + np.log(np.exp(z - z.max()).sum())
assert np.isfinite(lse)

# %% [markdown]
# ## Mini-project 1: QR three ways on an ill-conditioned matrix

# %%
section("Mini-project 1: Gram-Schmidt vs Householder")


def gram_schmidt_classical(A):
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def gram_schmidt_modified(A):
    m, n = A.shape
    V = A.astype(float).copy()
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        R[j, j + 1 :] = Q[:, j] @ V[:, j + 1 :]
        V[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])
    return Q, R


def householder_qr(A):
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        v = x.copy()
        v[0] += (1.0 if x[0] >= 0 else -1.0) * np.linalg.norm(x)  # sign choice avoids cancellation
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v /= nv
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    return Q[:, :n], np.triu(R[:n, :])


A = sla.hilbert(12)
results = {}
for name, f in [("classical GS", gram_schmidt_classical), ("modified GS", gram_schmidt_modified),
                ("Householder", householder_qr), ("np.linalg.qr", np.linalg.qr)]:
    Q, R = f(A)
    results[name] = (np.linalg.norm(Q.T @ Q - np.eye(12)), np.linalg.norm(Q @ R - A))
    show(name, f"orthogonality loss {results[name][0]:.1e}   residual {results[name][1]:.1e}")

assert results["Householder"][0] < 1e-13
assert results["classical GS"][0] > 1e3 * results["Householder"][0]  # classical GS loses orthogonality
assert results["modified GS"][0] < results["classical GS"][0]

# %% [markdown]
# ## Mini-project 2: PCA from scratch via SVD

# %%
section("Mini-project 2: PCA")


def pca(X, k):
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # SVD of data: no covariance matrix squared-cond issue
    var = s**2 / (len(X) - 1)
    return Xc @ Vt[:k].T, Vt[:k], var


latent = rng.normal(size=(500, 2)) * np.array([5.0, 1.0])
mix = rng.normal(size=(2, 6))
X = latent @ mix + 0.05 * rng.normal(size=(500, 6))
scores, comps, var = pca(X, 2)
cov_eigs = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
assert np.allclose(var, cov_eigs)  # SVD route == covariance-eigendecomposition route
assert var[:2].sum() / var.sum() > 0.99  # 2 latent factors explain ~everything
assert np.allclose(comps @ comps.T, np.eye(2))
show("variance explained by 2 PCs", f"{var[:2].sum() / var.sum():.4f}")

# %% [markdown]
# ### Your turn
# 1. Implement power iteration for the dominant eigenpair; compare with `eigh`.
# 2. Solve a 1D Poisson tridiagonal system with `scipy.linalg.solve_banded`; time vs dense `solve`.
# 3. Show that `inv` is worse than `solve` on a family of random matrices with growing condition number.

print("\nModule 3 complete: all assertions passed.")
