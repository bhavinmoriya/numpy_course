# %% [markdown]
# # Module 4: Random numbers and Monte Carlo simulation
# Run:  `uv run python -m numpy_course.m04_random`

# %%
import numpy as np
from scipy.special import ndtr  # standard normal CDF

from numpy_course.utils import section, show

# %% [markdown]
# ## 4.1 The modern API: Generator, not np.random.seed
# `default_rng(seed)` gives you an explicit, local generator: no hidden global state.

# %%
section("4.1 Generator basics")
rng1, rng2 = np.random.default_rng(123), np.random.default_rng(123)
assert np.array_equal(rng1.normal(size=5), rng2.normal(size=5))  # same seed -> same stream

rng = np.random.default_rng(2026)
assert rng.random(3).shape == (3,) and ((0 <= rng.random(1000)) & (rng.random(1000) < 1)).all()
assert set(rng.integers(0, 6, size=1000)) == {0, 1, 2, 3, 4, 5}  # high is EXCLUSIVE
x = rng.normal(loc=10, scale=2, size=200_000)
assert abs(x.mean() - 10) < 0.05 and abs(x.std() - 2) < 0.05
assert rng.exponential(2.0, 200_000).mean() > 1.9
assert rng.poisson(3.0, 200_000).mean() > 2.9

# Sampling / shuffling
pop = np.arange(10)
pick = rng.choice(pop, size=5, replace=False)
assert len(set(pick)) == 5
w = rng.choice(["a", "b"], size=100_000, p=[0.9, 0.1])
assert 0.08 < (w == "b").mean() < 0.12
perm = rng.permutation(10)  # returns a new array
assert sorted(perm) == list(range(10))
arr = np.arange(10)
rng.shuffle(arr)  # shuffles IN PLACE, returns None
assert sorted(arr) == list(range(10))

# %% [markdown]
# ## 4.2 Independent streams for parallel work: SeedSequence.spawn
# Never do `default_rng(seed + i)` for workers; spawn statistically independent children.

# %%
section("4.2 Independent streams")
ss = np.random.SeedSequence(42)
streams = [np.random.default_rng(child) for child in ss.spawn(4)]
draws = np.array([g.normal(size=100_000) for g in streams])
corr = np.corrcoef(draws)
assert np.abs(corr - np.eye(4)).max() < 0.02  # uncorrelated streams
# Reproducible: re-spawning from the same root gives the same children
again = [np.random.default_rng(c).normal(size=3) for c in np.random.SeedSequence(42).spawn(4)]
assert np.array_equal(again[0], np.random.default_rng(np.random.SeedSequence(42).spawn(4)[0]).normal(size=3))
show("max off-diagonal corr", f"{np.abs(corr - np.eye(4)).max():.4f}")

# %% [markdown]
# ## 4.3 Monte Carlo estimation and confidence intervals
# Error shrinks like 1/sqrt(N): 100x more samples buys 10x more accuracy.

# %%
section("4.3 Monte Carlo basics")


def mc_summary(samples):
    m = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    return m, se, (m - 1.96 * se, m + 1.96 * se)


rng = np.random.default_rng(7)
pts = rng.random((1_000_000, 2))
inside = (pts**2).sum(axis=1) < 1.0
pi_hat, se, ci = mc_summary(4.0 * inside)
show("pi estimate", f"{pi_hat:.5f} +- {1.96 * se:.5f}")
assert ci[0] < np.pi < ci[1] or abs(pi_hat - np.pi) < 4 * se

# Convergence rate check: error ~ N^-1/2
errs = []
for N in (10**3, 10**5):
    est = [4.0 * ((rng.random((N, 2)) ** 2).sum(axis=1) < 1.0).mean() for _ in range(40)]
    errs.append(np.mean(np.abs(np.array(est) - np.pi)))
assert 5 < errs[0] / errs[1] < 20  # expect ~ sqrt(100) = 10

# %% [markdown]
# ## 4.4 Option pricing: European call, closed form vs Monte Carlo

# %%
section("4.4 European option")
S0, K, r, sigma, T = 100.0, 105.0, 0.03, 0.2, 1.0


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


exact = bs_call(S0, K, r, sigma, T)
N = 400_000
rng = np.random.default_rng(1)
Z = rng.standard_normal(N)


def terminal(Z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


disc = np.exp(-r * T)
plain = disc * np.maximum(terminal(Z) - K, 0)
p_mean, p_se, _ = mc_summary(plain)

# Antithetic variates: pair Z with -Z, average within pairs (pairs are the independent samples)
anti = 0.5 * disc * (np.maximum(terminal(Z) - K, 0) + np.maximum(terminal(-Z) - K, 0))
a_mean, a_se, _ = mc_summary(anti)

# Control variate: S_T has known mean S0*e^{rT}
ST = terminal(Z)
payoff = np.maximum(ST - K, 0)
beta = np.cov(payoff, ST)[0, 1] / ST.var(ddof=1)
cv = disc * (payoff - beta * (ST - S0 * np.exp(r * T)))
c_mean, c_se, _ = mc_summary(cv)

show("Black-Scholes exact", f"{exact:.5f}")
show("plain MC", f"{p_mean:.5f}  (se {p_se:.5f})")
show("antithetic", f"{a_mean:.5f}  (se {a_se:.5f})")
show("control variate", f"{c_mean:.5f}  (se {c_se:.5f})")
for m, se in [(p_mean, p_se), (a_mean, a_se), (c_mean, c_se)]:
    assert abs(m - exact) < 4.5 * se
assert a_se < p_se and c_se < p_se

# %% [markdown]
# ## 4.5 Path-dependent: arithmetic Asian call with a geometric-Asian control variate

# %%
section("4.5 Asian option")
steps = 50
dt = T / steps
rng = np.random.default_rng(11)
Zp = rng.standard_normal((200_000, steps))
logS = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Zp, axis=1)  # (paths, steps)
paths = np.exp(logS)

arith = disc * np.maximum(paths.mean(axis=1) - K, 0)
geo = disc * np.maximum(np.exp(logS.mean(axis=1)) - K, 0)


def geometric_asian_call(S0, K, r, sigma, T, n):
    """Closed form for discretely monitored geometric average (times T/n, ..., T)."""
    m = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (m + v - np.log(K)) / np.sqrt(v)
    d2 = d1 - np.sqrt(v)
    return np.exp(-r * T) * (np.exp(m + 0.5 * v) * ndtr(d1) - K * ndtr(d2))


geo_exact = geometric_asian_call(S0, K, r, sigma, T, steps)
g_mean, g_se, _ = mc_summary(geo)
assert abs(g_mean - geo_exact) < 4.5 * g_se  # validates the simulator itself

beta = np.cov(arith, geo)[0, 1] / geo.var(ddof=1)
arith_cv = arith - beta * (geo - geo_exact)
a_m, a_s, _ = mc_summary(arith)
c_m, c_s, _ = mc_summary(arith_cv)
show("geometric Asian (exact)", f"{geo_exact:.5f}")
show("arithmetic plain", f"{a_m:.5f}  (se {a_s:.5f})")
show("arithmetic + control variate", f"{c_m:.5f}  (se {c_s:.5f})")
show("variance reduction", f"{(a_s / c_s) ** 2:.0f}x")
assert c_s < a_s / 5 and a_m >= g_mean  # AM >= GM pathwise

# %% [markdown]
# ## 4.6 Correlated multi-asset simulation via Cholesky

# %%
section("4.6 Correlated assets")
corr = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.3], [0.2, -0.3, 1.0]])
vols = np.array([0.20, 0.30, 0.25])
S0v = np.array([100.0, 50.0, 80.0])
L = np.linalg.cholesky(corr)
rng = np.random.default_rng(5)
Zc = rng.standard_normal((300_000, 3))
W = Zc @ L.T  # rows now have covariance L L^T = corr
assert np.abs(np.corrcoef(W.T) - corr).max() < 0.01
ST = S0v * np.exp((r - 0.5 * vols**2) * T + vols * np.sqrt(T) * W)
basket = ST.mean(axis=1)  # equal-weighted basket of the 3 terminal prices
basket_call = disc * np.maximum(basket - S0v.mean(), 0)
m, se, ci = mc_summary(basket_call)
show("basket call", f"{m:.4f} +- {1.96 * se:.4f}")
# Martingale check: discounted asset means should equal S0
assert np.allclose(disc * ST.mean(axis=0), S0v, rtol=0.01)

# %% [markdown]
# ### Your turn
# 1. Price an up-and-out barrier call; discuss the bias from discrete monitoring.
# 2. Combine antithetic + control variates on the Asian option; measure the extra gain.
# 3. Replace pseudo-random numbers with `scipy.stats.qmc.Sobol` and compare convergence.

print("\nModule 4 complete: all assertions passed.")
