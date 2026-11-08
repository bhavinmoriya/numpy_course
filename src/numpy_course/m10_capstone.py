# %% [markdown]
# # Module 10: Capstone, the `miniquant` library
# Run:  `uv run python -m numpy_course.m10_capstone`
# The library is in `miniquant.py`; its tests are in `tests/test_miniquant.py`.
# Capstone options you can build next in the same style: NumPy neural network (forward/backward + gradient
# check), SDE framework (GBM/Heston/OU with convergence tests), signal-processing suite (conv/FFT filters).

# %%
import numpy as np

from numpy_course import miniquant as mq
from numpy_course.utils import section, show

# %%
section("10.1 Black-Scholes, vectorised over a whole surface")
strikes = np.linspace(80, 120, 9)[:, None]  # (9,1)
maturities = np.array([0.25, 0.5, 1.0, 2.0])[None, :]  # (1,4)
calls = mq.bs_price(100.0, strikes, 0.03, 0.2, maturities)  # broadcasts to (9,4): zero loops
puts = mq.bs_price(100.0, strikes, 0.03, 0.2, maturities, kind="put")
assert calls.shape == (9, 4)
# Put-call parity: C - P = S e^{-qT} - K e^{-rT}
assert np.allclose(calls - puts, 100.0 - strikes * np.exp(-0.03 * maturities))
assert np.all(np.diff(calls, axis=0) < 0) and np.all(np.diff(calls, axis=1) > 0)  # monotone in K and T
show("ATM 1y call", f"{mq.bs_price(100, 100, 0.03, 0.2, 1.0):.4f}")

# %%
section("10.2 Greeks: closed form vs finite differences")
g = mq.bs_greeks(100, 105, 0.03, 0.25, 0.75)
d_fd, gam_fd, vega_fd = mq.fd_greeks(100, 105, 0.03, 0.25, 0.75)
assert np.isclose(g.delta, d_fd, atol=1e-6) and np.isclose(g.gamma, gam_fd, atol=1e-5) and np.isclose(g.vega, vega_fd, atol=1e-4)
show("delta / gamma / vega", f"{float(g.delta):.4f} / {float(g.gamma):.4f} / {float(g.vega):.4f}")

# %%
section("10.3 Implied volatility round trip over a grid")
true_vol = np.linspace(0.1, 0.6, 6)[None, :] * np.ones((9, 1))
prices = mq.bs_price(100.0, strikes, 0.03, true_vol, 1.0)
iv = mq.implied_vol(prices, 100.0, strikes, 0.03, 1.0)
assert np.allclose(iv, true_vol, atol=1e-6)
assert np.isnan(mq.implied_vol(0.0, 100, 100, 0.03, 1.0))  # no-arbitrage violation -> NaN

# %%
section("10.4 Monte Carlo engine vs analytic")
exact = float(mq.bs_price(100, 105, 0.03, 0.25, 0.75))
res = mq.mc_european(100, 105, 0.03, 0.25, 0.75, n=500_000, seed=3)
show("exact / MC", f"{exact:.4f} / {res.price:.4f} +- {1.96 * res.stderr:.4f}")
assert abs(res.price - exact) < 4.5 * res.stderr
d_mc = mq.mc_delta_crn(100, 105, 0.03, 0.25, 0.75)
show("MC delta (CRN) vs exact", f"{d_mc:.4f} vs {float(g.delta):.4f}")
assert abs(d_mc - float(g.delta)) < 0.01

# %%
section("10.5 Portfolio optimisation")
rng = np.random.default_rng(1)
n_assets = 6
A = rng.normal(size=(n_assets, n_assets))
cov = A @ A.T / n_assets * 0.04 + 0.01 * np.eye(n_assets)  # SPD by construction
mu = rng.uniform(0.02, 0.12, n_assets)
w_mv = mq.mean_variance_weights(mu, cov)
w_eq = np.full(n_assets, 1 / n_assets)
assert np.isclose(w_mv.sum(), 1.0)
assert w_mv @ cov @ w_mv <= w_eq @ cov @ w_eq + 1e-12  # global min variance beats equal weight
# Efficient frontier: lower risk aversion -> higher return and higher risk
frontier = [mq.portfolio_stats(mq.mean_variance_weights(mu, cov, lam), mu, cov) for lam in (20, 5, 1, 0.2)]
rets, risks = zip(*frontier)
assert list(rets) == sorted(rets) and list(risks) == sorted(risks)
show("min-variance risk vs equal-weight", f"{np.sqrt(w_mv @ cov @ w_mv):.4f} vs {np.sqrt(w_eq @ cov @ w_eq):.4f}")

# %% [markdown]
# ### Your turn (turn this into your own package)
# 1. Add Heston and a Longstaff-Schwartz American put to `miniquant`, with convergence tests.
# 2. Add `numba` kernels for the Asian MC and keep the NumPy version as the reference in tests.
# 3. Publish: `uv build`, add a README with benchmarks, and a GitHub Action that runs `uv run pytest`.

print("\nModule 10 complete: all assertions passed.")
