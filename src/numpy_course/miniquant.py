"""miniquant: a small, pure-NumPy quant library (capstone of the NumPy course).

Contents
--------
* Black-Scholes price + closed-form Greeks (vectorised over all inputs)
* Vectorised implied volatility (bisection)
* GBM Monte Carlo engine with antithetic variates and common random numbers
* Finite-difference Greeks (bump-and-revalue) for validating the closed forms
* Mean-variance portfolio optimisation via the KKT linear system
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def _d1_d2(S, K, r, q, sigma, T):
    S, K, r, q, sigma, T = np.broadcast_arrays(*map(np.asarray, (S, K, r, q, sigma, T)))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, r, sigma, T, q=0.0, kind="call"):
    """Black-Scholes price. All arguments broadcast against each other."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    S, K, r, sigma, T, q = (np.asarray(a, dtype=float) for a in (S, K, r, sigma, T, q))
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "call":
        return S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    return K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray


def bs_greeks(S, K, r, sigma, T, q=0.0, kind="call") -> Greeks:
    """Closed-form Greeks (vega per 1.00 vol, theta per year, rho per 1.00 rate)."""
    S, K, r, sigma, T, q = (np.asarray(a, dtype=float) for a in (S, K, r, sigma, T, q))
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r, df_q, pdf = np.exp(-r * T), np.exp(-q * T), _norm_pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    common_theta = -S * df_q * pdf * sigma / (2.0 * np.sqrt(T))
    if kind == "call":
        delta = df_q * ndtr(d1)
        theta = common_theta - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
        rho = K * T * df_r * ndtr(d2)
    elif kind == "put":
        delta = -df_q * ndtr(-d1)
        theta = common_theta + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
        rho = -K * T * df_r * ndtr(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return Greeks(delta, gamma, vega, theta, rho)


def implied_vol(price, S, K, r, T, q=0.0, kind="call", lo=1e-6, hi=5.0, iters=100):
    """Vectorised bisection. Returns NaN where the price is outside no-arbitrage bounds."""
    price, S, K, r, T, q = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, r, T, q)))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        too_high = bs_price(S, K, r, mid, T, q, kind) > price  # price is increasing in vol
        b = np.where(too_high, mid, b)
        a = np.where(too_high, a, mid)
    iv = 0.5 * (a + b)
    bad = (price < bs_price(S, K, r, lo, T, q, kind)) | (price > bs_price(S, K, r, hi, T, q, kind))
    return np.where(bad, np.nan, iv)


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float

    @property
    def ci95(self):
        return self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr


def mc_european(S, K, r, sigma, T, q=0.0, kind="call", n=200_000, seed=0, antithetic=True, z=None) -> MCResult:
    """GBM Monte Carlo. Pass `z` to reuse random numbers (common random numbers for Greeks)."""
    if z is None:
        z = np.random.default_rng(seed).standard_normal(n)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(zz):
        return np.maximum(sign * (S * np.exp(drift + vol * zz) - K), 0.0)

    samples = 0.5 * (payoff(z) + payoff(-z)) if antithetic else payoff(z)
    samples = np.exp(-r * T) * samples
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))))


def fd_greeks(S, K, r, sigma, T, q=0.0, kind="call", bump=1e-4):
    """Central-difference delta, gamma, vega via bump-and-revalue on the closed-form price."""
    f = lambda **kw: bs_price(**{**dict(S=S, K=K, r=r, sigma=sigma, T=T, q=q, kind=kind), **kw})  # noqa: E731
    hS, hv = bump * S, bump
    delta = (f(S=S + hS) - f(S=S - hS)) / (2 * hS)
    gamma = (f(S=S + hS) - 2 * f() + f(S=S - hS)) / hS**2
    vega = (f(sigma=sigma + hv) - f(sigma=sigma - hv)) / (2 * hv)
    return delta, gamma, vega


def mc_delta_crn(S, K, r, sigma, T, q=0.0, kind="call", n=400_000, seed=0, bump=0.01):
    """MC delta with common random numbers: same z in both bumped valuations."""
    z = np.random.default_rng(seed).standard_normal(n)
    hS = bump * S
    up = mc_european(S + hS, K, r, sigma, T, q, kind, z=z).price
    dn = mc_european(S - hS, K, r, sigma, T, q, kind, z=z).price
    return (up - dn) / (2 * hS)


def mean_variance_weights(mu, cov, risk_aversion=None):
    """Fully-invested (sum w = 1) portfolio via the KKT system.

    risk_aversion=None -> global minimum variance.
    otherwise          -> maximise  mu'w - (lambda/2) w'Cw.
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    n = len(mu)
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = cov if risk_aversion is None else risk_aversion * cov
    kkt[:n, n] = kkt[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    if risk_aversion is not None:
        rhs[:n] = mu
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n]


def portfolio_stats(w, mu, cov):
    w = np.asarray(w)
    return float(w @ mu), float(np.sqrt(w @ cov @ w))
