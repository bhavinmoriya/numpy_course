import numpy as np
import pytest

from numpy_course import miniquant as mq

PARAMS = dict(S=100.0, K=95.0, r=0.03, sigma=0.22, T=0.8, q=0.01)


def test_put_call_parity():
    c = mq.bs_price(**PARAMS, kind="call")
    p = mq.bs_price(**PARAMS, kind="put")
    S, K, r, T, q = (PARAMS[k] for k in ("S", "K", "r", "T", "q"))
    assert np.isclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    g = mq.bs_greeks(**PARAMS, kind=kind)
    d, gam, vega = mq.fd_greeks(**PARAMS, kind=kind)
    assert np.isclose(g.delta, d, atol=1e-6)
    assert np.isclose(g.gamma, gam, atol=1e-5)
    assert np.isclose(g.vega, vega, atol=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_and_rho_match_bumps(kind):
    h = 1e-6
    g = mq.bs_greeks(**PARAMS, kind=kind)
    f = lambda **kw: mq.bs_price(**{**PARAMS, **kw}, kind=kind)  # noqa: E731
    theta_fd = -(f(T=PARAMS["T"] + h) - f(T=PARAMS["T"] - h)) / (2 * h)  # theta = -dV/dT
    rho_fd = (f(r=PARAMS["r"] + h) - f(r=PARAMS["r"] - h)) / (2 * h)
    assert np.isclose(g.theta, theta_fd, rtol=1e-4)
    assert np.isclose(g.rho, rho_fd, rtol=1e-4)


def test_implied_vol_roundtrip():
    vols = np.array([0.05, 0.2, 0.8, 2.0])
    prices = mq.bs_price(100.0, 100.0, 0.02, vols, 1.0)
    assert np.allclose(mq.implied_vol(prices, 100.0, 100.0, 0.02, 1.0), vols, atol=1e-7)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_within_confidence_band(kind):
    exact = float(mq.bs_price(**PARAMS, kind=kind))
    res = mq.mc_european(**PARAMS, kind=kind, n=300_000, seed=11)
    assert abs(res.price - exact) < 4.5 * res.stderr


def test_antithetic_reduces_variance():
    plain = mq.mc_european(**PARAMS, n=100_000, seed=1, antithetic=False)
    anti = mq.mc_european(**PARAMS, n=100_000, seed=1, antithetic=True)
    assert anti.stderr < plain.stderr


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        mq.bs_price(**PARAMS, kind="straddle")


def test_min_variance_weights_properties():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5)
    w = mq.mean_variance_weights(np.zeros(5), cov)
    assert np.isclose(w.sum(), 1.0)
    for _ in range(20):  # no random fully-invested portfolio beats the minimum
        v = rng.normal(size=5)
        v /= v.sum()
        assert w @ cov @ w <= v @ cov @ v + 1e-12
