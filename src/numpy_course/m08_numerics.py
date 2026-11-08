# %% [markdown]
# # Module 8: Numerical methods toolkit
# Run:  `uv run python -m numpy_course.m08_numerics`

# %%
import time

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

from numpy_course.utils import section, show

rng = np.random.default_rng(0)
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2.0

# %% [markdown]
# ## 8.1 Polynomials: use `numpy.polynomial`, not legacy `poly1d`

# %%
section("8.1 Polynomials")
p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2  (LOW-to-high order, unlike np.polyfit!)
assert p(2) == 1 + 4 + 12 and p.deriv()(2) == 2 + 6 * 2
assert np.allclose(Polynomial.fromroots([1, 2, 3]).roots(), [1, 2, 3])
assert np.allclose(p.integ()(1.0), 1 + 1 + 1)  # integral 0..1 of p

# Least-squares fit: Polynomial.fit rescales the domain -> far better conditioned than np.polyfit
t = np.linspace(0, 10, 200)
y = 1.0 - 2.0 * t + 0.3 * t**2 + 0.05 * rng.normal(size=t.size)
fit = Polynomial.fit(t, y, deg=2)
assert np.allclose(fit.convert().coef, [1.0, -2.0, 0.3], atol=0.1)

# Runge phenomenon: equispaced interpolation diverges; Chebyshev nodes don't
f = lambda x: 1.0 / (1.0 + 25.0 * x**2)  # noqa: E731
xx = np.linspace(-1, 1, 2001)
deg = 15
equi_x = np.linspace(-1, 1, deg + 1)
equi = Polynomial.fit(equi_x, f(equi_x), deg)
cheb = Chebyshev.interpolate(f, deg, domain=[-1, 1])
err_equi, err_cheb = np.abs(equi(xx) - f(xx)).max(), np.abs(cheb(xx) - f(xx)).max()
show("max error, equispaced nodes", f"{err_equi:.3f}")
show("max error, Chebyshev nodes", f"{err_cheb:.3f}")
assert err_cheb < err_equi / 3

# %% [markdown]
# ## 8.2 FFT: convolution theorem and spectral differentiation

# %%
section("8.2 FFT")
a, b = rng.normal(size=64), rng.normal(size=64)
# circular convolution: ifft(fft(a) * fft(b))
circ = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
direct = np.array([sum(a[j] * b[(k - j) % 64] for j in range(64)) for k in range(64)])
assert np.allclose(circ, direct)
# linear convolution: zero-pad to length >= len(a)+len(b)-1
n_fft = 128
lin = np.fft.irfft(np.fft.rfft(a, n_fft) * np.fft.rfft(b, n_fft), n_fft)[:127]
assert np.allclose(lin, np.convolve(a, b))
# Parseval: energy is preserved
assert np.isclose((a**2).sum(), (np.abs(np.fft.fft(a)) ** 2).sum() / 64)
# Frequency detection
fs, N = 1000, 1000
tt = np.arange(N) / fs
sig = np.sin(2 * np.pi * 50 * tt) + 0.5 * np.sin(2 * np.pi * 120 * tt)
freqs = np.fft.rfftfreq(N, 1 / fs)
top2 = np.sort(freqs[np.argsort(np.abs(np.fft.rfft(sig)))[-2:]])
assert np.array_equal(top2, [50.0, 120.0])

# Spectral derivative of a smooth periodic function: error near machine precision
N = 64
x = 2 * np.pi * np.arange(N) / N
u = np.exp(np.sin(x))
k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers on [0, 2*pi)
du = np.fft.ifft(1j * k * np.fft.fft(u)).real
exact = np.cos(x) * np.exp(np.sin(x))
show("spectral derivative error", f"{np.abs(du - exact).max():.1e}")
assert np.abs(du - exact).max() < 1e-12

# %% [markdown]
# ## 8.3 Interpolation, integration, finite differences: observe the convergence order

# %%
section("8.3 Quadrature and differences")
xs = np.array([0.0, 1.0, 2.0])
assert np.interp(1.5, xs, [0.0, 10.0, 0.0]) == 5.0  # linear interpolation; xs must be increasing


def trap_error(n):
    x = np.linspace(0, np.pi, n + 1)
    return abs(trapezoid(np.sin(x), x) - 2.0)


e1, e2 = trap_error(100), trap_error(200)
show("trapezoid error ratio (n -> 2n)", f"{e1 / e2:.2f}  (expect ~4: second order)")
assert 3.8 < e1 / e2 < 4.2

h = 1e-3
x0 = 1.0
fwd = (np.sin(x0 + h) - np.sin(x0)) / h  # O(h)
ctr = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)  # O(h^2)
assert abs(ctr - np.cos(x0)) < abs(fwd - np.cos(x0)) / 100
xg = np.linspace(0, 2 * np.pi, 400)
assert np.abs(np.gradient(np.sin(xg), xg) - np.cos(xg))[1:-1].max() < 1e-3
# Step-size trade-off: truncation error falls, round-off error rises as h shrinks
errs = {h: abs((np.sin(1 + h) - np.sin(1 - h)) / (2 * h) - np.cos(1)) for h in (1e-1, 1e-5, 1e-12)}
assert errs[1e-5] < errs[1e-1] and errs[1e-5] < errs[1e-12]

# %% [markdown]
# ## 8.4 Vectorised SDE solvers: Euler-Maruyama vs Milstein (strong convergence orders)
# dS = mu S dt + sigma S dW has the exact solution S_T = S0 exp((mu - sigma^2/2)T + sigma W_T).
# We refine the step on the SAME Brownian paths and measure E|S_num(T) - S_exact(T)|.

# %%
section("8.4 SDE convergence")
S0, mu, sigma, T = 1.0, 0.05, 0.4, 1.0
N, n_fine = 20_000, 256
dW_fine = np.sqrt(T / n_fine) * rng.standard_normal((N, n_fine))
S_exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW_fine.sum(axis=1))


def euler_maruyama(dW, T):
    dt = T / dW.shape[1]
    S = np.full(dW.shape[0], S0)
    for j in range(dW.shape[1]):  # loop over TIME (few steps); vectorise over PATHS (many)
        S = S + mu * S * dt + sigma * S * dW[:, j]
    return S


def milstein(dW, T):
    dt = T / dW.shape[1]
    S = np.full(dW.shape[0], S0)
    for j in range(dW.shape[1]):
        S = S + mu * S * dt + sigma * S * dW[:, j] + 0.5 * sigma**2 * S * (dW[:, j] ** 2 - dt)
    return S


levels = [4, 8, 16, 32, 64]
err_em, err_mil = [], []
for n in levels:
    dW = dW_fine.reshape(N, n, n_fine // n).sum(axis=2)  # coarsen the SAME paths
    err_em.append(np.abs(euler_maruyama(dW, T) - S_exact).mean())
    err_mil.append(np.abs(milstein(dW, T) - S_exact).mean())
slope_em = np.polyfit(np.log(T / np.array(levels)), np.log(err_em), 1)[0]
slope_mil = np.polyfit(np.log(T / np.array(levels)), np.log(err_mil), 1)[0]
show("Euler-Maruyama strong order", f"{slope_em:.2f}  (theory 0.5)")
show("Milstein strong order", f"{slope_mil:.2f}  (theory 1.0; slightly low because the reference is itself discretised)")
assert 0.35 < slope_em < 0.7 and 0.8 < slope_mil < 1.25

# %% [markdown]
# ## 8.5 Missing data: NaN-aware functions and masked arrays

# %%
section("8.5 Missing data")
d = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
assert np.isnan(d.mean()) and np.nanmean(d) == 3.0 and np.nansum(d) == 9.0
assert np.nanargmax(d) == 4 and np.nanmax(d) == 5.0
assert np.array_equal(np.nan_to_num(d, nan=0.0), [1, 0, 3, 0, 5])
mk = np.ma.masked_invalid(d)
assert mk.mean() == 3.0 and mk.count() == 3
assert np.ma.masked_less(np.array([1, -5, 3]), 0).sum() == 4
# Interpolate over gaps
good = ~np.isnan(d)
filled = np.interp(np.arange(5), np.flatnonzero(good), d[good])
assert np.array_equal(filled, [1, 2, 3, 4, 5])

# %% [markdown]
# ## Mini-project: heat equation, finite differences vs spectral
# u_t = alpha u_xx on [0, 2*pi) periodic.  Exact: e^{-alpha t} sin x + 0.5 e^{-9 alpha t} sin 3x.

# %%
section("8.6 Heat equation")
alpha, Tend, Ngrid = 1.0, 0.5, 128
x = 2 * np.pi * np.arange(Ngrid) / Ngrid
dx = x[1] - x[0]
u0 = np.sin(x) + 0.5 * np.sin(3 * x)
u_exact = np.exp(-alpha * Tend) * np.sin(x) + 0.5 * np.exp(-9 * alpha * Tend) * np.sin(3 * x)


def heat_fd(u0, alpha, Tend, dx, cfl=0.4):
    dt = cfl * dx**2 / alpha  # explicit scheme is stable only for dt <= dx^2 / (2 alpha)
    steps = int(np.ceil(Tend / dt))
    dt = Tend / steps
    r = alpha * dt / dx**2
    u = u0.copy()
    for _ in range(steps):
        u += r * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
    return u, steps


def heat_spectral(u0, alpha, Tend):
    k = np.fft.fftfreq(len(u0), d=1.0 / len(u0))
    return np.fft.ifft(np.fft.fft(u0) * np.exp(-alpha * k**2 * Tend)).real  # exact in time for this linear PDE


t0 = time.perf_counter()
u_fd, nsteps = heat_fd(u0, alpha, Tend, dx)
t_fd = time.perf_counter() - t0
t0 = time.perf_counter()
u_sp = heat_spectral(u0, alpha, Tend)
t_sp = time.perf_counter() - t0
err_fd, err_sp = np.abs(u_fd - u_exact).max(), np.abs(u_sp - u_exact).max()
show("finite difference", f"error {err_fd:.1e}   {nsteps} steps   {t_fd:.1e} s")
show("spectral (FFT)", f"error {err_sp:.1e}   1 step      {t_sp:.1e} s")
assert err_sp < 1e-12 and err_fd < 1e-3 and err_sp < err_fd and t_sp < t_fd

# FD is second-order: halving dx cuts error ~4x
u_fd2, _ = heat_fd(np.sin(x[::2]) + 0.5 * np.sin(3 * x[::2]), alpha, Tend, 2 * dx)
err_coarse = np.abs(u_fd2 - u_exact[::2]).max()
assert 3.0 < err_coarse / err_fd < 5.0

# Stability: violate the CFL limit and the explicit scheme blows up
with np.errstate(over="ignore", invalid="ignore"):
    u_bad, _ = heat_fd(u0, alpha, 0.5, dx, cfl=0.6)  # > 0.5
assert np.abs(u_bad).max() > 1e3 or not np.isfinite(u_bad).all()

# %% [markdown]
# ### Your turn
# 1. Add Crank-Nicolson (implicit) with `scipy.linalg.solve_banded` and show it is stable for any dt.
# 2. Simulate an Ornstein-Uhlenbeck process and Heston (full truncation); test weak convergence of E[S_T].
# 3. Differentiate a noisy signal with FFT low-pass filtering vs `np.gradient`.

print("\nModule 8 complete: all assertions passed.")
