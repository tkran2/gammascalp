"""Raw SVI smile: fit one expiry's implied variances and test it for arbitrage.

The raw SVI parameterization (Gatheral, 2004) writes total implied variance
w = sigma_implied^2 * T as a function of log-moneyness k = ln(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Five numbers describe a whole smile: level a, wing slope b, tilt rho,
horizontal shift m, and curvature sigma at the money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from itertools import product

from scipy.optimize import minimize


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def implied_vol(self, k, T):
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    def slope(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def curvature(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * self.sigma**2 / ((k - self.m) ** 2 + self.sigma**2) ** 1.5

    def min_variance(self) -> float:
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def butterfly(self, k):
        """Gatheral's g(k). The smile implies a negative probability density,
        and therefore a butterfly arbitrage, wherever g(k) < 0."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        w1 = self.slope(k)
        w2 = self.curvature(k)
        return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2

    def wing_bound_ok(self) -> bool:
        """Roger Lee's moment bound: total variance cannot grow faster than 2|k|."""
        return self.b * (1.0 + abs(self.rho)) <= 2.0 + 1e-12


@dataclass(frozen=True)
class FitResult:
    params: SVI
    rmse_variance: float  # Root-mean-square error in total variance.
    rmse_vol: float       # Root-mean-square error in implied volatility.


def _box_least_squares(design, target, lower, upper):
    """Exact least squares with box bounds, for a handful of unknowns.

    The objective is a convex quadratic, so the constrained minimum is the
    unconstrained minimum of whichever face of the box it lies on. With three
    unknowns there are only 27 faces: solve each and keep the best feasible one.
    """
    x, *_ = np.linalg.lstsq(design, target, rcond=None)
    if np.all(x >= lower) and np.all(x <= upper):
        return x
    count = design.shape[1]
    best, best_sse = None, np.inf
    for state in product((0, 1, 2), repeat=count):  # 0 free, 1 at lower, 2 at upper
        x = np.where(np.array(state) == 1, lower, upper).astype(float)
        free = [i for i in range(count) if state[i] == 0]
        if free:
            fixed = [i for i in range(count) if state[i] != 0]
            rhs = target - design[:, fixed] @ x[fixed]
            solved, *_ = np.linalg.lstsq(design[:, free], rhs, rcond=None)
            if np.any(solved < lower[free]) or np.any(solved > upper[free]):
                continue
            x[free] = solved
        residual = design @ x - target
        sse = float(residual @ residual)
        if sse < best_sse:
            best, best_sse = x, sse
    return best


def _inner(k, w, weights, m, sigma):
    """Best (a, b, rho) for fixed (m, sigma).

    With y = (k - m) / sigma the model is linear:
        w = a + d * y + c * sqrt(y^2 + 1),   c = b sigma,   d = rho b sigma.
    Substituting p = c + d and n = c - d turns the cone |d| <= c into the box
    p, n >= 0, so an exact bounded linear least-squares solve applies. The
    two-level idea, linear inside and nonlinear outside, follows Zeliade
    Systems (2009).
    """
    y = (k - m) / sigma
    z = np.sqrt(y * y + 1.0)
    design = np.column_stack([np.ones_like(y), 0.5 * (z + y), 0.5 * (z - y)]) * weights[:, None]
    top = max(float(np.max(w)), 1e-12)
    target = w * weights
    lower = np.array([-top, 0.0, 0.0])
    # c + |d| <= 2 sigma is Lee's wing bound b (1 + |rho|) <= 2.
    upper = np.array([top, 2.0 * sigma, 2.0 * sigma])
    x = _box_least_squares(design, target, lower, upper)
    a, p, n = x
    c = 0.5 * (p + n)
    d = 0.5 * (p - n)
    b = c / sigma
    rho = 0.0 if c <= 1e-14 else float(np.clip(d / c, -0.999999, 0.999999))
    residual = design @ x - target
    return SVI(float(a), float(b), rho, float(m), float(sigma)), float(residual @ residual)


def fit(k, total_variance, T, weights=None) -> FitResult:
    """Least-squares raw-SVI fit to one expiry.

    k: log-moneyness ln(K / F). total_variance: implied vol^2 * T at each k.
    weights: optional per-point weights (for example, vega, or 1 / spread).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.shape != w.shape or k.size < 5:
        raise ValueError("need at least five (k, total variance) pairs")
    weights = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / np.mean(weights)
    span = float(np.max(k) - np.min(k))

    def loss(theta):
        m, log_sigma = theta
        sigma = float(np.exp(log_sigma))
        if not (0.002 <= sigma <= 5.0) or not (np.min(k) - span <= m <= np.max(k) + span):
            return 1e9
        params, sse = _inner(k, w, weights, m, sigma)
        if params.min_variance() < 0:
            return 1e9
        return sse

    # Stage 1: a coarse grid over the two nonlinear parameters. The inner solve
    # is exact, so the grid only has to land in the right basin.
    candidates = []
    for m in np.linspace(np.min(k) - 0.25 * span, np.max(k) + 0.25 * span, 25):
        for log_sigma in np.linspace(np.log(0.004), np.log(max(2.0 * span, 0.05)), 17):
            candidates.append((loss((m, log_sigma)), m, log_sigma))
    candidates.sort(key=lambda item: item[0])

    # Stage 2: polish the best few with a derivative-free search.
    best = None
    for _, m0, s0 in candidates[:3]:
        found = minimize(loss, x0=[m0, s0], method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-16, "maxiter": 600})
        if best is None or found.fun < best.fun:
            best = found
    params, _ = _inner(k, w, weights, best.x[0], float(np.exp(best.x[1])))
    residual = params.total_variance(k) - w
    vol_error = params.implied_vol(k, T) - np.sqrt(w / T)
    return FitResult(params, float(np.sqrt(np.mean(residual**2))), float(np.sqrt(np.mean(vol_error**2))))


def butterfly_violation(params: SVI, k_min=-1.5, k_max=1.5, points=3001) -> float:
    """Most negative value of g(k) on a grid, or 0.0 if the slice is clean."""
    grid = np.linspace(k_min, k_max, points)
    return float(min(0.0, np.min(params.butterfly(grid))))


def calendar_violation(slices, k_min=-1.5, k_max=1.5, points=601) -> float:
    """Largest drop in total variance from one expiry to the next (0.0 if none).

    `slices` is a list of SVI objects ordered by increasing expiry. Total
    variance that falls with maturity at some strike is a calendar arbitrage.
    """
    grid = np.linspace(k_min, k_max, points)
    worst = 0.0
    for near, far in zip(slices[:-1], slices[1:]):
        worst = min(worst, float(np.min(far.total_variance(grid) - near.total_variance(grid))))
    return worst
