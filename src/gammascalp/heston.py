"""Heston (1993) stochastic-volatility model: semi-analytic prices and simulation.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

It is used here to manufacture option chains with a realistic skew and known
ground truth, and as a "true world" in which a Black-Scholes hedger is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # Speed of mean reversion of variance.
    theta: float  # Long-run variance.
    xi: float     # Volatility of variance.
    rho: float    # Correlation between price and variance shocks.
    v0: float     # Initial variance.

    def feller(self) -> float:
        """2 kappa theta / xi^2. Above 1 the variance never touches zero."""
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


def char_fn(u, T, p: HestonParams):
    """E[exp(i u X_T)] for X_T = ln(S_T / F), F the forward price.

    Written in the form of Albrecher et al. (2007), which stays on the
    principal branch of the complex logarithm for all maturities.
    """
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b * b + p.xi * p.xi * (iu + u * u))
    g = (b - d) / (b + d)
    decay = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * decay) / (1.0 - g)))
    D = (b - d) / p.xi**2 * (1.0 - decay) / (1.0 - g * decay)
    return np.exp(C + D * p.v0)


def price(S, K, T, r, p: HestonParams, q=0.0, call=True):
    """European price by Lewis's (2001) single-integral formula."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)

    def integrand(u):
        return np.real(np.exp(1j * u * k) * char_fn(u - 0.5j, T, p)) / (u * u + 0.25)

    integral, _ = quad_vec(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-11, limit=400)
    call_value = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)
    put_value = call_value - np.exp(-r * T) * (F - K)
    return np.where(call, call_value, put_value)


def price_gil_pelaez(S, K, T, r, p: HestonParams, q=0.0, call=True):
    """The same price by Heston's original two-probability formula.

    Kept as an independent check on `price`: different integrand, same answer.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * np.exp((r - q) * T)
    k = np.log(F / K)

    def integrand(u):
        phase = np.exp(1j * u * k) / (1j * u)
        in_the_money = np.real(phase * char_fn(u, T, p))
        share_measure = np.real(phase * char_fn(u - 1j, T, p))
        return np.stack([share_measure, in_the_money])

    integral, _ = quad_vec(integrand, 1e-12, np.inf, epsabs=1e-11, epsrel=1e-11, limit=400)
    p1 = 0.5 + integral[0] / np.pi
    p2 = 0.5 + integral[1] / np.pi
    call_value = np.exp(-r * T) * (F * p1 - K * p2)
    put_value = call_value - np.exp(-r * T) * (F - K)
    return np.where(call, call_value, put_value)


def simulate_step(S, v, dt, r, q, p: HestonParams, z1, z2):
    """One full-truncation Euler step (Lord, Koekkoek and van Dijk, 2010).

    Negative variance is floored at zero inside the drift and diffusion only,
    which keeps the scheme simple and its bias small for small dt.
    """
    v_plus = np.maximum(v, 0.0)
    root = np.sqrt(v_plus * dt)
    w2 = p.rho * z1 + np.sqrt(1.0 - p.rho * p.rho) * z2
    S_next = S * np.exp((r - q - 0.5 * v_plus) * dt + root * z1)
    v_next = v + p.kappa * (p.theta - v_plus) * dt + p.xi * root * w2
    return S_next, v_next
