"""American options on a Cox-Ross-Rubinstein binomial tree.

`price` uses two standard refinements of the plain tree (Broadie and Detemple,
1996): the last step is valued with the Black-Scholes formula, which removes
the sawtooth error caused by the kink in the payoff, and two tree sizes are
combined by Richardson extrapolation.
"""

from __future__ import annotations

import numpy as np

from . import black_scholes as bs


def _tree(S, K, T, r, sigma, q, call, steps, american, smooth):
    dt = T / steps
    up = np.exp(sigma * np.sqrt(dt))
    growth = np.exp((r - q) * dt)
    p = (growth - 1.0 / up) / (up - 1.0 / up)
    if not 0.0 < p < 1.0:
        raise ValueError("time step too coarse for these rates; use more steps")
    discount = np.exp(-r * dt)

    def exercise(spot):
        return np.maximum(spot - K, 0.0) if call else np.maximum(K - spot, 0.0)

    def spots(level):
        return S * up ** (2.0 * np.arange(level + 1) - level)

    if smooth:
        # One step before expiry the continuation value is exactly a European
        # option with dt left, so use the closed form instead of two branches.
        level = steps - 1
        values = bs.price(spots(level), K, dt, r, sigma, q, call)
        if american:
            values = np.maximum(values, exercise(spots(level)))
    else:
        level = steps
        values = exercise(spots(level))
    while level > 0:
        level -= 1
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            values = np.maximum(values, exercise(spots(level)))
    return float(values[0])


def crr(S, K, T, r, sigma, q=0.0, call=False, steps=1000, american=True):
    """Plain CRR tree. Converges as O(1/steps), with oscillation."""
    return _tree(S, K, T, r, sigma, q, call, steps, american, smooth=False)


def price(S, K, T, r, sigma, q=0.0, call=False, steps=400, american=True):
    """Smoothed tree with Richardson extrapolation (two trees: steps and steps/2)."""
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even number of at least 4")
    fine = _tree(S, K, T, r, sigma, q, call, steps, american, smooth=True)
    coarse = _tree(S, K, T, r, sigma, q, call, steps // 2, american, smooth=True)
    return 2.0 * fine - coarse


def early_exercise_premium(S, K, T, r, sigma, q=0.0, call=False, steps=400):
    """How much the right to exercise early is worth."""
    european = float(bs.price(S, K, T, r, sigma, q, call))
    return price(S, K, T, r, sigma, q, call, steps) - european
