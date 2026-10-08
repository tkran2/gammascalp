"""Seeded price-path generators for the hedging experiments.

Every generator returns an array of shape (paths, steps + 1) whose first
column is S0. `mu` is the real-world drift; a hedger's P&L should not depend on
it to first order, and the experiments test exactly that.
"""

from __future__ import annotations

import numpy as np

from .heston import HestonParams, simulate_step


def gbm(rng, paths, steps, S0, mu, sigma, T):
    """Geometric Brownian motion, sampled exactly (no discretization error)."""
    dt = T / steps
    shocks = rng.standard_normal((paths, steps))
    log_returns = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shocks
    out = np.empty((paths, steps + 1))
    out[:, 0] = S0
    out[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return out


def merton(rng, paths, steps, S0, mu, sigma, T, intensity, jump_mean, jump_std):
    """Merton (1976) jump-diffusion.

    Jumps arrive as a Poisson process with `intensity` per year; each
    multiplies the price by exp(N(jump_mean, jump_std^2)). The drift is
    compensated so that E[S_T] = S0 exp(mu T) with or without jumps.
    """
    dt = T / steps
    compensator = intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    shocks = rng.standard_normal((paths, steps))
    counts = rng.poisson(intensity * dt, size=(paths, steps))
    jump_noise = rng.standard_normal((paths, steps))
    jumps = counts * jump_mean + np.sqrt(counts) * jump_std * jump_noise
    log_returns = (mu - compensator - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shocks + jumps
    out = np.empty((paths, steps + 1))
    out[:, 0] = S0
    out[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return out


def merton_total_variance(sigma, intensity, jump_mean, jump_std):
    """Annualized variance of log-returns: diffusion plus jumps."""
    return sigma * sigma + intensity * (jump_mean**2 + jump_std**2)


def heston(rng, paths, steps, S0, mu, T, params: HestonParams, substeps=4):
    """Heston paths on a grid of `steps`, simulated with `substeps` finer steps each."""
    dt = T / (steps * substeps)
    S = np.full(paths, float(S0))
    v = np.full(paths, float(params.v0))
    out = np.empty((paths, steps + 1))
    out[:, 0] = S0
    for i in range(steps):
        for _ in range(substeps):
            z1 = rng.standard_normal(paths)
            z2 = rng.standard_normal(paths)
            S, v = simulate_step(S, v, dt, mu, 0.0, params, z1, z2)
        out[:, i + 1] = S
    return out
