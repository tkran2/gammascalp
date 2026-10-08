"""Black-Scholes-Merton prices, Greeks, and a robust implied-volatility solver.

Conventions used everywhere in this package:
  S      spot price
  K      strike
  T      time to expiry in years
  r, q   continuously compounded risk-free rate and dividend yield
  sigma  annualized volatility
All functions accept NumPy arrays and broadcast.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def d1_d2(S, K, T, r, sigma, q=0.0):
    """The two standardized distances. At T = 0 they collapse to +/- infinity."""
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (S, K, T, sigma)))
    vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    # 0/0 happens only exactly at the money at expiry; any finite value gives
    # the right price there (the option is worth zero).
    d1 = np.where(np.isnan(d1), 0.0, d1)
    return d1, d1 - vol


def price(S, K, T, r, sigma, q=0.0, call=True):
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    forward_leg = np.asarray(S, dtype=float) * np.exp(-q * np.asarray(T, dtype=float))
    strike_leg = np.asarray(K, dtype=float) * np.exp(-r * np.asarray(T, dtype=float))
    call_value = forward_leg * ndtr(d1) - strike_leg * ndtr(d2)
    put_value = strike_leg * ndtr(-d2) - forward_leg * ndtr(-d1)
    return np.where(call, call_value, put_value)


def delta(S, K, T, r, sigma, q=0.0, call=True):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-q * np.asarray(T, dtype=float))
    return np.where(call, discount * ndtr(d1), -discount * ndtr(-d1))


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-q * T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where(T > 0, value, 0.0)


def vega(S, K, T, r, sigma, q=0.0):
    """Change in price per unit change in sigma (per 1.00, not per vol point)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T = np.asarray(T, dtype=float)
    return np.asarray(S, dtype=float) * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, call=True):
    """Change in price per year of calendar time passing (usually negative)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -S * np.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * np.sqrt(T))
    decay = np.where(T > 0, decay, 0.0)
    call_value = decay - r * K * np.exp(-r * T) * ndtr(d2) + q * S * np.exp(-q * T) * ndtr(d1)
    put_value = decay + r * K * np.exp(-r * T) * ndtr(-d2) - q * S * np.exp(-q * T) * ndtr(-d1)
    return np.where(call, call_value, put_value)


def implied_vol(option_price, S, K, T, r, q=0.0, call=True, tol=1e-14, max_iter=100):
    """Volatility that reproduces `option_price`, or NaN where none exists.

    The price is first converted to the out-of-the-money option with the same
    strike (put-call parity), because that is where the time value lives and
    floating-point error is smallest. The equation is then solved in total
    volatility s = sigma * sqrt(T) by Newton's method started at the inflection
    point sqrt(2 |ln(F/K)|), where the iteration is monotone (Manaster and
    Koehler, 1982). Every step is kept inside a shrinking bracket, falling back
    to bisection, so a flat vega far from the money cannot throw it off.

    NaN is returned when the price is outside the no-arbitrage bounds, and
    also when its time value is below double-precision noise (about 1e-13 of
    the forward): such a price is consistent with every small volatility.
    """
    option_price, S, K, T = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (option_price, S, K, T)))
    shape = option_price.shape
    call = np.broadcast_to(np.asarray(call, dtype=bool), shape).ravel()
    option_price, S, K, T = (a.ravel() for a in (option_price, S, K, T))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        F = S * np.exp((r - q) * T)
        undiscounted = option_price * np.exp(r * T)
        x = np.log(F / K)
        otm_is_call = K >= F
        # C - P = F - K, undiscounted.
        other_side = np.where(call, undiscounted - (F - K), undiscounted + (F - K))
        value = np.where(call == otm_is_call, undiscounted, other_side)
        upper = np.where(otm_is_call, F, K)
        valid = ((T > 0) & (S > 0) & (K > 0) & np.isfinite(value) &
                 (value > 1e-13 * np.maximum(F, K)) & (value < upper))

        result = np.full(option_price.shape, np.nan)
        index = np.flatnonzero(valid)
        if index.size:
            F, K, x, value, otm_is_call = (a[index] for a in (F, K, x, value, otm_is_call))
            s = np.where(np.abs(x) > 1e-6, np.sqrt(2.0 * np.abs(x)), _SQRT_2PI * value / F)
            s = np.clip(s, 1e-8, 10.0)
            low = np.zeros_like(s)
            high = np.full_like(s, 40.0)
            total_vol = np.empty_like(s)
            active = np.arange(s.size)
            for _ in range(max_iter):
                d1 = x / s + 0.5 * s
                d2 = d1 - s
                model = np.where(otm_is_call, F * ndtr(d1) - K * ndtr(d2),
                                 K * ndtr(-d2) - F * ndtr(-d1))
                f = model - value
                slope = F * norm_pdf(d1)
                high = np.where(f > 0, s, high)
                low = np.where(f <= 0, s, low)
                candidate = s - np.where(slope > 0, f / slope, np.inf)
                inside = np.isfinite(candidate) & (candidate > low) & (candidate < high)
                new_s = np.where(inside, candidate, 0.5 * (low + high))
                done = np.abs(new_s - s) <= tol * np.maximum(1.0, s)
                total_vol[active[done]] = new_s[done]
                # Keep iterating only on the options that have not converged.
                keep = ~done
                if not np.any(keep):
                    active = active[:0]
                    break
                active = active[keep]
                F, K, x, value, otm_is_call, low, high, s = (
                    a[keep] for a in (F, K, x, value, otm_is_call, low, high, new_s))
            if active.size:
                total_vol[active] = s  # Anything left is at its bisection limit.
            result[index] = total_vol / np.sqrt(T[index])
    return result.reshape(shape)
