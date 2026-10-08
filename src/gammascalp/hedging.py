"""Delta-hedge one option along simulated paths and account for every dollar.

The hedger holds the option position, a share position, and a cash account
that earns the risk-free rate. Shares are bought and sold from cash; trading
costs are paid from cash. Terminal P&L is whatever cash remains after the
shares are sold and the option is settled, so nothing can go missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import black_scholes as bs


@dataclass
class HedgeOutcome:
    pnl: np.ndarray        # Terminal P&L per path, after costs.
    costs: np.ndarray      # Trading costs per path, carried to expiry at the risk-free rate.
    trades: np.ndarray     # Number of rebalances that changed the share position.
    gamma_pnl: np.ndarray  # Gamma-rent explanation of the P&L (see `hedge`).


def whalley_wilmott_band(cost, risk_aversion):
    """No-trade band of Whalley and Wilmott (1997) for small proportional costs.

    Half-width, in shares: (3/2 * exp(-r tau) * cost * S * gamma^2 / lambda)^(1/3).
    Wider when trading is expensive or gamma is large; narrower when the
    hedger is more risk averse.
    """

    def band(S, tau, gamma, r):
        return (1.5 * np.exp(-r * tau) * cost * S * gamma * gamma / risk_aversion) ** (1.0 / 3.0)

    return band


def hedge(paths, K, T, r, sigma_hedge, premium, *, position=-1, call=True, q=0.0,
          cost=0.0, every=1, band=None, open_cost=True, close_cost=False) -> HedgeOutcome:
    """Delta-hedge to expiry.

    paths        array (n, M + 1): prices on an even grid from 0 to T
    sigma_hedge  volatility the hedger plugs into Black-Scholes for its delta
    premium      option price paid (position=+1) or received (position=-1) at time 0
    position     +1 long one option, -1 short one option
    cost         proportional one-way cost: trading x shares costs cost * |x| * S
    open_cost    charge that cost on the initial hedge too (close_cost: on the final unwind)
    every        rebalance at every `every`-th grid point (time-based hedging)
    band         None, or a function (S, tau, gamma, r) -> half-width in shares.
                 With a band the hedger checks every grid point, trades only when
                 the Black-Scholes delta leaves the band around its holding, and
                 then trades to the nearest edge of the band, not to the center.

    `gamma_pnl` is the textbook explanation of a delta-hedged P&L: for each
    interval between rebalances,
        position * 1/2 * gamma * S^2 * ( (dS/S)^2 - sigma_hedge^2 * dt ),
    carried to expiry. It is computed from the same paths but by a different
    route than `pnl`, so their agreement is a test, not an identity.
    """
    paths = np.asarray(paths, dtype=float)
    n, columns = paths.shape
    M = columns - 1
    dt = T / M
    growth = np.exp(r * dt)
    carry = np.exp(q * dt) - 1.0  # Dividends on shares held over one step.

    def target(S, tau):
        # A long option is hedged with a short share position, and vice versa.
        return -position * bs.delta(S, K, tau, r, sigma_hedge, q, call)

    S = paths[:, 0]
    held = target(S, T)
    costs = cost * np.abs(held) * S if open_cost else np.zeros(n)
    cash = -position * premium - held * S - costs
    trades = (np.abs(held) > 0).astype(np.int64)
    gamma_pnl = np.zeros(n)
    anchor_S = S.copy()
    anchor_gamma = bs.gamma(S, K, T, r, sigma_hedge, q)
    anchor_step = np.zeros(n, dtype=np.int64)

    for i in range(1, M + 1):
        S = paths[:, i]
        tau = T - i * dt
        cash = cash * growth + held * paths[:, i - 1] * carry
        costs = costs * growth
        gamma_pnl = gamma_pnl * growth
        if i == M:
            trade_now = np.ones(n, dtype=bool)  # Close the explanation at expiry.
            desired = held
        elif band is None:
            trade_now = np.full(n, i % every == 0)
            desired = target(S, tau) if trade_now[0] else held
        else:
            ideal = target(S, tau)
            width = band(S, tau, bs.gamma(S, K, tau, r, sigma_hedge, q), r)
            gap = ideal - held
            trade_now = np.abs(gap) > width
            desired = np.where(trade_now, ideal - np.sign(gap) * width, held)

        if np.any(trade_now):
            elapsed = (i - anchor_step) * dt
            move = S / anchor_S - 1.0
            rent = position * 0.5 * anchor_gamma * anchor_S**2 * (move**2 - sigma_hedge**2 * elapsed)
            gamma_pnl = np.where(trade_now, gamma_pnl + rent, gamma_pnl)
            if i < M:
                change = np.where(trade_now, desired - held, 0.0)
                fee = cost * np.abs(change) * S
                cash = cash - change * S - fee
                costs = costs + fee
                trades = trades + (np.abs(change) > 0)
                held = held + change
                anchor_S = np.where(trade_now, S, anchor_S)
                anchor_gamma = np.where(trade_now, bs.gamma(S, K, tau, r, sigma_hedge, q), anchor_gamma)
                anchor_step = np.where(trade_now, i, anchor_step)

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0) if call else np.maximum(K - S_T, 0.0)
    exit_fee = cost * np.abs(held) * S_T if close_cost else 0.0
    pnl = cash + held * S_T - exit_fee + position * payoff
    return HedgeOutcome(pnl=pnl, costs=costs + exit_fee, trades=trades, gamma_pnl=gamma_pnl)


def leland_volatility(sigma, cost, dt, position=-1):
    """Leland's (1985) adjusted volatility for hedging every `dt` with costs.

    A hedger who rebalances every dt pays, on average, cost * gamma * S^2 *
    sigma * sqrt(2 dt / pi) per step. That is exactly the gamma rent of a
    different volatility. An option seller must charge (and hedge with) the
    higher one; a buyer the lower one.
    """
    leland_number = np.sqrt(8.0 / np.pi) * cost / (sigma * np.sqrt(dt))
    adjusted = sigma * sigma * (1.0 - position * leland_number)
    return float(np.sqrt(adjusted)) if adjusted > 0 else float("nan")


def derman_kamal_std(S, K, T, r, sigma, rebalances, q=0.0):
    """Approximate standard deviation of the hedging error with `rebalances`
    evenly spaced hedges and no costs: sqrt(pi / 4) * vega * sigma / sqrt(N)
    (Derman and Kamal, 1999)."""
    return float(np.sqrt(np.pi / 4.0) * bs.vega(S, K, T, r, sigma, q) * sigma / np.sqrt(rebalances))
