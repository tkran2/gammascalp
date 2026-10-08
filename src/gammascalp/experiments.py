"""The five studies. Each one states a prediction from theory, simulates it
with seeded paths, and reports both numbers side by side.

Every function returns plain rows (lists of dicts) so results can be written
to CSV, tested, and plotted without rerunning the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import black_scholes as bs
from . import hedging, heston, paths, svi


@dataclass(frozen=True)
class Market:
    """Base case: a three-month at-the-money call on a $100 stock."""

    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25
    r: float = 0.03
    sigma: float = 0.20
    mu: float = 0.08  # Real-world drift. A delta hedger should not care.
    trading_days: int = 63


def moments(x):
    """Mean, standard deviation, and their standard errors, plus shape."""
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(np.mean(x))
    centered = x - mean
    var = float(np.mean(centered**2))
    std = float(np.sqrt(var))
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    return {
        "n": n,
        "mean": mean,
        "mean_se": std / np.sqrt(n),
        "std": std,
        # Delta method: Var(s) ~ (m4 - s^4) / (4 n s^2).
        "std_se": float(np.sqrt(max(m4 - var * var, 0.0) / (4.0 * n * var))) if var > 0 else 0.0,
        "skew": m3 / std**3 if std > 0 else 0.0,
        "excess_kurtosis": m4 / var**2 - 3.0 if var > 0 else 0.0,
    }


def _chunks(total, size):
    done = 0
    while done < total:
        take = min(size, total - done)
        yield take
        done += take


def _chunk_size(steps):
    return max(1000, min(50000, 8_000_000 // (steps + 1)))


# ---------------------------------------------------------------------------
# Study 1: how large is the hedging error when you can only hedge N times?


def leading_order_std(market: Market, rebalances, points=20001):
    """Standard deviation of the hedging error to leading order in 1/N.

    Over one interval the hedged book earns 1/2 * gamma * S^2 * sigma^2 * dt *
    (1 - Z^2) with Z standard normal, and Z^2 has variance 2. Errors in
    different intervals are uncorrelated, so the total variance is

        sum over hedges of  1/2 * sigma^4 * dt^2 * E[ (gamma * S^2)^2 ],

    each term carried to expiry at the riskless rate. The expectation is over
    the real-world distribution of the stock at each hedge date. The Derman-
    Kamal rule of thumb is this sum evaluated approximately for an at-the-money
    option; this function evaluates it for any strike.
    """
    z = np.linspace(-8.0, 8.0, points)
    weights = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * (z[1] - z[0])
    dt = market.T / rebalances
    variance = 0.0
    for i in range(rebalances):
        t = i * dt
        S = market.S0 * np.exp((market.mu - 0.5 * market.sigma**2) * t + market.sigma * np.sqrt(t) * z)
        dollar_gamma = bs.gamma(S, market.K, market.T - t, market.r, market.sigma) * S * S
        growth = np.exp(2.0 * market.r * (market.T - t - dt))
        variance += 0.5 * market.sigma**4 * dt * dt * growth * float(np.sum(weights * dollar_gamma**2))
    return float(np.sqrt(variance))


def discrete_hedging(market: Market = Market(), rebalances=(4, 8, 16, 32, 63, 126, 252, 504, 1008),
                     strikes=(100.0,), n_paths=200_000, seed=11):
    """Short one call at its Black-Scholes price, hedge N times, no costs.

    Prediction (Derman and Kamal, 1999): the error has mean zero and standard
    deviation sqrt(pi/4) * vega * sigma / sqrt(N), so quadrupling the number of
    hedges halves the risk. That rule of thumb is derived at the money, so each
    row also carries the exact leading-order figure from leading_order_std().
    """
    rows = []
    for K in strikes:
        m = replace(market, K=K)
        premium = float(bs.price(m.S0, m.K, m.T, m.r, m.sigma))
        for N in rebalances:
            rng = np.random.default_rng([seed, int(K * 100), N])
            pnl = []
            for take in _chunks(n_paths, _chunk_size(N)):
                grid = paths.gbm(rng, take, N, m.S0, m.mu, m.sigma, m.T)
                pnl.append(hedging.hedge(grid, m.K, m.T, m.r, m.sigma, premium).pnl)
            stats = moments(np.concatenate(pnl))
            predicted = hedging.derman_kamal_std(m.S0, m.K, m.T, m.r, m.sigma, N)
            leading = leading_order_std(m, N)
            rows.append({
                "strike": K, "rebalances": N, "paths": n_paths, "premium": premium,
                "mean": stats["mean"], "mean_se": stats["mean_se"],
                "std": stats["std"], "std_se": stats["std_se"],
                "predicted_std": predicted, "ratio": stats["std"] / predicted,
                "leading_order_std": leading, "leading_order_ratio": stats["std"] / leading,
                "std_pct_of_premium": 100.0 * stats["std"] / premium,
                "skew": stats["skew"], "excess_kurtosis": stats["excess_kurtosis"],
            })
    return rows


def scaling_exponent(rows, strike=100.0, min_rebalances=16):
    """Slope of log(std) against log(N). Theory says -1/2."""
    use = [r for r in rows if r["strike"] == strike and r["rebalances"] >= min_rebalances]
    x = np.log([r["rebalances"] for r in use])
    y = np.log([r["std"] for r in use])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    se = float(np.sqrt(np.sum(residual**2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))
    return float(slope), se


# ---------------------------------------------------------------------------
# Study 2: you bought volatility cheap. Where does the money come from?


def expected_dollar_gamma(market: Market, sigma_hedge, sigma_realized, mu, points=8001, time_points=400):
    """Integral over the option's life of E[ 1/2 * gamma * S^2 ], carried to expiry.

    Gamma is the hedger's (computed with sigma_hedge); the expectation is over
    the real-world distribution of S_t. This one number converts a variance
    gap into dollars.

    Two numerical details matter. Near expiry gamma is a narrow spike at the
    strike, so the expectation uses a fine uniform grid rather than a few
    Gauss-Hermite nodes. And the integrand grows like 1/sqrt(time left), so
    time is integrated in u = sqrt(time left), where it is smooth.
    """
    z = np.linspace(-8.0, 8.0, points)
    weights = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * (z[1] - z[0])
    edges = np.linspace(0.0, np.sqrt(market.T), time_points + 1)
    total = 0.0
    for u, width in zip(0.5 * (edges[:-1] + edges[1:]), np.diff(edges)):
        left = u * u                      # Time to expiry.
        t = market.T - left
        S = market.S0 * np.exp((mu - 0.5 * sigma_realized**2) * t + sigma_realized * np.sqrt(t) * z)
        dollar_gamma = bs.gamma(S, market.K, left, market.r, sigma_hedge) * S * S
        total += np.exp(market.r * left) * float(np.sum(weights * dollar_gamma)) * 2.0 * u * width
    return 0.5 * total


def expected_gamma_rent(market: Market, sigma_implied, sigma_realized, mu):
    """Continuous-hedging expectation of the P&L when hedging at implied vol.

    Long one option bought at sigma_implied and hedged with the sigma_implied
    delta earns 1/2 * gamma_i * S^2 * (sigma_r^2 - sigma_i^2) per unit time.
    """
    return (sigma_realized**2 - sigma_implied**2) * expected_dollar_gamma(
        market, sigma_implied, sigma_realized, mu)


def drift_bias(market: Market = Market(), drifts=(-0.5, 0.0, 0.5, 1.0), rebalances=(63, 252),
               n_paths=200_000, seed=12):
    """Is a delta hedger really indifferent to the stock's drift?

    In continuous time, yes. With N discrete hedges there is a second-order
    leftover: over one interval the squared move averages
    sigma^2 dt + (mu - r)^2 dt^2 rather than sigma^2 dt, and a short-gamma
    position pays for the extra piece. Prediction for a short option:
        mean P&L  ~  -(mu - r)^2 * (T / N) * integral of E[1/2 gamma S^2].
    """
    premium = float(bs.price(market.S0, market.K, market.T, market.r, market.sigma))
    rows = []
    for mu in drifts:
        exposure = expected_dollar_gamma(market, market.sigma, market.sigma, mu)
        for N in rebalances:
            rng = np.random.default_rng([seed, int(round(mu * 1000)) + 5000, N])
            pnl = []
            for take in _chunks(n_paths, _chunk_size(N)):
                grid = paths.gbm(rng, take, N, market.S0, mu, market.sigma, market.T)
                pnl.append(hedging.hedge(grid, market.K, market.T, market.r, market.sigma, premium).pnl)
            stats = moments(np.concatenate(pnl))
            rows.append({"drift": mu, "rebalances": N, "paths": n_paths,
                         "mean": stats["mean"], "mean_se": stats["mean_se"],
                         "predicted_mean": -(mu - market.r) ** 2 * market.T / N * exposure,
                         "mean_pct_of_premium": 100.0 * stats["mean"] / premium})
    return rows


def volatility_mismatch(market: Market = Market(), sigma_implied=0.20, sigma_realized=0.30,
                        drifts=(-0.40, 0.0, 0.08, 0.40), rebalances=1008, n_paths=100_000, seed=22):
    """Long a call bought at implied vol while the stock realizes a higher vol.

    Two hedgers, same paths:
      * hedge at REALIZED vol: P&L is locked in at V(sigma_r) - V(sigma_i),
        whatever the path (Ahmad and Wilmott, 2005).
      * hedge at IMPLIED vol: P&L is the gamma rent, which depends on where the
        stock spends its time and therefore on the drift.
    """
    premium = float(bs.price(market.S0, market.K, market.T, market.r, sigma_implied))
    locked = float(np.exp(market.r * market.T) *
                   (bs.price(market.S0, market.K, market.T, market.r, sigma_realized) - premium))
    rows = []
    for mu in drifts:
        rng = np.random.default_rng([seed, int(round(mu * 1000)) + 5000])
        at_realized, at_implied, rent = [], [], []
        for take in _chunks(n_paths, _chunk_size(rebalances)):
            grid = paths.gbm(rng, take, rebalances, market.S0, mu, sigma_realized, market.T)
            at_realized.append(hedging.hedge(grid, market.K, market.T, market.r, sigma_realized,
                                             premium, position=+1).pnl)
            outcome = hedging.hedge(grid, market.K, market.T, market.r, sigma_implied,
                                    premium, position=+1)
            at_implied.append(outcome.pnl)
            rent.append(outcome.gamma_pnl)
        at_realized = np.concatenate(at_realized)
        at_implied = np.concatenate(at_implied)
        rent = np.concatenate(rent)
        real = moments(at_realized)
        impl = moments(at_implied)
        rows.append({
            "drift": mu, "sigma_implied": sigma_implied, "sigma_realized": sigma_realized,
            "rebalances": rebalances, "paths": n_paths, "premium": premium,
            "locked_in_value": locked,
            "realized_hedge_mean": real["mean"], "realized_hedge_mean_se": real["mean_se"],
            "realized_hedge_std": real["std"],
            "implied_hedge_mean": impl["mean"], "implied_hedge_mean_se": impl["mean_se"],
            "implied_hedge_std": impl["std"],
            "implied_hedge_p05": float(np.quantile(at_implied, 0.05)),
            "implied_hedge_p95": float(np.quantile(at_implied, 0.95)),
            "theory_implied_hedge_mean": expected_gamma_rent(market, sigma_implied, sigma_realized, mu),
            "gamma_rent_correlation": float(np.corrcoef(at_implied, rent)[0, 1]),
            "gamma_rent_residual_std": float(np.std(at_implied - rent)),
            "share_of_paths_losing": float(np.mean(at_implied < 0)),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 3: trading costs. How often should you hedge, and by what rule?


def leland_check(market: Market = Market(), cost=0.001, rebalances=(16, 32, 63, 126, 252, 504),
                 n_paths=200_000, seed=33):
    """Short a call and hedge with proportional costs.

    Naive: charge the Black-Scholes price and hedge with sigma. The seller
    loses the cost of hedging on average.
    Leland (1985): charge and hedge with the adjusted volatility. Prediction:
    the average P&L returns to zero.
    """
    rows = []
    plain = float(bs.price(market.S0, market.K, market.T, market.r, market.sigma))
    for N in rebalances:
        dt = market.T / N
        adjusted = hedging.leland_volatility(market.sigma, cost, dt)
        leland_premium = float(bs.price(market.S0, market.K, market.T, market.r, adjusted))
        rng = np.random.default_rng([seed, N])
        naive, leland, paid = [], [], []
        for take in _chunks(n_paths, _chunk_size(N)):
            grid = paths.gbm(rng, take, N, market.S0, market.mu, market.sigma, market.T)
            a = hedging.hedge(grid, market.K, market.T, market.r, market.sigma, plain,
                              cost=cost, open_cost=False)
            b = hedging.hedge(grid, market.K, market.T, market.r, adjusted, leland_premium,
                              cost=cost, open_cost=False)
            naive.append(a.pnl)
            paid.append(a.costs)
            leland.append(b.pnl)
        naive, leland, paid = (np.concatenate(v) for v in (naive, leland, paid))
        n_stats, l_stats = moments(naive), moments(leland)
        rows.append({
            "rebalances": N, "cost_bps": cost * 1e4, "paths": n_paths,
            "leland_number": float(np.sqrt(8.0 / np.pi) * cost / (market.sigma * np.sqrt(dt))),
            "adjusted_vol": adjusted, "bs_premium": plain, "leland_premium": leland_premium,
            "predicted_cost": float(np.exp(market.r * market.T) * (leland_premium - plain)),
            "measured_cost": float(np.mean(paid)),
            "naive_mean": n_stats["mean"], "naive_mean_se": n_stats["mean_se"], "naive_std": n_stats["std"],
            "leland_mean": l_stats["mean"], "leland_mean_se": l_stats["mean_se"],
            "leland_std": l_stats["std"],
            "leland_mean_pct_of_premium": 100.0 * l_stats["mean"] / leland_premium,
        })
    return rows


def hedging_frontier(market: Market = Market(), cost=0.001, monitor_steps=1008,
                     intervals=(4, 8, 16, 24, 32, 48, 63, 96, 126, 252, 504, 1008),
                     risk_aversions=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2),
                     n_paths=40_000, seed=44):
    """Risk against cost for two rebalancing rules on the same paths.

    Clock rule: hedge N times at even intervals.
    Band rule (Whalley and Wilmott, 1997): watch continuously, trade only when
    the delta drifts outside a band, and trade just back to the band's edge.
    Each rule traces a curve of (average cost, standard deviation of P&L).
    """
    premium = float(bs.price(market.S0, market.K, market.T, market.r, market.sigma))
    rng = np.random.default_rng([seed])
    grid = paths.gbm(rng, n_paths, monitor_steps, market.S0, market.mu, market.sigma, market.T)
    rows = []
    for N in intervals:
        if monitor_steps % N:
            continue
        out = hedging.hedge(grid, market.K, market.T, market.r, market.sigma, premium,
                            cost=cost, every=monitor_steps // N, open_cost=False)
        stats = moments(out.pnl)
        rows.append({"rule": "clock", "setting": float(N), "mean_cost": float(np.mean(out.costs)),
                     "std": stats["std"], "std_se": stats["std_se"], "mean": stats["mean"],
                     "mean_trades": float(np.mean(out.trades)), "paths": n_paths})
    for lam in risk_aversions:
        out = hedging.hedge(grid, market.K, market.T, market.r, market.sigma, premium,
                            cost=cost, band=hedging.whalley_wilmott_band(cost, lam), open_cost=False)
        stats = moments(out.pnl)
        rows.append({"rule": "band", "setting": float(lam), "mean_cost": float(np.mean(out.costs)),
                     "std": stats["std"], "std_se": stats["std_se"], "mean": stats["mean"],
                     "mean_trades": float(np.mean(out.trades)), "paths": n_paths})
    return rows


def frontier_comparison(rows):
    """At each band point's cost, the clock rule's risk (log-log interpolated)."""
    clock = sorted((r for r in rows if r["rule"] == "clock"), key=lambda r: r["mean_cost"])
    cx = np.log([r["mean_cost"] for r in clock])
    cy = np.log([r["std"] for r in clock])
    out = []
    for r in rows:
        if r["rule"] != "band":
            continue
        x = np.log(r["mean_cost"])
        if not cx[0] <= x <= cx[-1]:
            continue
        clock_std = float(np.exp(np.interp(x, cx, cy)))
        out.append({"risk_aversion": r["setting"], "mean_cost": r["mean_cost"],
                    "band_std": r["std"], "clock_std_at_same_cost": clock_std,
                    "risk_reduction_pct": 100.0 * (1.0 - r["std"] / clock_std)})
    return out


# ---------------------------------------------------------------------------
# Study 4: jumps. What delta hedging cannot remove.


def merton_price(S, K, T, r, sigma, intensity, jump_mean, jump_std, call=True, terms=60):
    """Merton's (1976) series: a Poisson-weighted mix of Black-Scholes prices."""
    k_bar = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam = intensity * (1.0 + k_bar)
    total = 0.0
    weight = np.exp(-lam * T)
    for n in range(terms):
        sigma_n = np.sqrt(sigma**2 + n * jump_std**2 / T)
        r_n = r - intensity * k_bar + n * np.log(1.0 + k_bar) / T
        total += weight * float(bs.price(S, K, T, r_n, sigma_n, 0.0, call))
        weight *= lam * T / (n + 1)
    return total


def jump_risk(market: Market = Market(), diffusion_vol=0.14, intensity=4.0, jump_mean=-0.02,
              jump_std=0.06, rebalances=(16, 63, 252, 1008), n_paths=200_000, seed=55):
    """Short a call when the stock can gap, against a smooth stock with the
    same total variance. The option is sold at its fair (Merton) price in the
    jump world, so both hedgers break even on average; only the risk differs.
    Drift equals the risk-free rate so that the fair price is unambiguous.
    """
    total_vol = float(np.sqrt(paths.merton_total_variance(diffusion_vol, intensity, jump_mean, jump_std)))
    fair = merton_price(market.S0, market.K, market.T, market.r, diffusion_vol, intensity,
                        jump_mean, jump_std)
    smooth_premium = float(bs.price(market.S0, market.K, market.T, market.r, total_vol))
    rows = []
    for N in rebalances:
        rng = np.random.default_rng([seed, N])
        jumpy, smooth = [], []
        for take in _chunks(n_paths, _chunk_size(N)):
            a = paths.merton(rng, take, N, market.S0, market.r, diffusion_vol, market.T,
                             intensity, jump_mean, jump_std)
            b = paths.gbm(rng, take, N, market.S0, market.r, total_vol, market.T)
            jumpy.append(hedging.hedge(a, market.K, market.T, market.r, total_vol, fair).pnl)
            smooth.append(hedging.hedge(b, market.K, market.T, market.r, total_vol, smooth_premium).pnl)
        jumpy, smooth = np.concatenate(jumpy), np.concatenate(smooth)
        j, s = moments(jumpy), moments(smooth)
        rows.append({
            "rebalances": N, "paths": n_paths, "total_vol": total_vol, "fair_premium": fair,
            "smooth_premium": smooth_premium,
            "jump_mean": j["mean"], "jump_mean_se": j["mean_se"], "jump_std": j["std"],
            "jump_skew": j["skew"], "jump_p01": float(np.quantile(jumpy, 0.01)),
            "jump_p001": float(np.quantile(jumpy, 0.001)),
            "smooth_mean": s["mean"], "smooth_mean_se": s["mean_se"], "smooth_std": s["std"],
            "smooth_skew": s["skew"], "smooth_p01": float(np.quantile(smooth, 0.01)),
            "smooth_p001": float(np.quantile(smooth, 0.001)),
        })
    return rows


# ---------------------------------------------------------------------------
# Study 5: from prices to a smile, and whether the smile is arbitrage-free.


DEFAULT_HESTON = heston.HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.04)


def heston_chain(params=DEFAULT_HESTON, S0=100.0, r=0.03, q=0.01,
                 expiries=(1 / 12, 0.25, 0.5, 1.0, 2.0), width=2.5, strikes_per_expiry=25):
    """Out-of-the-money option prices on a grid of strikes for each expiry.

    Strikes span +/- `width` at-the-money standard deviations, like a listed
    chain that gets wider in dollar terms as expiry moves out.
    """
    chain = []
    for T in expiries:
        F = S0 * np.exp((r - q) * T)
        k = np.linspace(-width, width, strikes_per_expiry) * np.sqrt(params.theta * T)
        K = F * np.exp(k)
        is_call = K >= F
        prices = heston.price(S0, K, T, r, params, q, is_call)
        vols = bs.implied_vol(prices, S0, K, T, r, q, is_call)
        chain.append({"T": T, "F": F, "k": k, "K": K, "is_call": is_call, "price": prices,
                      "implied_vol": vols})
    return chain


def smile_fit(params=DEFAULT_HESTON, noise_vol_bps=(0.0, 10.0, 30.0), trials=20, seed=66):
    """Fit raw SVI to each expiry of a Heston chain, clean and with quote noise.

    Reported per expiry: fit error in volatility basis points against the
    quotes and against the true (noise-free) smile, and both arbitrage tests.
    """
    chain = heston_chain(params)
    rows = []
    for noise in noise_vol_bps:
        rng = np.random.default_rng([seed, int(noise * 10)])
        for _ in range(1 if noise == 0 else trials):
            fitted = []
            for leg in chain:
                true_vol = leg["implied_vol"]
                quoted = true_vol + rng.standard_normal(true_vol.shape) * noise * 1e-4
                result = svi.fit(leg["k"], quoted**2 * leg["T"], leg["T"])
                fitted.append(result.params)
                model = result.params.implied_vol(leg["k"], leg["T"])
                rows.append({
                    "noise_vol_bps": noise, "expiry": leg["T"],
                    "rmse_vs_quotes_bps": 1e4 * float(np.sqrt(np.mean((model - quoted) ** 2))),
                    "rmse_vs_truth_bps": 1e4 * float(np.sqrt(np.mean((model - true_vol) ** 2))),
                    "max_abs_error_bps": 1e4 * float(np.max(np.abs(model - true_vol))),
                    "atm_vol": float(result.params.implied_vol(0.0, leg["T"])),
                    "butterfly_min_g": svi.butterfly_violation(
                        result.params, float(leg["k"][0]), float(leg["k"][-1])),
                    "a": result.params.a, "b": result.params.b, "rho": result.params.rho,
                    "m": result.params.m, "sigma": result.params.sigma,
                    "calendar_violation": 0.0,
                })
            worst = svi.calendar_violation(fitted, -0.3, 0.3)
            for row in rows[-len(chain):]:
                row["calendar_violation"] = worst
    return chain, rows


def implied_vol_accuracy(n=1_000_000, seed=77):
    """Round-trip a million random options through price -> implied vol."""
    import time

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    total_vol = rng.uniform(0.02, 1.5, n)
    T = rng.uniform(0.02, 3.0, n)
    call = rng.random(n) < 0.5
    S, r, q = 100.0, 0.03, 0.01
    K = S * np.exp(-x)
    sigma = total_vol / np.sqrt(T)
    prices = bs.price(S, K, T, r, sigma, q, call)
    started = time.perf_counter()
    solved = bs.implied_vol(prices, S, K, T, r, q, call)
    seconds = time.perf_counter() - started
    F = S * np.exp((r - q) * T)
    time_value = np.where(K >= F, bs.price(S, K, T, r, sigma, q, True),
                          bs.price(S, K, T, r, sigma, q, False)) * np.exp(r * T) / F
    error = np.abs(solved - sigma)
    rows = []
    for floor in (1e-4, 1e-6, 1e-8, 1e-10):
        mask = time_value > floor
        rows.append({"time_value_above": floor, "options": int(mask.sum()),
                     "solved": int(np.isfinite(solved[mask]).sum()),
                     "max_abs_error": float(np.nanmax(error[mask])),
                     "median_abs_error": float(np.nanmedian(error[mask]))})
    return rows, {"options": n, "seconds": seconds, "options_per_second": n / seconds,
                  "returned_nan": int(np.isnan(solved).sum())}
