#!/usr/bin/env python3
"""Turn results/*.csv into figures and the research memo.

    python scripts/build_report.py            # reads results/, writes reports/

The memo's numbers are read from the CSV files and manifest, so rerunning the
study and then this script keeps the text and the data in agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gammascalp import black_scholes as bs  # noqa: E402
from gammascalp import experiments as ex  # noqa: E402
from gammascalp import hedging, paths  # noqa: E402

# One small, fixed palette. The three series colors stay distinguishable under
# the common forms of color blindness; text is always ink, never a series color.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # Light to dark, for ordered expiries.

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.markersize": 7, "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
    "svg.fonttype": "none",
})


def read(results, name):
    with open(os.path.join(results, name), newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plain_log_ticks(axis, values):
    axis.set_ticks(values)
    axis.set_ticklabels([f"{v:g}" for v in values])
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def figure_discrete(results, out):
    rows = read(results, "discrete_hedging.csv")
    atm = [r for r in rows if r["strike"] == 100.0]
    n = np.array([r["rebalances"] for r in atm])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.30})

    left.loglog(n, [r["predicted_std"] for r in atm], color=MUTED, linewidth=1.4, zorder=2,
                label="rule of thumb")
    left.loglog(n, [r["std"] for r in atm], "o", color=BLUE, zorder=3, label="simulated")
    left.set_title("Hedging error falls like 1/√N")
    left.set_xlabel("Number of hedges over three months")
    left.set_ylabel("Std of hedged P&L ($ per option)")
    plain_log_ticks(left.xaxis, [4, 16, 63, 252, 1008])
    plain_log_ticks(left.yaxis, [0.1, 0.2, 0.5, 1.0, 2.0])
    left.legend(loc="upper right")

    # Color says which formula; the strike is written at the end of each line.
    for column, color in (("ratio", ORANGE), ("leading_order_ratio", BLUE)):
        for strike in (90.0, 100.0, 110.0):
            part = [r for r in rows if r["strike"] == strike]
            right.semilogx([r["rebalances"] for r in part], [r[column] for r in part], "-o", color=color,
                           markersize=5.5, markeredgewidth=1.0,
                           label={"ratio": "rule of thumb", "leading_order_ratio": "exact leading-order sum"}[column]
                           if strike == 90.0 else None)
            if column == "ratio":
                right.annotate(f"K={int(strike)}", xy=(part[-1]["rebalances"], part[-1][column]),
                               xytext=(7, -3), textcoords="offset points", color=INK_2, fontsize=8.5)
    right.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    right.set_title("Simulated risk ÷ predicted risk, by strike")
    right.set_xlabel("Number of hedges over three months")
    right.set_ylabel("Simulated std / predicted std")
    right.set_xlim(3.2, 2600)
    plain_log_ticks(right.xaxis, [4, 16, 63, 252, 1008])
    right.legend(loc="center right", bbox_to_anchor=(1.0, 0.66))
    save(fig, out, "discrete_hedging")


def figure_mismatch(results, out):
    rows = read(results, "volatility_mismatch.csv")
    market = ex.Market()
    implied, realized = rows[0]["sigma_implied"], rows[0]["sigma_realized"]
    premium = rows[0]["premium"]
    grid = paths.gbm(np.random.default_rng(2024), 30000, 504, market.S0, market.mu, realized, market.T)
    at_realized = hedging.hedge(grid, market.K, market.T, market.r, realized, premium, position=+1).pnl
    at_implied = hedging.hedge(grid, market.K, market.T, market.r, implied, premium, position=+1).pnl

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.28})
    bins = np.linspace(0.0, 4.5, 91)
    left.hist(at_implied, bins=bins, color=ORANGE, alpha=0.55, label="hedge at implied vol (20%)")
    left.hist(at_realized, bins=bins, color=BLUE, alpha=0.55, label="hedge at realized vol (30%)")
    left.axvline(rows[0]["locked_in_value"], color=INK_2, linewidth=1.0)
    left.annotate("V(30%) − V(20%)", xy=(rows[0]["locked_in_value"], left.get_ylim()[1] * 0.93),
                  xytext=(8, 0), textcoords="offset points", color=INK_2, fontsize=9)
    left.set_title("Same edge, two very different P&L distributions")
    left.set_xlabel("P&L of a long call bought at 20% vol ($)")
    left.set_ylabel("Paths (of 30,000)")
    left.set_ylim(0, left.get_ylim()[1] * 1.22)
    left.legend(loc="upper left")

    drift = [r["drift"] for r in rows]
    dense = np.linspace(min(drift), max(drift), 33)
    theory = [ex.expected_gamma_rent(market, implied, realized, mu) for mu in dense]
    right.plot(dense, theory, color=MUTED, linewidth=1.4, zorder=2, label="continuous-hedging theory")
    right.errorbar(drift, [r["implied_hedge_mean"] for r in rows],
                   yerr=[2 * r["implied_hedge_mean_se"] for r in rows], fmt="o", color=ORANGE,
                   ecolor=ORANGE, elinewidth=1.2, label="hedge at implied vol", zorder=3)
    right.errorbar(drift, [r["realized_hedge_mean"] for r in rows],
                   yerr=[2 * r["realized_hedge_mean_se"] for r in rows], fmt="o", color=BLUE,
                   ecolor=BLUE, elinewidth=1.2, label="hedge at realized vol", zorder=3)
    right.set_title("Only the implied-vol hedge cares about drift")
    right.set_xlabel("Real-world drift of the stock (per year)")
    right.set_ylabel("Mean P&L ($), bars = ±2 standard errors")
    right.legend(loc="lower center")
    save(fig, out, "volatility_mismatch")


def figure_costs(results, out):
    leland = read(results, "leland.csv")
    frontier = read(results, "frontier.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.28})

    n = [r["rebalances"] for r in leland]
    left.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    left.errorbar(n, [r["naive_mean"] for r in leland], yerr=[2 * r["naive_mean_se"] for r in leland],
                  fmt="-o", color=ORANGE, ecolor=ORANGE, elinewidth=1.2,
                  label="Black-Scholes price and delta")
    left.errorbar(n, [r["leland_mean"] for r in leland], yerr=[2 * r["leland_mean_se"] for r in leland],
                  fmt="-o", color=BLUE, ecolor=BLUE, elinewidth=1.2,
                  label="Leland-adjusted price and delta")
    left.set_xscale("log")
    plain_log_ticks(left.xaxis, [16, 32, 63, 126, 252, 504])
    left.set_title("Leland pricing removes the cost drag")
    left.set_xlabel("Number of hedges (10 bp cost per trade)")
    left.set_ylabel("Mean P&L of the option seller ($)")
    left.legend(loc="lower left")

    clock = sorted((r for r in frontier if r["rule"] == "clock"), key=lambda r: r["mean_cost"])
    band = sorted((r for r in frontier if r["rule"] == "band"), key=lambda r: r["mean_cost"])
    right.plot([r["mean_cost"] for r in clock], [r["std"] for r in clock], "-o", color=BLUE,
               label="hedge on a clock")
    right.plot([r["mean_cost"] for r in band], [r["std"] for r in band], "-o", color=ORANGE,
               label="hedge when delta leaves a band")
    for r in clock:
        if r["setting"] in (16.0, 63.0, 1008.0):
            right.annotate(f"{int(r['setting'])} hedges", xy=(r["mean_cost"], r["std"]), xytext=(8, 6),
                           textcoords="offset points", color=INK_2, fontsize=8.5)
    right.set_xscale("log")
    right.set_yscale("log")
    plain_log_ticks(right.xaxis, [0.05, 0.1, 0.2, 0.5, 1.0])
    plain_log_ticks(right.yaxis, [0.25, 0.5, 1.0, 1.5])
    right.set_title("Bands beat the clock at equal cost")
    right.set_xlabel("Average trading cost paid ($ per option)")
    right.set_ylabel("Std of hedged P&L ($)")
    right.set_xlim(0.045, 2.6)
    right.legend(loc="upper right")
    save(fig, out, "trading_costs")


def figure_jumps(results, out):
    rows = read(results, "jump_risk.csv")
    market = ex.Market()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.28})
    n = [r["rebalances"] for r in rows]
    left.loglog(n, [r["jump_std"] for r in rows], "-o", color=ORANGE, label="stock with jumps")
    left.loglog(n, [r["smooth_std"] for r in rows], "-o", color=BLUE, label="smooth stock, same total variance")
    plain_log_ticks(left.xaxis, [16, 63, 252, 1008])
    plain_log_ticks(left.yaxis, [0.1, 0.2, 0.5, 1.0, 2.0])
    left.set_title("Hedging more often does not remove jump risk")
    left.set_xlabel("Number of hedges over three months")
    left.set_ylabel("Std of hedged P&L ($ per option)")
    left.legend(loc="lower left")

    total_vol = rows[0]["total_vol"]
    rng = np.random.default_rng(2025)
    jumpy = paths.merton(rng, 60000, 252, market.S0, market.r, 0.14, market.T, 4.0, -0.02, 0.06)
    smooth = paths.gbm(rng, 60000, 252, market.S0, market.r, total_vol, market.T)
    a = hedging.hedge(jumpy, market.K, market.T, market.r, total_vol, rows[0]["fair_premium"]).pnl
    b = hedging.hedge(smooth, market.K, market.T, market.r, total_vol, rows[0]["smooth_premium"]).pnl
    bins = np.linspace(-12, 4, 97)
    right.hist(a, bins=bins, color=ORANGE, alpha=0.55, label="stock with jumps")
    right.hist(b, bins=bins, color=BLUE, alpha=0.55, label="smooth stock")
    right.set_yscale("log")
    right.set_title("The seller's losses have a long left tail (252 hedges)")
    right.set_xlabel("P&L of a short, delta-hedged call ($)")
    right.set_ylabel("Paths (of 60,000, log scale)")
    right.legend(loc="upper left")
    save(fig, out, "jump_risk")


def figure_smile(results, out):
    chain = read(results, "heston_chain.csv")
    fits = [r for r in read(results, "smile_fit.csv") if r["noise_vol_bps"] == 0.0]
    from gammascalp import svi

    expiries = sorted({r["expiry"] for r in chain})
    shown = [expiries[0], expiries[2], expiries[4]]
    labels = {expiries[0]: "1 month", expiries[2]: "6 months", expiries[4]: "2 years"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.28})
    for T, color in zip(shown, BLUE_RAMP):
        leg = [r for r in chain if r["expiry"] == T]
        fit = next(r for r in fits if abs(r["expiry"] - T) < 1e-9)
        params = svi.SVI(fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
        k = np.array([r["log_moneyness"] for r in leg])
        vol = np.array([r["implied_vol"] for r in leg])
        dense = np.linspace(k[0], k[-1], 200)
        left.plot(dense, 100 * params.implied_vol(dense, T), color=color, label=labels[T])
        left.plot(k, 100 * vol, "o", color=color, markersize=5.5, markeredgewidth=1.0)
        right.plot(k, 1e4 * (params.implied_vol(k, T) - vol), "-o", color=color, markersize=5.5,
                   markeredgewidth=1.0, label=labels[T])
    left.set_title("Heston quotes (dots) and fitted SVI smiles (lines)")
    left.set_xlabel("Log-moneyness ln(K / F)")
    left.set_ylabel("Implied volatility (%)")
    left.legend(loc="upper right", title="Expiry")
    right.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    right.set_title("Fit error stays within a few basis points")
    right.set_xlabel("Log-moneyness ln(K / F)")
    right.set_ylabel("Fitted minus quoted vol (basis points)")
    right.legend(loc="upper right", title="Expiry")
    save(fig, out, "smile_fit")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" ---: " for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def memo(results, out):
    with open(os.path.join(results, "manifest.json")) as f:
        manifest = json.load(f)
    s = manifest["summary"]
    all_discrete = read(results, "discrete_hedging.csv")
    discrete = [r for r in all_discrete if r["strike"] == 100.0]
    drift = read(results, "drift_bias.csv")
    mismatch = read(results, "volatility_mismatch.csv")
    leland = read(results, "leland.csv")
    compared = read(results, "frontier_comparison.csv")
    jumps = read(results, "jump_risk.csv")
    accuracy = read(results, "implied_vol_accuracy.csv")
    smile = read(results, "smile_fit.csv")
    market = ex.Market()
    premium = float(bs.price(market.S0, market.K, market.T, market.r, market.sigma))
    total_paths = int(sum(r["paths"] for r in read(results, "discrete_hedging.csv")) +
                      sum(r["paths"] for r in drift) + 2 * sum(r["paths"] for r in mismatch) +
                      2 * sum(r["paths"] for r in leland) +
                      sum(r["paths"] for r in read(results, "frontier.csv")) +
                      2 * sum(r["paths"] for r in jumps))

    def clean(noise):
        part = [r for r in smile if r["noise_vol_bps"] == noise]
        return float(np.mean([r["rmse_vs_truth_bps"] for r in part])), float(
            np.mean([r["rmse_vs_quotes_bps"] for r in part]))

    best = max(compared, key=lambda r: r["risk_reduction_pct"])
    off = [r for r in discrete if abs(r["mean"]) > 2 * r["mean_se"]]
    if off:
        biased = ("The mean is within two standard errors of zero except at "
                  + " and ".join(f"{int(r['rebalances'])} hedges ({100 * r['mean'] / r['premium']:+.2f}% of premium)"
                                 for r in off)
                  + ". The sign is the one the next subsection predicts for so few hedges.")
    else:
        biased = "The mean is within two standard errors of zero in every row."
    text = f"""# GammaScalp research memo

**Question.** The Black-Scholes argument says an option can be replicated by
trading the stock continuously, for free. Nobody can do either. How much risk
is left when hedging is discrete, costly, and done with the wrong volatility,
and do the classical correction formulas predict it?

**Method.** One short (or long) three-month at-the-money call on a $100 stock,
20% volatility, 3% rate, premium ${premium:.4f}. Every study states a prediction
from the literature, simulates seeded paths, and reports both numbers.
{total_paths:,} hedging runs in total (a run is one hedger on one price path; some
studies put two hedgers or several rules on the same paths). All figures below
are synthetic: they test the formulas and the code, not any market.

## 1. Discrete hedging: risk falls like 1/sqrt(N)

Derman and Kamal (1999) predict a hedging error with mean zero and standard
deviation `sqrt(pi/4) * vega * sigma / sqrt(N)` for N evenly spaced hedges.

{table(["Hedges", "Mean P&L", "± s.e.", "Std", "Predicted", "Ratio", "Std / premium"],
       [[f"{int(r['rebalances'])}", f"{r['mean']:+.4f}", f"{r['mean_se']:.4f}", f"{r['std']:.4f}",
         f"{r['predicted_std']:.4f}", f"{r['ratio']:.3f}", f"{r['std_pct_of_premium']:.1f}%"]
        for r in discrete])}

- A log-log regression of std on N (16 hedges and up) gives a slope of
  **{s['discrete_hedging']['scaling_exponent']:.3f} ± {s['discrete_hedging']['scaling_exponent_se']:.3f}**; theory says -0.5.
- At 63 hedges (daily) the simulated std is {100 * s['discrete_hedging']['ratio_to_theory_at_63']:.1f}% of the formula;
  at 1,008 hedges it is {100 * s['discrete_hedging']['ratio_to_theory_at_1008']:.1f}%. The formula is an
  asymptotic one and overstates risk when hedges are few.
- Daily hedging still leaves a standard deviation of about
  {next(r['std_pct_of_premium'] for r in discrete if r['rebalances'] == 63):.0f}% of the premium.
- {biased}

**The rule of thumb is an at-the-money result.** The same experiment at
strikes 90 and 110 shows it understating risk: at 1,008 hedges the simulated
standard deviation is {100 * (min(s['discrete_hedging']['rule_of_thumb_ratio_off_the_money']) - 1):.0f}% to {100 * (max(s['discrete_hedging']['rule_of_thumb_ratio_off_the_money']) - 1):.0f}% above the formula, and the gap does not close with more hedges.
The step it skips can be done exactly: each interval contributes
`1/2 * sigma^4 * dt^2 * E[(gamma * S^2)^2]` of variance, and summing that over
the hedge dates by quadrature (`leading_order_std`) gives a prediction for any
strike.

{table(["Strike", "Hedges", "Simulated std", "Rule of thumb", "Ratio", "Exact sum", "Ratio"],
       [[f"{int(r['strike'])}", f"{int(r['rebalances'])}", f"{r['std']:.4f}", f"{r['predicted_std']:.4f}",
         f"{r['ratio']:.3f}", f"{r['leading_order_std']:.4f}", f"{r['leading_order_ratio']:.3f}"]
        for r in all_discrete if r['rebalances'] in (63, 1008)])}

At 1,008 hedges the exact sum is within {100 * max(abs(x - 1) for x in s['discrete_hedging']['leading_order_ratio_at_1008']):.1f}% of the simulation at all three strikes;
at 63 hedges, within {100 * max(abs(x - 1) for x in s['discrete_hedging']['leading_order_ratio_at_63']):.1f}%. It is a leading-order result, so it too drifts when hedges are few.

![Discrete hedging](figures/discrete_hedging.png)

### A second-order effect: the drift does matter a little

In continuous time a delta hedger is indifferent to the stock's drift. With N
discrete hedges the squared move over one interval averages
`sigma^2 dt + (mu - r)^2 dt^2`, so a short-gamma position pays a small extra
amount. Integrating gives a prediction for a short option:
`mean = -(mu - r)^2 * (T / N) * E[integral of 1/2 gamma S^2]`.

{table(["Drift", "Hedges", "Mean P&L", "± s.e.", "Predicted", "% of premium"],
       [[f"{r['drift']:+.2f}", f"{int(r['rebalances'])}", f"{r['mean']:+.4f}", f"{r['mean_se']:.4f}",
         f"{r['predicted_mean']:+.4f}", f"{r['mean_pct_of_premium']:+.2f}%"] for r in drift])}

The largest gap between simulation and this prediction is
{s['drift_bias']['largest_gap_to_prediction_in_se']:.1f} standard errors. For realistic drifts the effect is
invisible; it becomes about {abs(s['drift_bias']['worst_mean_pct_of_premium']):.1f}% of premium only at a drift of 100% a year
with daily hedging.

## 2. Buying volatility cheap: where the money comes from

A call is bought at 20% implied volatility; the stock then realizes 30%. The
value of that edge is `V(30%) - V(20%)` = ${mismatch[0]['locked_in_value']:.4f} at expiry. Two hedgers
trade the same paths (Ahmad and Wilmott, 2005).

{table(["Drift", "Hedge at realized: mean", "std", "Hedge at implied: mean", "± s.e.", "theory", "std", "5% to 95%"],
       [[f"{r['drift']:+.2f}", f"{r['realized_hedge_mean']:.4f}", f"{r['realized_hedge_std']:.3f}",
         f"{r['implied_hedge_mean']:.4f}", f"{r['implied_hedge_mean_se']:.4f}",
         f"{r['theory_implied_hedge_mean']:.4f}", f"{r['implied_hedge_std']:.3f}",
         f"{r['implied_hedge_p05']:.2f} to {r['implied_hedge_p95']:.2f}"] for r in mismatch])}

- Hedging at **realized** volatility locks in the edge: the mean equals
  `V(30%) - V(20%)` at every drift, and the spread around it is only the
  discrete-hedging noise of study 1.
- Hedging at **implied** volatility earns the gamma rent
  `1/2 * gamma * S^2 * (sigma_r^2 - sigma_i^2)` each instant. Its total depends
  on how long the stock stays near the strike, so the mean moves with drift.
  The quadrature of that integral matches the simulated mean in every row.
- Per path, the gamma-rent sum explains the simulated P&L with a correlation of
  at least {s['volatility_mismatch']['min_gamma_rent_correlation']:.4f}.
- Share of paths on which the implied-vol hedger lost money: at most
  {100 * s['volatility_mismatch']['max_share_of_paths_losing']:.2f}%. The outcome is uncertain in size, not in sign.

![Volatility mismatch](figures/volatility_mismatch.png)

## 3. Trading costs: what to charge, and when to trade

With a proportional cost of 10 basis points per trade, a seller who charges
the Black-Scholes price loses the cost of hedging. Leland (1985) says to price
and hedge with `sigma^2 * (1 + sqrt(8/pi) * cost / (sigma * sqrt(dt)))`.

{table(["Hedges", "Leland number", "Adjusted vol", "Leland markup", "Plain hedge cost", "Naive mean", "Leland mean", "± s.e."],
       [[f"{int(r['rebalances'])}", f"{r['leland_number']:.3f}", f"{100 * r['adjusted_vol']:.2f}%",
         f"{r['predicted_cost']:.4f}", f"{r['measured_cost']:.4f}", f"{r['naive_mean']:+.4f}",
         f"{r['leland_mean']:+.4f}", f"{r['leland_mean_se']:.4f}"] for r in leland])}

- The adjusted price brings the seller's mean P&L back to within
  {s['leland']['max_abs_leland_mean_pct_of_premium']:.2f}% of the premium at every frequency tested.
- The markup is what the adjusted hedge costs, not what the unadjusted hedge
  costs. At {int(leland[-1]['rebalances'])} hedges the seller who keeps the plain Black-Scholes delta pays
  {100 * (leland[-1]['measured_cost'] / leland[-1]['predicted_cost'] - 1):.1f}% more than the markup, because hedging at the higher volatility
  also flattens gamma and so trades less. At {int(leland[0]['rebalances'])} hedges the two differ by
  {abs(100 * (leland[0]['measured_cost'] / leland[0]['predicted_cost'] - 1)):.1f}%.

Costs make "hedge more often" a trade-off. Two rules were run on the same
{int(read(results, 'frontier.csv')[0]['paths']):,} paths: hedge on a clock, or watch continuously and trade only
when delta leaves a band (Whalley and Wilmott, 1997), trading back to the
band's edge.

{table(["Risk aversion", "Avg cost", "Band std", "Clock std at same cost", "Risk reduction"],
       [[f"{r['risk_aversion']:g}", f"{r['mean_cost']:.4f}", f"{r['band_std']:.4f}",
         f"{r['clock_std_at_same_cost']:.4f}", f"{r['risk_reduction_pct']:.1f}%"] for r in compared])}

At equal average cost the band rule carried {s['frontier']['risk_reduction_pct_min']:.0f}% to
{s['frontier']['risk_reduction_pct_max']:.0f}% less risk than the clock rule (largest at a cost of
${best['mean_cost']:.2f}). The clock rule's risk also stops improving past a few hundred
hedges, because the cost itself becomes a source of variance. The clock
curve is interpolated between tested frequencies, and the band is the
small-cost asymptotic formula, not a numerically optimal one.

![Trading costs](figures/trading_costs.png)

## 4. Jumps: the risk delta hedging cannot remove

A stock with four jumps a year on average (mean -2%, std 6%) is compared with a
smooth stock of the same total variance ({100 * jumps[0]['total_vol']:.1f}% volatility). The
option is sold at its fair Merton price, so both sellers break even on average.

{table(["Hedges", "Smooth std", "Jump std", "Jump skew", "Smooth 1% worst", "Jump 1% worst", "Jump mean", "± s.e."],
       [[f"{int(r['rebalances'])}", f"{r['smooth_std']:.3f}", f"{r['jump_std']:.3f}", f"{r['jump_skew']:.2f}",
         f"{r['smooth_p01']:.2f}", f"{r['jump_p01']:.2f}", f"{r['jump_mean']:+.4f}", f"{r['jump_mean_se']:.4f}"]
        for r in jumps])}

Going from {int(jumps[0]['rebalances'])} to {int(jumps[-1]['rebalances'])} hedges cut the smooth stock's risk by
{s['jump_risk']['smooth_std_reduction_pct']:.0f}% and the jumping stock's by {s['jump_risk']['jump_std_reduction_pct']:.0f}%.
At the highest frequency the seller's 1%-worst outcome is
{s['jump_risk']['p01_ratio_at_most_frequent']:.0f} times larger with jumps. A gap moves the stock before
the hedge can follow; only owning other options removes that risk.

![Jump risk](figures/jump_risk.png)

## 5. From prices to an arbitrage-free smile

**Implied volatility.** {int(s['implied_vol']['options']):,} random options were priced and
inverted at {s['implied_vol']['options_per_second']:,.0f} options per second on the recorded
machine. Accuracy is limited by floating point, not by the solver:

{table(["Time value above (× forward)", "Options", "Solved", "Max abs error", "Median abs error"],
       [[f"{r['time_value_above']:g}", f"{int(r['options']):,}", f"{int(r['solved']):,}",
         f"{r['max_abs_error']:.1e}", f"{r['median_abs_error']:.1e}"] for r in accuracy])}

**Smile fitting.** Option chains were generated from a Heston model
(kappa 2, theta 0.04, xi 0.6, rho -0.7), five expiries of 25 strikes each,
and a raw SVI slice was fitted to each expiry.

{table(["Quote noise (vol bp)", "Fits", "RMSE vs quotes (bp)", "RMSE vs true smile (bp)"],
       [[f"{noise:g}", f"{sum(r['noise_vol_bps'] == noise for r in smile)}", f"{clean(noise)[1]:.2f}",
         f"{clean(noise)[0]:.2f}"] for noise in sorted({r['noise_vol_bps'] for r in smile})])}

- Clean quotes are fitted to {s['smile_fit']['clean_rmse_bps_mean']:.1f} basis points on average
  (worst expiry {s['smile_fit']['clean_rmse_bps_max']:.1f}). SVI is not the Heston smile, so the error is not zero.
- With noisy quotes the fit is closer to the true smile than the quotes are:
  five parameters average out the noise of 25 strikes.
- {s['smile_fit']['fits_with_butterfly_arbitrage']} of {s['smile_fit']['fits']} fitted slices show butterfly arbitrage
  inside the quoted strike range, and the worst calendar violation is {s['smile_fit']['worst_calendar_violation']:.1g}.
  The test suite confirms the detector does fire on a published arbitrageable smile.

![Smile fit](figures/smile_fit.png)

## Limitations

- Every result is a simulation under a stated model. Nothing here is a backtest
  or evidence that a strategy makes money in a real market.
- One option, one maturity, constant rates, no dividends in the hedging
  studies, no bid-ask spread on the option, and costs proportional to notional.
- The jump and volatility-mismatch studies give the hedger a volatility number
  it could not know in advance.
- Standard errors describe Monte Carlo noise only. Comparing many table rows
  to theory was not corrected for multiple comparisons.
- The Leland and Whalley-Wilmott formulas are small-cost approximations; the
  comparison with them is a check of their accuracy in one configuration.

## Reproduce

```bash
python -m gammascalp research        # about {manifest['seconds'] / 60:.0f} minutes; writes results/
python scripts/build_report.py       # figures and this memo
```

Versions used for the delivered run: Python {manifest['python']}, NumPy {manifest['numpy']},
SciPy {manifest['scipy']}, Matplotlib {manifest['matplotlib']}. Seeds are fixed in
`experiments.py`; other library versions can change the last digits.

## References

- Ahmad, R. and Wilmott, P. (2005). Which free lunch would you like today, sir? *Wilmott Magazine*.
- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*.
- Broadie, M. and Detemple, J. (1996). American option valuation. *Review of Financial Studies* 9.
- Derman, E. and Kamal, M. (1999). When you cannot hedge continuously. Goldman Sachs Quantitative Strategies Research Notes.
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies* 6.
- Leland, H. (1985). Option pricing and replication with transactions costs. *Journal of Finance* 40.
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Levy processes.
- Manaster, S. and Koehler, G. (1982). The calculation of implied variances from the Black-Scholes model. *Journal of Finance* 37.
- Merton, R. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics* 3.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs. *Mathematical Finance* 7.
"""
    with open(os.path.join(out, "GammaScalp_Research_Memo.md"), "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    figures = os.path.join(args.out, "figures")
    figure_discrete(args.results, figures)
    figure_mismatch(args.results, figures)
    figure_costs(args.results, figures)
    figure_jumps(args.results, figures)
    figure_smile(args.results, figures)
    memo(args.results, args.out)
    print(f"wrote {args.out}/GammaScalp_Research_Memo.md and five figures")


if __name__ == "__main__":
    main()
