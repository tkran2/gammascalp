"""Command line: quick calculations and the full reproducible study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from . import __version__, american
from . import black_scholes as bs
from . import experiments as ex


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()})


def cmd_price(args):
    call = not args.put
    greeks = {
        "price": float(bs.price(args.spot, args.strike, args.expiry, args.rate, args.vol, args.dividend, call)),
        "delta": float(bs.delta(args.spot, args.strike, args.expiry, args.rate, args.vol, args.dividend, call)),
        "gamma": float(bs.gamma(args.spot, args.strike, args.expiry, args.rate, args.vol, args.dividend)),
        "vega_per_vol_point": float(bs.vega(args.spot, args.strike, args.expiry, args.rate, args.vol, args.dividend)) / 100,
        "theta_per_day": float(bs.theta(args.spot, args.strike, args.expiry, args.rate, args.vol, args.dividend, call)) / 365,
    }
    if args.american:
        greeks["american_price"] = american.price(args.spot, args.strike, args.expiry, args.rate,
                                                  args.vol, args.dividend, call)
        greeks["early_exercise_premium"] = greeks["american_price"] - greeks["price"]
    for name, value in greeks.items():
        print(f"{name:24} {value: .6f}")


def cmd_iv(args):
    vol = float(bs.implied_vol(args.price, args.spot, args.strike, args.expiry, args.rate,
                               args.dividend, not args.put))
    if np.isnan(vol):
        print("no implied volatility: the price is outside the no-arbitrage bounds")
        return 1
    print(f"implied_vol {vol:.6f}")
    return 0


def cmd_hedge(args):
    market = ex.Market(sigma=args.vol)
    rows = ex.discrete_hedging(market, rebalances=(args.rebalances,), n_paths=args.paths, seed=args.seed)
    row = rows[0]
    print(f"short one 3-month at-the-money call, premium {row['premium']:.4f}, "
          f"{args.rebalances} hedges, {args.paths} paths")
    print(f"mean P&L          {row['mean']: .4f}  (standard error {row['mean_se']:.4f})")
    print(f"std of P&L        {row['std']: .4f}  ({row['std_pct_of_premium']:.1f}% of premium)")
    print(f"theory            {row['predicted_std']: .4f}  (ratio {row['ratio']:.3f})")


def cmd_research(args):
    """Run all five studies and write CSVs plus a manifest."""
    scale = 0.1 if args.quick else 1.0
    out = args.output
    started = time.time()
    summary = {}

    def paths(n):
        return max(2000, int(n * scale))

    print("1/5 discrete hedging error ...", flush=True)
    rows = ex.discrete_hedging(strikes=(90.0, 100.0, 110.0), n_paths=paths(200_000))
    _write_csv(os.path.join(out, "discrete_hedging.csv"), rows)
    slope, slope_se = ex.scaling_exponent(rows)
    atm = [r for r in rows if r["strike"] == 100.0]
    summary["discrete_hedging"] = {
        "scaling_exponent": slope, "scaling_exponent_se": slope_se,
        "ratio_to_theory_at_63": next(r["ratio"] for r in atm if r["rebalances"] == 63),
        "ratio_to_theory_at_1008": next(r["ratio"] for r in atm if r["rebalances"] == 1008),
        "largest_abs_mean_in_se": max(abs(r["mean"]) / r["mean_se"] for r in rows),
        # Away from the money, at the most frequent hedging tested.
        "rule_of_thumb_ratio_off_the_money": [
            r["ratio"] for r in rows if r["strike"] != 100.0 and r["rebalances"] == 1008],
        "leading_order_ratio_at_1008": [r["leading_order_ratio"] for r in rows if r["rebalances"] == 1008],
        "leading_order_ratio_at_63": [r["leading_order_ratio"] for r in rows if r["rebalances"] == 63],
    }

    rows = ex.drift_bias(n_paths=paths(200_000))
    _write_csv(os.path.join(out, "drift_bias.csv"), rows)
    extreme = [r for r in rows if abs(r["drift"]) >= 0.5]
    summary["drift_bias"] = {
        "largest_gap_to_prediction_in_se": max(
            abs(r["mean"] - float(r["predicted_mean"])) / r["mean_se"] for r in rows),
        "worst_mean_pct_of_premium": min(r["mean_pct_of_premium"] for r in extreme),
    }

    print("2/5 volatility mismatch ...", flush=True)
    rows = ex.volatility_mismatch(n_paths=paths(100_000))
    _write_csv(os.path.join(out, "volatility_mismatch.csv"), rows)
    summary["volatility_mismatch"] = {
        "locked_in_value": rows[0]["locked_in_value"],
        "realized_hedge_means": [r["realized_hedge_mean"] for r in rows],
        "implied_hedge_means": [r["implied_hedge_mean"] for r in rows],
        "theory_implied_hedge_means": [float(r["theory_implied_hedge_mean"]) for r in rows],
        "min_gamma_rent_correlation": min(r["gamma_rent_correlation"] for r in rows),
        "max_share_of_paths_losing": max(r["share_of_paths_losing"] for r in rows),
    }

    print("3/5 trading costs ...", flush=True)
    rows = ex.leland_check(n_paths=paths(200_000))
    _write_csv(os.path.join(out, "leland.csv"), rows)
    summary["leland"] = {
        "max_abs_leland_mean_pct_of_premium": max(abs(r["leland_mean_pct_of_premium"]) for r in rows),
        "naive_mean_at_252": next(r["naive_mean"] for r in rows if r["rebalances"] == 252),
        "leland_mean_at_252": next(r["leland_mean"] for r in rows if r["rebalances"] == 252),
    }
    rows = ex.hedging_frontier(n_paths=paths(40_000))
    _write_csv(os.path.join(out, "frontier.csv"), rows)
    compared = ex.frontier_comparison(rows)
    _write_csv(os.path.join(out, "frontier_comparison.csv"), compared)
    summary["frontier"] = {
        "risk_reduction_pct_min": min(r["risk_reduction_pct"] for r in compared),
        "risk_reduction_pct_max": max(r["risk_reduction_pct"] for r in compared),
        "best_clock_std": min(r["std"] for r in rows if r["rule"] == "clock"),
        "best_band_std": min(r["std"] for r in rows if r["rule"] == "band"),
    }

    print("4/5 jump risk ...", flush=True)
    rows = ex.jump_risk(n_paths=paths(200_000))
    _write_csv(os.path.join(out, "jump_risk.csv"), rows)
    first, last = rows[0], rows[-1]
    summary["jump_risk"] = {
        "smooth_std_reduction_pct": 100 * (1 - last["smooth_std"] / first["smooth_std"]),
        "jump_std_reduction_pct": 100 * (1 - last["jump_std"] / first["jump_std"]),
        "p01_ratio_at_most_frequent": last["jump_p01"] / last["smooth_p01"],
        "largest_abs_jump_mean_in_se": max(abs(r["jump_mean"]) / r["jump_mean_se"] for r in rows),
    }

    print("5/5 implied volatility and smile fitting ...", flush=True)
    accuracy, speed = ex.implied_vol_accuracy(n=int(1_000_000 * scale))
    _write_csv(os.path.join(out, "implied_vol_accuracy.csv"), accuracy)
    chain, rows = ex.smile_fit(trials=4 if args.quick else 20)
    _write_csv(os.path.join(out, "smile_fit.csv"), rows)
    chain_rows = []
    for leg in chain:
        for i in range(len(leg["k"])):
            chain_rows.append({"expiry": leg["T"], "forward": leg["F"], "strike": leg["K"][i],
                               "log_moneyness": leg["k"][i], "is_call": bool(leg["is_call"][i]),
                               "price": leg["price"][i], "implied_vol": leg["implied_vol"][i]})
    _write_csv(os.path.join(out, "heston_chain.csv"), chain_rows)

    def mean_where(noise, key):
        return float(np.mean([r[key] for r in rows if r["noise_vol_bps"] == noise]))

    summary["implied_vol"] = {**speed, "accuracy": accuracy}
    summary["smile_fit"] = {
        "clean_rmse_bps_mean": mean_where(0.0, "rmse_vs_truth_bps"),
        "clean_rmse_bps_max": max(r["rmse_vs_truth_bps"] for r in rows if r["noise_vol_bps"] == 0.0),
        "noise10_rmse_vs_truth_bps": mean_where(10.0, "rmse_vs_truth_bps"),
        "noise30_rmse_vs_truth_bps": mean_where(30.0, "rmse_vs_truth_bps"),
        "fits": len(rows),
        "fits_with_butterfly_arbitrage": sum(r["butterfly_min_g"] < 0 for r in rows),
        "worst_calendar_violation": min(r["calendar_violation"] for r in rows),
    }

    import matplotlib
    import scipy

    manifest = {
        "version": __version__, "quick": bool(args.quick), "seconds": round(time.time() - started, 1),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "summary": summary,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)
    print(f"done in {manifest['seconds']} s; results in {out}/")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gammascalp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def option_args(p):
        p.add_argument("--spot", type=float, default=100.0)
        p.add_argument("--strike", type=float, default=100.0)
        p.add_argument("--expiry", type=float, default=0.25, help="years")
        p.add_argument("--rate", type=float, default=0.03)
        p.add_argument("--dividend", type=float, default=0.0)
        p.add_argument("--put", action="store_true")

    p = sub.add_parser("price", help="Black-Scholes price and Greeks")
    option_args(p)
    p.add_argument("--vol", type=float, default=0.20)
    p.add_argument("--american", action="store_true", help="also price early exercise on a tree")
    p.set_defaults(run=cmd_price)

    p = sub.add_parser("iv", help="implied volatility from a price")
    option_args(p)
    p.add_argument("--price", type=float, required=True)
    p.set_defaults(run=cmd_iv)

    p = sub.add_parser("hedge", help="one quick discrete-hedging experiment")
    p.add_argument("--rebalances", type=int, default=63)
    p.add_argument("--paths", type=int, default=50_000)
    p.add_argument("--vol", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(run=cmd_hedge)

    p = sub.add_parser("research", help="run all five studies")
    p.add_argument("--output", default="results")
    p.add_argument("--quick", action="store_true", help="one tenth of the paths, for a smoke test")
    p.set_defaults(run=cmd_research)

    args = parser.parse_args(argv)
    return args.run(args) or 0
