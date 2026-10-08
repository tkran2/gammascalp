"""Hedging simulator tests: exact accounting first, then theory within noise."""

import unittest

import numpy as np

from gammascalp import black_scholes as bs
from gammascalp import experiments as ex
from gammascalp import hedging, paths

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.03, 0.20
PREMIUM = float(bs.price(S0, K, T, R, SIGMA))


def sample(seed, n, steps, mu=0.08, sigma=SIGMA):
    return paths.gbm(np.random.default_rng(seed), n, steps, S0, mu, sigma, T)


class Accounting(unittest.TestCase):
    def test_buy_and_hold_hedge_is_exact(self):
        # Hedge once at the start and never again: the P&L has a closed form.
        grid = sample(1, 500, 40)
        out = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, every=10**9)
        delta0 = float(bs.delta(S0, K, T, R, SIGMA))
        S_T = grid[:, -1]
        expected = (PREMIUM - delta0 * S0) * np.exp(R * T) + delta0 * S_T - np.maximum(S_T - K, 0)
        np.testing.assert_allclose(out.pnl, expected, atol=1e-10)
        np.testing.assert_array_equal(out.trades, 1)

    def test_long_is_the_mirror_of_short(self):
        grid = sample(2, 300, 63)
        short = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, position=-1)
        long = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, position=+1)
        np.testing.assert_allclose(long.pnl, -short.pnl, atol=1e-10)
        np.testing.assert_allclose(long.gamma_pnl, -short.gamma_pnl, atol=1e-10)

    def test_costs_are_exactly_the_difference(self):
        grid = sample(3, 300, 63)
        free = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM)
        charged = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, cost=0.002, close_cost=True)
        self.assertTrue(np.all(charged.costs > 0))
        np.testing.assert_allclose(free.pnl - charged.pnl, charged.costs, atol=1e-10)
        no_open = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, cost=0.002, open_cost=False)
        self.assertTrue(np.all(no_open.costs < charged.costs))

    def test_put_hedge_uses_negative_delta(self):
        grid = sample(4, 2000, 126)
        premium = float(bs.price(S0, K, T, R, SIGMA, call=False))
        out = hedging.hedge(grid, K, T, R, SIGMA, premium, call=False)
        self.assertLess(abs(out.pnl.mean()), 4 * out.pnl.std() / np.sqrt(out.pnl.size))
        self.assertLess(out.pnl.std(), 0.5)

    def test_paths_start_at_spot_and_are_reproducible(self):
        a = sample(5, 10, 20)
        b = sample(5, 10, 20)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], S0)
        self.assertFalse(np.array_equal(a, sample(6, 10, 20)))

    def test_moments_helper(self):
        x = np.random.default_rng(0).standard_exponential(50000)
        m = ex.moments(x)
        self.assertAlmostEqual(m["mean"], x.mean(), places=12)
        self.assertAlmostEqual(m["std"], x.std(), places=12)
        self.assertAlmostEqual(m["skew"], 2.0, delta=0.15)          # Exponential distribution.
        self.assertAlmostEqual(m["excess_kurtosis"], 6.0, delta=1.2)
        self.assertAlmostEqual(m["mean_se"], x.std() / np.sqrt(x.size), places=12)


class Theory(unittest.TestCase):
    def test_frequent_hedging_replicates_the_option(self):
        out = hedging.hedge(sample(10, 4000, 2000), K, T, R, SIGMA, PREMIUM)
        self.assertLess(out.pnl.std(), 0.09)  # Under 2% of a 4.36 premium.
        self.assertLess(abs(out.pnl.mean()), 4 * out.pnl.std() / np.sqrt(4000))

    def test_error_matches_derman_kamal_and_halves_when_hedges_quadruple(self):
        coarse = hedging.hedge(sample(11, 30000, 63), K, T, R, SIGMA, PREMIUM).pnl.std()
        fine = hedging.hedge(sample(12, 30000, 252), K, T, R, SIGMA, PREMIUM).pnl.std()
        self.assertAlmostEqual(coarse / hedging.derman_kamal_std(S0, K, T, R, SIGMA, 63), 1.0, delta=0.04)
        self.assertAlmostEqual(fine / hedging.derman_kamal_std(S0, K, T, R, SIGMA, 252), 1.0, delta=0.04)
        self.assertAlmostEqual(coarse / fine, 2.0, delta=0.08)

    def test_rule_of_thumb_is_at_the_money_only_and_the_exact_sum_is_not(self):
        # At the money the two formulas agree to about 1%.
        base = ex.Market()
        self.assertAlmostEqual(ex.leading_order_std(base, 252) /
                               hedging.derman_kamal_std(S0, K, T, R, SIGMA, 252), 1.0, delta=0.02)
        # Ten percent out of the money the rule of thumb is about a fifth too low.
        strike = 110.0
        premium = float(bs.price(S0, strike, T, R, SIGMA))
        simulated = hedging.hedge(sample(21, 40000, 252), strike, T, R, SIGMA, premium).pnl.std()
        thumb = hedging.derman_kamal_std(S0, strike, T, R, SIGMA, 252)
        exact = ex.leading_order_std(ex.Market(K=strike), 252)
        self.assertGreater(simulated / thumb, 1.15)
        self.assertAlmostEqual(simulated / exact, 1.0, delta=0.04)
        # Four times the hedges gives close to half the risk (not exactly: the
        # sum samples gamma at the hedge dates, which move with N).
        self.assertAlmostEqual(ex.leading_order_std(ex.Market(K=strike), 63) / exact, 2.0, delta=0.07)

    def test_mean_does_not_depend_on_realistic_drift(self):
        for mu in (-0.10, 0.0, 0.10):
            pnl = hedging.hedge(sample(13, 30000, 126, mu=mu), K, T, R, SIGMA, PREMIUM).pnl
            self.assertLess(abs(pnl.mean()), 4 * pnl.std() / np.sqrt(pnl.size))

    def test_extreme_drift_leaves_the_predicted_second_order_bias(self):
        # Continuous hedging ignores drift; N discrete hedges leave
        # -(mu - r)^2 * (T / N) * E[integral of 1/2 gamma S^2] for a short option.
        rows = ex.drift_bias(drifts=(1.0,), rebalances=(63, 252), n_paths=60000)
        for row in rows:
            self.assertLess(row["mean"], -10 * row["mean_se"])  # Clearly not zero.
            self.assertAlmostEqual(row["mean"], float(row["predicted_mean"]),
                                   delta=4 * row["mean_se"] + 0.05 * abs(row["predicted_mean"]))
        self.assertAlmostEqual(rows[0]["mean"] / rows[1]["mean"], 4.0, delta=0.5)  # Scales like 1 / N.

    def test_gamma_rent_explains_the_pnl(self):
        out = hedging.hedge(sample(14, 20000, 504), K, T, R, SIGMA, PREMIUM)
        self.assertGreater(np.corrcoef(out.pnl, out.gamma_pnl)[0, 1], 0.99)

    def test_hedging_at_realized_vol_locks_in_the_edge(self):
        implied, realized = 0.20, 0.30
        premium = float(bs.price(S0, K, T, R, implied))
        locked = np.exp(R * T) * (float(bs.price(S0, K, T, R, realized)) - premium)
        grid = sample(15, 20000, 504, sigma=realized)
        at_realized = hedging.hedge(grid, K, T, R, realized, premium, position=+1).pnl
        at_implied = hedging.hedge(grid, K, T, R, implied, premium, position=+1).pnl
        self.assertAlmostEqual(at_realized.mean(), locked, delta=4 * at_realized.std() / np.sqrt(20000))
        self.assertLess(at_realized.std(), 0.35 * at_implied.std())
        # Hedging at implied is path dependent but, long gamma with realized
        # above implied, it earns rent every step and essentially never loses.
        self.assertLess(np.mean(at_implied < 0), 0.001)

    def test_expected_gamma_rent_matches_simulation(self):
        market = ex.Market()
        rows = ex.volatility_mismatch(market, drifts=(0.0,), rebalances=504, n_paths=20000)
        row = rows[0]
        self.assertAlmostEqual(row["implied_hedge_mean"], float(row["theory_implied_hedge_mean"]),
                               delta=4 * row["implied_hedge_mean_se"] + 0.01)

    def test_leland_volatility_formula(self):
        # sigma^2 * (1 + sqrt(8/pi) * c / (sigma * sqrt(dt))) for a seller.
        adjusted = hedging.leland_volatility(0.2, 0.001, 1 / 252)
        expected = np.sqrt(0.04 * (1 + np.sqrt(8 / np.pi) * 0.001 / (0.2 * np.sqrt(1 / 252))))
        self.assertAlmostEqual(adjusted, expected, places=14)
        self.assertGreater(adjusted, 0.2)
        self.assertLess(hedging.leland_volatility(0.2, 0.001, 1 / 252, position=+1), 0.2)
        self.assertTrue(np.isnan(hedging.leland_volatility(0.2, 0.05, 1e-6, position=+1)))

    def test_leland_price_removes_the_cost_bias(self):
        rows = ex.leland_check(rebalances=(63,), n_paths=40000)
        row = rows[0]
        self.assertLess(row["naive_mean"], -10 * row["naive_mean_se"])        # Clearly loses money.
        self.assertLess(abs(row["leland_mean"]), 4 * row["leland_mean_se"])   # Indistinguishable from zero.
        self.assertAlmostEqual(row["measured_cost"], row["predicted_cost"], delta=0.05 * row["predicted_cost"])

    def test_band_rule_dominates_the_clock(self):
        rows = ex.hedging_frontier(n_paths=6000, intervals=(16, 63, 252), risk_aversions=(0.8, 3.2))
        compared = ex.frontier_comparison(rows)
        self.assertEqual(len(compared), 2)
        for row in compared:
            self.assertGreater(row["risk_reduction_pct"], 15.0)

    def test_band_trades_to_the_edge_not_the_center(self):
        grid = sample(16, 2000, 504)
        band = hedging.whalley_wilmott_band(0.001, 1.0)
        out = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, cost=0.001, band=band, open_cost=False)
        clock = hedging.hedge(grid, K, T, R, SIGMA, PREMIUM, cost=0.001, every=1, open_cost=False)
        self.assertLess(out.costs.mean(), 0.5 * clock.costs.mean())
        self.assertLess(out.trades.mean(), clock.trades.mean())

    def test_jumps_leave_risk_that_more_hedging_cannot_remove(self):
        rows = ex.jump_risk(rebalances=(16, 504), n_paths=30000)
        coarse, fine = rows
        self.assertLess(fine["smooth_std"], 0.25 * coarse["smooth_std"])   # Smooth world: big gain.
        self.assertGreater(fine["jump_std"], 0.8 * coarse["jump_std"])     # Jump world: almost none.
        self.assertLess(fine["jump_skew"], -1.0)
        self.assertLess(abs(fine["jump_mean"]), 4 * fine["jump_mean_se"])  # Fairly priced.

    def test_scaling_exponent(self):
        rows = ex.discrete_hedging(rebalances=(16, 63, 252), n_paths=20000)
        slope, error = ex.scaling_exponent(rows)
        self.assertAlmostEqual(slope, -0.5, delta=0.03)
        self.assertLess(error, 0.03)


if __name__ == "__main__":
    unittest.main()
