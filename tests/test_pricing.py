"""Pricing library tests: closed forms, limits, parities, and independent methods."""

import unittest

import numpy as np

from gammascalp import american, heston, paths
from gammascalp import black_scholes as bs
from gammascalp import experiments as ex


class BlackScholes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 2000
        self.S = rng.uniform(50, 150, n)
        self.K = rng.uniform(50, 150, n)
        self.T = rng.uniform(0.05, 3.0, n)
        self.sigma = rng.uniform(0.05, 0.9, n)
        self.r = 0.04
        self.q = 0.015

    def test_textbook_value(self):
        self.assertAlmostEqual(float(bs.price(100, 100, 1.0, 0.05, 0.2)), 10.450583572185565, places=12)
        self.assertAlmostEqual(float(bs.price(100, 100, 1.0, 0.05, 0.2, call=False)), 5.573526022256971, places=12)

    def test_put_call_parity(self):
        call = bs.price(self.S, self.K, self.T, self.r, self.sigma, self.q, True)
        put = bs.price(self.S, self.K, self.T, self.r, self.sigma, self.q, False)
        forward = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        np.testing.assert_allclose(call - put, forward, atol=1e-11)

    def test_no_arbitrage_bounds(self):
        call = bs.price(self.S, self.K, self.T, self.r, self.sigma, self.q, True)
        lower = np.maximum(self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T), 0)
        self.assertTrue(np.all(call >= lower - 1e-12))
        self.assertTrue(np.all(call <= self.S * np.exp(-self.q * self.T) + 1e-12))

    def test_greeks_match_finite_differences(self):
        args = (self.K, self.T, self.r, self.sigma, self.q)
        for call in (True, False):
            h = 1e-3
            up = bs.price(self.S + h, *args, call)
            down = bs.price(self.S - h, *args, call)
            mid = bs.price(self.S, *args, call)
            np.testing.assert_allclose(bs.delta(self.S, *args, call), (up - down) / (2 * h), atol=1e-6)
            np.testing.assert_allclose(bs.gamma(self.S, *args), (up - 2 * mid + down) / h**2, atol=2e-5)
            e = 1e-5
            vega = (bs.price(self.S, self.K, self.T, self.r, self.sigma + e, self.q, call) -
                    bs.price(self.S, self.K, self.T, self.r, self.sigma - e, self.q, call)) / (2 * e)
            np.testing.assert_allclose(bs.vega(self.S, *args), vega, rtol=1e-6, atol=1e-7)
            t = 1e-5
            theta = (bs.price(self.S, self.K, self.T - t, self.r, self.sigma, self.q, call) -
                     bs.price(self.S, self.K, self.T + t, self.r, self.sigma, self.q, call)) / (2 * t)
            np.testing.assert_allclose(bs.theta(self.S, *args, call), theta, rtol=1e-5, atol=1e-6)

    def test_black_scholes_equation_holds(self):
        # theta + 1/2 sigma^2 S^2 gamma + (r - q) S delta - r V = 0
        args = (self.K, self.T, self.r, self.sigma, self.q)
        for call in (True, False):
            residual = (bs.theta(self.S, *args, call) + 0.5 * self.sigma**2 * self.S**2 * bs.gamma(self.S, *args) +
                        (self.r - self.q) * self.S * bs.delta(self.S, *args, call) -
                        self.r * bs.price(self.S, *args, call))
            np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_expiry_is_intrinsic_value(self):
        S = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(bs.price(S, 100.0, 0.0, 0.05, 0.2), [0.0, 0.0, 20.0])
        np.testing.assert_allclose(bs.price(S, 100.0, 0.0, 0.05, 0.2, call=False), [20.0, 0.0, 0.0])
        np.testing.assert_allclose(bs.delta(S, 100.0, 0.0, 0.05, 0.2), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(bs.gamma(S, 100.0, 0.0, 0.05, 0.2), 0.0)


class ImpliedVolatility(unittest.TestCase):
    def test_round_trip_calls_and_puts(self):
        rng = np.random.default_rng(2)
        n = 20000
        K = 100 * np.exp(rng.uniform(-0.6, 0.6, n))
        T = rng.uniform(0.05, 2.0, n)
        sigma = rng.uniform(0.08, 0.8, n)
        call = rng.random(n) < 0.5
        prices = bs.price(100.0, K, T, 0.03, sigma, 0.01, call)
        solved = bs.implied_vol(prices, 100.0, K, T, 0.03, 0.01, call)
        self.assertEqual(solved.shape, prices.shape)
        # Compare where the time value is large enough to carry information.
        informative = bs.vega(100.0, K, T, 0.03, sigma, 0.01) > 1e-3
        self.assertGreater(informative.mean(), 0.9)
        np.testing.assert_allclose(solved[informative], sigma[informative], atol=1e-9)
        self.assertFalse(np.any(np.isnan(solved[informative])))

    def test_scalar_and_shapes(self):
        self.assertAlmostEqual(float(bs.implied_vol(10.450583572185565, 100, 100, 1.0, 0.05)), 0.2, places=12)
        grid = bs.implied_vol(np.full((3, 4), 10.450583572185565), 100, 100, 1.0, 0.05)
        self.assertEqual(grid.shape, (3, 4))

    def test_prices_outside_bounds_have_no_volatility(self):
        self.assertTrue(np.isnan(bs.implied_vol(0.0, 100, 100, 1.0, 0.05)))
        self.assertTrue(np.isnan(bs.implied_vol(101.0, 100, 100, 1.0, 0.05)))            # Above the stock.
        self.assertTrue(np.isnan(bs.implied_vol(4.0, 100, 100, 1.0, 0.05)))              # Below intrinsic.
        self.assertTrue(np.isnan(bs.implied_vol(96.0, 100, 100, 1.0, 0.05, call=False)))  # Above the strike's PV.
        self.assertTrue(np.isnan(bs.implied_vol(5.0, 100, 100, 0.0, 0.05)))              # Already expired.

    def test_far_wings_and_extreme_volatility(self):
        for sigma in (0.01, 0.05, 2.5, 6.0):
            for K in (60.0, 100.0, 170.0):
                price = float(bs.price(100.0, K, 0.5, 0.02, sigma))
                solved = float(bs.implied_vol(price, 100.0, K, 0.5, 0.02))
                if np.isnan(solved):
                    # Legitimate only when the time value is below double precision.
                    time_value = price - max(100.0 - K * np.exp(-0.01), 0.0)
                    self.assertLess(abs(time_value), 1e-10)
                else:
                    self.assertAlmostEqual(solved, sigma, delta=1e-6 * max(1.0, sigma))

    def test_accuracy_report(self):
        rows, speed = ex.implied_vol_accuracy(n=50000, seed=3)
        self.assertEqual(rows[0]["solved"], rows[0]["options"])
        self.assertLess(rows[0]["max_abs_error"], 1e-10)   # Time value above 1e-4 of the forward.
        self.assertLess(rows[2]["max_abs_error"], 1e-7)    # Above 1e-8.
        self.assertGreater(speed["options_per_second"], 10000)


class American(unittest.TestCase):
    def test_call_without_dividends_is_european(self):
        tree = american.price(100, 105, 0.75, 0.05, 0.25, call=True)
        self.assertAlmostEqual(tree, float(bs.price(100, 105, 0.75, 0.05, 0.25)), delta=2e-3)

    def test_put_is_worth_more_than_european_and_intrinsic(self):
        for S in (80.0, 95.0, 100.0, 110.0):
            tree = american.price(S, 100, 1.0, 0.05, 0.2)
            self.assertGreaterEqual(tree, float(bs.price(S, 100, 1.0, 0.05, 0.2, call=False)) - 1e-9)
            self.assertGreaterEqual(tree, max(100 - S, 0) - 1e-9)
        self.assertAlmostEqual(american.price(60, 100, 1.0, 0.05, 0.2), 40.0, delta=1e-9)  # Exercise now.

    def test_matches_a_very_fine_plain_tree(self):
        reference = american.crr(100, 100, 1.0, 0.05, 0.2, steps=20000)
        smoothed = american.price(100, 100, 1.0, 0.05, 0.2, steps=400)
        plain = american.crr(100, 100, 1.0, 0.05, 0.2, steps=600)  # Same total work as 400 + 200.
        self.assertAlmostEqual(smoothed, reference, delta=6e-4)
        self.assertLess(abs(smoothed - reference), abs(plain - reference))

    def test_put_call_symmetry(self):
        # McDonald and Schroder (1998): C(S, K, r, q) = P(K, S, q, r).
        for S, K, r, q in ((100, 110, 0.05, 0.02), (90, 80, 0.01, 0.06), (100, 100, 0.03, 0.03)):
            call = american.price(S, K, 0.5, r, 0.3, q, call=True)
            put = american.price(K, S, 0.5, q, 0.3, r, call=False)
            self.assertAlmostEqual(call, put, delta=2e-3)

    def test_european_tree_converges_to_black_scholes(self):
        tree = american.price(100, 95, 0.5, 0.03, 0.3, 0.01, call=False, american=False)
        self.assertAlmostEqual(tree, float(bs.price(100, 95, 0.5, 0.03, 0.3, 0.01, call=False)), delta=1e-3)

    def test_early_exercise_premium_grows_with_rates(self):
        low = american.early_exercise_premium(100, 100, 1.0, 0.01, 0.2)
        high = american.early_exercise_premium(100, 100, 1.0, 0.08, 0.2)
        self.assertGreater(low, 0)
        self.assertGreater(high, low)

    def test_rejects_bad_step_counts(self):
        with self.assertRaises(ValueError):
            american.price(100, 100, 1.0, 0.05, 0.2, steps=7)


class Heston(unittest.TestCase):
    params = heston.HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.05)
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])

    def test_characteristic_function_is_a_martingale(self):
        self.assertAlmostEqual(abs(heston.char_fn(0.0, 1.3, self.params)), 1.0, places=14)
        self.assertAlmostEqual(abs(heston.char_fn(-1j, 1.3, self.params) - 1.0), 0.0, places=12)

    def test_two_integral_formulas_agree(self):
        for T in (0.1, 1.0, 5.0):
            a = heston.price(100, self.strikes, T, 0.03, self.params, 0.01)
            b = heston.price_gil_pelaez(100, self.strikes, T, 0.03, self.params, 0.01)
            np.testing.assert_allclose(a, b, atol=1e-8)

    def test_collapses_to_black_scholes_without_vol_of_vol(self):
        flat = heston.HestonParams(kappa=1.0, theta=0.09, xi=1e-4, rho=0.0, v0=0.09)
        np.testing.assert_allclose(heston.price(100, self.strikes, 1.0, 0.02, flat),
                                   bs.price(100, self.strikes, 1.0, 0.02, 0.3), atol=1e-6)

    def test_put_call_parity(self):
        call = heston.price(100, self.strikes, 0.7, 0.03, self.params, 0.01, True)
        put = heston.price(100, self.strikes, 0.7, 0.03, self.params, 0.01, False)
        np.testing.assert_allclose(call - put, 100 * np.exp(-0.007) - self.strikes * np.exp(-0.021), atol=1e-9)

    def test_negative_correlation_produces_a_downward_skew(self):
        prices = heston.price(100, self.strikes, 0.5, 0.0, self.params)
        vols = bs.implied_vol(prices, 100, self.strikes, 0.5, 0.0)
        self.assertTrue(np.all(np.diff(vols[:4]) < 0))

    def test_monte_carlo_agrees(self):
        rng = np.random.default_rng(5)
        T, r, n = 0.5, 0.02, 200_000
        grid = paths.heston(rng, n, 50, 100.0, r, T, self.params, substeps=4)
        S_T = grid[:, -1]
        for K in (90.0, 100.0, 110.0):
            payoff = np.exp(-r * T) * np.maximum(S_T - K, 0.0)
            estimate = payoff.mean()
            error = payoff.std() / np.sqrt(n)
            exact = float(heston.price(100, K, T, r, self.params)[0])
            # Four standard errors plus a small allowance for time-stepping bias.
            self.assertLess(abs(estimate - exact), 4 * error + 0.02)


class Merton(unittest.TestCase):
    def test_series_price_matches_simulation(self):
        rng = np.random.default_rng(6)
        args = dict(sigma=0.14, intensity=4.0, jump_mean=-0.02, jump_std=0.06)
        S_T = paths.merton(rng, 400_000, 1, 100.0, 0.03, args["sigma"], 0.25,
                           args["intensity"], args["jump_mean"], args["jump_std"])[:, -1]
        payoff = np.exp(-0.03 * 0.25) * np.maximum(S_T - 100.0, 0.0)
        exact = ex.merton_price(100.0, 100.0, 0.25, 0.03, **args)
        self.assertLess(abs(payoff.mean() - exact), 4 * payoff.std() / np.sqrt(payoff.size))
        # Drift is compensated: the forward is unchanged by jumps.
        self.assertLess(abs(S_T.mean() - 100 * np.exp(0.0075)), 4 * S_T.std() / np.sqrt(S_T.size))

    def test_no_jumps_is_black_scholes(self):
        self.assertAlmostEqual(ex.merton_price(100, 105, 0.5, 0.02, 0.25, 0.0, 0.0, 0.1),
                               float(bs.price(100, 105, 0.5, 0.02, 0.25)), places=10)

    def test_total_variance(self):
        rng = np.random.default_rng(7)
        grid = paths.merton(rng, 20000, 63, 100.0, 0.0, 0.14, 0.25, 4.0, -0.02, 0.06)
        realized = np.var(np.diff(np.log(grid), axis=1)) * 63 / 0.25
        self.assertAlmostEqual(realized, paths.merton_total_variance(0.14, 4.0, -0.02, 0.06), delta=0.002)


if __name__ == "__main__":
    unittest.main()
