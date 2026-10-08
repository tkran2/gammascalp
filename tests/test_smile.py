"""SVI fitting and arbitrage checks."""

import unittest

import numpy as np
from scipy.optimize import lsq_linear

from gammascalp import experiments as ex
from gammascalp import svi


class Fit(unittest.TestCase):
    def test_recovers_exact_parameters(self):
        for true in (svi.SVI(0.02, 0.4, -0.5, 0.05, 0.2), svi.SVI(0.01, 0.15, 0.3, -0.1, 0.08),
                     svi.SVI(0.04, 0.9, -0.85, 0.2, 0.35)):
            k = np.linspace(-0.8, 0.8, 41)
            result = svi.fit(k, true.total_variance(k), 1.0)
            self.assertLess(result.rmse_vol, 1e-7)
            for name in ("a", "b", "rho", "m", "sigma"):
                self.assertAlmostEqual(getattr(result.params, name), getattr(true, name), delta=2e-4)

    def test_noise_is_averaged_out(self):
        true = svi.SVI(0.02, 0.4, -0.5, 0.05, 0.2)
        k = np.linspace(-0.6, 0.6, 41)
        vol = true.implied_vol(k, 1.0)
        rng = np.random.default_rng(3)
        errors = []
        for _ in range(10):
            noisy = vol + rng.standard_normal(k.size) * 0.002  # 20 vol basis points.
            fitted = svi.fit(k, noisy**2, 1.0).params
            errors.append(np.sqrt(np.mean((fitted.implied_vol(k, 1.0) - vol) ** 2)))
        self.assertLess(np.mean(errors), 0.001)  # The fit is closer to the truth than the quotes.

    def test_weights_pull_the_fit(self):
        true = svi.SVI(0.02, 0.4, -0.5, 0.05, 0.2)
        k = np.linspace(-0.6, 0.6, 31)
        w = true.total_variance(k)
        w[-1] *= 1.3  # One bad quote in the right wing.
        plain = svi.fit(k, w, 1.0).params
        weights = np.ones_like(k)
        weights[-1] = 1e-3
        weighted = svi.fit(k, w, 1.0, weights).params
        inner = slice(0, -1)
        plain_error = np.max(np.abs(plain.total_variance(k[inner]) - true.total_variance(k[inner])))
        weighted_error = np.max(np.abs(weighted.total_variance(k[inner]) - true.total_variance(k[inner])))
        self.assertLess(weighted_error, 0.2 * plain_error)

    def test_needs_enough_points(self):
        with self.assertRaises(ValueError):
            svi.fit(np.array([0.0, 0.1, 0.2]), np.array([0.04, 0.04, 0.04]), 1.0)

    def test_box_solver_matches_scipy(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            design = rng.normal(size=(20, 3))
            target = rng.normal(size=20)
            lower = -rng.uniform(0, 1, 3)
            upper = rng.uniform(0, 1, 3)
            mine = svi._box_least_squares(design, target, lower, upper)
            theirs = lsq_linear(design, target, bounds=(lower, upper), tol=1e-12).x
            self.assertTrue(np.all(mine >= lower - 1e-12) and np.all(mine <= upper + 1e-12))
            self.assertAlmostEqual(float(np.sum((design @ mine - target) ** 2)),
                                   float(np.sum((design @ theirs - target) ** 2)), places=8)


class Arbitrage(unittest.TestCase):
    def test_known_butterfly_arbitrage_is_detected(self):
        # The parameters Axel Vogt reported; discussed in Gatheral and Jacquier (2014).
        vogt = svi.SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
        self.assertGreater(vogt.min_variance(), 0)   # It looks harmless: variance stays positive
        self.assertTrue(vogt.wing_bound_ok())        # and the wings obey Lee's bound.
        self.assertLess(svi.butterfly_violation(vogt), -0.01)

    def test_ordinary_smile_is_clean(self):
        self.assertEqual(svi.butterfly_violation(svi.SVI(0.02, 0.2, -0.4, 0.0, 0.2)), 0.0)

    def test_butterfly_function_matches_density_sign(self):
        # g(k) has the sign of the risk-neutral density implied by the smile, so
        # a negative g must coincide with a call price that is not convex in strike.
        from gammascalp import black_scholes as bs

        vogt = svi.SVI(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
        k = np.linspace(-1.4, 1.4, 2801)
        strikes = np.exp(k)
        calls = bs.price(1.0, strikes, 1.0, 0.0, vogt.implied_vol(k, 1.0))
        convexity = np.gradient(np.gradient(calls, strikes), strikes)
        g = vogt.butterfly(k)
        inner = slice(5, -5)
        self.assertTrue(np.any(convexity[inner] < 0))
        np.testing.assert_array_equal(np.sign(convexity[inner][np.abs(g[inner]) > 0.01]),
                                      np.sign(g[inner][np.abs(g[inner]) > 0.01]))

    def test_calendar_arbitrage_is_detected(self):
        near = svi.SVI(0.03, 0.3, -0.3, 0.0, 0.2)
        far_ok = svi.SVI(0.05, 0.3, -0.3, 0.0, 0.2)
        far_bad = svi.SVI(0.02, 0.3, -0.3, 0.0, 0.2)
        self.assertEqual(svi.calendar_violation([near, far_ok]), 0.0)
        self.assertLess(svi.calendar_violation([near, far_bad]), -0.005)

    def test_lee_bound_flag(self):
        self.assertFalse(svi.SVI(0.0, 1.5, 0.5, 0.0, 0.1).wing_bound_ok())


class HestonSmiles(unittest.TestCase):
    def test_heston_chain_fits_within_a_few_basis_points(self):
        chain, rows = ex.smile_fit(noise_vol_bps=(0.0,))
        self.assertEqual(len(rows), 5)
        for leg in chain:
            self.assertFalse(np.any(np.isnan(leg["implied_vol"])))
            self.assertGreater(leg["implied_vol"][0], leg["implied_vol"][len(leg["k"]) // 2])  # Put skew.
        for row in rows:
            self.assertLess(row["rmse_vs_truth_bps"], 10.0)
            self.assertEqual(row["butterfly_min_g"], 0.0)
            self.assertEqual(row["calendar_violation"], 0.0)
            self.assertLessEqual(row["b"] * (1 + abs(row["rho"])), 2.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
