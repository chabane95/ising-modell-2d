import math
import unittest

import numpy as np

from DataSet import DataSet, autocorrelation_diagnostics, summarize_replicas
from study import measure_point


class StatisticsTests(unittest.TestCase):
    def test_legacy_standard_error(self):
        data = DataSet([1.0, 3.0, 5.0, 100.0])
        self.assertEqual(data.calc_mean(3), 3)
        self.assertAlmostEqual(data.calc_NaiveError(3), 2 / math.sqrt(3))
        with self.assertRaises(ValueError):
            data.calc_NaiveError(1)

    def test_pooled_fluctuations_and_jackknife(self):
        energies = np.array([[-8, -8], [-4, -4], [0, 0], [4, 4]])
        magnetizations = np.array([[-4, -2], [2, 0], [0, 2], [4, -2]])
        result = summarize_replicas(energies, magnetizations, spins=4, temperature=2.0)
        m = magnetizations.ravel() / 4
        self.assertAlmostEqual(result["e"], -0.5)
        # Replicas, not the eight (pairwise repeated) measurements, set this SE.
        self.assertAlmostEqual(result["e_se"], np.std(energies[:, 0] / 4, ddof=1) / 2)
        self.assertAlmostEqual(result["cv"], np.var(energies) / 16)
        self.assertAlmostEqual(result["chi"], np.var(magnetizations) / 8)
        self.assertAlmostEqual(result["chi_abs"], np.var(abs(magnetizations)) / 8)
        self.assertAlmostEqual(result["abs_m"], np.mean(abs(m)))
        self.assertAlmostEqual(result["binder"], 1 - np.mean(m**4) / (3 * np.mean(m**2)**2))
        loo = [np.var(np.delete(energies, r, axis=0)) / 16 for r in range(4)]
        self.assertAlmostEqual(result["cv_se"], math.sqrt(3 * np.var(loo)))

    def test_autocorrelation_time_for_ar1(self):
        rng = np.random.default_rng(1701)
        series = rng.normal(size=(8, 8192))
        for i in range(1, series.shape[1]):
            series[:, i] += 0.8 * series[:, i - 1]
        tau, effective, reliable = autocorrelation_diagnostics(series)
        expected_tau = (1 + 0.8) / (2 * (1 - 0.8))
        self.assertLess(abs(tau - expected_tau), 0.7)
        self.assertAlmostEqual(effective, series.size / (2 * tau))
        self.assertTrue(reliable)
        shifted = series + np.arange(len(series))[:, None] * 100
        self.assertAlmostEqual(autocorrelation_diagnostics(shifted)[0], tau)

    def test_undefined_and_invalid_input(self):
        frozen = np.zeros((4, 24))
        out = summarize_replicas(frozen, frozen, spins=4, temperature=1)
        self.assertTrue(math.isnan(out["binder"]))
        self.assertTrue(math.isnan(out["tau_e"]))
        self.assertFalse(out["tau_e_reliable"])
        with self.assertRaises(ValueError):
            summarize_replicas(frozen[:1], frozen[:1], spins=4, temperature=1)
        with self.assertRaises(ValueError):
            summarize_replicas(frozen, frozen[:, 1:], spins=4, temperature=1)

    def test_seeded_replicas(self):
        kw = dict(warmup=8, samples=32, replicas=4, seed=91)
        first = measure_point(4, 2.4, **kw)
        again = measure_point(4, 2.4, **kw)
        self.assertEqual(first["binder"], again["binder"])
        self.assertGreater(first["abs_m_se"], 0)
        self.assertGreater(first["chi_cluster"], 0)
        # Symmetrized <M²>/(NT) differs from Var(M)/(NT) by the nonnegative
        # pooled sample-mean correction, even at exactly zero external field.
        self.assertGreaterEqual(first["chi_cluster"] - first["chi_cluster_minus_sym"] + 1e-12,
                                first["chi"])
        metropolis = measure_point(4, 2.4, algorithm="metropolis", **kw)
        self.assertNotIn("chi_cluster", metropolis)


if __name__ == "__main__":
    unittest.main()
