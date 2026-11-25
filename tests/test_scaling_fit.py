import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scaling_fit import L_REF, chi2_survival, extrapolate, fit_power, read_replicas, resample_replicas
from study import CRITICAL_T


class FiniteSizeFitTests(unittest.TestCase):
    def test_variable_projection_recovers_free_exponent_and_amplitudes(self):
        sizes = np.array([12, 16, 24, 32, 48, 96, 192])
        ratio = sizes / L_REF
        exponents = np.array([0.093, 0.312])
        amplitude = np.array([0.71, 0.65])
        errors = np.linspace(0.0005, 0.0014, len(sizes))
        for omega in (None, 1.0, 1.75, 2.0):
            correction = np.array([0.04, -0.02]) if omega is not None else np.zeros(2)
            m = amplitude[:, None] * ratio**(-exponents[:, None])
            if omega is not None:
                m *= 1 + correction[:, None] * ratio**(-omega)
            x, A, b, chi2, prediction = fit_power(sizes, m, errors, omega)
            np.testing.assert_allclose(x, exponents, atol=3e-9)
            np.testing.assert_allclose(A, amplitude, atol=3e-9)
            np.testing.assert_allclose(b, correction, atol=3e-9)
            np.testing.assert_allclose(prediction, m, atol=2e-10)
            self.assertTrue(np.all(chi2 < 1e-12))

    def test_replica_weights_and_bootstrap_scale(self):
        # Give the two strata different means and unequal numbers of replicas.
        a = np.linspace(0.73, 0.78, 12)
        b = np.linspace(0.64, 0.71, 15)
        groups = {16: {"up": a, "random": b}}
        _, mean, se, draws, _ = resample_replicas(groups, 4096, 403)
        target_se = math.sqrt(len(a) * a.var(ddof=1) + len(b) * b.var(ddof=1)) / (len(a) + len(b))
        self.assertAlmostEqual(mean[0], np.r_[a, b].mean())
        self.assertAlmostEqual(se[0], target_se)
        self.assertLess(abs(draws.mean() - mean[0]), 0.06 * target_se)
        self.assertAlmostEqual(draws.std() / target_se, 1.0, delta=0.05)

    def test_null_fit_and_shared_resamples(self):
        groups = {}
        sizes = (8, 16, 24, 32, 48, 64, 96, 128)
        for size in sizes:
            ratio = size / L_REF
            target = 0.70 * ratio**(-0.24) * (1 + 0.012 * ratio**(-2))
            perturbation = np.linspace(-0.001, 0.001, 12)
            groups[size] = {"up": target + perturbation, "random": target + 2 * perturbation}
        result = extrapolate(groups, 100, 489)
        main = result["primary"]
        self.assertEqual(main["l_min"], 16)
        self.assertEqual(main["dof"], 4)
        self.assertAlmostEqual(main["x"], 0.24, places=8)
        self.assertGreater(main["x_se"], 0)
        self.assertAlmostEqual(main["p_bootstrap"], 1)
        self.assertEqual(main["delta_x_se"], 0)
        self.assertEqual(result["effective"][0]["L2"], 16)
        self.assertTrue(all(f["dof"] >= 2 for f in result["fits"]))

    def test_chisquare_upper_tail(self):
        # Well-known 95% quantiles for odd/even degrees of freedom.
        for dof, quantile in [(1, 3.841459), (2, 5.991465), (5, 11.070498), (8, 15.507313)]:
            self.assertAlmostEqual(chi2_survival(quantile, dof), 0.05, places=7)
        self.assertEqual(chi2_survival(0, 6), 1.0)

    def test_input_rejects_duplicate_chain_and_mixed_temperature(self):
        fields = ["L", "replica", "T", "initial", "samples", "abs_m"]
        row = [16, 0, CRITICAL_T, "random", 1000, 0.7]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp) / "replicas.csv"
            for second in [row, [16, 1, 2.25, "up", 1000, 0.71], [16, 1, CRITICAL_T, "up", 500, 0.68]]:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerows([fields, row, second])
                with self.assertRaises(ValueError):
                    read_replicas(path)
