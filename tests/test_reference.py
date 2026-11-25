import math
import unittest

from validate import exact_thermodynamics


class ReferenceTests(unittest.TestCase):
    def test_two_by_two_spectrum_in_field(self):
        # The 16 states comprise uniform, single-defect, striped and checkerboard states.
        J, h, T = 1.2, 0.4, 2.0
        levels = [(-8 * J - 4 * h, 4, 1), (-8 * J + 4 * h, -4, 1),
                  (-2 * h, 2, 4), (2 * h, -2, 4), (0, 0, 4), (8 * J, 0, 2)]
        weights = [g * math.exp(-e / T) for e, m, g in levels]
        partition = sum(weights)

        def mean(observable):
            return sum(w * observable(e, m) for (e, m, _), w in zip(levels, weights)) / partition

        result = exact_thermodynamics(2, T, J, h)
        energy = mean(lambda e, m: e)
        magnetization = mean(lambda e, m: m)
        m2 = mean(lambda e, m: m**2)
        self.assertAlmostEqual(result["e"], energy / 4)
        self.assertAlmostEqual(result["abs_m"], mean(lambda e, m: abs(m)) / 4)
        self.assertAlmostEqual(result["cv"], mean(lambda e, m: (e - energy)**2) / (4 * T**2))
        self.assertAlmostEqual(result["chi"], (m2 - magnetization**2) / (4 * T))
        self.assertAlmostEqual(result["binder"], 1 - mean(lambda e, m: m**4) / (3 * m2**2))
