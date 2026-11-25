import math
import unittest

from exact_torus import TC, compare, critical_energy
from validate import exact_thermodynamics


class CriticalEnergyTests(unittest.TestCase):
    def test_even_and_odd_squares_against_all_configurations(self):
        for L in (2, 3, 4):
            with self.subTest(L=L):
                self.assertAlmostEqual(critical_energy(L), exact_thermodynamics(L, TC)["e"], places=14)

    def test_asymptotic_coefficients_for_the_square(self):
        # Salas (2001), Eq. (3.22) and Table 1, aspect ratio one.
        q = math.exp(-math.pi)
        theta2 = 2 * sum(q**((n + 0.5)**2) for n in range(8))
        theta3 = 1 + 2 * sum(q**(n*n) for n in range(1, 9))
        theta4 = 1 + 2 * sum((-1)**n * q**(n*n) for n in range(1, 9))
        e1 = 2 * theta2 * theta3 * theta4 / (theta2 + theta3 + theta4)
        e3 = -0.206683145336864
        for L in (32, 64, 128, 256, 4096):
            with self.subTest(L=L):
                scaled = L * (-critical_energy(L) - math.sqrt(2))
                # E5=-0.73018..., hence the remainder after multiplying by L
                # falls as 1/L^4. The tolerance includes double-precision roundoff.
                self.assertLess(abs(scaled - e1 - e3 / L**2), 0.8 / L**4 + 5e-10)

    def test_diagnostic_requires_critical_temperature_and_positive_error(self):
        row = dict(L=8, T=TC, e=-1.490, e_se=0.001)
        result = compare([row])[0]
        self.assertAlmostEqual(result["z"], (-1.49 - critical_energy(8)) / 0.001)
        for override in ({"T": 2.26}, {"e_se": 0}):
            with self.assertRaises(ValueError):
                compare([row | override])
