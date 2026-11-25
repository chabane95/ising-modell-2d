import itertools
import unittest
from collections import Counter

import numpy as np

import config
import functions as mc


class LatticeTests(unittest.TestCase):
    def test_two_by_two_densities_and_every_local_flip(self):
        # Four sites, eight bonds. Each like/unlike stripe has E=0 at h=0.
        counts = Counter()
        J, h = 1.3, -0.25
        for state in itertools.product((-1, 1), repeat=4):
            spins = np.array(state, dtype=np.int8).reshape(2, 2)
            counts[config.energy(spins)] += 1
            e = config.energy(spins, J, h)
            for x, y in np.ndindex(2, 2):
                change = spins.copy()
                change[x, y] *= -1
                neighbors = mc.find_neigboorSpins(spins, x, y)
                expected = 2 * spins[x, y] * (J * sum(neighbors) + h)
                self.assertAlmostEqual(config.energy(change, J, h) - e, expected)
        self.assertEqual(counts, {-8.0: 2, 0.0: 12, 8.0: 2})

    def test_legacy_observables_and_initialization(self):
        lattice = config.Lattice(3)
        lattice.lattice[1, 1] = 1
        self.assertIs(lattice.init_lattice(), lattice.lattice)
        self.assertTrue(np.all(lattice.lattice == -1))
        spins = np.array([[-1, -1], [1, -1]])
        self.assertEqual(config.SpinConfiguration(spins).get_magnetization(), 0.5)
        obs = config.Observables(spins)
        self.assertEqual(obs.get_magnetization(), -2)
        self.assertEqual(obs.get_energy(h=0.1), 0.2)
        self.assertEqual(obs.get_specificHeat([-8, 0], beta=0.5), 1.0)
        self.assertEqual(obs.get_suscebility([-4, 4], beta=0.5), 2.0)

    def test_inputs(self):
        for params in ({"L": 1}, {"L": 3.5}, {"temperature": 0}, {"temperature": np.nan},
                       {"J": 0}, {"h": np.inf}, {"initial": np.ones((2, 3))},
                       {"initial": np.zeros((3, 3))}, {"initial": "other"}):
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                mc.IsingSampler(**({"L": 3, "temperature": 2} | params))
        for options in ({"stride": 0}, {"samples": 0}, {"warmup": -1}, {"algorithm": "other"},
                        {"algorithm": "wolff", "h": 0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                mc.sample_chain(2, 2.2, **options)


class SamplingTests(unittest.TestCase):
    def test_integer_bookkeeping_for_even_odd_and_parallel_bonds(self):
        for L, method, h in itertools.product((2, 3, 8), ("metropolis", "wolff"), (0, -0.35)):
            if method == "wolff" and h != 0:
                continue
            with self.subTest(L=L, method=method, h=h):
                s = mc.IsingSampler(L, 2.2, J=0.73, h=h, seed=16)
                for _ in range(180):
                    if method == "metropolis":
                        self.assertLessEqual(s.metropolis_sweep(), L * L)
                    else:
                        previous = s.lattice.copy()
                        size = s.wolff_step()
                        self.assertEqual(size, np.count_nonzero(previous != s.lattice))
                        self.assertGreater(size, 0)
                    self.assertEqual(s.energy, config.energy(s.lattice, J=0.73, h=h))
                    self.assertEqual(s.magnetization, s.lattice.sum())

    def test_wolff_parallel_bonds_offer_two_independent_chances(self):
        class ScriptedRNG:
            def __init__(self):
                self.trials = iter((0.99, 0.01))

            def integers(self, high):
                return 0

            def random(self):
                return next(self.trials)

        s = mc.IsingSampler(2, 2.0, initial=[[-1, 1], [-1, 1]])
        s.rng = ScriptedRNG()
        self.assertEqual(s.wolff_step(), 2)
        self.assertEqual(s.energy, -8.0)
        self.assertTrue(np.all(s.lattice == 1))

    def test_external_field_spin_inversion_symmetry(self):
        s = np.array([[1, -1, 1], [-1, 1, -1], [1, 1, 1]])
        options = dict(warmup=20, samples=60, stride=3, seed=32)
        plus = mc.sample_chain(3, 2.1, h=0.4, initial=s, **options)
        minus = mc.sample_chain(3, 2.1, h=-0.4, initial=-s, **options)
        np.testing.assert_equal(plus.magnetization, -minus.magnetization)
        np.testing.assert_equal(plus.energy, minus.energy)
        np.testing.assert_equal(plus.lattice, -minus.lattice)
        self.assertIsNone(plus.mean_cluster_size)

    def test_cluster_measurements_at_fixed_update_counts(self):
        solver = mc.IsingSampler(5, 2.3, seed=42)
        for _ in range(11):
            solver.wolff_step()
        e, m, cluster = [], [], []
        for _ in range(17):
            for _ in range(3):
                cluster.append(solver.wolff_step())
            e.append(solver.energy)
            m.append(solver.magnetization)
        chain = mc.sample_chain(5, 2.3, algorithm="wolff", seed=42,
                                warmup=11, samples=17, stride=3)
        np.testing.assert_equal(chain.energy, e)
        np.testing.assert_equal(chain.magnetization, m)
        np.testing.assert_equal(chain.lattice, solver.lattice)
        self.assertEqual(chain.mean_cluster_size, np.mean(cluster))

    def test_original_entry_points(self):
        spins = np.ones((4, 4), dtype=int)
        self.assertIs(mc.metropolis(spins, 1, np.inf, rng=42), spins)
        self.assertTrue(np.all(spins == 1))
        betas, magnetizations = mc.Lsquared(spins, 15, 1, seed=42)
        self.assertEqual(len(betas), len(magnetizations))
        self.assertEqual(len(betas), 8)
        self.assertTrue(np.all((magnetizations >= 0) & (magnetizations <= 1)))


if __name__ == "__main__":
    unittest.main()
