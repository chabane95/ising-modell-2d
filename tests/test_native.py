import collections
import math
import unittest

import numpy as np

import native_wolff
from validate import exact_thermodynamics


class NativeWolffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            native_wolff.build_library()
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_first_cluster_on_two_by_two_torus(self):
        # Start all-up. Translation symmetry fixes the seed at 0. Independently
        # enumerate the occupied-bond graphs, including distinct parallel edges.
        T = 2.4
        p = -math.expm1(-2 / T)
        edges = [(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0), (1, 3), (3, 1)]
        expected = collections.Counter()
        for mask in range(2**len(edges)):
            neighbors = [[] for _ in range(4)]
            for j, (a, b) in enumerate(edges):
                if mask & (1 << j):
                    neighbors[a].append(b)
                    neighbors[b].append(a)
            reached = {0}
            todo = [0]
            while todo:
                for site in neighbors[todo.pop()]:
                    if site not in reached:
                        reached.add(site)
                        todo.append(site)
            spins = [-1 if i in reached else 1 for i in range(4)]
            e = -sum(spins[a] * spins[b] for a, b in edges)
            occupied = mask.bit_count()
            expected[e, sum(spins)] += p**occupied * (1 - p)**(len(edges) - occupied)
        observed = collections.Counter()
        replicas = 8192
        for seed in range(replicas):
            chain = native_wolff.sample_chain(2, T, seed=seed, initial="up", warmup=0, samples=1)
            observed[int(chain.energy[0]), int(chain.magnetization[0])] += 1
        self.assertEqual(set(expected), set(observed))
        for state, prob in expected.items():
            self.assertLess(abs(observed[state] / replicas - prob),
                            5.5 * math.sqrt(prob * (1 - prob) / replicas))

    def test_equilibrium_against_finite_partition_sum(self):
        T, L = 2.3, 4
        exact = exact_thermodynamics(L, T)
        estimates = []
        for seed in range(16):
            chain = native_wolff.sample_chain(L, T, warmup=256, samples=2048,
                                              seed=seed, initial="random" if seed % 2 else "up")
            m = chain.magnetization.astype(float)
            estimates.append([chain.energy.mean() / L**2, np.abs(m).mean() / L**2,
                              np.mean(m**2) / (L**2 * T), chain.mean_cluster_size / T])
        values = np.array(estimates)
        reference = [exact["e"], exact["abs_m"], exact["chi"], exact["chi"]]
        tolerance = 5.5 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
        self.assertTrue(np.all(np.abs(values.mean(axis=0) - reference) < tolerance))

    def test_update_spacing_and_spin_inversion(self):
        kw = dict(L=9, temperature=2.4, warmup=11, seed=117)
        original = native_wolff.sample_chain(**kw, initial="up", samples=91)
        thinned = native_wolff.sample_chain(**kw, initial="up", samples=13, stride=7)
        inverse = native_wolff.sample_chain(**kw, initial="down", samples=13, stride=7)
        np.testing.assert_array_equal(thinned.magnetization, original.magnetization[6::7])
        np.testing.assert_array_equal(thinned.energy, original.energy[6::7])
        np.testing.assert_array_equal(inverse.energy, thinned.energy)
        np.testing.assert_array_equal(inverse.magnetization, -thinned.magnetization)
        self.assertEqual(original.mean_cluster_size, thinned.mean_cluster_size)

    def test_input_range_before_ffi(self):
        for override in ({"L": -1}, {"temperature": math.nan}, {"seed": 2**64},
                         {"warmup": -1}, {"samples": 2**32 + 1}, {"stride": 0}):
            kwargs = dict(L=2, temperature=2, seed=0, warmup=0, samples=1)
            with self.assertRaises(ValueError):
                native_wolff.sample_chain(**(kwargs | override))
