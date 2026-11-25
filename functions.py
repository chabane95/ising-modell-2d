"""Metropolis and Wolff sampling on a square torus (k_B = 1)."""

from dataclasses import dataclass
from functools import lru_cache
import math
import operator

import numpy as np

import config


@lru_cache(maxsize=8)
def _neighbors(L):
    # Keep all four bonds. On the 2x2 torus two distinct bonds can share an endpoint.
    return tuple((((x + 1) % L) * L + y, x * L + (y - 1) % L,
                  x * L + (y + 1) % L, ((x - 1) % L) * L + y)
                 for x in range(L) for y in range(L))


def find_neigboorSpins(lattice, pos_x, pos_y):
    """Return the four neighboring spin values, preserving the original public spelling."""
    L = config.lattice_size(len(lattice))
    return [lattice[i][j] for i, j in (
        ((pos_x + 1) % L, pos_y % L), (pos_x % L, (pos_y - 1) % L),
        (pos_x % L, (pos_y + 1) % L), ((pos_x - 1) % L, pos_y % L))]


def _acceptance_table(beta, J, h):
    if math.isnan(beta) or beta < 0:
        raise ValueError("Inverse temperature must be nonnegative.")
    probabilities = {}
    for spin in (-1, 1):
        delta = [2 * spin * (J * nn + h) for nn in (-4, -2, 0, 2, 4)]
        probabilities[spin] = tuple(1.0 if de <= 0 else math.exp(-beta * de) for de in delta)
    return probabilities


def _metropolis_sweep(spins, neighbors, probabilities, rng):
    """One random-site sweep. Return the integer changes in bond sum and total spin."""
    n = len(spins)
    sites = rng.integers(n, size=n).tolist()
    uniforms = rng.random(n).tolist()
    delta_bonds = delta_spin = accepted = 0
    for site, u in zip(sites, uniforms):
        a, b, c, d = neighbors[site]
        nn = spins[a] + spins[b] + spins[c] + spins[d]
        old = spins[site]
        if u < probabilities[old][nn // 2 + 2]:
            spins[site] = -old
            delta_bonds -= 2 * old * nn
            delta_spin -= 2 * old
            accepted += 1
    return delta_bonds, delta_spin, accepted


def metropolis(lattice, J, beta, *, rng=None, h=0.0):
    """Update the caller's lattice in place using N random-site proposals with replacement."""
    J, h = config.couplings(J, h)
    a = config.spin_array(lattice)
    work = np.ascontiguousarray(a, dtype=np.int8)
    _metropolis_sweep(memoryview(work.reshape(-1)), _neighbors(len(a)),
                      _acceptance_table(float(beta), J, h), np.random.default_rng(rng))
    if isinstance(lattice, np.ndarray):
        if work is not lattice:
            lattice[...] = work
    else:
        for i, row in enumerate(lattice):
            row[:] = work[i].tolist()
    return lattice


class IsingSampler:
    def __init__(self, L, temperature, *, J=1.0, h=0.0, initial="random", seed=None):
        self.L = config.lattice_size(L)
        self.J, self.h = config.couplings(J, h)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and strictly positive.")
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        shape = (self.L, self.L)
        if isinstance(initial, str):
            if initial == "random":
                self._lattice = self.rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1
            elif initial in ("up", "down"):
                self._lattice = np.full(shape, 1 if initial == "up" else -1, dtype=np.int8)
            else:
                raise ValueError("Choose initial='random', 'up', 'down', or provide a spin array.")
        else:
            a = config.spin_array(initial)
            if a.shape != shape:
                raise ValueError("The initial lattice must match L.")
            self._lattice = np.array(a, dtype=np.int8, order="C", copy=True)
        self._spins = memoryview(self._lattice.reshape(-1))
        self._neighbors = _neighbors(self.L)
        self._bonds = config.bond_sum(self._lattice)
        self.magnetization = int(self._lattice.sum(dtype=np.int64))
        self._acceptance = _acceptance_table(1 / self.temperature, self.J, self.h)
        self._p_bond = -math.expm1(-2 * self.J / self.temperature)

    @property
    def lattice(self):
        """Read-only view of the live lattice; copy when saving a snapshot."""
        view = self._lattice.view()
        view.setflags(write=False)
        return view

    @property
    def energy(self):
        return -self.J * self._bonds - self.h * self.magnetization

    def metropolis_sweep(self):
        db, dm, accepted = _metropolis_sweep(self._spins, self._neighbors, self._acceptance, self.rng)
        self._bonds += db
        self.magnetization += dm
        return accepted

    def wolff_step(self):
        """Flip a single Wolff cluster and return its size. Valid at h=0 for J>0."""
        if self.h != 0:
            raise ValueError("The Wolff kernel requires zero external field.")
        spins, neighbors = self._spins, self._neighbors
        seed = int(self.rng.integers(len(spins)))
        old = spins[seed]
        # Flipping on insertion both marks visited sites and permits exact integer
        # bookkeeping: the energy changes of these individual flips telescope.
        a, b, c, d = neighbors[seed]
        delta_bonds = -2 * old * (spins[a] + spins[b] + spins[c] + spins[d])
        spins[seed] = -old
        stack = [seed]
        size = 1
        uniform, p_bond = self.rng.random, self._p_bond
        while stack:
            for site in neighbors[stack.pop()]:
                if spins[site] != old or uniform() >= p_bond:
                    continue
                a, b, c, d = neighbors[site]
                delta_bonds -= 2 * old * (spins[a] + spins[b] + spins[c] + spins[d])
                spins[site] = -old
                stack.append(site)
                size += 1
        self._bonds += delta_bonds
        self.magnetization -= 2 * old * size
        return size


@dataclass
class Chain:
    energy: np.ndarray
    magnetization: np.ndarray
    lattice: np.ndarray
    mean_cluster_size: float | None


def sample_chain(L, temperature, *, warmup=100, samples=1000, stride=1, algorithm="metropolis",
                 J=1.0, h=0.0, seed=None, initial="random"):
    """Return extensive E and signed M after a fixed number of updates per observation."""
    warmup, samples, stride = map(operator.index, (warmup, samples, stride))
    if warmup < 0 or samples < 1 or stride < 1:
        raise ValueError("Require warmup >= 0, samples >= 1 and stride >= 1.")
    sampler = IsingSampler(L, temperature, J=J, h=h, initial=initial, seed=seed)
    if algorithm == "metropolis":
        update = sampler.metropolis_sweep
    elif algorithm == "wolff":
        update = sampler.wolff_step
    else:
        raise ValueError("Choose algorithm='metropolis' or 'wolff'.")
    for _ in range(warmup):
        update()
    e = np.empty(samples)
    m = np.empty(samples, dtype=np.int64)
    size_sum = 0
    for index in range(samples):
        for _ in range(stride):
            size_sum += update()
        e[index] = sampler.energy
        m[index] = sampler.magnetization
    cluster_size = size_sum / (samples * stride) if algorithm == "wolff" else None
    return Chain(e, m, sampler.lattice.copy(), cluster_size)


def Lsquared(init_lattice, sweep_max, J, *, seed=None):
    """Original beta scan; each point is the mean absolute magnetization per spin."""
    lattice = config.spin_array(init_lattice)
    sweeps = operator.index(sweep_max)
    if sweeps < 1:
        raise ValueError("sweep_max must be positive.")
    rng = np.random.default_rng(seed)
    betas = np.linspace(0, 1, 8)
    warmup = sweeps // 4
    samples = sweeps - warmup
    mags = np.empty(len(betas))
    # At beta=0 spins are independent. A binomial draw gives their exact equilibrium count.
    n = lattice.size
    mags[0] = np.mean(np.abs(2 * rng.binomial(n, 0.5, size=samples) - n)) / n
    for i, beta in enumerate(betas[1:], 1):
        chain = sample_chain(len(lattice), 1 / beta, J=J, seed=rng, initial=lattice,
                             warmup=warmup, samples=samples)
        mags[i] = np.mean(np.abs(chain.magnetization)) / n
    return betas, mags
