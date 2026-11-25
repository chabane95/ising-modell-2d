"""Square Ising lattice, periodic bonds, and observables of a spin configuration."""

import math
import operator

import numpy as np


def lattice_size(L):
    L = operator.index(L)
    if L < 2:
        raise ValueError("The periodic square lattice requires L >= 2.")
    return L


def spin_array(spins):
    a = np.asarray(spins)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square spin lattice.")
    lattice_size(a.shape[0])
    if not np.all((a == 1) | (a == -1)):
        raise ValueError("Each spin must be +1 or -1.")
    return a


def couplings(J, h):
    J, h = float(J), float(h)
    if not math.isfinite(J) or J <= 0 or not math.isfinite(h):
        raise ValueError("J must be positive and finite, h must be finite.")
    return J, h


def bond_sum(spins):
    """Sum s_i s_j over the 2*L² bonds, counting right and down from each site."""
    a = spin_array(spins)
    horizontal = np.sum(a * np.roll(a, -1, axis=1), dtype=np.int64)
    vertical = np.sum(a * np.roll(a, -1, axis=0), dtype=np.int64)
    return int(horizontal + vertical)


def energy(spins, J=1.0, h=0.0):
    """Total Hamiltonian H = -J sum_<ij> s_i s_j - h sum_i s_i."""
    J, h = couplings(J, h)
    return -J * bond_sum(spins) - h * int(np.sum(spins, dtype=np.int64))


class Lattice:
    def __init__(self, L):
        self.L = lattice_size(L)
        self.lattice = np.full((self.L, self.L), -1, dtype=np.int8)

    def init_lattice(self):
        """Reset the original lattice object to the all-down configuration."""
        self.lattice.fill(-1)
        return self.lattice


class SpinConfiguration:
    def __init__(self, config_mat):
        self.config_mat = spin_array(config_mat)
        self.N = self.config_mat.size

    def get_configuration(self):
        return self.config_mat

    def get_magnetization(self):
        """Absolute magnetization per spin, |sum_i s_i|/N."""
        return abs(float(np.mean(self.config_mat)))


class Observables(SpinConfiguration):
    def get_magnetization(self):
        """Signed, extensive magnetization, sum_i s_i."""
        return int(np.sum(self.config_mat, dtype=np.int64))

    def get_energy(self, J=1.0, h=0.0):
        return energy(self.config_mat, J, h)

    def get_specificHeat(self, energies, beta):
        """Heat capacity per site estimated from a series of total energies, k_B=1."""
        return beta**2 * np.var(energies) / self.N

    def get_suscebility(self, magnetizations, beta):
        """Uniform susceptibility per site from a series of signed total magnetizations."""
        return beta * np.var(magnetizations) / self.N
