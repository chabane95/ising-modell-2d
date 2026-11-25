"""Statistics for equally long independent Markov chains of total E and signed M."""

from math import isfinite, sqrt

import numpy as np


class DataSet:
    """Original one-series interface; calc_NaiveError assumes independent draws."""

    def __init__(self, data_array):
        self.data_array = np.asarray(data_array, dtype=float)

    def print_data(self):
        print(self.data_array)

    def _prefix(self, N, minimum):
        if not isinstance(N, (int, np.integer)) or not minimum <= N <= len(self.data_array):
            raise ValueError(f"N must be between {minimum} and the length of the series")
        return self.data_array[:N]

    def calc_mean(self, N):
        return float(np.mean(self._prefix(N, 1)))

    def calc_NaiveError(self, N):
        return float(np.std(self._prefix(N, 2), ddof=1) / sqrt(N))


def autocorrelation_diagnostics(chains):
    """Return (tau_int, effective sample count, resolved) in saved-measurement units.

    Each chain has its own mean removed. Pool autocovariances at the same lag,
    then apply Geyer's initial monotone sequence of adjacent lag pairs. Includes
    zero lag with weight 1/2, so white noise has tau_int = 0.5. The length check
    is a diagnostic heuristic; it cannot detect all equilibration failures.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or not np.all(np.isfinite(chains)):
        raise ValueError("Expected a finite matrix of Markov chains")
    replicas, n = chains.shape
    if n < 4 or replicas == 0:
        return float("nan"), float("nan"), False
    centered = chains - chains.mean(axis=1, keepdims=True)
    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_len, axis=1)
    autocov = np.fft.irfft(np.sum(np.abs(spectrum) ** 2, axis=0), n=fft_len)[:n]
    if autocov[0] <= 0:
        return float("nan"), float("nan"), False
    rho = autocov / autocov[0]
    pairs = rho[:n - n % 2].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(pairs <= 0)
    kept = pairs[:stop[0]] if stop.size else pairs
    tau = float(np.minimum.accumulate(kept).sum() - 0.5)
    if tau <= 0:
        return float("nan"), float("nan"), False
    return tau, replicas * n / (2 * tau), bool(stop.size and n >= 50 * tau)


def _observables(moments, spins, temperature):
    e, e2, m, abs_m, m2, m4 = moments
    # Roundoff may put a variance just below zero for a constant series.
    return {
        "e": e,
        "abs_m": abs_m,
        "cv": spins * max(0.0, e2 - e * e) / temperature**2,
        "chi": spins * max(0.0, m2 - m * m) / temperature,
        "chi_abs": spins * max(0.0, m2 - abs_m * abs_m) / temperature,
        "binder": 1.0 - m4 / (3 * m2**2) if m2 > 0 else float("nan"),
    }


def summarize_replicas(energies, magnetizations, *, spins, temperature):
    """Pool raw moments, then jackknife the entire estimator over replicate chains.

    Inputs are total energy and total signed magnetization, one row per independent
    replica and one column per retained draw; spins is the number of lattice sites.
    """
    if not isinstance(spins, (int, np.integer)) or spins < 1 or not isfinite(temperature) or temperature <= 0:
        raise ValueError("spins must be a positive integer and temperature must be positive and finite")
    energy = np.asarray(energies, dtype=float) / spins
    magnet = np.asarray(magnetizations, dtype=float) / spins
    if energy.ndim != 2 or energy.shape != magnet.shape or min(energy.shape) < 2:
        raise ValueError("Expected matching arrays with at least two replicas and two draws per replica")
    if not np.isfinite(energy).all() or not np.isfinite(magnet).all():
        raise ValueError("All energies and magnetizations must be finite")
    replicas, samples = energy.shape
    moments = np.column_stack([x.mean(axis=1) for x in
                               (energy, energy**2, magnet, abs(magnet), magnet**2, magnet**4)])
    total = moments.sum(axis=0)
    central = _observables(total / replicas, spins, temperature)
    omitted = [_observables((total - row) / (replicas - 1), spins, temperature) for row in moments]
    result = {"replicas": replicas, "samples": samples}
    for name, value in central.items():
        leave_one_out = np.array([item[name] for item in omitted])
        result[name] = float(value)
        result[name + "_se"] = float(np.sqrt((replicas - 1) * np.var(leave_one_out)))
    for name, observable in (("e", energy), ("abs_m", abs(magnet))):
        tau, effective, reliable = autocorrelation_diagnostics(observable)
        result["tau_" + name] = tau
        result["ess_" + name] = effective
        result["tau_" + name + "_reliable"] = reliable
    return result
