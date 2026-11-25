"""Compare both Markov chains with an independent finite-volume Boltzmann sum."""

import argparse
import csv
from pathlib import Path

import numpy as np

from study import measure_point


def exact_thermodynamics(L, T, J=1.0, h=0.0):
    """Enumerate all spin bitstrings; count disagreeing bonds including wrapped ones."""
    if not isinstance(L, int) or not 2 <= L <= 4:
        raise ValueError("Exact enumeration supports L=2,3,4")
    if not (np.isfinite(T) and T > 0 and np.isfinite(J) and np.isfinite(h)):
        raise ValueError("T must be positive; T, J and h must be finite")
    n = L * L
    states = np.arange(1 << n, dtype=np.uint32)
    bits = [(states >> i) & 1 for i in range(n)]
    disagreements = np.zeros_like(states)
    for y in range(L):
        for x in range(L):
            spin = bits[y*L + x]
            disagreements += spin ^ bits[y*L + (x+1) % L]
            disagreements += spin ^ bits[((y+1) % L)*L + x]
    m = 2 * np.sum(bits, axis=0, dtype=np.float64) - n
    energy = -2 * J * n + 2 * J * disagreements.astype(float) - h*m
    weights = np.exp(-(energy - energy.min()) / T)
    p = weights / weights.sum()
    def avg(values):
        return float(np.sum(p * values))

    em, mm, m2 = avg(energy), avg(m), avg(m * m)
    return {
        "e": em / n,
        "abs_m": avg(np.abs(m)) / n,
        "cv": avg((energy - em)**2) / (n * T * T),
        "chi": avg((m - mm)**2) / (n * T),
        "binder": 1 - avg(m**4) / (3 * m2 * m2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "exact_check.csv")
    args = parser.parse_args()
    L, T, replicas, warmup, samples, seed = 4, 2.3, 12, 256, 2048, 47821
    exact = exact_thermodynamics(L, T)
    rows = []
    for index, method in enumerate(("metropolis", "wolff")):
        point = measure_point(L, T, replicas=replicas, warmup=warmup, samples=samples,
                              algorithm=method, seed=seed+index)
        for key, reference in exact.items():
            estimate, se = point[key], point[key+"_se"]
            z = (estimate-reference)/se
            rows.append(dict(L=L, T=T, method=method, seed=seed+index, replicas=replicas,
                             warmup=warmup, samples=samples, observable=key,
                             exact=reference, estimate=estimate, se=se, z=z))
            print(f"{method:10} {key:7} exact={reference: .6f}  MC={estimate: .6f} "
                  f"+/- {se:.6f}  z={z:+.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
