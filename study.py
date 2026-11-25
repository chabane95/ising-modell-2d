"""Reproduce the zero-field finite-size study of the ferromagnetic square lattice."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import tempfile

import numpy as np

from DataSet import summarize_replicas
from functions import sample_chain


CRITICAL_T = 2.0 / math.log1p(math.sqrt(2.0))  # J = k_B = 1
SIZES = (8, 16, 24)
TEMPERATURES = (1.8, 2.0, 2.12, 2.18, 2.22, CRITICAL_T, 2.30, 2.34, 2.40, 2.55, 2.8, 3.2)
DEFAULT_SEED = 20261125


def measure_point(L, T, *, replicas=8, warmup=768, samples=3072, stride=1,
                  algorithm="wolff", J=1.0, h=0.0, seed=DEFAULT_SEED):
    """Measure one (L,T) using independent equal-length chains and alternating starts."""
    if not isinstance(replicas, (int, np.integer)) or replicas < 2 or samples < 2:
        raise ValueError("At least two independent replicas and two retained measurements are required")
    energies, magnetizations, cluster_sizes = [], [], []
    # Encode T itself, not its index, so changing the order of a scan leaves each run reproducible.
    temperature_bits = int(np.float64(T).view(np.uint64))
    for replica in range(replicas):
        sequence = np.random.SeedSequence([seed, L, temperature_bits, replica])
        child_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        chain = sample_chain(L, T, warmup=warmup, samples=samples, stride=stride,
                             algorithm=algorithm, J=J, h=h, seed=child_seed,
                             initial="random" if replica % 2 == 0 else "up")
        energies.append(chain.energy)
        magnetizations.append(chain.magnetization)
        if chain.mean_cluster_size is not None:
            cluster_sizes.append(chain.mean_cluster_size)
    summary = summarize_replicas(energies, magnetizations, spins=L * L, temperature=T)
    averages = np.mean(energies, axis=1) / L**2
    hot, cold = averages[::2], averages[1::2]
    summary["e_hot_minus_cold"] = float(hot.mean() - cold.mean())
    summary["e_hot_minus_cold_se"] = (float(np.sqrt(hot.var(ddof=1) / len(hot) + cold.var(ddof=1) / len(cold)))
                                     if min(len(hot), len(cold)) >= 2 else float("nan"))
    if cluster_sizes:
        # At zero field the Wolff seed samples clusters with size-biased weights.
        # <|C_seed|>/T and <M²>/(N T) estimate the same susceptibility by symmetry.
        cluster_chi = np.array(cluster_sizes) / T
        signed_m2 = np.square(np.asarray(magnetizations, dtype=float)).mean(axis=1)
        difference = cluster_chi - signed_m2 / (L**2 * T)
        for key, values in (("chi_cluster", cluster_chi), ("chi_cluster_minus_sym", difference)):
            summary[key] = float(values.mean())
            summary[key + "_se"] = float(values.std(ddof=1) / math.sqrt(replicas))
    return {"L": L, "T": T, **summary}


def _measure_pair(pair, **settings):
    return measure_point(*pair, **settings)


def _critical_scaling(rows):
    points = sorted((row for row in rows if row["T"] == CRITICAL_T), key=lambda row: row["L"])
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    if min(first["abs_m"], last["abs_m"]) <= 0:
        return None
    log_sizes = math.log(last["L"] / first["L"])
    exponent = math.log(first["abs_m"] / last["abs_m"]) / log_sizes
    # Different sizes have independent RNG streams. This SE omits size corrections.
    error = math.hypot(first["abs_m_se"] / first["abs_m"], last["abs_m_se"] / last["abs_m"]) / log_sizes
    return {"T": CRITICAL_T, "sizes": [first["L"], last["L"]],
            "beta_over_nu_effective": exponent, "monte_carlo_se": error}


def _write_results(out, rows, metadata):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode("utf-8")
    metadata["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as staging:
        csv_path = Path(staging) / "temperature_scan.csv"
        meta_path = Path(staging) / "metadata.json"
        csv_path.write_bytes(csv_bytes)
        meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        csv_path.replace(out / csv_path.name)
        meta_path.replace(out / meta_path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=768)
    parser.add_argument("--samples", type=int, default=3072)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if (args.workers < 1 or args.replicas < 2 or args.warmup < 0 or args.samples < 2
            or args.stride < 1 or args.seed < 0 or min(args.sizes) < 2
            or not all(math.isfinite(t) and t > 0 for t in args.temperatures)):
        parser.error("Check simulation lengths, positive temperatures, lattice sizes, seed and worker count.")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("Each lattice size and temperature must occur only once.")
    settings = {"replicas": args.replicas, "warmup": args.warmup, "samples": args.samples,
                "stride": args.stride, "seed": args.seed, "algorithm": "wolff", "J": 1.0, "h": 0.0}
    points = [(L, T) for L in sorted(args.sizes) for T in sorted(args.temperatures)]
    measure = partial(_measure_pair, **settings)
    if args.workers == 1:
        rows = list(map(measure, points))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(points))) as pool:
            rows = list(pool.map(measure, points))
    metadata = {
        **settings,
        "sizes": sorted(args.sizes), "temperatures": sorted(args.temperatures),
        "update_unit": "one Wolff cluster flip; store every stride flips after warmup flips",
        "initialization": "even replica indices random independent spins; odd indices all +1",
        "seeds": "uint64 from NumPy SeedSequence([seed,L,float64(T).view(uint64),replica]).generate_state(1,dtype=uint64)",
        "uncertainties": "One standard error from leave-one-replica-out jackknife of the pooled-moment estimators.",
        "observables": "N=L^2. e=<E>/N; abs_m=<|M|>/N; cv=Var(E)/(NT^2); chi=Var(M)/(NT); "
                       "chi_abs=Var(|M|)/(NT); binder=1-<M^4>/(3<M^2>^2). k_B=1.",
        "cluster_check": "At h=0, chi_cluster=<size of seed cluster>/T. chi_cluster_minus_sym subtracts "
                         "<M^2>/(N*T) using <M>=0 by symmetry. Its SE uses paired replica differences.",
        "autocorrelation": "Within-replica centered, biased FFT autocovariances pooled over replicas; Geyer "
                           "initial monotone sequence. tau in retained measurements; ESS=replicas*samples/(2*tau). "
                           "Reliable flag requires a truncation point and samples >= 50*tau; not an equilibrium test. "
                           "nan denotes an undefined estimate.",
        "equilibration_check": "Difference of the energy-per-spin means for the random and ordered starts; "
                               "SE from independent within-group replica means. This checks a single observable.",
        "critical_scaling": _critical_scaling(rows),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    _write_results(args.output, rows, metadata)
    print(f"Wrote {len(rows)} temperature/size points to {args.output}")


if __name__ == "__main__":
    main()
