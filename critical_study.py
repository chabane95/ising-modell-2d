"""Collect independent Wolff chains at the known critical temperature of the square lattice."""

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import numpy as np

from DataSet import autocorrelation_diagnostics
from study import CRITICAL_T


ROOT = Path(__file__).resolve().parent
SIZES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
MASTER_SEED = 451976231


def seed_for(size, replica, master):
    seq = np.random.SeedSequence([master, size, replica])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def atomic_write(path, write):
    """Publish a complete file; concurrent identical chains can finish in either order."""
    with NamedTemporaryFile(dir=path.parent, delete=False, prefix=path.name + ".") as stream:
        temp = Path(stream.name)
        try:
            write(stream)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def run_replica(cache, size, replica, settings):
    from native_wolff import sample_chain

    path = cache / f"L{size}-r{replica:03d}.npz"
    if path.is_file():
        return
    seed = seed_for(size, replica, settings["master_seed"])
    state = "random" if replica % 2 == 0 else "up"
    chain = sample_chain(size, CRITICAL_T, warmup=settings["warmup"], samples=settings["samples"],
                         seed=seed, initial=state, stride=1)
    atomic_write(path, lambda f: np.savez_compressed(f, E=chain.energy, M=chain.magnetization,
                                                     mean_cluster=chain.mean_cluster_size))


def se(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def mean_se(values):
    return float(np.mean(values)), se(values)


def summarize_size(cache, size, settings):
    n = size**2
    replicas, samples = settings["replicas"], settings["samples"]
    rows, energies, abs_mags = [], [], []
    for replica in range(replicas):
        path = cache / f"L{size}-r{replica:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            e = np.asarray(data["E"], dtype=float) / n
            m = np.asarray(data["M"], dtype=float) / n
            cluster = float(data["mean_cluster"])
        if (e.shape != (samples,) or m.shape != (samples,) or
                not (np.all(np.abs(e) <= 2) and np.all(np.abs(m) <= 1) and 1 <= cluster <= n)):
            raise ValueError(f"Invalid cached time series in {path}")
        a = np.abs(m)
        energies.append(e)
        abs_mags.append(a)
        rows.append({"L": size, "T": CRITICAL_T, "replica": replica,
                     "initial": "random" if replica % 2 == 0 else "up",
                     "seed": seed_for(size, replica, settings["master_seed"]),
                     "warmup": settings["warmup"], "samples": samples,
                     "abs_m": float(a.mean()), "e": float(e.mean()), "m2": float(np.mean(m*m)),
                     "mean_cluster": cluster})
    diag = {"L": size, "T": CRITICAL_T, "replicas": replicas}
    for name, values in (("abs_m", abs_mags), ("e", energies)):
        series = np.asarray(values)
        chain_mean = series.mean(axis=1)
        hot, cold = chain_mean[::2], chain_mean[1::2]
        h = samples // 2
        shift = series[:, h:].mean(axis=1) - series[:, :h].mean(axis=1)
        diag[name] = float(chain_mean.mean())
        diag[name + "_se"] = float(np.hypot(se(hot), se(cold)) / 2)
        diag[name + "_hot_minus_cold"] = float(hot.mean() - cold.mean())
        diag[name + "_hot_minus_cold_se"] = float(np.hypot(se(hot), se(cold)))
        diag[name + "_late_minus_early"], diag[name + "_late_minus_early_se"] = mean_se(shift)
        tau, ess, resolved = autocorrelation_diagnostics(series)
        diag["tau_" + name], diag["ess_" + name], diag["tau_" + name + "_resolved"] = tau, ess, resolved
    # At h=0 each seed cluster is size biased and <|C|>/N equals <m²>.
    paired = [r["mean_cluster"] / n - r["m2"] for r in rows]
    diag["cluster_over_N_minus_m2"], diag["cluster_over_N_minus_m2_se"] = mean_se(paired)
    return rows, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--replicas", type=int, default=24)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--series-dir", type=Path, default=ROOT / "build" / "critical-series")
    args = parser.parse_args()
    sizes = sorted(args.sizes)
    if (len(set(sizes)) != len(sizes) or min(sizes) < 2 or args.replicas < 4 or args.replicas % 2
            or args.warmup < 0 or args.samples < 8 or args.samples % 2 or args.seed < 0 or args.workers < 1):
        parser.error("Require distinct L >= 2, even replicas >= 4, even samples >= 8, nonnegative seed/warmup and workers >= 1.")

    from native_wolff import build_library
    library = build_library()
    sources = [ROOT / "native_wolff.py", ROOT / "critical_study.py"]
    sources.extend(sorted(path for path in (ROOT / "native").rglob("*")
                          if path.suffix in {".h", ".c", ".cpp", ".hpp"}))
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                     for p in sources}
    settings = {"T": CRITICAL_T, "J": 1, "h": 0, "master_seed": args.seed, "warmup": args.warmup,
                "samples": args.samples, "stride": 1, "source_sha256_lf": source_hashes,
                "numpy_version": np.__version__}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cache = args.series_dir / digest
    cache.mkdir(parents=True, exist_ok=True)
    settings["replicas"] = args.replicas
    print(f"Critical run on {len(sizes)} sizes, {args.replicas} replicas each. Cache {cache}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_replica, cache, size, replica, settings): size
                   for size in reversed(sizes) for replica in range(args.replicas)}
        counts = Counter()
        for future in as_completed(futures):
            future.result()
            size = futures[future]
            counts[size] += 1
            if counts[size] == args.replicas:
                print(f"L={size}: {counts[size]} chains complete", flush=True)

    replicas, diagnostics = [], []
    for size in sizes:
        rows, diag = summarize_size(cache, size, settings)
        replicas.extend(rows)
        diagnostics.append(diag)
    metadata = {**settings, "sizes": sizes, "python_version": platform.python_version(),
                "native_binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "platform": platform.platform(), "prng": "xoshiro256++ 1.0, SplitMix64 seed expansion",
                "boundary": "square torus; nearest neighbours", "update_unit": "one Wolff single-cluster flip",
                "initial": "even replica: independent random spins; odd replica: all +1",
                "rng_seed": "uint64 from SeedSequence([master_seed,L,replica]).generate_state(1,dtype=uint64)",
                "samples_definition": "one measurement per cluster flip after the discarded updates",
                "cache_key": digest, "cache_fields": "E, M are extensive integer time series; mean_cluster is over the retained flips",
                "diagnostics": "e=E/N, abs_m=|M|/N. Means across chains; equal-weight ordered/random means and within-start replica SE. Hot-minus-cold is random minus ordered. Late-minus-early is paired within a chain. Cluster-size identity uses paired replica differences.",
                "autocorrelation": "Geyer initial monotone sequence, chain-centered autocovariances pooled across replicas; tau in cluster updates. ESS=replicas*samples/(2*tau). Resolution requires a truncation and samples >= 50*tau.",
                "sha256": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("critical_replicas.csv", replicas), ("critical_diagnostics.csv", diagnostics)):
        payload = csv_bytes(rows)
        atomic_write(args.output / filename, lambda f: f.write(payload))
        metadata["sha256"][filename] = hashlib.sha256(payload).hexdigest()
    manifest = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(args.output / "critical_run.json", lambda f: f.write(manifest))


if __name__ == "__main__":
    main()
