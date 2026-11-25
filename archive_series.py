"""Publish the retained E and M time series, or verify the archived replica means."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from critical_study import ROOT, atomic_write


def save_size(path, cache, rows):
    energy, magnetization = [], []
    for row in rows:
        source = cache / f"L{row['L']}-r{int(row['replica']):03d}.npz"
        with np.load(source, allow_pickle=False) as chain:
            energy.append(chain["E"])
            magnetization.append(chain["M"])
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=np.uint64)
    # One row per replica, one column per cluster flip. Keep exact signed totals.
    atomic_write(path, lambda f: np.savez_compressed(f, E=np.stack(energy), M=np.stack(magnetization),
                                                      seed=seeds))


def verify_size(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        E, M, seeds = archive["E"], archive["M"], archive["seed"]
    size, samples = int(rows[0]["L"]), int(rows[0]["samples"])
    shape = (len(rows), samples)
    if E.dtype != np.int32 or M.dtype != np.int32 or E.shape != shape or M.shape != shape:
        raise ValueError(f"Unexpected series shape or dtype in {path}")
    expected_seeds = np.asarray([int(row["seed"]) for row in rows], dtype=np.uint64)
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError(f"Replica seed mismatch in {path}")
    n = size**2
    scaled = M.astype(np.float64) / n
    computed = {"e": np.mean(E.astype(np.float64) / n, axis=1), "abs_m": np.mean(abs(scaled), axis=1),
                "m2": np.mean(scaled**2, axis=1)}
    for key, values in computed.items():
        expected = [float(row[key]) for row in rows]
        if not np.allclose(values, expected, rtol=1e-14, atol=1e-15):
            raise ValueError(f"{key} differs from the replica table in {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--series-dir", type=Path, default=ROOT / "build" / "critical-series")
    parser.add_argument("--verify", action="store_true", help="Read published NPZs; no native compiler or run cache needed.")
    args = parser.parse_args()
    run = json.loads((args.results / "critical_run.json").read_text(encoding="utf-8"))
    replica_file = args.results / "critical_replicas.csv"
    if hashlib.sha256(replica_file.read_bytes()).hexdigest() != run["sha256"][replica_file.name]:
        raise ValueError("Replica CSV does not match the run manifest")
    with replica_file.open(encoding="utf-8", newline="") as stream:
        replicas = list(csv.DictReader(stream))
    output = args.results / "series"
    manifest = output / "sha256.json"
    if not args.verify:
        output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for size in run["sizes"]:
        rows = sorted((r for r in replicas if int(r["L"]) == size), key=lambda r: int(r["replica"]))
        if [int(row["replica"]) for row in rows] != list(range(run["replicas"])):
            raise ValueError(f"Missing or duplicate replica at L={size}")
        path = output / f"L{size:03}.npz"
        if not args.verify:
            save_size(path, args.series_dir / run["cache_key"], rows)
        hashes[path.name] = verify_size(path, rows)
    if args.verify:
        if hashes != json.loads(manifest.read_text(encoding="utf-8")):
            raise ValueError("A time-series archive differs from the published checksum")
    else:
        atomic_write(manifest, lambda f: f.write((json.dumps(hashes, indent=2) + "\n").encode()))
    print(f"Verified {len(replicas)} time series pairs across {len(hashes)} lattice sizes.")


if __name__ == "__main__":
    main()
