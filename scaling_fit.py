"""Extrapolate the critical magnetization using independent Wolff-replica means.

The leading exponent x is fitted to m = <|M|>/L² at the exact critical
temperature. L_ref only sets the units in which the amplitudes are reported.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from critical_study import atomic_write
from study import CRITICAL_T


ROOT = Path(__file__).resolve().parent
L_REF = 16
PRIMARY_MIN = 16
PRIMARY_OMEGA = 2.0
OMEGAS = (PRIMARY_OMEGA, 1.75, 1.0, None)
BOOTSTRAP_SEED = 967241
BOOTSTRAP_DRAWS = 4096


def read_replicas(path):
    """Reject duplicate chains, mixed temperatures and unequal exposure within a size."""
    grouped = defaultdict(lambda: defaultdict(dict))
    lengths = defaultdict(set)
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"L", "T", "replica", "initial", "samples", "abs_m"}
        if required.difference(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {required.difference(reader.fieldnames or [])}")
        for row in reader:
            size, replica, samples = int(row["L"]), int(row["replica"]), int(row["samples"])
            if (size < 2 or replica < 0 or samples < 1 or row["initial"] not in ("random", "up")
                    or not math.isclose(float(row["T"]), CRITICAL_T, rel_tol=0, abs_tol=1e-12)
                    or float(row.get("h", 0)) != 0 or float(row.get("J", 1)) != 1):
                raise ValueError(f"Unexpected critical ensemble at L={size}, replica={replica}")
            key = (size, replica)
            if key in seen:
                raise ValueError(f"Duplicate replica {key}")
            seen.add(key)
            m = float(row["abs_m"])
            if not 0 < m <= 1:
                raise ValueError(f"Invalid absolute magnetization at {key}")
            grouped[size][row["initial"]][replica] = m
            lengths[size].add(samples)
    if any(len(numbers) != 1 for numbers in lengths.values()):
        raise ValueError("Each size needs equal numbers of retained measurements per replica.")
    return {size: {initial: [means[r] for r in sorted(means)]
                   for initial, means in sorted(strata.items())} for size, strata in sorted(grouped.items())}


def resample_replicas(grouped, draws, seed):
    """Stratified means and standard errors, keeping independent replicas intact.

    The empirical distribution with n atoms has variance (n-1)/n times the
    unbiased sample variance. Rescale its centered atoms before resampling.
    Bootstrap SEs from the same resampled atoms enter the studentized null test.
    """
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    sizes = np.array(sorted(grouped), dtype=int)
    means, errors = np.zeros(len(sizes)), np.zeros(len(sizes))
    boot = np.zeros((draws, len(sizes)))
    boot_errors = np.zeros_like(boot)
    for i, size in enumerate(sizes):
        strata = grouped[size]
        total = sum(len(values) for values in strata.values())
        for index, (initial, data) in enumerate(sorted(strata.items())):
            values = np.array(data, dtype=float)
            n = len(values)
            if n < 2 or values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"At least two finite independent replicates per stratum at L={size}, {initial}.")
            weight = n / total
            center = values.mean()
            means[i] += weight * center
            errors[i] += weight**2 * values.var(ddof=1) / n
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(size), index]))
            indices = rng.integers(n, size=(draws, n))
            sampled = (values[indices] - center) * math.sqrt(n / (n - 1))
            boot[:, i] += weight * (center + sampled.mean(axis=1))
            boot_errors[:, i] += weight**2 * sampled.var(axis=1, ddof=1) / n
    if np.any(errors <= 0) or np.any(boot_errors <= 0):
        raise ValueError("Each size needs a resolved between-replica variance, also in the bootstrap.")
    return sizes, means, np.sqrt(errors), boot, np.sqrt(boot_errors)


def fit_power(sizes, y, se, omega):
    """Fit rows of magnetizations in original units by weighted variable projection.

    Given x, A and B=A*b are linear. Center the correction column in its
    weighted inner product to obtain B without subtracting nearly equal sums.
    Search a broad physical range for x; endpoints indicate a failed fit.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    se = np.atleast_2d(np.asarray(se, dtype=float))
    log_l = np.log(np.asarray(sizes, dtype=float) / L_REF)
    if (y.shape[1] != len(sizes) or se.shape[1] != len(sizes)
            or se.shape[0] not in (1, len(y)) or not np.isfinite(y).all()
            or not np.isfinite(se).all() or np.any(se <= 0)
            or len(set(sizes)) < (2 if omega is None else 3)):
        raise ValueError("Need independent sizes and matching observations with positive finite errors.")
    weights = 1 / se**2
    correction = None if omega is None else np.exp(-omega * log_l)

    def profile(x):
        g = np.exp(-x[:, None] * log_l)
        wgg = weights * g**2
        wgy = weights * g * y
        norm = wgg.sum(axis=1)
        A = wgy.sum(axis=1) / norm
        B = np.zeros_like(A)
        if correction is not None:
            qbar = (wgg * correction).sum(axis=1) / norm
            centered_q = correction - qbar[:, None]
            B = (centered_q * wgy).sum(axis=1) / (wgg * centered_q**2).sum(axis=1)
            A -= B * qbar
            prediction = g * (A[:, None] + B[:, None] * correction)
        else:
            prediction = g * A[:, None]
        chi2 = (weights * (prediction - y)**2).sum(axis=1)
        return chi2, A, B, prediction

    # Bracket each draw on its own coarse grid before golden-section refinement.
    grid = np.linspace(0, 1, 17)
    values = np.stack([profile(np.full(len(y), x))[0] for x in grid])
    best = np.argmin(values, axis=0)
    lo, hi = grid[np.maximum(0, best - 1)], grid[np.minimum(len(grid) - 1, best + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    p, q = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fp, fq = profile(p)[0], profile(q)[0]
    for _ in range(44):
        left = fp <= fq
        lo, hi = np.where(left, lo, p), np.where(left, q, hi)
        p, q = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fresh = profile(np.where(left, p, q))[0]
        fp, fq = np.where(left, fresh, fq), np.where(left, fp, fresh)
    x = (lo + hi) / 2
    chi2, A, B, predicted = profile(x)
    if np.any((x < 1e-8) | (x > 1 - 1e-8) | (A <= 0)) or np.any(predicted <= 0):
        raise ValueError("Exponent at search boundary or nonphysical leading amplitude/prediction.")
    return x, A, B / A, chi2, predicted


def chi2_survival(chi2, dof):
    """Upper chi-squared tail for integer dof via the half-integer gamma recurrence."""
    if not isinstance(dof, (int, np.integer)) or dof < 1 or chi2 < 0:
        raise ValueError("Expected positive integer degrees of freedom and nonnegative chi-square.")
    z = chi2 / 2
    if z == 0:
        return 1.0
    shape = 0.5 if dof % 2 else 1.0
    tail = math.erfc(math.sqrt(z)) if dof % 2 else math.exp(-z)
    while shape < dof / 2:
        tail += math.exp(shape * math.log(z) - z - math.lgamma(shape + 1))
        shape += 1
    return min(1.0, tail)


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def extrapolate(grouped, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    sizes, m, se, boot, boot_se = resample_replicas(grouped, draws, seed)
    large = sizes[sizes >= PRIMARY_MIN]
    if len(large) < 5:
        raise ValueError("The main corrected fit requires at least five sizes with L >= 16.")
    cutoffs = list(large[:min(4, len(large) - 4)])
    if sizes[0] < large[0]:
        cutoffs.append(sizes[0])
    records = []
    exponent_draws = []
    for cutoff in cutoffs:
        keep = sizes >= cutoff
        lattice = sizes[keep]
        for omega in OMEGAS:
            x, A, b, chi2, predicted = fit_power(lattice, m[keep], se[keep], omega)
            xb, Ab, bb, _, _ = fit_power(lattice, boot[:, keep], se[keep], omega)
            dof = len(lattice) - (2 if omega is None else 3)
            record = {
                "model": "power" if omega is None else "corrected", "omega": omega,
                "l_min": int(cutoff), "l_max": int(lattice[-1]), "sizes": lattice.tolist(),
                "x": float(x[0]), "x_se": float(xb.std(ddof=1)), "x_ci95": interval(xb),
                "A": float(A[0]), "A_se": float(Ab.std(ddof=1)),
                "b": float(b[0]), "b_se": float(bb.std(ddof=1)),
                "chi2": float(chi2[0]), "dof": dof, "p_approx": chi2_survival(chi2[0], dof),
            }
            if cutoff == large[0] and omega == PRIMARY_OMEGA:
                reweighted = fit_power(lattice, boot[:, keep], boot_se[:, keep], omega)[0]
                record["x_se_reweighted"] = float(reweighted.std(ddof=1))
                record["x_ci95_reweighted"] = interval(reweighted)
                null_samples = predicted + (boot[:, keep] - m[keep])
                null_chi2 = fit_power(lattice, null_samples, boot_se[:, keep], omega)[3]
                p = (1 + np.count_nonzero(null_chi2 >= chi2[0])) / (1 + draws)
                record["p_bootstrap"] = float(p)
                record["p_bootstrap_mc_se"] = math.sqrt(p * (1 - p) / draws)
                record["weighted_residuals"] = ((m[keep] - predicted[0]) / se[keep]).tolist()
                primary = record
                main_draws = xb
            records.append(record)
            exponent_draws.append(xb)
    for record, xb in zip(records, exponent_draws):
        record["delta_x"] = record["x"] - primary["x"]
        record["delta_x_se"] = float(np.std(xb - main_draws, ddof=1))
    effective = []
    by_size = {int(size): i for i, size in enumerate(sizes)}
    for size, i in by_size.items():
        if 2 * size in by_size:
            j = by_size[2 * size]
            values = np.log(boot[:, i] / boot[:, j]) / math.log(2)
            effective.append({"L1": size, "L2": 2 * size, "x": math.log(m[i] / m[j], 2),
                              "x_se": float(values.std(ddof=1)), "x_ci95": interval(values)})
    points = [{"L": int(size), "replicas": sum(map(len, grouped[size].values())),
               "abs_m": float(value), "abs_m_se": float(error)}
              for size, value, error in zip(sizes, m, se)]
    return {
        "T": CRITICAL_T, "L_ref": L_REF,
        "model": "m(L)=A*(L/L_ref)^(-x)*(1+b*(L/L_ref)^(-omega)); x=beta/nu",
        "primary_rule": "omega=2, all sizes L>=16. L_min, omega and pure powers varied independently.",
        "bootstrap": {
            "draws": draws, "seed": seed,
            "method": "Resample whole independent chain means within (L, initial). Rescale centered "
                      "empirical atoms by sqrt(n/(n-1)). Keep each stratum's sample count fixed. "
                      "Shared resamples for all cutoffs, models and effective slopes. Conditional fits "
                      "use inverse variances from the original chains; x_ci95 is the percentile interval. "
                      "For the main fit x_ci95_reweighted repeats estimation of the weights in every draw.",
            "goodness": "p_approx uses the chi-square tail with estimated original weights. For "
                        "p_bootstrap the resampled means are recentered on the fitted null curve. "
                        "Re-estimate stratum variances and refit each synthetic dataset before "
                        "counting its chi-square in the upper tail; use the plus-one correction.",
        },
        "sizes": points, "primary": primary, "fits": records, "effective": effective,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "critical_replicas.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "critical_fit.json")
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    grouped = read_replicas(args.input)
    result = extrapolate(grouped, args.draws, args.seed)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    report = (json.dumps(result, allow_nan=False, indent=2) + "\n").encode("utf-8")
    atomic_write(args.output, lambda stream: stream.write(report))
    fit = result["primary"]
    print(f"x={fit['x']:.7f} +/- {fit['x_se']:.7f}, 95% {fit['x_ci95']}, "
          f"chi2/dof={fit['chi2']:.2f}/{fit['dof']}, bootstrap p={fit['p_bootstrap']:.3f}")


if __name__ == "__main__":
    main()
