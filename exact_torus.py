"""Critical energy per spin on a finite periodic square (J = k_B = 1, h = 0).

Ferdinand & Fisher, Phys. Rev. 185, 832 (1969), doi:10.1103/PhysRev.185.832.
Salas, J. Phys. A 34, 1311 (2001), doi:10.1088/0305-4470/34/7/307,
Eqs. (2.3), (2.16a), (2.17), (3.1)–(3.3).  https://arxiv.org/abs/cond-mat/0009054
"""

import argparse
import csv
import math
import operator
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results"
TC = 2 / math.log1p(math.sqrt(2))


def critical_energy(L):
    """Evaluate -d log Z/dK per site at Kc from the four torus sectors."""
    L = operator.index(L)
    if L < 2:
        raise ValueError("The periodic Ising lattice requires L >= 2")

    # a_k = L gamma_k / 2. Reflection about k=L keeps both ends accurate.
    def mode(k):
        angle = (math.pi / (2 * L)) * min(k, 2 * L - k)
        return L * math.asinh(math.sin(angle))

    odd = [mode(k) for k in range(1, 2 * L, 2)]
    even = [mode(k) for k in range(0, 2 * L, 2)]
    # Factor exp(sum(even)) out of every sector before multiplying. Only the
    # O(1) difference of the odd and even mode sums enters the final ratio.
    offset = math.fsum(a - b for a, b in zip(odd, even))

    def sector(modes, sign, log_scale=0.0):
        return math.exp(log_scale + math.fsum(math.log1p(sign * math.exp(-2 * a))
                                              for a in modes))

    z1 = sector(odd, 1, offset)
    z2 = sector(odd, -1, offset)
    z3 = 2 * sector(even[1:], 1)
    # Z4 vanishes at Kc; differentiating its zero mode gives 4L while
    # all nonzero modes have zero first derivative. Division by N contributes 1/L².
    z4_slope = 4 * sector(even[1:], -1)
    return -math.sqrt(2) - z4_slope / (L * (z1 + z2 + z3))


def compare(diagnostics):
    rows = []
    for point in diagnostics:
        L = int(point["L"])
        temperature = float(point.get("T", TC))
        if not math.isclose(temperature, TC, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"The exact reference is valid only at Tc; L={L}, T={temperature}")
        e, se = float(point["e"]), float(point["e_se"])
        if not (math.isfinite(e) and math.isfinite(se) and se > 0):
            raise ValueError(f"Invalid energy or standard error for L={L}")
        exact = critical_energy(L)
        rows.append(dict(L=L, T=temperature, e_mc=e, e_se=se, e_exact=exact, z=(e-exact)/se))
    if not rows or len({r["L"] for r in rows}) != len(rows):
        raise ValueError("Expected one critical-energy measurement per lattice size")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS / "critical_diagnostics.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "critical_energy_check.csv")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = compare(csv.DictReader(stream))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
