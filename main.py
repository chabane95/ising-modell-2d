"""Measure the square-lattice Ising model at one temperature."""

import argparse

from study import measure_point


def main():
    parser = argparse.ArgumentParser(description="2D-Ising-Modell: Messung an einem Temperaturpunkt (k_B=1)")
    parser.add_argument("--size", type=int, default=12, help="Lineare Gittergröße L")
    parser.add_argument("--temperature", type=float, default=2.3, help="Temperatur T")
    parser.add_argument("--coupling", type=float, default=1.0, help="Ferromagnetische Kopplung J")
    parser.add_argument("--field", type=float, default=0.0, help="Homogenes Magnetfeld h")
    parser.add_argument("--algorithm", choices=("wolff", "metropolis"),
                        help="Standard: Wolff im Nullfeld, sonst Metropolis")
    parser.add_argument("--replicas", type=int, default=8, help="Anzahl unabhängiger Ketten")
    parser.add_argument("--warmup", type=int, default=256, help="Verworfene Updates pro Kette")
    parser.add_argument("--samples", type=int, default=1024, help="Messungen pro Kette")
    parser.add_argument("--stride", type=int, default=1, help="Updates zwischen Messungen")
    parser.add_argument("--seed", type=int, default=20261125, help="Seed für unabhängige Zufallsströme")
    args = parser.parse_args()
    algorithm = args.algorithm or ("wolff" if args.field == 0 else "metropolis")
    try:
        row = measure_point(args.size, args.temperature, replicas=args.replicas,
                            warmup=args.warmup, samples=args.samples, stride=args.stride,
                            algorithm=algorithm, J=args.coupling, h=args.field, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f"L={args.size}  T={args.temperature:g}  J={args.coupling:g}  h={args.field:g}  {algorithm}")
    print(f"{args.replicas} Repliken, je {args.warmup} Updates Einlaufzeit und {args.samples} Messungen")
    print(f"{'Messgröße':12} {'Mittelwert':>13} {'Standardfehler':>16}")
    for key, label in (("e", "E/N"), ("abs_m", "|M|/N"), ("cv", "C/N"),
                       ("chi", "chi pro Spin"), ("binder", "U4")):
        print(f"{label:12} {row[key]:13.6f} {row[key+'_se']:16.6f}")


if __name__ == "__main__":
    main()
