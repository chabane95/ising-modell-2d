"""Render the zero-field Ising study from its archived CSV (PNG and SVG)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from study import CRITICAL_T


ROOT = Path(__file__).resolve().parent
STYLES = (("#202124", "o"), ("#236697", "s"), ("#b24832", "^"))
TEMPERATURE = r"$k_\mathrm{B}T/J$"
CRITICAL = {"color": "#797979", "linestyle": (0, (2, 2.5)), "linewidth": 1, "zorder": 0}
THEORY = {"color": "#555555", "linestyle": (0, (5, 2)), "linewidth": 1.45, "zorder": 1}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.85,
    "axes.unicode_minus": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.3,
    "ytick.minor.size": 2.3,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "svg.hashsalt": "ising-square-figures",
}


def comma(value, _):
    """German decimal comma and a typographic minus for linear or log ticks."""
    return f"{value:g}".replace(".", ",").replace("-", "−")


def configure_axes(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, pad=7)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))


def measurement(ax, x, y, se, style, *, connect=True):
    color, marker = style
    return ax.errorbar(
        x, y, yerr=se, fmt=marker, linestyle="-" if connect else "none",
        color=color, linewidth=1.05, markersize=4.2, markerfacecolor="white",
        markeredgewidth=1.05, elinewidth=1, capsize=2.0, zorder=3,
    )


def size_legend(groups):
    return [
        Line2D([], [], color=color, marker=marker, markerfacecolor="white",
               markersize=4.2, linewidth=1.05, label=rf"$L={size}$")
        for size, _, (color, marker) in groups
    ]


def add_key(fig, handles):
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.6,
               handlelength=2.1, handletextpad=0.55)


def thermodynamics(groups):
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 6.0), sharex=True)
    fig.subplots_adjust(left=0.095, right=0.983, bottom=0.098, top=0.87,
                        wspace=0.28, hspace=0.33)
    panels = (
        ("abs_m", "(a)  Magnetisierung", r"$\langle |m| \rangle$"),
        ("cv", "(b)  Wärmekapazität", r"$C_V/(N k_\mathrm{B})$"),
        ("chi_abs", "(c)  Suszeptibilität aus $|M|$", r"$J\chi_{|M|}$"),
        ("binder", "(d)  Binder-Kumulante", r"$U_4$"),
    )
    lower = min(rows["T"].min() for _, rows, _ in groups)
    upper = max(rows["T"].max() for _, rows, _ in groups)
    for ax, (field, title, ylabel) in zip(axes.flat, panels):
        configure_axes(ax, title, ylabel)
        ax.axvline(CRITICAL_T, **CRITICAL)
        for _, rows, style in groups:
            measurement(ax, rows["T"], rows[field], rows[field + "_se"], style)
        span = upper - lower
        ax.set_xlim(lower - 0.025 * span, upper + 0.025 * span)
    for ax in axes[-1]:
        ax.set_xlabel(TEMPERATURE)
    # Resolve the nearly vertical approach to Tc without evaluating above Tc.
    if lower < CRITICAL_T:
        distances = np.geomspace(CRITICAL_T - lower, CRITICAL_T * 1e-14, 400)
        t = CRITICAL_T - distances
        m = np.maximum(0, -np.expm1(-4 * np.log(np.sinh(2 / t)))) ** 0.125
        axes[0, 0].plot(np.append(t, CRITICAL_T), np.append(m, 0), **THEORY)
    axes[0, 0].set_ylim(-0.035, 1.035)
    add_key(fig, size_legend(groups) + [
        Line2D([], [], label=r"Yang, $L\to\infty$", **THEORY),
        Line2D([], [], label=r"$T_c$", **CRITICAL),
    ])
    return fig


def scaling(groups):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.45))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.775, wspace=0.35)
    configure_axes(left, "(a)  Skalierung um $T_c$", r"$L^{1/8}\langle |m| \rangle$")
    configure_axes(right, "(b)  Magnetisierung bei $T_c$", r"$\langle |m| \rangle_{T_c}$")
    left.set_xlabel(r"$L(T-T_c)/T_c$")
    left.axvline(0, **CRITICAL)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Gittergröße $L$")
    sizes, critical_m = [], []
    for size, rows, style in groups:
        x = size * (rows["T"] - CRITICAL_T) / CRITICAL_T
        nearby = np.abs(x) <= 1.65
        factor = size ** 0.125
        measurement(left, x[nearby], rows["abs_m"][nearby] * factor,
                    rows["abs_m_se"][nearby] * factor, style)
        critical_rows = rows[np.isclose(rows["T"], CRITICAL_T, rtol=0, atol=1e-12)]
        if len(critical_rows) != 1:
            raise ValueError(f"Expected exactly one critical-temperature measurement for L={size}")
        row = critical_rows[0]
        measurement(right, size, row["abs_m"], row["abs_m_se"], style, connect=False)
        sizes.append(size)
        critical_m.append(row["abs_m"])
    left.set_xlim(-1.7, 1.7)
    left.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

    # Show the predicted slope, fixing only the amplitude at the middle size.
    anchor = len(sizes) // 2
    xref = np.geomspace(sizes[0] / 1.1, sizes[-1] * 1.1, 100)
    ref = critical_m[anchor] * (xref / sizes[anchor]) ** (-0.125)
    right.plot(xref, ref, **THEORY)
    right.set_xlim(xref[0], xref[-1])
    right.xaxis.set_major_locator(FixedLocator(sizes))
    right.xaxis.set_major_formatter(FuncFormatter(comma))
    right.xaxis.set_minor_formatter(NullFormatter())
    # Place readable decimal labels within this narrow log interval.
    lo, hi = right.get_ylim()
    decade = 10 ** np.floor(np.log10(hi - lo))
    step = decade / 4 if (hi - lo) < 1.5 * decade else decade / 2
    ticks = np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step
    right.yaxis.set_major_locator(FixedLocator(ticks))
    right.yaxis.set_major_formatter(FuncFormatter(comma))
    right.yaxis.set_minor_formatter(NullFormatter())
    add_key(fig, size_legend(groups) + [
        Line2D([], [], label=rf"$L^{{-1/8}}$, normiert bei $L={sizes[anchor]}$", **THEORY),
    ])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "temperature_scan.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    fields = ("L", "T", "abs_m", "cv", "chi_abs", "binder")
    fields += tuple(field + "_se" for field in fields[2:])
    data = np.genfromtxt(args.input, delimiter=",", names=True, usecols=fields,
                         dtype=None, encoding="utf-8", ndmin=1)
    groups = []
    for index, size in enumerate(np.unique(data["L"])):
        rows = np.sort(data[data["L"] == size], order="T")
        groups.append((size, rows, STYLES[index % len(STYLES)]))
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        for render in (thermodynamics, scaling):
            fig = render(groups)
            fig.savefig(args.output / f"{render.__name__}.svg", metadata={"Date": None})
            fig.savefig(args.output / f"{render.__name__}.png", dpi=300, facecolor="white")
            plt.close(fig)


if __name__ == "__main__":
    main()
