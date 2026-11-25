"""Render the temperature scan and the critical-size study as PNG and SVG."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
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


def fitted_magnetization(length, fit, reference_size):
    relative = np.asarray(length, dtype=float) / reference_size
    correction = 1 + fit["b"] * relative ** -fit["omega"] if fit["omega"] is not None else 1
    return fit["A"] * relative ** -fit["x"] * correction


def size_axis(ax, sizes):
    """Label powers of two, retaining the other simulated sizes as minor ticks."""
    lower, upper = min(sizes), max(sizes)
    ax.set_xscale("log", base=2)
    ax.set_xlim(lower / 1.14, upper * 1.14)
    powers = 2.0 ** np.arange(np.ceil(np.log2(lower)), np.floor(np.log2(upper)) + 1)
    ax.xaxis.set_major_locator(FixedLocator(powers))
    ax.xaxis.set_minor_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.xaxis.set_minor_formatter(NullFormatter())


def scaling(groups, critical):
    fig = plt.figure(figsize=(7.9, 3.8))
    columns = fig.add_gridspec(1, 2, left=0.10, right=0.985, bottom=0.16, top=0.90, wspace=0.34)
    left = fig.add_subplot(columns[0])
    right_column = columns[1].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.09)
    right = fig.add_subplot(right_column[0])
    residuals = fig.add_subplot(right_column[1], sharex=right)
    configure_axes(left, "(a)  Kollaps um $T_c$", r"$L^{1/8}\langle |m| \rangle$")
    configure_axes(right, "(b)  Magnetisierung bei $T_c$", r"$\langle |m| \rangle_{T_c}$")
    left.set_xlabel(r"$L(T-T_c)/T_c$")
    left.axvline(0, **CRITICAL)
    for size, rows, style in groups:
        x = size * (rows["T"] - CRITICAL_T) / CRITICAL_T
        nearby = np.abs(x) <= 1.65
        factor = size ** 0.125
        measurement(left, x[nearby], rows["abs_m"][nearby] * factor,
                    rows["abs_m_se"][nearby] * factor, style)
    left.set_xlim(-1.7, 1.7)
    left.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    left.legend(handles=size_legend(groups), loc="lower left", handlelength=1.9)

    primary = critical["primary"]
    sizes = [row["L"] for row in critical["sizes"]]
    size_axis(right, sizes)
    right.set_yscale("log")
    right.tick_params(labelbottom=False)
    for row in critical["sizes"]:
        style = STYLES[0] if row["L"] in primary["sizes"] else ("#737373", "s")
        measurement(right, row["L"], row["abs_m"], row["abs_m_se"], style, connect=False)
    grid = np.geomspace(primary["l_min"], primary["l_max"], 256)
    fit_line, = right.plot(grid, fitted_magnetization(grid, primary, critical["L_ref"]),
                           color=STYLES[1][0], linewidth=1.65, zorder=2,
                           label=rf"Fit ab $L={primary['l_min']}$")
    right.legend(handles=[fit_line], loc="upper right")
    # Use decimal labels for magnetization despite the logarithmic axes.
    lo, hi = right.get_ylim()
    ticks = MaxNLocator(nbins=6, steps=(1, 2, 2.5, 5, 10)).tick_values(lo, hi)
    right.yaxis.set_major_locator(FixedLocator(ticks[(ticks >= lo) & (ticks <= hi)]))
    right.yaxis.set_major_formatter(FuncFormatter(comma))
    right.yaxis.set_minor_formatter(NullFormatter())
    residuals.axhline(0, **CRITICAL)
    residuals.plot(primary["sizes"], primary["weighted_residuals"], "o", color=STYLES[1][0],
                   markerfacecolor="white", markersize=3.7, markeredgewidth=1)
    extent = max(2, np.ceil(max(np.abs(primary["weighted_residuals"]))))
    residuals.set_yticks([-extent, 0, extent])
    residuals.set_ylim(-1.2 * extent, 1.2 * extent)
    residuals.set_xlabel("Gittergröße $L$")
    residuals.set_ylabel(r"$r_i/\sigma_i$")
    return fig


def critical_fit(critical):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.65))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.78, wspace=0.34)
    configure_axes(left, "(a)  Verdopplung des Gitters", r"$x_{\mathrm{eff}}(L,2L)$")
    configure_axes(right, "(b)  Stabilität des Fits", r"$x=\beta/\nu$")
    for ax in (left, right):
        ax.axhline(1 / 8, **THEORY)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    primary = critical["primary"]
    pairs = critical["effective"]
    starts = np.array([pair["L1"] for pair in pairs])
    size_axis(left, starts)
    left.set_xlabel("Gittergröße $L$")
    measurement(left, starts, [p["x"] for p in pairs], [p["x_se"] for p in pairs],
                STYLES[0], connect=False)

    # Only draw size pairs for which both sizes lie in the primary fit window.
    lo, hi = primary["l_min"], primary["l_max"] / 2
    if lo < hi:
        grid = np.geomspace(lo, hi, 256)
        m = fitted_magnetization(grid, primary, critical["L_ref"])
        m2 = fitted_magnetization(2 * grid, primary, critical["L_ref"])
        left.plot(grid, np.log(m / m2) / np.log(2), color=STYLES[1][0], linewidth=1.4)

    model_styles = {
        None: (STYLES[0][0], "o", "ohne Korrektur"),
        1.0: ("#598477", "D", r"$\omega=1$"),
        1.75: (STYLES[2][0], "^", r"$\omega=7/4$"),
        2.0: (STYLES[1][0], "s", r"$\omega=2$"),
    }
    cuts = sorted({fit["l_min"] for fit in critical["fits"]})
    model_handles = []
    for offset, (omega, (color, marker, label)) in zip(np.linspace(-0.24, 0.24, 4), model_styles.items()):
        rows = [fit for fit in critical["fits"] if fit["omega"] == omega]
        if not rows:
            continue
        x = [cuts.index(row["l_min"]) + offset for row in rows]
        measurement(right, x, [row["x"] for row in rows], [row["x_se"] for row in rows],
                    (color, marker), connect=False)
        if omega == primary["omega"]:
            right.plot(cuts.index(primary["l_min"]) + offset, primary["x"], marker=marker,
                       color=color, linestyle="none", markersize=4.2, zorder=4)
        model_handles.append(Line2D([], [], color=color, marker=marker, markersize=4.5,
                                    markerfacecolor="white", linestyle="none", label=label))
    right.set_xticks(range(len(cuts)), labels=[str(cut) for cut in cuts])
    right.set_xlim(-0.65, len(cuts) - 0.35)
    right.xaxis.set_minor_locator(FixedLocator([]))
    right.set_xlabel(r"Kleinste Gittergröße $L_{\mathrm{min}}$")

    left.legend(handles=[Line2D([], [], color=STYLES[1][0], linewidth=1.4, label="Hauptfit"),
                         Line2D([], [], label=r"$x=1/8$", **THEORY)],
                loc="lower left", bbox_to_anchor=(-0.02, 1.16), ncol=2, borderaxespad=0)
    right.legend(handles=model_handles, loc="lower left", bbox_to_anchor=(-0.02, 1.15),
                 ncol=2, columnspacing=0.9, handlelength=1.0, borderaxespad=0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "temperature_scan.csv")
    parser.add_argument("--critical-fit", type=Path, default=ROOT / "results" / "critical_fit.json")
    parser.add_argument("--figure", nargs="+", choices=("thermodynamics", "scaling", "critical_fit"),
                        default=("thermodynamics", "scaling", "critical_fit"))
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    fields = ("L", "T", "abs_m", "cv", "chi_abs", "binder")
    fields += tuple(field + "_se" for field in fields[2:])
    groups = []
    if "thermodynamics" in args.figure or "scaling" in args.figure:
        data = np.genfromtxt(args.input, delimiter=",", names=True, usecols=fields,
                             dtype=None, encoding="utf-8", ndmin=1)
        for index, size in enumerate(np.unique(data["L"])):
            rows = np.sort(data[data["L"] == size], order="T")
            groups.append((size, rows, STYLES[index % len(STYLES)]))
    if "scaling" in args.figure or "critical_fit" in args.figure:
        critical = json.loads(args.critical_fit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        for name in args.figure:
            if name == "thermodynamics":
                fig = thermodynamics(groups)
            elif name == "scaling":
                fig = scaling(groups, critical)
            else:
                fig = critical_fit(critical)
            fig.savefig(args.output / f"{name}.svg", metadata={"Date": None})
            fig.savefig(args.output / f"{name}.png", dpi=300, facecolor="white")
            plt.close(fig)


if __name__ == "__main__":
    main()
