"""Generate every report figure from results/*.csv (run analyse.py first for cost_model.csv).

Usage (from the repository root):
    python scripts/make_plots.py [--results DIR] [--out DIR]

Colour roles (validated with the dataviz palette checker, light surface):
  blocked = blue #2a78d6, Strassen = orange #eb6834, third series = aqua #1baf7a
  (aqua is < 3:1 contrast, so every series also has its own marker and a direct label);
  ordered quantities (problem size, antagonist MB, recursion depth) use one-hue ramps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import mmdata as md

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
ORANGE_RAMP = ["#f0915f", "#d8561f", "#9c3a0f"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PRED = "#e1e0d9"  # shaded band for a pre-registered prediction


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "lines.markersize": 5,
    })


def log2_x(ax, ticks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def label_end(ax, x, y, text, color=INK2, dx=4, dy=0) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=8, color=color)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name)
    plt.close(fig)
    print("wrote", out / name)


# ---------------------------------------------------------------------------
def fig_scaling(R, out):
    e3 = md.summarise(md.load("E3_scaling.csv", R), ["n", "algorithm"])
    w = e3.pivot(index="n", columns="algorithm", values=["median_s", "eff_gflops"])
    n = w.index.to_numpy()
    ratio = (w["median_s"]["strassen"] / w["median_s"]["blocked"]).to_numpy()
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a.plot(n, w["eff_gflops"]["blocked"], "-o", color=BLUE, label="Blocked (A)")
    a.plot(n, w["eff_gflops"]["strassen"], "-s", color=ORANGE, label="Strassen (B)")
    label_end(a, n[-1], w["eff_gflops"]["blocked"].iloc[-1], "Blocked", BLUE)
    label_end(a, n[-1], w["eff_gflops"]["strassen"].iloc[-1], "Strassen", ORANGE)
    a.set_ylabel("Effective GFLOP/s  (2n³ / time)")
    a.set_ylim(0, None)
    a.set_title("E3: throughput vs size (tuned tile, tuned cutoff)", loc="left")
    a.legend(loc="lower right")
    b.axhspan(1 / 1.8, 1 / 1.5, xmin=0.88, xmax=1.0, color=PRED, zorder=0)
    b.axhline(1.0, color=AXIS, lw=1)
    b.plot(n, ratio, "-o", color=ORANGE)
    b.fill_between(n, ratio, 1.0, where=ratio < 1, color=ORANGE, alpha=0.08, lw=0)
    b.text(4300, 1 / 1.5 + 0.012, "shaded: predicted\n1.5–1.8× (P5)", ha="right", va="bottom", fontsize=7.5,
           color=INK2)
    b.annotate(f"{1 / ratio[-1]:.2f}× faster", (n[-1], ratio[-1]), xytext=(-6, -10), textcoords="offset points",
               ha="right", fontsize=8, color=INK)
    b.text(800, 1.015, "blocked faster ↑", fontsize=8, color=MUTED, va="bottom")
    b.text(800, 0.985, "Strassen faster ↓", fontsize=8, color=MUTED, va="top")
    b.set_ylabel("Time ratio  Strassen / Blocked")
    b.set_xlabel("Matrix size n")
    log2_x(b, [64, 128, 256, 512, 1024, 2048, 4096])
    b.set_ylim(0.45, 1.08)
    save(fig, out, "fig1_scaling.png")


def fig_cutoff(R, out):
    e2 = md.summarise(md.load("E2_cutoff.csv", R), ["n", "key"])
    s = e2[e2.key.str.startswith("strassen")].copy()
    s["c"] = s.key.str.split("c=").str[1].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(64 / 1.2, 128 * 1.2, color=PRED, zorder=0)
    ax.text(90, 1.72, "predicted\nc* (P2)", ha="center", va="top", fontsize=8, color=INK2)
    sizes = sorted(s.n.unique())
    for col, nn, mk in zip(BLUE_RAMP[1:], sizes, "oDs^"):
        g = s[s.n == nn].sort_values("c")
        rel = g.median_s / g.median_s.min()
        ax.plot(g.c, rel, "-" + mk, color=col, label=f"n = {nn}")
    ax.axhline(1.0, color=AXIS, lw=1)
    log2_x(ax, [16, 32, 64, 128, 256, 512])
    ax.set_xlabel("Strassen cutoff c (recurse while size > c)")
    ax.set_ylabel("Time / best cutoff at that n")
    ax.set_title("E2: the best cutoff is 128 at every size", loc="left")
    ax.legend(title="problem size", title_fontsize=8, loc="upper right")
    save(fig, out, "fig2_cutoff.png")


def fig_tile(R, out):
    e1 = md.summarise(md.load("E1_tile.csv", R), ["n", "variant"])
    t = e1[e1.variant.str.startswith("T=")].copy()
    t["T"] = t.variant.str[2:].astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(32 / 1.15, 64 * 1.15, color=PRED, zorder=0)
    ax.text(45, 3, "predicted best T\n(B tile fits L1, P6)", ha="center", fontsize=8, color=INK2)
    ax.axvline(64, color=MUTED, lw=0.8, ls=":")
    ax.axvline(362, color=MUTED, lw=0.8, ls=":")
    ax.text(64, 31.5, " 8T² = 32 KB (L1)", fontsize=7.5, color=MUTED, va="top")
    ax.text(362, 31.5, " 8T² = 1 MB\n (L2)", fontsize=7.5, color=MUTED, va="top")
    sizes = sorted(t.n.unique())
    for col, nn, mk in zip(BLUE_RAMP[1::1], sizes, "oDs"):
        g = t[(t.n == nn) & (t["T"] > 1)].sort_values("T")
        ax.plot(g["T"], g.eff_gflops, "-" + mk, color=col, label=f"n = {nn}")
        un = e1[(e1.n == nn) & (e1.variant == "untiled")].eff_gflops
        if len(un):
            ax.plot([700], un, mk, color=col, mfc="white", mew=1.4)
    ax.plot([], [], "o", color=MUTED, mfc="white", label="untiled (T = n, one tile)")
    log2_x(ax, [8, 16, 32, 64, 128, 256, 700])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "untiled"])
    ax.set_xlim(6.5, 900)
    ax.set_ylim(0, 32)
    ax.set_xlabel("Tile edge T")
    ax.set_ylabel("Effective GFLOP/s")
    ax.set_title("E1: tile size for the blocked kernel", loc="left")
    ax.legend(loc="lower right")
    save(fig, out, "fig3_tile.png")


def fig_padding(R, out):
    e4 = md.summarise(md.load("E4_padding.csv", R), ["tag", "n", "key"])
    p = e4.pivot_table(index=["tag", "n"], columns="key", values="median_s")
    bk = [c for c in p.columns if c.startswith("blocked")][0]
    nk = [c for c in p.columns if "nextpow2" in c][0]
    mk = [c for c in p.columns if c.startswith("strassen") and "nextpow2" not in c][0]
    rel = p.div(p[bk], axis=0)
    cutoff = int(mk.split("c=")[1])
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 6.0), gridspec_kw={"height_ratios": [1, 1.1]})
    edges = rel[rel.index.get_level_values(0) != "dense"].sort_index(level=1)
    x = np.arange(len(edges))
    a.bar(x - 0.2, edges[mk], 0.38, color=ORANGE, label="Strassen, minimal padding")
    a.bar(x + 0.2, edges[nk], 0.38, color=AQUA, label="Strassen, next-power-of-two padding", hatch="////",
          edgecolor="white", lw=0)
    a.axhline(1, color=INK2, lw=1)
    a.text(-0.55, 1.05, "blocked = 1", ha="left", fontsize=8, color=INK2)
    a.set_ylim(0, 4.6)
    a.grid(axis="x", visible=False)
    for xi, v in zip(x, edges[nk]):
        if v > 1.5:
            a.text(xi + 0.2, v + 0.05, f"{v:.1f}×", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(x, [str(n) for n in edges.index.get_level_values(1)])
    a.set_ylabel("Time / blocked")
    a.set_title("E4a: sizes either side of a power of two", loc="left")
    a.legend(loc="upper center", ncol=2, fontsize=7.5)
    d = rel.loc["dense"]
    leaf = [md.predict(int(nn), cutoff)["base_size"] for nn in d.index]
    b.axhline(1, color=INK2, lw=1)
    b.plot(d.index, d[mk], "-", color=ORANGE, lw=1.2)
    aligned = np.array([lf % 4 == 0 for lf in leaf])
    b.plot(d.index[aligned], d[mk][aligned], "o", color=ORANGE, ms=6, label="leaf size a multiple of 4")
    b.plot(d.index[~aligned], d[mk][~aligned], "o", color=ORANGE, mfc="white", mew=1.4, ms=6,
           label="leaf size not a multiple of 4")
    prev = None
    for nn, v, lf in zip(d.index, d[mk], leaf):
        if lf != prev:
            b.annotate(f"leaf {lf}", (nn, v), xytext=(0, 9 if v < 1 else -13), textcoords="offset points",
                       ha="left", fontsize=7, color=INK2)
        prev = lf
    b.set_xlabel("Matrix size n  (labels: leaf size where the recursion stops)")
    b.set_ylabel("Time / blocked")
    b.set_title("E4b: minimal padding, dense sweep 1000–1100", loc="left")
    b.set_ylim(0.6, 1.2)
    b.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig4_padding.png")


def fig_stride(R, out):
    e5 = md.summarise(md.load("E5_stride.csv", R), ["n", "tag", "key"])
    e5["base_n"] = e5["n"].where(e5["tag"] != "n+8", e5["n"] - 8)
    tags = [("packed", "n, packed (stride n)", BLUE_RAMP[1], ""), ("ld+8", "n, stride n + 8", BLUE_RAMP[3], ""),
            ("n+8", "size n + 8, packed", AQUA, "////")]
    keys = [k for k in sorted(e5.key.unique()) if "temp_pad" not in k]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    for ax, k in zip(axes, keys):
        g = e5[e5.key == k]
        ns = sorted(g.base_n.unique())
        x = np.arange(len(ns))
        for i, (tag, lab, col, hatch) in enumerate(tags):
            v = [g[(g.base_n == nn) & (g.tag == tag)].eff_gflops.iloc[0] for nn in ns]
            ax.bar(x + (i - 1) * 0.27, v, 0.25, color=col, label=lab, hatch=hatch, edgecolor="white", lw=0)
        ax.set_xticks(x, [str(nn) for nn in ns])
        ax.set_title(("Blocked" if k.startswith("blocked") else "Strassen"), loc="left")
        ax.set_xlabel("n (power of two)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Effective GFLOP/s")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("E5: row-stride padding barely matters; growing n by 8 hurts Strassen", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig5_stride.png")


def fig_pressure(R, out):
    e6 = md.summarise(md.load("E6_cache_pressure.csv", R), ["n", "antagonist_mb", "algorithm"])
    mbs = sorted(e6.antagonist_mb.unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    for ax, alg, title in zip(axes, ["blocked", "strassen"], ["Blocked", "Strassen"]):
        for col, nn, mk in zip(BLUE_RAMP[1::1], sorted(e6.n.unique()), "oDs"):
            g = e6[(e6.algorithm == alg) & (e6.n == nn)].set_index("antagonist_mb").loc[mbs]
            ax.plot(range(len(mbs)), g.median_s / g.median_s.iloc[0], "-" + mk, color=col, label=f"n = {nn}")
        ax.axhline(1, color=AXIS, lw=1)
        ax.set_xticks(range(len(mbs)), ["off", "1\n(control)"] + [str(m) for m in mbs[2:]])
        ax.set_xlabel("L3 occupied by antagonist (MB of 96)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Time / no antagonist")
    axes[1].legend(loc="upper left")
    fig.suptitle("E6: occupying L3 slows Strassen about twice as much as blocked", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig6_cache_pressure.png")


def fig_allocation(R, out):
    e7 = md.summarise(md.load("E7_allocation.csv", R), ["n", "key"])
    p = e7.pivot(index="n", columns="key", values="median_s")
    ak = [c for c in p.columns if "alloc_each_call" in c][0]
    pk = [c for c in p.columns if "plan_reused" in c][0]
    ov = (p[ak] / p[pk] - 1) * 100
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(ov))
    ax.bar(x, ov, 0.55, color=ORANGE)
    ax.grid(axis="x", visible=False)
    for xi, v in zip(x, ov):
        ax.text(xi, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
    ax.axhline(2, color=MUTED, lw=0.8, ls=":")
    ax.text(len(x) - 0.6, 2.1, "P11: < 2% for n ≥ 1024", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, [str(n) for n in ov.index])
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Extra time from allocating\nper call vs reused Plan (%)")
    ax.set_title("E7: cost of allocating Strassen's workspace", loc="left")
    save(fig, out, "fig7_allocation.png")


def fig_accuracy(R, out):
    e8 = pd.read_csv(R / "E8_accuracy.csv")
    g = e8.groupby(["family", "n", "variant"]).agg(fro=("rel_err_fro", "max"),
                                                    comp=("max_componentwise_ratio", "max"),
                                                    gamma=("gamma_n", "first")).reset_index()
    cuts = sorted({int(v[2:]) for v in g.variant if v.startswith("c=")}, reverse=True)  # shallow -> deep
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    for ax, fam, col, title in [(axes[0], "uniform", "fro", "uniform inputs: ‖C − Ĉ‖F / ‖C‖F"),
                                 (axes[1], "widerange", "comp", "wide-range inputs: max |C − Ĉ| / (|A||B|)")]:
        f = g[g.family == fam]
        b = f[f.variant.str.startswith("tuned")].sort_values("n")
        ax.plot(b.n, b[col], "-o", color=BLUE, label="Blocked")
        for c, oc, mk in zip(cuts, ORANGE_RAMP, "sD^"):
            s = f[f.variant == f"c={c}"].sort_values("n")
            ax.plot(s.n, s[col], "-" + mk, color=oc, label=f"Strassen c={c}")
        if col == "comp":
            ax.plot(b.n, b.gamma, ":", color=MUTED, label="classical bound γₙ")
        ax.set_yscale("log")
        log2_x(ax, [128, 256, 512, 1024])
        ax.set_xlabel("Matrix size n")
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("Error vs double-double reference")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("E8: deeper recursion costs accuracy; wide-range inputs expose the normwise bound",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig8_accuracy.png")


def fig_drift(R, out):
    d = pd.read_csv(R / "summary" / "drift_monitor.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.axhspan(95, 105, color=PRED, zorder=0)
    ax.plot(d.index, d.pct, "o", color=BLUE, ms=3.5)
    ax.grid(axis="x", visible=False)
    for exp, grp in d.groupby("exp"):
        ax.axvline(grp.index.min() - 0.5, color=AXIS, lw=0.6)
        if len(grp) >= 10:
            ax.text(grp.index.min() + 1, 101.2, exp, fontsize=7.5, color=INK2)
    ax.text(1, 101.2, "E0–E3", fontsize=7.5, color=INK2)
    ax.set_ylim(88, 102.5)
    ax.text(len(d) - 1, 88.6, "band = ±5% of the start-of-run baseline", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xlabel("Drift-monitor reading (one per configuration, in run order)")
    ax.set_ylabel("Reference speed\n(% of baseline)")
    ax.set_title("Drift monitor during the official run", loc="left")
    save(fig, out, "fig9_drift_monitor.png")


def fig_cost_model(R, out):
    cm = pd.read_csv(R / "summary" / "cost_model.csv")
    e3 = md.summarise(md.load("E3_scaling.csv", R), ["n", "algorithm"]).pivot(index="n", columns="algorithm",
                                                                           values="median_s")
    cm["t_blocked"] = e3.loc[cm.n, "blocked"].to_numpy()
    leaves = cm.t_leaves_model / cm.t_blocked
    rest = (cm.t_measured - cm.t_leaves_model) / cm.t_blocked
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(cm))
    ax.bar(x, leaves, 0.6, color=ORANGE, label="leaf multiplications (modelled at blocked's rate for that size)")
    ax.bar(x, rest, 0.6, bottom=leaves, color=ORANGE_RAMP[0], label="additions, copies, overhead (residual)",
           hatch="////", edgecolor="white", lw=0)
    ax.axhline(1, color=INK2, lw=1)
    ax.text(-0.55, 1.02, "blocked = 1", ha="left", fontsize=8, color=INK2)
    ax.grid(axis="x", visible=False)
    for xi, l, r_ in zip(x, leaves, rest):
        ax.text(xi, l + r_ + 0.015, f"{r_ / (l + r_) * 100:.0f}%", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{n}\nd={d}" for n, d in zip(cm.n, cm.depth)])
    ax.set_ylabel("Strassen time / blocked time")
    ax.set_xlabel("n and recursion depth d   (% above bars: share of Strassen time outside the leaves)")
    ax.set_title("Where Strassen's time goes", loc="left")
    ax.set_ylim(0, 1.1)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, fontsize=7.5, bbox_to_anchor=(0.5, -0.27))
    save(fig, out, "fig10_cost_model.png")


def fig_alignment(R, out):
    if not (R / "E9_alignment.csv").exists():
        print("skip fig11 (no E9_alignment.csv)")
        return
    info = md.run_info(R, "E9")
    cutoff = int(info["tuned_cutoff"].split()[0])
    e9 = md.summarise(md.load("E9_alignment.csv", R), ["tag", "n", "key"])
    p = e9.pivot_table(index=["tag", "n"], columns="key", values="median_s")
    bk = [c for c in p.columns if c.startswith("blocked")][0]
    rel = p.div(p[bk], axis=0)
    tags = sorted(rel.index.get_level_values(0).unique(), key=lambda t: int(t.split("_")[1]))
    fig, axes = plt.subplots(len(tags), 1, figsize=(6.4, 2.9 * len(tags)))
    axes = np.atleast_1d(axes)
    series = [("minimal", "minimal padding", ORANGE, "o", "minimal", 4),
              ("aligned4", "aligned: leaf a multiple of 4", AQUA, "s", "aligned", 4),
              ("aligned8", "aligned: leaf a multiple of 8", YELLOW, "^", "aligned", 8)]
    for ax, tag in zip(axes, tags):
        d = rel.loc[tag]
        ax.axhline(1, color=INK2, lw=1)
        for suffix, lab, col, mk, pad, al in series:
            key = [c for c in d.columns if c.startswith("strassen") and
                   (c.endswith(suffix) if suffix != "minimal" else "+" not in c.split(":")[1])][0]
            leaf = np.array([md.predict(int(n), cutoff, pad, al)["base_size"] for n in d.index])
            ax.plot(d.index, d[key], "-", color=col, lw=1.2)
            ok = leaf % 4 == 0
            ax.plot(d.index[ok], d[key][ok], mk, color=col, ms=5.5, label=lab)
            if (~ok).any():
                ax.plot(d.index[~ok], d[key][~ok], mk, color=col, mfc="white", mew=1.3, ms=5.5)
        ax.set_ylabel("Time / blocked")
        ax.set_title(f"n = {d.index.min()}–{d.index.max()}", loc="left")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Matrix size n   (hollow markers: leaf size not a multiple of 4)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("E9: rounding the leaf size up to a multiple of 4 or 8", x=0.02, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig11_alignment.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=md.RESULTS)
    ap.add_argument("--out", type=Path, default=md.FIGURES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    for f in (fig_scaling, fig_cutoff, fig_tile, fig_padding, fig_stride, fig_pressure, fig_allocation,
              fig_accuracy, fig_drift, fig_cost_model, fig_alignment):
        f(args.results, args.out)


if __name__ == "__main__":
    main()
