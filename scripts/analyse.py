"""Turn results/*.csv into summary tables and a predictions-vs-results check.

Usage (from the repository root):
    python scripts/analyse.py [--results DIR]

Writes results/summary/*.csv and results/summary/summary.md. Every number in
the report's results section comes from these outputs.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import mmdata as md

pd.set_option("display.width", 200)


def bootstrap_ci(x: np.ndarray, k: int, rng: np.random.Generator, draws: int = 4000) -> float:
    """95% CI half-width of the median of k draws, as % of the full-sample median."""
    meds = np.median(rng.choice(x, size=(draws, k), replace=True), axis=1)
    return float((np.percentile(meds, 97.5) - np.percentile(meds, 2.5)) / 2 / np.median(x) * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=md.RESULTS)
    args = ap.parse_args()
    R = args.results
    out = R / "summary"
    out.mkdir(parents=True, exist_ok=True)
    info = md.run_info(R, "E0")
    tile = int(info["tuned_tile"].split()[0])
    cutoff = int(info["tuned_cutoff"].split()[0])
    lines: list[str] = []
    P: dict[str, tuple[str, str]] = {}  # prediction -> (measured, verdict)

    def section(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
        lines.append(df.to_markdown(floatfmt=floatfmt.strip("{}:")) if hasattr(df, "to_markdown") else df.to_string())
        lines.append("")

    lines.append(f"# Results summary (commit {info.get('git_commit')}, tuned tile {tile}, tuned cutoff {cutoff})")
    lines.append(f"\nMachine: {info.get('cpu')}; {info.get('os')}; {info.get('compiler')}; power plan {info.get('power_plan')}.")

    # ---------------- drift monitor
    refs = []
    for f in sorted(R.glob("E[0-9]_*.csv")):
        d = pd.read_csv(f)
        if "ref_gflops" not in d:
            continue  # E8 is not timed
        call = (d["ref_gflops"] != d["ref_gflops"].shift()).cumsum()
        exp = f.name[:2]
        base_exp = float(md.run_info(R, exp)["drift_monitor_baseline"].split()[0])  # that run's own baseline
        refs.append(d.groupby(call)["ref_gflops"].first().to_frame().assign(exp=exp, baseline=base_exp))
    refs = pd.concat(refs, ignore_index=True)
    base = float(info["drift_monitor_baseline"].split()[0])
    refs["pct"] = refs["ref_gflops"] / refs["baseline"] * 100
    refs.to_csv(out / "drift_monitor.csv", index_label="reading")
    section("Drift monitor")
    lines.append(f"{len(refs)} readings; main-run baseline {base:.2f} GFLOP/s (follow-up runs use their own); "
                 f"{(refs.pct.between(95, 105)).mean() * 100:.1f}% within ±5%; range {refs.pct.min():.1f}–{refs.pct.max():.1f}%.\n")
    table(refs.groupby("exp")["pct"].agg(["min", "median", "max", "count"]), "{:.1f}")

    # ---------------- E0 pilot
    e0 = md.load("E0_pilot.csv", R)
    rng = np.random.default_rng(1)
    rows = []
    for (key, n), g in e0.groupby(["key", "n"]):
        x = g["seconds"].to_numpy()
        rows.append({"variant": key, "n": n, "cv_pct": x.std(ddof=1) / x.mean() * 100,
                     **{f"ci_k{k}": bootstrap_ci(x, k, rng) for k in (3, 5, 7, 10, 15)}})
    e0s = pd.DataFrame(rows)
    e0s.to_csv(out / "E0_variance.csv", index=False)
    section("E0 pilot: variance and repetition count")
    table(e0s, "{:.2f}")

    # ---------------- E1 tile
    e1 = md.load("E1_tile.csv", R)
    e1s = md.summarise(e1, ["n", "variant"])
    piv = e1s.pivot(index="variant", columns="n", values="eff_gflops")
    order = sorted([v for v in piv.index if v.startswith("T=")], key=lambda v: int(v[2:])) + ["untiled"]
    piv = piv.loc[order]
    piv.to_csv(out / "E1_tile_gflops.csv")
    section("E1 tile sweep: effective GFLOP/s (blocked)")
    table(piv, "{:.2f}")
    best_by_n = piv.drop(index=["T=1", "untiled"], errors="ignore").idxmax()
    lines.append("Best tile per n: " + ", ".join(f"n={n}: {v}" for n, v in best_by_n.items()) + "\n")
    p1_rate = e1s[(e1s.variant == f"T={tile}")]["eff_gflops"]
    P["P6"] = (f"tuned T={tile}; best per n: {', '.join(best_by_n.values)}; T=64 is "
               f"{(1 - piv.loc['T=64'] / piv.drop(index=['T=1','untiled'], errors='ignore').max()).max() * 100:.1f}% slower at worst",
               "refuted" if tile > 64 else "supported")

    # ---------------- E2 cutoff
    e2 = md.load("E2_cutoff.csv", R)
    e2s = md.summarise(e2, ["n", "key"])
    piv2 = e2s.pivot(index="key", columns="n", values="median_s")
    rel_best = piv2.drop(index=[k for k in piv2.index if k.startswith("blocked")]) / \
        piv2.drop(index=[k for k in piv2.index if k.startswith("blocked")]).min()
    rel_best = rel_best.loc[sorted(rel_best.index, key=lambda k: int(k.split("c=")[1]))]
    rel_blocked = piv2 / piv2.loc[[k for k in piv2.index if k.startswith("blocked")][0]]
    rel_best.to_csv(out / "E2_cutoff_rel_best.csv")
    rel_blocked.to_csv(out / "E2_cutoff_rel_blocked.csv")
    section("E2 cutoff sweep: time relative to the best cutoff at each n")
    table(rel_best, "{:.3f}")
    section("E2 cutoff sweep: time relative to blocked")
    table(rel_blocked.loc[sorted(rel_blocked.index)], "{:.3f}")
    best_c = rel_best.idxmin()
    lines.append("Best cutoff per n: " + ", ".join(f"n={n}: {v.split(':')[1]}" for n, v in best_c.items()) + "\n")
    cvals = [int(k.split("c=")[1]) for k in rel_best.index]
    i_best = cvals.index(cutoff)
    neigh = [rel_best.iloc[j] for j in (i_best - 1, i_best + 1) if 0 <= j < len(cvals)]
    worst_neigh = max(float(s.max()) for s in neigh) if neigh else float("nan")
    P["P2"] = (f"c*={cutoff}; neighbours within one power of two are up to {(worst_neigh - 1) * 100:.1f}% slower",
               ("supported" if cutoff in (64, 128) else "refuted") + ("" if worst_neigh <= 1.10 else "; flatness (±10%) refuted"))
    P["P3"] = ("best per n: " + ", ".join(v.split(':')[1] for v in best_c.values),
               "supported" if len({v for v in best_c.values}) == 1 else "partly supported")

    # ---------------- E3 scaling
    e3 = md.load("E3_scaling.csv", R)
    e3s = md.summarise(e3, ["n", "algorithm"])
    w = e3s.pivot(index="n", columns="algorithm", values=["median_s", "eff_gflops", "iqr_pct"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w["ratio_S_over_B"] = w["median_s_strassen"] / w["median_s_blocked"]
    w.to_csv(out / "E3_scaling.csv")
    section("E3 scaling")
    table(w[["eff_gflops_blocked", "eff_gflops_strassen", "ratio_S_over_B", "iqr_pct_blocked", "iqr_pct_strassen"]], "{:.3f}")
    r = w["ratio_S_over_B"]
    first_win = int(r[r < 1.0].index.min())
    sustained = int(next(n for n in r.index if (r.loc[n:] < 0.97).all()))
    lines.append(f"First n with Strassen faster: {first_win}. From n = {sustained} on Strassen is always >= 3% faster.\n")
    P1_lo, P1_hi = e3s[(e3s.algorithm == "blocked") & (e3s.n >= 256)]["eff_gflops"].agg(["min", "max"])
    P["P1"] = (f"blocked {P1_lo:.1f}–{P1_hi:.1f} GFLOP/s for n >= 256", "refuted (faster than predicted)" if P1_lo > 20 else "supported")
    P["P4"] = (f"first win n={first_win} (ratio {r.loc[first_win]:.3f}); sustained >=3% from n={sustained}; "
               f"ratio at 256: {r.loc[256]:.3f}", "supported" if 128 <= sustained <= 384 else "refuted")
    sp4096 = 1 / r.loc[4096]
    P["P5"] = (f"speedup at 4096 = {sp4096:.2f}x", "supported" if 1.5 <= sp4096 <= 1.8 else
               ("exceeded" if sp4096 > 1.8 else "refuted"))
    gb = w["eff_gflops_blocked"]
    peak = gb[gb.index >= 256].max()
    P["P7a"] = (f"blocked at 4096 is {gb.loc[4096] / peak * 100:.1f}% of its peak ({peak:.1f}); "
                f"at 2048 {gb.loc[2048] / peak * 100:.1f}%", "supported" if gb.loc[4096] / peak >= 0.9 else "refuted")

    # cost model: t_S = calls * t_leaf(base) + tau_add * (add + 2/3 copy)
    leaf_rate = w["eff_gflops_blocked"]  # GFLOP/s of a packed blocked multiply at that size
    rows = []
    for n in w.index:
        pr = md.predict(int(n), cutoff)
        if pr["depth"] == 0 or pr["base_size"] not in leaf_rate.index:
            continue
        t_leaf = pr["base_flops"] / (leaf_rate.loc[pr["base_size"]] * 1e9)
        a = pr["add_flops"] + 2 / 3 * pr["copy_elems"]
        rows.append({"n": n, "padded_n": pr["padded_n"], "depth": pr["depth"], "base": pr["base_size"],
                     "t_measured": w.loc[n, "median_s_strassen"], "t_leaves_model": t_leaf,
                     "add_ops": a, "resid": w.loc[n, "median_s_strassen"] - t_leaf})
    cm = pd.DataFrame(rows)
    cm["tau_add_ns"] = cm["resid"] / cm["add_ops"] * 1e9
    cm["leaf_share"] = cm["t_leaves_model"] / cm["t_measured"]
    tau_mul_ns = 1 / gb.loc[4096]  # ns per flop of the blocked kernel at 4096
    cm["r_implied"] = cm["tau_add_ns"] / tau_mul_ns
    cm.to_csv(out / "cost_model.csv", index=False)
    section("Cost model check: Strassen time = leaves (at measured blocked rate) + tau_add x additions")
    table(cm[["n", "padded_n", "depth", "base", "t_measured", "t_leaves_model", "leaf_share", "tau_add_ns", "r_implied"]], "{:.4f}")

    # ---------------- E4 padding
    e4 = md.load("E4_padding.csv", R)
    e4s = md.summarise(e4, ["tag", "n", "key"])
    p4 = e4s.pivot_table(index=["tag", "n"], columns="key", values="median_s")
    bkey = [c for c in p4.columns if c.startswith("blocked")][0]
    rel4 = p4.div(p4[bkey], axis=0)
    rel4.to_csv(out / "E4_padding_rel_blocked.csv")
    section("E4 padding: time relative to blocked")
    table(rel4[rel4.index.get_level_values(0) != "dense"], "{:.3f}")
    nk = [c for c in p4.columns if "nextpow2" in c][0]
    mk = [c for c in p4.columns if c.startswith("strassen") and "nextpow2" not in c][0]
    jumps = []
    for tag in sorted({t for t in p4.index.get_level_values(0) if t.startswith("edge")}):
        k = int(tag.split("k")[1]); p = 1 << k
        jumps.append((p, p4.loc[(tag, p + 1), nk] / p4.loc[(tag, p), nk], rel4.loc[(tag, p + 1), nk]))
    dense = rel4.loc["dense"]
    lines.append("nextpow2 time jump t(2^k+1)/t(2^k): " + ", ".join(f"{p}: {j:.2f}x (vs blocked {b:.2f}x)" for p, j, b in jumps) + "\n")
    lines.append(f"Dense sweep 1000–1100, minimal padding: Strassen/blocked {dense[mk].min():.3f}–{dense[mk].max():.3f}; "
                 f"slower than blocked at {(dense[mk] > 1).sum()} of {len(dense)} sizes.\n")
    P["P9"] = ("jumps " + ", ".join(f"{j:.2f}x" for _, j, _ in jumps) + "; vs blocked at 2^k+1: " +
               ", ".join(f"{b:.2f}x" for _, _, b in jumps), "supported" if all(5 <= j <= 9 for _, j, _ in jumps) else "partly supported")
    pmin = []
    for n in dense.index:
        pr = md.predict(int(n), cutoff)
        pmin.append(pr["padded_n"] / n)
    P["P10"] = (f"minimal padding Strassen/blocked {dense[mk].min():.2f}–{dense[mk].max():.2f} over 1000–1100 "
                f"(slower than blocked at {(dense[mk] > 1).sum()}/{len(dense)} sizes)",
                "supported" if (dense[mk] > 1).sum() == 0 else "refuted at some sizes")

    # ---------------- E5 stride
    e5 = md.load("E5_stride.csv", R)
    e5s = md.summarise(e5, ["n", "tag", "key"])
    e5s["base_n"] = e5s["n"].where(e5s["tag"] != "n+8", e5s["n"] - 8)
    p5 = e5s.pivot_table(index=["base_n", "key"], columns="tag", values="eff_gflops")
    p5["ld8_vs_packed"] = p5["ld+8"] / p5["packed"]
    p5.to_csv(out / "E5_stride.csv")
    section("E5 stride: effective GFLOP/s")
    table(p5, "{:.3f}")
    bl = p5.xs([k for k in p5.index.get_level_values(1) if k.startswith("blocked")][0], level=1)
    P["P8"] = ("blocked GFLOP/s gain from padding the stride: " +
               ", ".join(f"n={n}: {(v - 1) * 100:+.1f}%" for n, v in bl["ld8_vs_packed"].items()),
               "refuted" if (bl["ld8_vs_packed"] < 1.10).all() else "supported")

    # ---------------- E6 cache pressure
    e6 = md.load("E6_cache_pressure.csv", R)
    e6s = md.summarise(e6, ["n", "antagonist_mb", "algorithm"])
    p6 = e6s.pivot_table(index=["n", "antagonist_mb"], columns="algorithm", values="median_s")
    for a in ("blocked", "strassen"):
        p6[f"{a}_rel_0MB"] = p6[a] / p6[a].xs(0, level=1).reindex(p6.index.get_level_values(0)).to_numpy()
    p6["speedup"] = p6["blocked"] / p6["strassen"]
    p6["speedup_rel_0MB"] = p6["speedup"] / p6["speedup"].xs(0, level=1).reindex(p6.index.get_level_values(0)).to_numpy()
    p6.to_csv(out / "E6_cache_pressure.csv")
    section("E6 cache pressure (relative to no antagonist)")
    table(p6, "{:.3f}")
    worst = (1 - p6["speedup_rel_0MB"]).max() * 100
    ctrl = (p6.xs(1, level=1)[["blocked_rel_0MB", "strassen_rel_0MB"]] - 1).abs().max().max() * 100
    P["P7b"] = (f"Strassen speedup falls by at most {worst:.1f}% at 80 MB; 1 MB control changes times by <= {ctrl:.1f}%",
                "supported" if worst < 10 else "refuted")

    # ---------------- E7 allocation
    e7 = md.load("E7_allocation.csv", R)
    e7s = md.summarise(e7, ["n", "key"])
    p7 = e7s.pivot(index="n", columns="key", values="median_s")
    ak = [c for c in p7.columns if "alloc_each_call" in c][0]
    pk = [c for c in p7.columns if "plan_reused" in c][0]
    p7["alloc_overhead_pct"] = (p7[ak] / p7[pk] - 1) * 100
    p7.to_csv(out / "E7_allocation.csv")
    section("E7 allocation overhead")
    table(p7, "{:.5f}")
    small = p7.loc[p7.index <= 256, "alloc_overhead_pct"]
    large = p7.loc[p7.index >= 1024, "alloc_overhead_pct"]
    P["P11"] = (f"overhead n<=256: {', '.join(f'{v:.1f}%' for v in small)}; n>=1024: {', '.join(f'{v:.1f}%' for v in large)}",
                "supported" if (large.abs() < 2).all() and (small > 5).any() else "partly supported")

    # ---------------- E8 accuracy
    e8 = pd.read_csv(R / "E8_accuracy.csv")
    e8s = e8.groupby(["family", "n", "variant"]).agg(
        depth=("depth", "first"), rel_err_fro=("rel_err_fro", "max"), normwise=("normwise_err", "max"),
        componentwise=("max_componentwise_ratio", "max"), entry=("max_entry_rel_err", "max"),
        gamma_n=("gamma_n", "first"), higham=("higham_factor", "first")).reset_index()
    e8s.to_csv(out / "E8_accuracy.csv", index=False)
    section("E8 accuracy (max over seeds)")
    table(e8s, "{:.2e}")
    def ratio(fam: str, col: str) -> pd.Series:
        f = e8s[e8s.family == fam]
        b = f[f.variant.str.startswith("tuned")].set_index("n")[col]
        s = f[f.variant.str.startswith("c=")]
        return s.assign(r=s[col].to_numpy() / b.reindex(s.n).to_numpy())
    u = ratio("uniform", "rel_err_fro")
    wr = ratio("widerange", "componentwise")
    wf = ratio("widerange", "rel_err_fro")
    P["P12"] = (f"smallint max error {e8s[e8s.family == 'smallint'].rel_err_fro.max():.1e}", "supported")
    P["P13"] = (f"uniform Strassen/blocked Frobenius error ratio {u.r.min():.1f}–{u.r.max():.1f}x; max {u.rel_err_fro.max():.1e}",
                "supported" if u.rel_err_fro.max() < 1e-12 else "refuted")
    d3 = wr[wr.depth >= 3]
    P["P14"] = (f"widerange componentwise ratio at depth>=3: {d3.r.min():.0f}–{d3.r.max():.0f}x; Frobenius ratio {wf.r.min():.1f}–{wf.r.max():.1f}x",
                "supported" if d3.r.min() >= 1e3 else "partly supported")

    # ---------------- E9 alignment follow-up (tests the explanation of P10's failure)
    if (R / "E9_alignment.csv").exists():
        e9 = md.load("E9_alignment.csv", R)
        cut9 = int(md.run_info(R, "E9")["tuned_cutoff"].split()[0])  # the cutoff E9 actually ran with
        e9s = md.summarise(e9, ["tag", "n", "key"])
        p9 = e9s.pivot_table(index=["tag", "n"], columns="key", values="median_s")
        b9 = [c for c in p9.columns if c.startswith("blocked")][0]
        rel9 = p9.div(p9[b9], axis=0).drop(columns=[b9])
        rel9.columns = [c.split("+")[-1] if "+" in c else "minimal" for c in rel9.columns]
        for col in rel9.columns:
            pad = "minimal" if col == "minimal" else "aligned"
            al = int(col.replace("aligned", "")) if col.startswith("aligned") else 4
            rel9[f"leaf_{col}"] = [md.predict(int(n), cut9, pad, al)["base_size"] for n in rel9.index.get_level_values(1)]
        rel9.to_csv(out / "E9_alignment.csv")
        section("E9 follow-up: leaf alignment (time relative to blocked)")
        table(rel9, "{:.3f}")
        variants = [c for c in rel9.columns if not c.startswith("leaf_")]
        rows = []
        for tag, grp in rel9.groupby(level=0):
            for v in variants:
                rows.append({"range": tag, "variant": v, "sizes": len(grp), "slower_than_blocked": int((grp[v] > 1).sum()),
                             "median_ratio": grp[v].median(), "worst_ratio": grp[v].max()})
        e9sum = pd.DataFrame(rows)
        e9sum.to_csv(out / "E9_summary.csv", index=False)
        table(e9sum.set_index(["range", "variant"]), "{:.3f}")
        mn = rel9["minimal"]
        aligned_leaf = rel9["leaf_minimal"] % 4 == 0
        lines.append(f"Minimal padding, leaf a multiple of 4: median ratio {mn[aligned_leaf].median():.3f} "
                     f"({aligned_leaf.sum()} sizes); otherwise {mn[~aligned_leaf].median():.3f} ({(~aligned_leaf).sum()} sizes).\n")
        a4 = e9sum[e9sum.variant == "aligned4"]
        mnm = e9sum[e9sum.variant == "minimal"]
        P["P10-E9"] = (f"minimal slower than blocked at {mnm.slower_than_blocked.sum()}/{mnm.sizes.sum()} sizes; "
                       f"aligned4 at {a4.slower_than_blocked.sum()}/{a4.sizes.sum()}; aligned8 at "
                       f"{e9sum[e9sum.variant == 'aligned8'].slower_than_blocked.sum()}/{a4.sizes.sum()}",
                       "alignment explanation supported" if a4.slower_than_blocked.sum() < 0.25 * mnm.slower_than_blocked.sum()
                       else "alignment explanation NOT supported")

    # ---------------- E9b: identical variants, both orders
    if (R / "E9b_order.csv").exists():
        e9b = md.load("E9b_order.csv", R)
        k1 = [k for k in e9b.key.unique() if k.endswith("#alloc1")][0]
        k2 = [k for k in e9b.key.unique() if k.endswith("#alloc2")][0]
        pr = md.paired_ratio(e9b, k2, k1, ["n", "tag"])
        cut9b = int(md.run_info(R, "E9b")["tuned_cutoff"].split()[0])
        pr["padded_copies"] = [md.predict(int(n), cut9b, "aligned", 4)["padded_n"] != n for n in pr.n]
        pr.to_csv(out / "E9b_order.csv", index=False)
        section("E9b follow-up: alloc2 / alloc1 for two identical aligned-4 variants (paired by repetition)")
        lines.append("Order A runs alloc2 right after alloc1 (aligned8's slot in E9); order B runs alloc1 right after alloc2.\n")
        table(pr, "{:.4f}")
        def sig(row):  # -1 alloc2 faster, +1 alloc1 faster, 0 indistinguishable
            return -1 if row.ci_hi < 1 else (1 if row.ci_lo > 1 else 0)
        pr["sign"] = pr.apply(sig, axis=1)
        test = pr[pr.padded_copies]
        a = test[test.tag == "order_A"].set_index("n")["sign"]
        b = test[test.tag == "order_B"].set_index("n")["sign"]
        carry = int(((a == -1) & (b == 1)).sum())       # the follower wins in both orders
        place = int(((a == b) & (a != 0)).sum())        # the same allocation wins in both orders
        none_ = int(((a == 0) & (b == 0)).sum())
        ctrl = pr[~pr.padded_copies]
        verdict = ("carry-over (position) effect" if carry > max(place, none_) else
                   "memory-placement effect" if place > max(carry, none_) else
                   "no reproducible difference" if none_ >= max(carry, place) else "mixed")
        lines.append(f"Sizes with padded copies: carry-over pattern at {carry}, placement pattern at {place}, "
                     f"no significant difference at {none_} (of {len(a)}). Control (no padded copies): ratios "
                     + ", ".join(f"{r.tag} {r.ratio:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" for r in ctrl.itertuples())
                     + f". Verdict: {verdict}.\n")
        ra = test[test.tag == "order_A"].set_index("n")["ratio"]
        rb = test[test.tag == "order_B"].set_index("n")["ratio"]
        direction = int(((ra < 1) & (rb > 1)).sum())  # point estimates: the follower is faster in both orders
        lines.append(f"Direction of the point estimates: the variant running second of the two is faster in both "
                     f"orders at {direction} of {len(ra)} sizes (both CIs exclude 1 at {carry}).\n")
        P["E9b"] = (f"follower faster in both orders at {direction}/{len(ra)} sizes (CI-significant in both at {carry}); "
                    f"placement pattern {place}/{len(a)}; "
                    f"median |ratio-1| with padded copies {(test.ratio - 1).abs().median() * 100:.1f}%", verdict)

    # ---------------- predictions table
    section("Predictions (report §2.8) vs results")
    order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7a", "P7b", "P8", "P9", "P10", "P10-E9", "P11", "P12", "P13", "P14", "E9b"]
    ptab = pd.DataFrame([{"prediction": k, "measured": P[k][0], "verdict": P[k][1]} for k in order if k in P])
    ptab.to_csv(out / "predictions_vs_results.csv", index=False)
    table(ptab.set_index("prediction"))

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
