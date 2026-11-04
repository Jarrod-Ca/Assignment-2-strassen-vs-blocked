"""Shared loading helpers and the Strassen cost model for the analysis scripts.

Everything here reads the CSVs written by mm_bench; nothing is hand-entered.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
SUMMARY = RESULTS / "summary"


def load(name: str, results: Path = RESULTS) -> pd.DataFrame:
    """Load one experiment CSV and add a unique 'key' = algorithm:variant."""
    df = pd.read_csv(results / name)
    if "variant" in df and "algorithm" in df:
        df["key"] = df["algorithm"] + ":" + df["variant"].astype(str)
    return df


def run_info(results: Path = RESULTS, experiment: str = "E0") -> dict[str, str]:
    """Parse the last *completed* run section of run_info.txt that ran `experiment`.

    Follow-up experiments (e.g. E9) are run as separate invocations, each with its
    own section, drift-monitor baseline and timestamps."""
    text = (results / "run_info.txt").read_text()
    sections = [s for s in text.split("==== run started") if "run finished" in s
                and any(line.startswith("experiments:") and experiment in line.split()
                        for line in s.splitlines())]
    if not sections:
        raise ValueError(f"no completed run of {experiment} in {results / 'run_info.txt'}")
    info: dict[str, str] = {}
    for line in sections[-1].splitlines():
        if ": " in line and not line.startswith(" "):
            k, v = line.split(": ", 1)
            info[k.strip()] = v.strip()
    return info


def iqr_pct(x: pd.Series) -> float:
    """Interquartile range as a percentage of the median."""
    return float((x.quantile(0.75) - x.quantile(0.25)) / x.median() * 100.0)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median seconds, effective GFLOP/s and IQR% per group."""
    g = df.groupby(by)
    out = pd.DataFrame({
        "median_s": g["seconds"].median(),
        "eff_gflops": g["eff_gflops"].median(),
        "iqr_pct": g["seconds"].agg(iqr_pct),
        "reps": g.size(),
    })
    return out.reset_index()


# ---------------------------------------------------------------------------
# Strassen cost model - a direct port of strassen::predict() (src/algo_strassen),
# whose agreement with measured operation counts is checked by the C++ tests.
# ---------------------------------------------------------------------------
def padded_size(n: int, cutoff: int, padding: str = "minimal", leaf_align: int = 4) -> int:
    if n <= cutoff:
        return n
    if padding == "nextpow2":
        return 1 << math.ceil(math.log2(n))
    s, d = n, 0
    while s > cutoff:
        s = (s + 1) // 2
        d += 1
    if padding == "aligned":
        def up(x: int) -> int:
            return -(-x // leaf_align) * leaf_align
        while up(s) > cutoff:
            s = (s + 1) // 2
            d += 1
        s = up(s)
    return s << d


def predict(n: int, cutoff: int, padding: str = "minimal", leaf_align: int = 4) -> dict[str, float]:
    m = padded_size(n, cutoff, padding, leaf_align)
    s, nodes, add, copy, depth = m, 1, 0, 0, 0
    while s > cutoff:
        h = s // 2
        add += nodes * 18 * h * h
        copy += nodes * h * h
        nodes *= 7
        s //= 2
        depth += 1
    return {"padded_n": m, "depth": depth, "base_size": s, "base_calls": nodes,
            "base_flops": nodes * 2 * s ** 3, "add_flops": add, "copy_elems": copy}
