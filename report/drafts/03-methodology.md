# 3. Experimental methodology

## 3.1 Platform

Every measurement was taken on one machine:
- **CPU:** AMD Ryzen 7 7800X3D (Zen 4; 32 KB L1d and 1 MB L2 per core, 96 MB shared L3).
- **Memory and OS:** 31 GB RAM; Windows build 10.0.26200; Ultimate Performance power plan; mains power.
- **Toolchain:** MSVC 19.51 (Visual Studio Community 18), CMake with Ninja, preset `windows-release`.
- **Flags:** `/O2 /Ob2 /arch:AVX2 /fp:precise /permissive- /EHsc`. These are identical for both algorithms. Strict floating point keeps the measured numerical error honest.

Both algorithms run single-threaded. The benchmark thread is pinned to one logical CPU at `HIGH_PRIORITY_CLASS` / `THREAD_PRIORITY_HIGHEST`. `results/run_info.txt` records all of this for every run, along with the git commit and a flag for uncommitted changes. The official run (E0–E8) was commit `73d07cf`, 15:56–16:18 on 4 November 2026. The follow-ups E9 and E9b were separate runs, on commits `58ea095` and `ffb7a43`.

## 3.2 Inputs

Inputs come from a portable xoshiro256\*\* generator (Blackman & Vigna, 2021), using only integer arithmetic and exact conversions; MSVC's and GCC's `<random>` distributions give different values from the same seed. A unit test checks hard-coded values on both compilers. Seeds depend only on n, so **every experiment and every algorithm at a given size multiplies the same matrices**. There are three input families:
- **`uniform`** U[−1, 1): used for all timings. The values have no denormals or special cases, so time reflects the algorithm alone.
- **`smallint`**, integers in [−8, 8]: every partial sum is exact in double precision, so both algorithms must match the reference bit for bit.
- **`widerange`**, ±m·2ᵉ with e ∈ [−20, 20]: built to expose Strassen's normwise (rather than componentwise) error bound (§2.7).

Errors are measured against a double-double reference: about 106-bit accumulation, using exact FMA products and TwoSum (Knuth, 1997).

## 3.3 Experiments

Each experiment targets specific predictions from §2 (P1–P14):

| ID | Varies | Tests |
|---|---|---|
| E0 | 30 repetitions at fixed parameters (n = 512, 1000) | variance; sets the repetition count |
| E1 | tile T ∈ {1, 8, …, 256, untiled}, n ∈ {256, 1000, 2000} | P1, P6; selects the tile |
| E2 | cutoff c ∈ {16, …, 512}, n ∈ {512, …, 4096} | P2, P3; selects the cutoff |
| E3 | 23 sizes, n = 64–4096 | P1, P4, P5 |
| E4 | `minimal` vs `nextpow2` at 2ᵏ − 1, 2ᵏ, 2ᵏ + 1 (k = 9–11), plus n = 1000–1100 in steps of 4 | P9, P10 |
| E5 | packed vs row stride n + 8 vs size n + 8, at n = 512, 1024, 2048 | P8 |
| E6 | antagonist occupying 0 / 1 / 32 / 64 / 80 MB of L3, n ∈ {1000, 2000, 3000} | P7 |
| E7 | reused `Plan` vs allocating on every call, n = 128–4096 | P11 |
| E8 | error, three input families, n = 128–1024, c ∈ {8, 32, 128} | P12–P14 |
| E9, E9b | follow-ups to E4 and E9 (§5), each with predictions committed before it ran | — |

**Tuning parameters are measured, not chosen.** E1 picks the tile and E2 the cutoff by a fixed rule. For each configuration, every candidate's median is divided by the best median in that configuration; the winner has the smallest geometric mean of those ratios across configurations. Later experiments read the winners from `results/tuned_params.txt`: **T = 128** and **c = 128**.

E6's antagonist is a thread on a *different physical core* that reads one double per cache line of a private buffer, sharing only the L3. Its **1 MB control** fits in its own L2, separating "a neighbouring core is busy" (which can lower the boost clock) from "part of L3 is occupied".

## 3.4 Timing

The harness keeps the PA1 harness's design rules. Its `mm_bench` executable runs a 163-check exact correctness gate before any timing, and refuses to run in a Debug build.

- **Clock:** `std::chrono::steady_clock`, which on Windows is QueryPerformanceCounter. Its resolution is 100 ns and each call costs 16.8 ns. Both are measured and reported; neither is subtracted.
- **Sample length:** each timed sample loops the call until it lasts at least 20 ms, so clock resolution is irrelevant. The number of inner calls is recorded.
- **Warm-up:** each algorithm gets one untimed warm-up call per configuration. It also sizes that inner loop.
- **Nothing extra in the timed region.** Inputs, outputs and `Plan`s are created first; no allocation, I/O or formatting is timed, except E7's labelled allocation rows.
- **Repetitions:** 7 per algorithm per configuration (30 in E0, 15 in E9b), and we report the median. E0 justifies 7: the coefficient of variation was 0.3–1.6%, and the 95% bootstrap interval on a 7-sample median was ±0.3–1.5%. Going to 15 samples narrows that to ±0.1–1.4%, which is not worth doubling the run time.
- **Order:** within each repetition, the algorithm order rotates. For E5 and E6, which compare *different configurations*, each repetition is one round over all configurations, rotated and reversed on alternate rounds, so the conditions being compared are measured seconds apart.
- **Outputs consumed:** every output is checksummed, and its relative difference from the first algorithm in its configuration is recorded. The largest difference in any timed configuration was 4.4×10⁻¹⁴.

## 3.5 Machine state, and what a shakedown run taught us

A first full run (archived in `results/shakedown_2026-11-04/`) was spoiled by a game started partway through: the same blocked multiplication fell from 26.0 to 21.8 GFLOP/s between E3 and E7. It also exposed a design flaw. E6 had measured every 0 MB configuration first and every 80 MB one last, so drift was indistinguishable from cache pressure. Two changes followed:
1. **Round-robin execution for E5 and E6.**
2. **A drift monitor.** Before every configuration, it times a fixed workload (blocked, n = 384, T = 64) and writes the reading to each row.

In the official and follow-up runs, 89% of the 288 readings were within ±5% of a warmed-up baseline (Figure 9). The lowest readings are the *first* configuration of each run (90–92%). Readings are also consistently low during sustained load, with medians of about 96% through E6's two-core load and through E9. That pattern reflects the boost clock responding to temperature and to the number of active cores. It is not interference. Effects under about 5% *between* configurations are therefore treated as inconclusive. Effects *within* a configuration are protected by the rotated order.

## 3.6 Reproducing

Build with `windows-release`, run `ctest`, then run `mm_bench.exe`, `mm_bench.exe --only E9` and `mm_bench.exe --only E9b`. Then run `python scripts/analyse.py` and `python scripts/make_plots.py`. Those two scripts regenerate every table in `results/summary/` and every figure from the CSVs alone. Every number in §4–5 comes from those 3,661 rows across 11 CSV files.
