# 3. Experimental methodology

## 3.1 Platform

All measurements come from one machine: an AMD Ryzen 7 7800X3D (Zen 4; 32 KB L1d and 1 MB L2 per core, 96 MB shared L3) with 31 GB RAM, running Windows build 10.0.26200 on the Ultimate Performance power plan. The code was built with MSVC 19.51 (Visual Studio Community 18) using CMake preset `windows-release`, with flags `/O2 /Ob2 /arch:AVX2 /fp:precise /permissive- /EHsc`. The flags are identical for both algorithms, and strict floating point keeps the measured error honest. Both algorithms run single-threaded, pinned to one logical CPU at high priority. `results/run_info.txt` records all of this for every run, together with the git commit. The official run (E0–E8) was commit `73d07cf`; the follow-ups E9 and E9b were commits `58ea095` and `ffb7a43`.

## 3.2 Inputs

Inputs come from a portable xoshiro256\*\* generator (Blackman & Vigna, 2021) that uses only integer arithmetic, so MSVC and GCC produce identical values; a unit test checks this. Seeds depend only on n, so **every experiment and every algorithm at a given size multiplies the same matrices**. There are three input families:
- **`uniform`** U[−1, 1): used for all timings, because it has no denormals or special cases.
- **`smallint`**, integers in [−8, 8]: every partial sum is exact, so both algorithms must match the reference bit for bit.
- **`widerange`**, ±m·2ᵉ with e ∈ [−20, 20]: built to expose Strassen's normwise error bound (§2.7).

Errors are measured against a double-double reference with about 106-bit accumulation (exact FMA products and TwoSum; Knuth, 1997).

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

**Tuning parameters are measured, not chosen.** E1 picks the tile and E2 the cutoff by a fixed rule. In each configuration, every candidate's median is divided by the best median; the winner has the smallest geometric mean of those ratios across configurations. Later experiments read the winners from `results/tuned_params.txt`: **T = 128** and **c = 128**.

E6's antagonist is a thread on a *different physical core* that reads one double per cache line of a private buffer, so it shares only the L3. Its **1 MB control** fits in its own L2. That separates "a neighbouring core is busy", which can lower the boost clock, from "part of L3 is occupied".

## 3.4 Timing

`mm_bench` follows the PA1 harness's design rules. It runs a 163-check exact correctness gate before any timing, and refuses to run in a Debug build.

- **Clock and samples.** `std::chrono::steady_clock` (QueryPerformanceCounter) has 100 ns resolution and costs 16.8 ns per call. Both are reported; neither is subtracted. Each sample loops the call for at least 20 ms, after one untimed warm-up call per algorithm per configuration.
- **Fairness.** Inputs, outputs and Strassen `Plan`s are created before timing starts, so no allocation, I/O or formatting is timed, apart from E7's labelled allocation rows. Every output is checksummed and compared with the other algorithm's; the largest difference was 4.4×10⁻¹⁴.
- **Repetitions.** Each algorithm gets 7 repetitions per configuration (30 in E0, 15 in E9b), and we report the median. E0 justifies 7: the coefficient of variation was 0.3–1.6%, and the 95% bootstrap interval on a 7-sample median was ±0.3–1.5%. Going to 15 would only narrow it to ±0.1–1.4%.
- **Order.** The algorithm order rotates every repetition. E5 and E6 compare *different configurations*, so each of their repetitions is a full round over all configurations, rotated and reversed on alternate rounds. The conditions being compared are therefore measured seconds apart.

## 3.5 Machine state

A first full run (archived in `results/shakedown_2026-11-04/`) was spoiled by additional workload on the PC, which caused 16–20% drift. It also showed that E6's original sequential ordering could not separate that drift from cache pressure. That led to the round-robin ordering above and to a **drift monitor**: before every configuration, a fixed workload (blocked, n = 384, T = 64) is timed and its reading written to each row. In the official and follow-up runs, 89% of 288 readings were within ±5% of a warmed-up baseline (Figure 9). The low readings, the first of each run (90–92%) and those during sustained two-core load (medians about 96%), reflect the boost clock rather than interference. Effects under about 5% *between* configurations are therefore treated as inconclusive. Effects *within* a configuration are protected by the rotated order.

## 3.6 Reproducing

Build `windows-release`, run `ctest`, and run `mm_bench.exe`, which runs E0–E9b. Then run `python scripts/analyse.py` and `python scripts/make_plots.py`. These regenerate every table in `results/summary/` and every figure from the 3,661 CSV rows alone. Every number in §4–5 comes from them.
