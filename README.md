# Matrix Multiplication: Cache-Blocked Iterative vs Strassen

Programming Assignment 2 (Advanced Algorithms): an empirical comparison of two
algorithms for dense square matrix multiplication `C = A·B` over `double`:

| Module | Algorithm | Complexity | Library target |
|---|---|---|---|
| `src/algo_blocked/` | **A:** cache-blocked (tiled) iterative multiplication | Θ(n³) | `mm_blocked` |
| `src/algo_strassen/` | **B:** Strassen's recursive algorithm with cutoff to A's kernel | Θ(n^log₂7) ≈ Θ(n^2.807) | `mm_strassen` |
| `src/reference/` | Naive i-j-k loop + double-double reference | Θ(n³) | `mm_common` (**oracle only, not compared**) |

Author: Jarrod Carmont (solo).

## Repository layout

```
src/common/        matrix.hpp     aligned row-major storage, strided views, alias check, scratch buffer
                   platform.hpp   portability macros (MM_RESTRICT)
                   rng.hpp        portable xoshiro256** RNG (bit-identical on MSVC and g++)
                   generators.*   seeded input families: uniform, smallint, widerange, identity
                   compare.*      exact/abs/Frobenius comparisons, relative error
src/reference/     naive.*        correctness oracle and high-accuracy error reference
src/algo_blocked/  blocked.*      Algorithm A
src/algo_strassen/ strassen.*     Algorithm B (+ Plan, padding modes, operation-count model)
tests/                            dependency-free unit tests (run by CTest)
bench/                            unified benchmark harness (mm_bench): environment capture,
                                  timing runner, cache-pressure thread, experiments E0-E8
scripts/                          analyse.py (summary tables, cost model, predictions check),
                                  make_plots.py (all figures), mmdata.py (shared loading + cost model)
results/                          raw CSVs, run_info.txt, runtimes.csv, tuned_params.txt
figures/                          generated plots (fig1-fig10)
report/                           drafts/*.md (the report source, incl. pre-registrations),
                                  build_report.py, report.docx (generated)
```

## Requirements

- **Windows (benchmark platform):** Visual Studio 2022 or later with the
  *Desktop development with C++* workload. This includes MSVC, CMake and Ninja.
- **Linux (development / cross-check):** g++ ≥ 9, CMake ≥ 3.21, make.
- C++17. No third-party libraries; nothing is downloaded at build time.

## Build and test

### Windows: Visual Studio IDE

1. **File → Open → Folder…** and select this repository's root folder. Do not
   create a Visual Studio solution; `CMakeLists.txt` *is* the project.
2. In the configuration drop-down on the toolbar, choose
   **Windows MSVC x64 Release (benchmarks)**.
3. **Build → Build All**.
4. **Test → Run CTests for MatMulComparison**, or run
   `.\build\windows-release\mm_tests.exe` directly.

### Windows: command line

Open **Developer PowerShell for VS** (Start menu) **for x64**. For example, run
`Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64`, as described under
*Running the benchmarks*. Then `cd` to the repository root and run:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
```

### Linux

```bash
cmake --preset linux-release
cmake --build --preset linux-release -j
ctest --preset linux-release
```

Expected test output ends with `N/N tests passed`.

## Algorithm parameters

Every tuning knob is a runtime parameter, so the benchmarks can sweep it as an
experimental variable. None are hard-coded.

| Parameter | Applies to | Default | Meaning |
|---|---|---|---|
| `blocked::Params::tile` | A, and B's base case | 64 | Tile edge T. Each tile step touches 3·T²·8 bytes. |
| `strassen::Params::cutoff` | B | 128 | Recurse only while size > cutoff; at or below it, use A's kernel |
| `strassen::Params::leaf_align` | B | 4 | `aligned` padding only: the leaf size is rounded up to a multiple of this (E9) |
| `strassen::Params::temp_pad` | B | 0 | Extra row-stride elements on every recursion temporary (tests power-of-two aliasing, E5) |
| `strassen::Params::padding` | B | `minimal` | `minimal`: pad to s·2^d with s ≤ cutoff (1025 → 1056 at cutoff 64). `nextpow2`: pad to the next power of two (1025 → 2048). `aligned`: like `minimal`, but with the leaf size rounded up to a multiple of `leaf_align` (1028 → 1088 at cutoff 128); added after E4. |

The defaults only exist so the API is usable. The values used for each
experiment are chosen from parameter sweeps and recorded in every results row.

`strassen::Plan` performs all allocation (recursion workspace and padded copies)
when it is constructed, and `execute()` allocates nothing. `strassen::multiply()`
builds a Plan and runs it once. This lets the benchmarks time Strassen's
arithmetic with and without its memory-allocation cost.

`strassen::predict(n, params)` returns the exact operation counts (leaf calls,
leaf flops, addition flops, copies, workspace and padding bytes) without running
anything. The test `strassen_measured_operation_counts_match_cost_model` checks
these against counts recorded during real executions, so the report's cost model
is verified rather than assumed.

## Compiler flags (identical for both algorithms)

| | MSVC | g++ |
|---|---|---|
| Optimisation | `/O2` (CMake Release default) | `-O3` (CMake Release default) |
| ISA | `/arch:AVX2` | `-mavx2 -mfma` |
| Floating point | `/fp:precise` | strict IEEE, `-ffp-contract=off` |
| Warnings | `/W4 /permissive-` | `-Wall -Wextra -Wpedantic` |

Strict floating point is deliberate. Numerical error is one of the measured
quantities, and fast-math reassociation would change it. It would also break
the error-free transforms that the double-double reference relies on.

## Reproducibility notes

- All inputs come from `src/common/rng.hpp` (xoshiro256** seeded by SplitMix64),
  with exact integer-to-double conversions. `<random>` distributions are not used,
  because MSVC and libstdc++ produce different values from the same seed.
  The test `rng_matches_golden_values_across_platforms` checks this on every build.
- Both algorithms receive the same `Matrix` objects generated from the same seeds.

## Running the benchmarks

Build the **windows-release** preset in the Visual Studio IDE with
**Build → Build All**. Building only the test target does not build `mm_bench`,
and Debug builds refuse to run the full experiments.

> **Command-line builds:** the terminal built into Visual Studio may start a
> *32-bit* developer environment. The link then fails with hundreds of
> unresolved standard-library symbols and `warning LNK4272: library machine type
> 'x86' conflicts with target machine type 'x64'`. Build from the IDE, or first
> switch the terminal to 64-bit with
> `& "$env:VSINSTALLDIR\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation`.

Then run the harness from the repository root. (Running it does not depend on
the terminal's architecture.) In PowerShell the `.\` prefix is required:

```powershell
.\build\windows-release\mm_bench.exe --estimate   # per-experiment runtime estimate, runs nothing
.\build\windows-release\mm_bench.exe --quick      # whole pipeline at tiny sizes (seconds), writes results/quick/
.\build\windows-release\mm_bench.exe              # correctness gate, then E0-E8 (full run)
```

Before a full run: plug in mains power, select the *High performance* or
*Ultimate Performance* power plan (recorded automatically), and close other
applications.

| Option | Effect |
|---|---|
| `--only E0,E3` | Run only the listed experiments |
| `--skip-large` | Drop every configuration with n > 2048 |
| `--reps N` | Timed repetitions per configuration (default 7; choose from E0) |
| `--tile T`, `--cutoff C` | Override the tuned values |
| `--bench-cpu N` | Logical CPU the benchmark thread is pinned to (default 2 = physical core 1) |
| `--antagonist-cpu N` | Logical CPU for E6's cache-pressure thread (default 6 = physical core 3) |
| `--out DIR` | Results directory (default `<repo>/results`) |

### Experiments

| ID | Question (prediction tested) | Output |
|---|---|---|
| E0 | Run-to-run variance and drift at fixed parameters; justifies `--reps` | `E0_pilot.csv` |
| E1 | Tile size for the blocked kernel (P1, P6); selects the tuned tile | `E1_tile.csv` |
| E2 | Strassen cutoff (P2, P3); selects the tuned cutoff | `E2_cutoff.csv` |
| E3 | Scaling 64–4096 and the crossover (P1, P4, P5) | `E3_scaling.csv` |
| E4 | `nextpow2` vs `minimal` padding around 2^k and a dense sweep (P9, P10) | `E4_padding.csv` |
| E5 | Power-of-two leading-dimension aliasing; row-stride and temporary padding (P8) | `E5_stride.csv` |
| E6 | L3 occupied by a pinned antagonist thread: 0 / 1 (control) / 32 / 64 / 80 MB (P7) | `E6_cache_pressure.csv` |
| E7 | Allocation cost: reused `Plan` vs `strassen::multiply` per call (P11) | `E7_allocation.csv` |
| E8 | Accuracy vs the double-double reference; Frobenius, normwise and componentwise (P12–P14) | `E8_accuracy.csv` |
| E9 | *Follow-up, added after E4.* Is minimal padding slow when the leaf size is not a multiple of 4? Compares `minimal` with `aligned` padding (leaf rounded up to a multiple of 4 or 8) over n = 1000–1100 and 2000–2100. Included in a full run; the official data came from a separate `--only E9` run. | `E9_alignment.csv` |
| E9b | *Follow-up, added after E9.* Two identical `aligned4` variants run in both execution orders, to tell a carry-over effect from memory placement. Included in a full run; the official data came from a separate `--only E9b` run. | `E9b_order.csv` |

Tuned parameters are selected from the data, not chosen by hand. For each
configuration, every candidate's median time is divided by the best median in
that configuration. The winner has the smallest geometric mean of those ratios.
E1 selects the tile and E2 the cutoff. Both are written to
`results/tuned_params.txt`, and E3–E8 use them, so `--only` runs of later
experiments stay consistent with the tuning.

### Measurement rules

- **Same inputs for every algorithm.** Seeds depend only on `n`, so every
  experiment and every algorithm at a given size multiplies the same matrices.
- **Inputs before timing.** All inputs, outputs and Strassen `Plan`s are created
  before any timing window opens. E7 is the deliberate exception and labels its
  allocating rows.
- **Warm-up.** Each algorithm gets one untimed warm-up call per configuration.
  That call also sizes an inner loop, so each timed sample lasts at least 20 ms
  (`inner_calls` column). This keeps the 100 ns `steady_clock` resolution
  irrelevant.
- **Rotated order.** The order of algorithms rotates every repetition, so slow
  drift affects all of them equally.
- **Round-robin for cross-configuration comparisons.** E5 (stride vs stride)
  and E6 (pressure level vs pressure level) compare *different* configurations.
  For these, every repetition is one round over all configurations, rotated
  each round and reversed on odd rounds. Configurations being compared are then
  measured seconds apart, never minutes.
- **Drift monitor.** Before every configuration, a fixed reference workload
  (blocked, n = 384, T = 64, at least 50 ms) is timed with no antagonist
  running. It is recorded as `ref_gflops` on every row and compared with a
  warmed-up baseline taken at start-up. A warning is printed past ±5%, and each
  experiment's range goes into `runtimes.csv` and `run_info.txt`. Conditions
  are measured, not assumed. `results/shakedown_2026-11-04/` shows why: a run
  where other workload on the PC partway through caused 16–20% drift.
- **Outputs consumed.** Every output is checksummed, and its relative difference
  from the first algorithm in the configuration is recorded
  (`rel_diff_vs_first`).
- **Timer reported, not subtracted.** Timer resolution and overhead are measured
  and written to `run_info.txt`, not subtracted from results.
- **Run records.** `run_info.txt` (appended per run) records CPU, OS, RAM,
  compiler, flags, git commit (with a dirty flag), power plan, pinning, priority,
  settings and calibration. `runtimes.csv` records each experiment's estimated
  and actual runtime.
- **Pinning.** The benchmark runs single-threaded, pinned to one logical CPU at
  high priority. E6's antagonist is pinned to a *different physical core*, so it
  shares only the L3. Its 1 MB control condition fits in the antagonist's own
  L2, so it separates "a neighbouring core is busy", which can lower the boost
  clock, from "L3 is occupied".
- **Machine idle during runs.** Nothing else should run during a full run: no
  games, browsers or updates.

## Reproducing the experiments

1. Build `windows-release`, then run `ctest --preset windows-release`.
2. Run `.\build\windows-release\mm_bench.exe` (E0–E9b, about 30 minutes). This regenerates every CSV in `results/`.
3. Run `python scripts/analyse.py`. This writes the summary tables, the
   cost-model check and the predictions-vs-results table to `results/summary/`.
4. Run `python scripts/make_plots.py`. This regenerates every figure in `figures/`.

Steps 3 and 4 need Python 3.9+ with `pandas`, `numpy`, `matplotlib` and `tabulate`
(`pip install pandas numpy matplotlib tabulate`; Anaconda includes them). Both
scripts read only the CSVs, so they run on any machine.

## Building the report

`report/drafts/*.md` is the report's source. Their git history shows that
§2 and the E9 and E9b pre-registrations were committed before the
measurements they predict. To regenerate the Word document:

```
python report/build_report.py      # needs pandoc 2.9+ and python-docx
```

This joins the drafts, embeds `figures/*.png` with their captions, renders
`$...$` maths as native Word equations and writes `report/report.docx`.

