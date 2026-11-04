# Matrix Multiplication: Cache-Blocked Iterative vs Strassen

Programming Assignment 2 (Advanced Algorithms): an empirical comparison of two
algorithms for dense square matrix multiplication `C = A·B` over `double`:

| Module | Algorithm | Complexity | Status |
|---|---|---|---|
| `src/algo_blocked/` | Cache-blocked (tiled) iterative multiplication | Θ(n³) | Phase 2 |
| `src/algo_strassen/` | Strassen's recursive algorithm, tuned cutoff | Θ(n^log₂7) ≈ Θ(n^2.807) | Phase 2 |
| `src/reference/` | Naive i-j-k loop + double-double reference | Θ(n³) | **Oracle only, not compared** |

Author: Jarrod Carmont (solo).

## Repository layout

```
src/common/        matrix.hpp     aligned row-major storage + strided views
                   rng.hpp        portable xoshiro256** RNG (bit-identical on MSVC and g++)
                   generators.*   seeded input families: uniform, smallint, widerange, identity
                   compare.*      exact/abs/Frobenius comparisons, relative error
src/reference/     naive.*        correctness oracle and high-accuracy error reference
src/algo_blocked/  (phase 2)      Algorithm A
src/algo_strassen/ (phase 2)      Algorithm B
tests/                            dependency-free unit tests (run by CTest)
bench/             (phase 4)      unified benchmark harness
scripts/           (phase 4-5)    experiment runner, analysis and plotting
results/, figures/ (phase 5)      raw CSVs and generated plots
report/            (phase 3, 6)   report drafts and final .docx
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
   `build\windows-release\mm_tests.exe` directly.

### Windows: command line

Open **Developer PowerShell for VS** (Start menu), `cd` to the repository root, then run:

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

## Reproducing the experiments

*(Phase 4–5: `scripts/run_experiments.py` will regenerate every CSV in `results/`
and every figure in `figures/` from a clean build.)*
