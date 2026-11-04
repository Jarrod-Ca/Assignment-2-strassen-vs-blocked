# Shakedown run (2026-11-04 15:22–15:43). Not used for the report's results

This was the first full run of `mm_bench` on the benchmark machine, at commit
`dcf671a` plus uncommitted fixes (hence `DIRTY` in `run_info.txt`). It is kept
because it shows two problems that changed the harness before the official run.

1. **Machine-state drift.** Additional workload started on the PC partway through the run. The
   same blocked multiplication (tile 192, n = 2048) ran at 26.0–26.1 GFLOP/s in
   E2–E3 and at 21.8–21.9 GFLOP/s in E5–E7, about 16% slower. Strassen slowed by
   the same proportion. Comparisons *within* a configuration were protected,
   because algorithm order rotates every repetition, but comparisons *between*
   configurations or experiments were not.
2. **E6 design flaw.** E6 ran every 0 MB configuration first and every 80 MB
   configuration last, minutes apart, so drift was indistinguishable from cache
   pressure. The data even shows 64 MB slower than 80 MB, which no cache
   explanation can produce. E5's stride comparisons had the same weakness, to a
   smaller degree.

**Harness changes made in response (commit after `6d0ee4b`):**
- **Drift monitor.** A fixed reference workload is timed before every
  configuration, recorded as `ref_gflops` in every row, and summarised per
  experiment in `runtimes.csv`. A warning is printed if it moves more than 5%.
- **Round-robin E5 and E6.** Every repetition is a full pass over the
  configurations, in rotated order and reversed on odd rounds. E6 is ordered
  with n outer and pressure level inner.
- **Unique variant labels** (`tuned_T=192`, `tuned_c=128`) and `n/a` for the
  Strassen-specific bound on blocked rows in E8.

The pilot (E0) from this run was used to keep `--reps 7`. Its 95% bootstrap
interval on the median at 7 repetitions is ±0.8–2.0%.
