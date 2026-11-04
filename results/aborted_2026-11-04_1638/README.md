# Aborted run (2026-11-04 16:38). Not used

`mm_bench.exe` was started without `--only E9` by mistake, then cancelled part
of the way through E1. `E0_pilot.csv` here is that run's complete E0, and
`E1_tile.csv` is a partial E1. Both had overwritten the official run's files
(commit `73d07cf`), which were restored from git. `tuned_params.txt` was not
affected, because E1 did not finish. The run's section in `run_info.txt` has no
`run finished` line, so the analysis scripts ignore it. Its E0 row in
`runtimes.csv` is the only other trace.
