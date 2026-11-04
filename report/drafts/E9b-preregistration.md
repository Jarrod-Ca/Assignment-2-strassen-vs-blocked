# E9b follow-up: pre-registered hypotheses

*Written and committed before E9b was run. Not edited afterwards.*

## What prompted it

In E9, `aligned4` and `aligned8` produced **identical computations** at
n = 1000, 1004, 1008, 1012, 1016, 1092, 1096 and 1100: same leaf size, same
padded size, same code path. Yet `aligned8` was consistently 1–6% faster, and
`aligned4` had 2–3× the interquartile range. Where no padded copies are made
(n = 1020, 1024, 2036, 2048) the two agreed within 0.4%. A difference between
identical computations is a measurement artefact, and its size sets a noise
floor for comparing Strassen variants.

Two explanations fit E9's design:

- **Carry-over (position).** Rotating the order every repetition keeps each
  variant's *predecessor* fixed: `aligned4` always ran right after `minimal`,
  and `aligned8` always right after `aligned4`. A variant that follows a
  similar one may inherit a favourable cache, TLB or clock state.
- **Memory placement.** Each `Plan` allocates its own padded copies and
  workspace. Where those buffers land, relative to each other and to the shared
  inputs, could make one Plan's layout systematically faster: cache-set
  conflicts, 4 KB aliasing, or page placement.

## Design

Two *identical* `aligned4` variants. `alloc1` is constructed (allocated) first
and `alloc2` second, alongside blocked and `minimal`, as in E9.

- **Order A:** blocked → minimal → alloc1 → alloc2. Here alloc2 sits in
  `aligned8`'s E9 slot, right after the other aligned variant.
- **Order B:** blocked → minimal → alloc2 → alloc1. The positions are swapped;
  the allocations are not.

There are 15 repetitions per configuration, with the order rotated each
repetition, at n = 1000, 1012, 1016, 1092 and 1100 (all with padded copies)
and n = 1024 (the control: no padded copies). The statistic is the median over
repetitions of time(alloc2) / time(alloc1), paired within each repetition, with
a 95% bootstrap confidence interval.

## Predictions

- **If carry-over:** in order A, alloc2 is faster (ratio < 1). In order B,
  alloc1 is faster (ratio > 1). The advantage follows the *position*.
- **If placement:** the same allocation is faster in both orders. The ratio is
  on the same side of 1 in A and in B.
- **Control (n = 1024):** ratio within 1 ± 0.01 in both orders, whichever
  explanation holds.
- **My expectation, recorded before the run:** carry-over. E9's faster variant
  was always the one running right after a near-identical variant, and
  separately allocated buffers of the same size usually get layouts that are
  just as good.
- **If neither appears** (every confidence interval includes 1): E9's gap was
  not reproducible, and the report will treat it as noise of about ±5% rather
  than a systematic effect.
