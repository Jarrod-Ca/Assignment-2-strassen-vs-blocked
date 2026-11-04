# E9 follow-up: pre-registered hypothesis and predictions

*Written and committed before E9 was run, in the same spirit as §2. This file
is the record of what we expected. It is not edited after the results exist.*

## What prompted it

In E4's dense sweep (n = 1000–1100, step 4, `minimal` padding, cutoff 128),
Strassen was **slower than blocked at 14 of 26 sizes** (worst 1.10× blocked's
time). That refutes P10. Splitting the sizes by the leaf size the recursion
stops at gives a sharp pattern:

| Leaf size | Time / blocked |
|---|---|
| 128 (n = 1020, 1024) | 0.71–0.74 |
| 68 (n = 1076–1088) | 0.75–0.78 |
| 66, 126 | 0.95–0.99 |
| 65, 67, 69, 125, 127 | 1.00–1.10 |

## Hypothesis

The slowdown comes from leaf sizes that are not a multiple of 4. One AVX2
register holds 4 doubles. A leaf of width 65 leaves a scalar remainder on
every row of every leaf multiplication. More importantly, the recursion
temporaries then have row strides (65, 130, 260, … doubles) that are not
multiples of 4, so leaf rows start at addresses that are not 32-byte aligned.
The extra flops of `minimal` padding are not the cause: they are at most 3%
here.

## Test

Add an `aligned` padding mode that rounds the leaf size up to a multiple of
`leaf_align`, so every leaf width and every temporary stride is a whole number
of vectors. Compare it with `minimal` and blocked on the same instances, with 7
repetitions, over n = 1000–1100 (step 4) and n = 2000–2100 (step 12).
Alignment 4 corresponds to one AVX2 vector; alignment 8 to one 64-byte cache
line.

## Predictions

- **H1.** With `aligned` padding at alignment 4, Strassen is faster than
  blocked at **every** size in 1000–1100 (at most 1 of 26 exceptions), with a
  median time ratio between 0.72 and 0.85. This holds even though it does up to
  about 13% more arithmetic than `minimal` (leaf 65 → 68: (68/65)^2.81 ≈ 1.135).
- **H2.** At alignment 8 there is no further improvement over alignment 4.
  Wherever it pads more (e.g. 1028 → 1152 rather than 1088), it is slower than
  alignment 4. In other words, vector-width alignment is what matters, not
  cache-line alignment.
- **H3.** The same pattern appears in 2000–2100. `minimal` is slow at leaf
  sizes that are not multiples of 4, and `aligned` at alignment 4 removes the
  slow points.
- **Falsifier.** If `aligned` at alignment 4 is still slower than blocked at 5
  or more of the 1000–1100 sizes, the alignment explanation is wrong, or at best
  incomplete, and the report will say so.
