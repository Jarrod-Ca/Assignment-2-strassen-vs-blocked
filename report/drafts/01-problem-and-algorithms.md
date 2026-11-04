# 1. The problem and the algorithms

## 1.1 Problem statement

Given dense square matrices $A, B \in \mathbb{R}^{n \times n}$, stored as IEEE-754 doubles in row-major order with any row stride ≥ n, compute

$$C = AB, \qquad c_{ij} = \sum_{k=1}^{n} a_{ik}\, b_{kj},$$

into an output that shares no memory with either input. Matrices are square because Strassen's algorithm partitions square blocks. Correctness means bit-for-bit agreement with a reference on inputs whose partial sums are exact integers, and staying within floating-point error bounds otherwise. That error is itself compared (§2.7).

## 1.2 Algorithm A: cache-blocked iterative multiplication

**Key idea.** The triple loop can run in any order and over any partition of the index space. The arithmetic stays the same, but the order decides how often each element must be fetched from slow memory. A plain i-j-k loop walks down a column of B for every output element, touching a new cache line at every step, so for large n it runs at memory speed. Blocking splits the iteration space into T × T × T tiles:

```
for ii, kk, jj in steps of T:               # tile of C/A rows, shared dimension, C/B columns
  for i in ii .. ii+T-1:
    for k in kk .. kk+T-1:
      a = A[i][k]
      for j in jj .. jj+T-1:                # contiguous in C and B -> vectorised
        C[i][j] += a * B[k][j]
```

A tile step reuses a T × T block of B for each of T rows of C, so each element of B fetched from a slow level is used about T times. The innermost loop multiplies one scalar of A by a contiguous row segment of B and adds it into a row of C. That is the pattern the compiler vectorises: with AVX2, four doubles per instruction.

**Complexity.** Θ(n³) time, exactly 2n³ flops, O(1) extra memory. T changes only the constant factor, so it is an experimental variable (E1), not a fixed constant. The module (`src/algo_blocked/`) deliberately has no hand-written intrinsics or BLAS-style register micro-kernel. Both algorithms get equal implementation effort, and Strassen's leaves call exactly this kernel.

## 1.3 Algorithm B: Strassen's algorithm

**Key idea.** Split each matrix into four (n/2) × (n/2) quadrants. The direct block product needs eight quadrant products and gains nothing. Strassen (1969) showed seven suffice:

| Product | Combination |
|---|---|
| $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$ | $C_{11} = M_1 + M_4 - M_5 + M_7$ |
| $M_2 = (A_{21}+A_{22})\,B_{11}$ | $C_{12} = M_3 + M_5$ |
| $M_3 = A_{11}\,(B_{12}-B_{22})$ | $C_{21} = M_2 + M_4$ |
| $M_4 = A_{22}\,(B_{21}-B_{11})$ | $C_{22} = M_1 - M_2 + M_3 + M_6$ |
| $M_5 = (A_{11}+A_{12})\,B_{22}$ | |
| $M_6 = (A_{21}-A_{11})(B_{11}+B_{12})$ | |
| $M_7 = (A_{12}-A_{22})(B_{21}+B_{22})$ | |

The cross terms cancel: for example, $M_1 + M_4 - M_5 + M_7$ expands to exactly $A_{11}B_{11} + A_{12}B_{21}$. The derivation never uses commutativity, so the scheme applies recursively to the quadrants. Forming operands and combining products takes 18 quadrant additions.

**Complexity.** $T(n) = 7T(n/2) + 18(n/2)^2 = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$. The price is Θ(n²) extra memory, additions with very low arithmetic intensity, and weaker numerical stability (§2.7).

**What makes it practical.** Three runtime parameters, each measured by the experiments:

1. **Cutoff.** Below size c the recursion stops and calls Algorithm A's kernel. With d levels there are 7ᵈ leaves, so the multiplication work is (7/8)ᵈ of classical: 51.3% at n = 4096 with c = 128 (d = 5), plus 1.1% in additions.
2. **Padding.** Every level needs an even size, so n is zero-padded to some m ≥ n, which is exact. `nextpow2` pads to the next power of two (1025 → 2048). `minimal` pads to m = s · 2ᵈ, with d the fewest halvings that bring s = ⌈n/2ᵈ⌉ to at most c (1025 → 1040). `aligned`, added after E4 (§5.2), also rounds s up to a multiple of 4 or 8.
3. **Memory.** Each node needs three (n/2)² temporaries, under n² doubles in total. A `Plan` allocates them, and any padded copies, before execution, and the recursion never allocates.

`strassen::predict()` computes exact operation and memory counts in closed form. The tests check these against counters recorded during real executions, so the cost model in §2 is a verified property of the code (`src/algo_strassen/`).

## 1.4 How they differ

A keeps the arithmetic fixed and attacks the *memory* cost. B attacks the *arithmetic*, cutting the exponent from 3 to 2.807, at the price of additions that move 24 bytes per flop and reuse nothing, plus temporaries and padding. Because B's leaves are A's kernel, the *only* difference between them is whether the top d levels are done Strassen's way or classically. The question is when a one-eighth cut in multiplication work per level is worth 18 memory-bound additions per node on real hardware.
