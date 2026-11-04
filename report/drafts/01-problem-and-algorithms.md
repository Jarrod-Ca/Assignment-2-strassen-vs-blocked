# 1. The problem and the algorithms

## 1.1 Problem statement

Given dense square matrices $A, B \in \mathbb{R}^{n \times n}$, stored as IEEE-754 doubles in row-major order with any row stride ≥ n, compute

$$C = AB, \qquad c_{ij} = \sum_{k=1}^{n} a_{ik}\, b_{kj},$$

into an output that shares no memory with either input. The matrices are square because Strassen's algorithm partitions square blocks. Correctness means bit-for-bit agreement with a reference on inputs whose partial sums are exact integers, and staying within floating-point error bounds otherwise. That error is itself one of the quantities compared (§2.7).

Every classical algorithm performs the same ~2n³ floating-point operations (flops) and differs only in their *order*. Strassen's algorithm does something different in kind: fewer multiplications, paid for with extra additions and memory. The two algorithms represent these two ideas. One reorganises the same arithmetic to suit the memory hierarchy; the other changes the arithmetic itself.

## 1.2 Algorithm A: cache-blocked iterative multiplication

**Key idea.** The triple loop can run in any order and over any partition of the index space. The arithmetic stays the same, but the order decides how often each element must be fetched from slow memory. A plain i-j-k loop walks down a column of B for every output element. In row-major storage every step lands on a new cache line, so for large n the loop runs at memory speed. Blocking splits the iteration space into T × T × T tiles:

```
for ii, kk, jj in steps of T:               # tile of C/A rows, shared dimension, C/B columns
  for i in ii .. ii+T-1:
    for k in kk .. kk+T-1:
      a = A[i][k]
      for j in jj .. jj+T-1:                # contiguous in C and B -> vectorised
        C[i][j] += a * B[k][j]
```

A tile step reuses a T × T block of B for each of T rows of C. Each element of B fetched from a slow level is therefore used about T times instead of once. Inside the tile the order is i-k-j. The innermost loop multiplies one scalar of A by a contiguous row segment of B and adds it into a row segment of C (an "axpy"), which the compiler vectorises: with AVX2, four doubles per instruction.

**Complexity.** Θ(n³) time, exactly 2n³ flops, O(1) extra memory. T changes only the constant factor, so it is treated as an experimental variable (E1), not a fixed constant.

**Implementation.** `src/algo_blocked/` is about 60 lines of C++17. The inner loop takes `__restrict` pointer parameters, so it vectorises without a runtime aliasing check (confirmed from the compiler's vectorisation report). It deliberately has no hand-written intrinsics and no BLAS-style register micro-kernel. Both algorithms get equal implementation effort, and Strassen's leaves call exactly this kernel.

## 1.3 Algorithm B: Strassen's algorithm

**Key idea.** Split each matrix into four (n/2) × (n/2) quadrants. The direct block product needs eight quadrant products, which recurses to Θ(n³) and gains nothing. Strassen (1969) showed seven suffice:

| Product | Combination |
|---|---|
| $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$ | $C_{11} = M_1 + M_4 - M_5 + M_7$ |
| $M_2 = (A_{21}+A_{22})\,B_{11}$ | $C_{12} = M_3 + M_5$ |
| $M_3 = A_{11}\,(B_{12}-B_{22})$ | $C_{21} = M_2 + M_4$ |
| $M_4 = A_{22}\,(B_{21}-B_{11})$ | $C_{22} = M_1 - M_2 + M_3 + M_6$ |
| $M_5 = (A_{11}+A_{12})\,B_{22}$ | |
| $M_6 = (A_{21}-A_{11})(B_{11}+B_{12})$ | |
| $M_7 = (A_{12}-A_{22})(B_{21}+B_{22})$ | |

The cross terms cancel: for example, $M_1 + M_4 - M_5 + M_7$ expands to exactly $A_{11}B_{11} + A_{12}B_{21}$. The derivation never uses commutativity, so the scheme applies recursively to the quadrants. Forming operands and combining products takes 18 quadrant additions, each of (n/2)² flops.

**Complexity.** $T(n) = 7T(n/2) + 18(n/2)^2 = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$. The price is Θ(n²) extra memory, additions with very low arithmetic intensity, and weaker numerical stability (§2.7).

**What makes it practical.** Three choices, each a runtime parameter that the experiments measure:

1. **Cutoff.** Below size c the recursion stops and calls Algorithm A's kernel; recursing to 1 × 1 would be dominated by additions and call overhead. With d levels there are 7^d leaves, so the multiplication work is (7/8)^d of classical. At n = 4096 with c = 128 (d = 5) that is 51.3%, plus 1.1% in additions.
2. **Padding.** Every level needs an even size, so n is zero-padded to some m ≥ n. Zero padding is exact. There are three rules:
   - `nextpow2`: the next power of two (1025 → 2048).
   - `minimal`: m = s · 2^d, with d the fewest halvings that bring s = ⌈n/2^d⌉ to at most c (1025 → 1040 at c = 128).
   - `aligned`: added after experiment E4 (§5.2). It rounds the leaf size s up to a multiple of 4 or 8.
3. **Memory schedule.** Each node needs three (n/2)² temporaries. Three of the seven products are written straight into output quadrants, so each node needs only one quadrant copy. The workspace totals under n² doubles, about one extra matrix. A `Plan` allocates it, and any padded copies, before execution; the recursion never allocates.

**Implementation.** `src/algo_strassen/` is about 220 lines. Quadrants are zero-copy *views* that keep the parent's stride. `strassen::predict()` computes exact operation and memory counts in closed form. The tests check these against counters recorded during real executions, across more than 80 size, cutoff and padding combinations. So the cost model in §2 is a verified property of the code, not an assumption.

## 1.4 How they differ

A keeps the arithmetic fixed and attacks the *memory* cost: each element fetched from far away is used about T times. B attacks the *arithmetic*, cutting the exponent from 3 to 2.807. In exchange it adds operations that are almost pure memory traffic: each quadrant addition moves 24 bytes per flop and reuses nothing. It also allocates temporaries and pads awkward sizes. Because B's leaves are A's kernel, the *only* difference between the two is whether the top d levels are done Strassen's way or classically. The project asks when trading a one-eighth cut in multiplication work per level for 18 memory-bound additions per node pays off on real hardware.
