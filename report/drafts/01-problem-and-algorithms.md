# 1. The problem and the algorithms

## 1.1 Problem statement

Given two dense square matrices $A, B \in \mathbb{R}^{n \times n}$ stored as IEEE-754 double-precision values in row-major order, compute

$$C = AB, \qquad c_{ij} = \sum_{k=1}^{n} a_{ik}\, b_{kj}, \quad 1 \le i, j \le n.$$

The output must not share memory with either input, and the inputs may have any row stride (leading dimension) of at least $n$. The comparison uses *square* matrices because Strassen's algorithm is defined on square block partitions. (Algorithm A also handles rectangular shapes, and the test suite checks those too.) Correctness means agreeing with a reference implementation. For inputs whose products and partial sums are exactly representable integers, the result must match bit for bit. Otherwise it must be within floating-point error bounds, and the size of that error is itself one of the quantities I compare (§2.4).

Every classical algorithm performs the same $n^3$ multiplications and $n^3 - n^2$ additions, about $2n^3$ floating-point operations (flops). They differ only in the *order* of those operations. Strassen's algorithm does something different in kind: it performs fewer multiplications, at the price of extra additions and extra memory. The two algorithms below were chosen because they represent these two ideas. One reorganises the same arithmetic to suit the memory hierarchy. The other changes the arithmetic itself to reduce its asymptotic count.

## 1.2 Algorithm A: cache-blocked iterative multiplication

**Key idea.** The triple loop that computes $C = AB$ can run its three loops in any order and over any partition of the index space. The total arithmetic is always the same, but the order determines how often each matrix element has to be fetched from slow memory. A straightforward $i$-$j$-$k$ loop walks down a column of $B$ for every output element. In row-major storage each step of that walk lands on a different cache line, and for large $n$ a line is evicted long before it is used again. So the loop runs at the speed of main memory rather than the speed of the arithmetic units.

Blocking (tiling) fixes this by splitting the iteration space into $T \times T \times T$ tiles:

```
for ii in 0, T, 2T, ... < n:            # tile row of C and A
  for kk in 0, T, 2T, ... < n:          # tile of the shared dimension
    for jj in 0, T, 2T, ... < n:        # tile column of C and B
      for i in ii .. ii+T-1:
        for k in kk .. kk+T-1:
          a = A[i][k]
          for j in jj .. jj+T-1:        # contiguous in C and B -> vectorised
            C[i][j] += a * B[k][j]
```

One tile step reads a $T \times T$ block of $A$ and of $B$ and updates a $T \times T$ block of $C$. The block of $B$ is reused for each of the $T$ rows $i$. If it stays in cache, each element of $B$ fetched from a slow level is used about $T$ times instead of once. Inside the tile the loop order is $i$-$k$-$j$. The innermost loop multiplies one scalar of $A$ by a contiguous row segment of $B$ and adds it into a contiguous row segment of $C$ (an "axpy"). This is the access pattern a compiler can vectorise. With AVX2, one instruction processes four doubles.

**Complexity.** $\Theta(n^3)$ time, exactly $2n^3$ flops (one multiply and one add per inner iteration), and $O(1)$ extra memory. The tile size $T$ changes none of these. It only changes the constant factor, by changing which cache level serves the inner loop. That's why $T$ is treated as an experimental variable (§3) rather than a fixed constant.

**Implementation notes.** `src/algo_blocked/` is about 60 lines of C++17. The inner loop is a separate function whose pointer parameters are declared `__restrict`, which lets the compiler vectorise it without a runtime aliasing check. I confirmed this from GCC's vectorisation report. I deliberately did *not* write hand-tuned intrinsics or a register-blocked micro-kernel of the kind used in BLAS libraries. Both algorithms get the same level of implementation effort, and Strassen's base case calls exactly this kernel, so any improvement to A would also improve B.

## 1.3 Algorithm B: Strassen's algorithm

**Key idea.** Partition each matrix into four $\frac{n}{2} \times \frac{n}{2}$ quadrants:

$$\begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22}\end{pmatrix} = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22}\end{pmatrix}\begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22}\end{pmatrix}.$$

Computed directly, this takes eight quadrant products ($C_{11} = A_{11}B_{11} + A_{12}B_{21}$, and so on), so recursing gives $T(n) = 8T(n/2) + \Theta(n^2) = \Theta(n^3)$ and nothing is gained. Strassen (1969) showed that seven products suffice:

| Product | Combination |
|---|---|
| $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$ | $C_{11} = M_1 + M_4 - M_5 + M_7$ |
| $M_2 = (A_{21}+A_{22})\,B_{11}$ | $C_{12} = M_3 + M_5$ |
| $M_3 = A_{11}\,(B_{12}-B_{22})$ | $C_{21} = M_2 + M_4$ |
| $M_4 = A_{22}\,(B_{21}-B_{11})$ | $C_{22} = M_1 - M_2 + M_3 + M_6$ |
| $M_5 = (A_{11}+A_{12})\,B_{22}$ | |
| $M_6 = (A_{21}-A_{11})(B_{11}+B_{12})$ | |
| $M_7 = (A_{12}-A_{22})(B_{21}+B_{22})$ | |

Forming the operands takes 10 quadrant additions or subtractions, and combining the products takes 8 more: 18 in total, each costing $(n/2)^2$ flops. The trick works because the cross terms cancel. For example, expanding $M_1 + M_4 - M_5 + M_7$ leaves exactly $A_{11}B_{11} + A_{12}B_{21}$. It relies only on the ring axioms (it never uses commutativity of multiplication), which is why it can be applied recursively to the quadrants themselves.

**Complexity.** The recurrence is $T(n) = 7T(n/2) + 18(n/2)^2$, so $T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$. Asymptotically this beats every classical ordering. The price is $\Theta(n^2)$ extra memory for temporaries, extra additions with poor arithmetic intensity, and weaker numerical stability (§2.4).

**What makes it practical.** Three choices turn the textbook recursion into something that can compete. Each is a parameter of the implementation, not a constant, so each can be measured.

1. *Cutoff.* Recursing all the way to $1 \times 1$ would spend almost all its time on additions and function calls. Below a cutoff size $c$, the implementation stops recursing and multiplies with Algorithm A's blocked kernel. With $d$ recursion levels it performs $7^d$ leaf multiplications of size $n/2^d$, so the multiplication work is $(7/8)^d$ of the classical $2n^3$. At $n = 4096$ with $c = 128$ (so $d = 5$), that's 51.3% of classical, plus 1.1% for the additions.
2. *Padding.* The recursion needs the size to be even at every level. Sizes that aren't powers of two are padded with zeros to a size $m \ge n$, using one of two rules. `nextpow2` pads to the next power of two, the textbook approach (1025 becomes 2048). `minimal` pads to $m = s \cdot 2^d$, where $d$ is the smallest number of halvings that brings $s = \lceil n/2^d \rceil$ down to at most $c$ (with $c = 64$, 1025 becomes 1056). Zero padding is exact: the extra rows and columns contribute zero to every entry of the true product.
3. *Memory schedule.* Each recursion node needs three temporaries of size $(n/2)^2$: two for operand sums and one for a product. $M_1$, $M_2$ and $M_3$ are written directly into the output quadrants, so each node needs only one quadrant copy. Summed over the levels, the workspace is $3\left(\frac{n^2}{4} + \frac{n^2}{16} + \cdots\right) < n^2$ doubles, about one extra matrix. A `Plan` object allocates all of it once, before execution, and the recursion itself never allocates. That lets the benchmarks separate the cost of Strassen's arithmetic from the cost of obtaining its memory.

**Implementation notes.** `src/algo_strassen/` is about 210 lines of C++17. Quadrants are zero-copy *views* that keep the parent's stride. `strassen::predict(n, params)` computes the algorithm's exact operation counts from closed-form expressions: leaf calls, leaf flops, addition flops, copies, workspace bytes and padding bytes. The test suite checks these against counters recorded during real executions, for 80 combinations of size, cutoff and padding mode. The cost model used in §2 is therefore a verified property of the code, not an assumption.

## 1.4 How the two algorithms differ

The two algorithms embody opposite strategies. Algorithm A keeps the arithmetic fixed at $2n^3$ flops and attacks the *memory* cost, arranging for each fetched element to be reused as many times as possible. Its inner loop is long, contiguous and branch-free. Algorithm B attacks the *arithmetic* cost by reducing the exponent from 3 to 2.807. In exchange it adds operations that are almost pure memory traffic. Each quadrant addition reads two values and writes one for every flop, which is 24 bytes moved per flop, and none of those values is reused. They come from whichever cache level, or main memory, the quadrants currently live in. In A's kernel, by contrast, each element of $B$ that is fetched from far away is used about $T$ times while it sits in L1 or L2. B also allocates temporaries and needs padding for awkward sizes. Because B uses A's kernel at its leaves, the comparison is cleaner than two unrelated codes would allow: the *only* difference between them is whether the top $d$ levels are done Strassen's way or classically. The question this project asks is when trading a 1/8 cut in multiplication work per level for 18 memory-bound quadrant additions per node is worth it on real hardware.
