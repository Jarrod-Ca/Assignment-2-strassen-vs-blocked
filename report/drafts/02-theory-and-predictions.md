# 2. Theoretical comparison and predictions

*Written and committed to the repository before any timing measurement was taken. The commit history shows this. Nothing in this section has been revised in light of results. Where the results disagree, §4 and §5 discuss why.*

## 2.1 Target hardware

All timings will come from one machine: an AMD Ryzen 7 7800X3D (Zen 4) running Windows, compiled with MSVC `/O2 /arch:AVX2 /fp:precise`. Exact versions are recorded in §3. Every run is single-threaded. The cache sizes that matter for the reasoning below are a 32 KB L1 data cache and a 1 MB L2 per core, and a 96 MB shared L3. The L3 is unusually large: the chip stacks extra "3D V-Cache" on top of the usual 32 MB.

## 2.2 Asymptotic and exact operation counts

| | Algorithm A (blocked) | Algorithm B (Strassen, cutoff $c$, $d$ levels) |
|---|---|---|
| Time | $\Theta(n^3)$ | $\Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$ |
| Multiply-kernel flops | $2n^3$ | $(7/8)^d \cdot 2n^3$ |
| Addition flops | 0 | $\sum_{\ell=0}^{d-1} 7^\ell \cdot 18\,(n/2^{\ell+1})^2$ |
| Extra memory | none | $< n^2$ doubles workspace, plus $3m^2$ if padded to $m$ |

From the verified cost model (`strassen::predict`), these are the fractions of classical work that Strassen performs:

| $n$ | cutoff | depth $d$ | kernel flops / $2n^3$ | addition flops / $2n^3$ | workspace |
|---|---|---|---|---|---|
| 256 | 64 | 2 | 0.766 | 0.024 | 0.5 MB |
| 1024 | 64 | 4 | 0.586 | 0.025 | 8 MB |
| 1024 | 128 | 3 | 0.670 | 0.013 | 8 MB |
| 4096 | 64 | 6 | 0.449 | 0.020 | 128 MB |
| 4096 | 128 | 5 | 0.513 | 0.011 | 128 MB |

In flop terms, the additions are never more than a few percent of the work. Strassen pays for them in a different currency: memory traffic.

## 2.3 Constant factors: a two-rate cost model

Counting flops treats every flop as equally expensive. They are not, so we model the run time as

$$t_B \approx \tau_{\text{mul}} \cdot F_{\text{kernel}} + \tau_{\text{add}} \cdot F_{\text{add}}, \qquad t_A \approx \tau_{\text{mul}} \cdot 2n^3,$$

where $\tau_{\text{mul}}$ is the time per flop in the blocked kernel and $\tau_{\text{add}}$ is the time per flop of a streaming quadrant addition. Their ratio $r = \tau_{\text{add}}/\tau_{\text{mul}}$ decides everything.

**Estimating $\tau_{\text{mul}}$.** Each vector iteration of A's inner loop loads 4 doubles of $B$ and 4 of $C$, multiplies, adds and stores 4 doubles back to $C$. That's two loads and one store for 8 flops. Zen 4 can retire at most about one 256-bit store per cycle, so the kernel is bounded by roughly 8 flops per cycle, about 40 GFLOP/s at 5 GHz. It will not reach that bound: loop overhead, partial tiles and L2 traffic for the $C$ and $A$ rows all eat into it. Because the kernel has no register blocking, every flop pays for a load and a store. **Prediction P1: Algorithm A sustains 10–20 GFLOP/s for $n \ge 256$, so $\tau_{\text{mul}} \approx 0.05$–$0.1$ ns.** (A tuned BLAS reaches several times this. That gap is the main reason we expect the crossover to land much earlier than published Strassen crossovers (e.g. Huss-Lederman et al., 1996), which were measured against tuned kernels.)

**Estimating $\tau_{\text{add}}$.** A quadrant addition moves 24 bytes per flop. One core can stream on the order of 100 GB/s from L2 or L3 and roughly 40–60 GB/s from DDR5, which gives $\tau_{\text{add}} \approx 0.25$ ns when the operands are cached and about 0.5 ns from DRAM. **So $r \approx 3$–$10$.**

**Break-even for one level.** One Strassen level at size $n$, with $h = n/2$, saves one quadrant product ($2h^3$ flops) and adds $18h^2$ addition flops plus one $h^2$ copy, worth about $2/3$ of an addition. Recursing pays off when

$$2h^3 \,\tau_{\text{mul}} > 18.7\,h^2\,\tau_{\text{add}} \iff h > 9.3\,r.$$

With $r = 3$–$10$ that gives $h \approx 28$–$93$, so recursion should be worth it once $n \gtrsim 56$–$186$.

**Prediction P2: the optimal cutoff $c^*$ is 64 or 128**, at the centre of that range. The curve of time against cutoff will be flat near the optimum: within ±1 power of two of $c^*$, run time will vary by less than 10%. The model contains nothing that depends on $n$, so **P3: $c^*$ does not change with $n$** across 512–4096, to within one power of two.

**Prediction P4: with its best cutoff, Strassen beats blocked from about $n \approx 256$ upward. Below about 128 they are effectively the same algorithm**, because at those sizes Strassen with $c^* \ge 128$ doesn't recurse at all. Near the crossover the margin will be under 5%, so the measured crossover will be a fuzzy band rather than a sharp point.

**Prediction P5: at $n = 4096$, Strassen with the best cutoff is 1.5–1.8× faster than blocked.** From the model: $1/(0.513 + 0.011r)$ gives 1.76× at $r = 5$ and 1.60× at $r = 10$. Allowing for some loss of kernel efficiency at the smaller leaf size pushes this towards the lower end.

## 2.4 Memory hierarchy: what should and shouldn't matter

**Tile size.** Inside a tile, the $T \times T$ block of $B$ is reused across $T$ rows, so it should fit in L1: $8T^2 \le 32$ KB gives $T \le 64$. **P6: the best tile size is between 32 and 64. $T \ge 256$ performs measurably worse because the B tile spills out of L1, and $T = 1$ (unvectorisable) is several times slower than the best.**

**The 96 MB L3.** We originally expected the large L3 to move the crossover, and said so in the project proposal. Working through the cost model changed our view. Algorithm A reads each element of $B$ from far memory only once per $T$ uses. At $T = 64$ that's about 16 flops per byte, so even DRAM bandwidth is barely touched, and A should be insensitive to L3 size. Algorithm B's additions are bandwidth-bound, but only the top level's working set (about $6 \times 8 (n/2)^2$ bytes: 96 MB at $n \approx 2900$) can overflow the L3, and the top level holds only $18(n/2)^2 / 2n^3 = 2.25/n$ of the work, about 0.05% at $n = 4096$. **P7: Algorithm A's GFLOP/s stays within 10% of its peak all the way to $n = 4096$, with no cliff at the point where the matrices outgrow L3 ($3 \times 8n^2 = 96$ MB at $n \approx 2000$). If we artificially shrink the effective L3 (§3 describes a cache-pressure thread that streams through a large buffer on another core), Strassen's speedup over blocked changes by less than 10%.** This prediction is deliberately falsifiable. If the L3 matters more than this, the cost model is missing something.

**Power-of-two strides.** L1 is 8-way set-associative with 64 sets of 64-byte lines, so addresses 4 KB apart map to the same set. When $n$ is a multiple of 512, consecutive rows of a $B$ tile are $8n \ge 4$ KB apart, and the $T$ rows of a tile all compete for the same few sets. At $T = 64$ that's 64 lines for 8 ways, so the tile cannot actually stay in L1. **P8: Algorithm A runs 10–40% slower per flop at $n = 1024, 2048, 4096$ than at $n + 8$, and padding the row stride (leading dimension) by one cache line removes the penalty.** Strassen's quadrant views and power-of-two-sized temporaries inherit the same strides, so **it suffers similarly**.

## 2.5 Padding

**P9 (`nextpow2`):** at $n = 2^k + 1$, Strassen does one more level of work than at $2^k$, so it takes about $7\times$ longer than at $n = 2^k$. At $n = 1025$ that should make it about 4× slower than blocked, which is a complete loss. Blocked shows no step at all, because $(1 + 1/n)^3 \approx 1$.

**P10 (`minimal`):** the padding overhead is a sawtooth in $n$. Its worst case, just past a size of the form $s \cdot 2^d$, is $(m/n)^{2.807}$ plus the $O(n^2)$ copies, which is at most about 10% at $c = 64$ (1025 pads to 1056: $1.030^{2.807} = 1.087$). So with minimal padding, Strassen's advantage over blocked should survive at every size, not just at powers of two.

## 2.6 Memory and allocation

**P11:** blocked needs no extra memory. Strassen needs about $n^2$ extra doubles of workspace (128 MB at $n = 4096$), plus $3m^2$ for padded copies when $m \ne n$. The cost of *allocating* that memory is mostly first-touch page faults: about 32,000 4 KB pages at $n = 4096$, at a fraction of a microsecond each, so roughly 5–20 ms against a run time of several seconds. **The difference between `strassen::multiply` (which allocates) and a reused `Plan` will be under 2% for $n \ge 1024$, but noticeable (over 5%) for $n \le 256$.**

## 2.7 Numerical accuracy

The classical algorithm satisfies a componentwise bound, $|C - \hat C| \le \gamma_n |A||B|$ with $\gamma_n = nu/(1-nu)$ and unit roundoff $u = 2^{-53}$. Every entry is accurate relative to the sizes of the terms that make it up. Strassen satisfies only a *normwise* bound, due to Brent (1970) and proved in Higham (2002, §23.2.2), here in the max norm:

$$\|C - \hat C\|_{\max} \le \left[\left(\tfrac{n}{c}\right)^{\log_2 12}\left(c^2 + 5c\right) - 5n\right] u\,\|A\|_{\max}\|B\|_{\max} + O(u^2),$$

with $\log_2 12 \approx 3.585$. That error is relative to the *largest* entries of $A$ and $B$, not to the entries that actually contribute to a given $c_{ij}$. This happens because operands like $A_{11} + A_{22}$ mix entries of different magnitudes before multiplying.

- **P12 (`smallint`):** both algorithms are exact. This is already confirmed by the test suite, since it's a correctness property rather than a measurement.
- **P13 (`uniform`):** with entries of similar magnitude the normwise bound isn't hurt by mixing. Strassen's Frobenius-relative error will be larger than blocked's by a factor that grows with depth: about 3–10× at $d = 3$ and about 10–100× at $d = 5$–$7$. Both remain below $10^{-12}$.
- **P14 (`widerange`, entries spanning $2^{\pm 20}$):** this is where the bound's weakness shows. Measured *componentwise*, Strassen's worst-case relative error will be at least $10^3\times$ worse than blocked's at $d \ge 3$, and some small entries of $C$ will have no correct digits at all. Measured by Frobenius norm, which is dominated by the large entries, the gap will look modest. This is why §3 reports both metrics.

## 2.8 Summary of predictions

| # | Prediction | Confidence |
|---|---|---|
| P1 | Blocked sustains 10–20 GFLOP/s for $n \ge 256$ | Medium |
| P2 | Optimal Strassen cutoff $c^*$ is 64 or 128; flat (±10%) within one power of two | Medium |
| P3 | $c^*$ does not depend on $n$ (512–4096) | Medium |
| P4 | Strassen (best $c$) wins from about $n = 256$; margin under 5% near the crossover | Medium |
| P5 | Strassen 1.5–1.8× faster than blocked at $n = 4096$ | Medium |
| P6 | Best tile $T$ in 32–64; $T \ge 256$ worse; $T = 1$ several times slower | High |
| P7 | No L3 cliff for blocked; shrinking effective L3 changes Strassen's speedup by under 10% | Low (deliberately falsifiable) |
| P8 | Power-of-two $n$ costs blocked 10–40% per flop; fixed by stride padding; Strassen also affected | Medium |
| P9 | `nextpow2`: about 7× jump at $2^k+1$; Strassen about 4× slower than blocked at 1025 | High |
| P10 | `minimal` padding overhead under about 10%; Strassen's advantage survives at all sizes | High |
| P11 | Allocation cost under 2% for $n \ge 1024$, over 5% for $n \le 256$ | Low |
| P12 | `smallint`: both exact | Certain (tested) |
| P13 | `uniform`: Strassen error about 3–100× blocked depending on depth; both $< 10^{-12}$ | Medium |
| P14 | `widerange`: componentwise error over $10^3\times$ worse for Strassen; Frobenius gap modest | Medium |
