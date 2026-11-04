// experiments.hpp - settings, experiment plans and the runner.
//
// Design rules (carried over from the PA1 harness, adapted to matrix multiplication):
//   - every input matrix is generated, from a recorded seed, before any timing
//     window opens; every algorithm in a configuration gets the same instances;
//   - no allocation, I/O or formatting inside a timed region (Strassen Plans and
//     output matrices are built beforehand; experiment E7 deliberately times
//     allocation, and says so in its `alloc` column);
//   - one untimed warm-up call per algorithm per configuration;
//   - every output is consumed (checksummed) and compared against the first
//     variant of its configuration, so dead-code elimination is impossible and
//     disagreement between algorithms would be visible in the CSV;
//   - algorithm order is rotated every repetition, so slow drift (thermal,
//     background load) affects all algorithms equally;
//   - timer resolution and overhead are measured and reported, not subtracted;
//   - each experiment's estimated and actual runtime is printed and recorded.
// Every number the report quotes lives in Settings.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "algo_strassen/strassen.hpp"
#include "environment.hpp"

namespace bench {

struct Settings {
    bool quick = false;
    bool skip_large = false;          // drop every configuration with n > 2048
    int reps = 7;                     // timed repetitions per algorithm per configuration
    int pilot_reps = 30;              // E0
    double min_sample_s = 0.02;       // each timed sample loops the call until >= this long
    std::uint64_t base_seed = 20261104;
    unsigned bench_cpu = 2;           // logical CPU for the benchmark thread (physical core 1)
    unsigned antagonist_cpu = 6;      // logical CPU for E6's cache-pressure thread (physical core 3)

    // E0 pilot: variance at fixed, pre-tuning parameters.
    std::vector<std::size_t> e0_sizes{512, 1000};
    std::size_t e0_tile = 64, e0_cutoff = 128;
    // E1 tile sweep (blocked). T = 1 is only run at e1_t1_size (it is very slow).
    std::vector<std::size_t> e1_sizes{256, 1000, 2000};
    std::vector<std::size_t> e1_tiles{8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    std::size_t e1_t1_size = 256;
    // E2 cutoff sweep (Strassen, minimal padding, tuned tile).
    std::vector<std::size_t> e2_sizes{512, 1024, 2048, 4096};
    std::vector<std::size_t> e2_cutoffs{16, 32, 64, 128, 256, 512};
    // E3 scaling / crossover.
    std::vector<std::size_t> e3_sizes{64,  96,  128, 160,  192,  224,  256,  320,  384,  448,  512, 640,
                                      768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096};
    // E4 padding: n = 2^k - 1, 2^k, 2^k + 1 for each k, plus a dense sweep.
    std::vector<int> e4_k{9, 10, 11};
    std::size_t e4_dense_from = 1000, e4_dense_to = 1100, e4_dense_step = 4;
    int e4_dense_reps = 3;
    // E5 leading-dimension aliasing.
    std::vector<std::size_t> e5_sizes{512, 1024, 2048};
    std::size_t e5_pad = 8;           // elements (8 doubles = one cache line)
    // E6 cache pressure (MB occupied by the antagonist; 0 = off, 1 = L2-resident control).
    std::vector<std::size_t> e6_sizes{1000, 2000, 3000};
    std::vector<std::size_t> e6_mb{0, 1, 32, 64, 80};
    // E7 allocation cost.
    std::vector<std::size_t> e7_sizes{128, 256, 512, 1024, 2048, 4096};
    // E8 accuracy (no timing).
    std::vector<std::size_t> e8_sizes{128, 256, 512, 1024};
    std::vector<std::size_t> e8_cutoffs{8, 32, 128};
    int e8_seeds = 3;
    int e8_max_exp = 20;              // widerange: entries span 2^-20 .. 2^21
    std::int64_t e8_int_bound = 8;    // smallint

    // Overrides for the tuned parameters (0 = take them from E1/E2 or tuned_params.txt).
    std::size_t tile_override = 0, cutoff_override = 0;
    std::string results_dir;
};

Settings quick_settings();  // same code paths, tiny sizes: a smoke test, not for analysis

struct Calibration {
    double blocked_flops_per_s = 1e10;  // measured blocked kernel rate
    double dd_seconds_per_n3 = 1e-8;    // double-double reference cost per n^3
};

struct Tuned {
    std::size_t tile = 0, cutoff = 0;   // 0 = unknown
    std::string tile_source, cutoff_source;
};

struct Context {
    Settings s;
    Environment env;
    Calibration cal;
    Tuned tuned;
    std::ostream* log;
    std::string mode() const { return s.quick ? "quick" : "full"; }

    // Drift monitor: a fixed reference workload timed before every configuration.
    mm::Matrix ref_A, ref_B, ref_C;
    double ref_baseline_gflops = 0.0;           // first measurement of the run
    double exp_ref_min = 0.0, exp_ref_max = 0.0;  // range within the current experiment
};

// Drift monitor. Times a fixed workload (blocked, n = 384, T = 64, looped to at
// least 50 ms) and returns its GFLOP/s. The first call of a run sets the
// baseline; any later reading more than 5% away from it is reported as a
// warning. Every timing row records the reading taken just before its
// configuration, so drift in machine state (thermal, background load) is
// measured rather than assumed absent.
double measure_reference(Context& ctx);
// Sets the baseline from a warmed-up machine: discards the first readings
// (clock ramp-up, first-touch page faults) and takes the median of the rest.
void establish_reference_baseline(Context& ctx);
constexpr double kDriftWarnFraction = 0.05;

struct ExperimentInfo {
    std::string id, title, csv;
};
const std::vector<ExperimentInfo>& all_experiments();

Calibration calibrate();
double estimate_seconds(const Context& ctx, const std::string& id);
void run_experiment(Context& ctx, const std::string& id);  // writes <results_dir>/<csv>

// Differential check of both algorithms against the naive reference, run before
// any experiment. Returns false (after printing what failed) on any mismatch.
bool correctness_gate(std::ostream& log);

Tuned load_tuned(const std::string& results_dir);
void save_tuned(const std::string& results_dir, const Tuned& t);

std::string timestamp();
std::string format_duration(double seconds);

}  // namespace bench
