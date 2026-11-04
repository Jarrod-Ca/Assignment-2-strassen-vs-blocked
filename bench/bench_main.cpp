// mm_bench - the unified benchmark harness.
//
//   mm_bench                     correctness gate, then every experiment (E0..E8)
//   mm_bench --quick             same pipeline at tiny sizes (smoke test, not for analysis)
//   mm_bench --estimate          print the per-experiment runtime estimate and exit
//   mm_bench --only E3,E4        run selected experiments (tuned values come from
//                                results/tuned_params.txt or --tile / --cutoff)
//
// Run `mm_bench --help` for all options.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "build_info.hpp"
#include "environment.hpp"
#include "experiments.hpp"

namespace {

void usage() {
    std::cout <<
        "usage: mm_bench [options]\n"
        "  --quick              tiny sizes, whole pipeline in seconds (results/quick/)\n"
        "  --estimate           print estimated runtime per experiment, then exit\n"
        "  --only LIST          comma-separated experiments, e.g. E0 or E3,E5 (default: all)\n"
        "  --skip-large         skip every configuration with n > 2048\n"
        "  --reps N             timed repetitions per configuration (default 7)\n"
        "  --tile T             override the tuned tile size\n"
        "  --cutoff C           override the tuned Strassen cutoff\n"
        "  --bench-cpu N        logical CPU for the benchmark thread (default 2)\n"
        "  --antagonist-cpu N   logical CPU for the E6 cache-pressure thread (default 6)\n"
        "  --out DIR            results directory (default <repo>/results)\n";
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back(item);
    return out;
}

void write_run_header(std::ostream& o, const bench::Context& ctx, const std::vector<std::string>& ids,
                      bool pinned, const std::string& priority) {
    const auto& e = ctx.env;
    const auto& s = ctx.s;
    o << "==== run started " << bench::timestamp() << " ====\n"
      << "mode: " << (s.quick ? "QUICK SMOKE TEST - not for the report" : "full") << "\n"
      << "experiments:";
    for (const auto& id : ids) o << " " << id;
    o << "\ncpu: " << e.cpu << "\nlogical_cpus: " << e.logical_cpus << "\nos: " << e.os
      << "\nram_gb: " << std::fixed << std::setprecision(1) << e.ram_gb << std::defaultfloat << std::setprecision(6)
      << "\ncompiler: " << e.compiler << "\nbuild: " << e.build_type << "\nflags: " << e.build_flags
      << "\ngit_commit: " << e.git_commit << (e.git_dirty ? " (DIRTY: uncommitted changes)" : "")
      << "\npower_plan: " << e.power_plan << "\ntimer: steady_clock tick " << e.timer_tick_ns
      << " ns, resolution " << e.timer_resolution_ns << " ns, overhead " << e.timer_overhead_ns << " ns/call"
      << "\nbench_cpu: " << s.bench_cpu << (pinned ? " (pinned)" : " (NOT pinned)")
      << "\nantagonist_cpu: " << s.antagonist_cpu << "\npriority: " << priority << "\nreps: " << s.reps
      << "\npilot_reps: " << s.pilot_reps << "\nmin_sample_s: " << s.min_sample_s << "\nbase_seed: " << s.base_seed
      << "\nskip_large: " << (s.skip_large ? "yes" : "no")
      << "\ncalibration: blocked " << ctx.cal.blocked_flops_per_s / 1e9 << " GFLOP/s, dd reference "
      << ctx.cal.dd_seconds_per_n3 * 1e9 << " ns per n^3"
      << "\ndrift_monitor_baseline: " << ctx.ref_baseline_gflops << " GFLOP/s (blocked n=384 T=64; warn at +/-"
      << 100 * bench::kDriftWarnFraction << "%)\n";
}

}  // namespace

int main(int argc, char** argv) {
    bench::Settings s;
    bool quick = false, estimate_only = false, skip_large = false;
    std::vector<std::string> only;
    std::string out_dir;
    int reps = 0;
    std::size_t tile = 0, cutoff = 0;
    unsigned bench_cpu = s.bench_cpu, antagonist_cpu = s.antagonist_cpu;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a);
                return argv[++i];
            };
            if (a == "--quick") quick = true;
            else if (a == "--estimate") estimate_only = true;
            else if (a == "--skip-large") skip_large = true;
            else if (a == "--only") only = split(value());
            else if (a == "--reps") reps = std::stoi(value());
            else if (a == "--tile") tile = std::stoul(value());
            else if (a == "--cutoff") cutoff = std::stoul(value());
            else if (a == "--bench-cpu") bench_cpu = static_cast<unsigned>(std::stoul(value()));
            else if (a == "--antagonist-cpu") antagonist_cpu = static_cast<unsigned>(std::stoul(value()));
            else if (a == "--out") out_dir = value();
            else if (a == "--help" || a == "-h") { usage(); return 0; }
            else throw std::invalid_argument("unknown option " + a);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage();
        return 2;
    }

    if (quick) s = bench::quick_settings();
    s.skip_large = skip_large;
    if (reps > 0) s.reps = reps;
    s.bench_cpu = bench_cpu;
    s.antagonist_cpu = antagonist_cpu;
    s.tile_override = tile;
    s.cutoff_override = cutoff;
    s.results_dir = !out_dir.empty() ? out_dir
                                     : (std::filesystem::path(MM_SOURCE_DIR) / "results" / (quick ? "quick" : ""))
                                           .lexically_normal()
                                           .string();

    std::vector<std::string> ids;
    for (const auto& e : bench::all_experiments()) ids.push_back(e.id);
    if (!only.empty()) {
        for (const auto& id : only)
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                std::cerr << "error: unknown experiment " << id << "\n";
                return 2;
            }
        ids = only;
    }

    std::ostream& log = std::cout;
    if (!bench::release_build() && !quick && !estimate_only) {
        log << "Refusing to run the experiments in a Debug build: the timings would be meaningless.\n"
               "Build the windows-release (or linux-release) preset, or use --quick for a smoke test.\n";
        return 1;
    }

    bench::Context ctx;
    ctx.s = s;
    ctx.log = &log;
    const bool pinned = bench::pin_current_thread(s.bench_cpu);
    const std::string priority = bench::raise_priority();
    log << "Capturing environment and calibrating...\n";
    ctx.env = bench::capture_environment();
    ctx.cal = bench::calibrate();
    bench::establish_reference_baseline(ctx);  // drift-monitor baseline for this run
    std::filesystem::create_directories(s.results_dir);
    ctx.tuned = bench::load_tuned(s.results_dir);
    if (tile) { ctx.tuned.tile = tile; ctx.tuned.tile_source = "--tile override"; }
    if (cutoff) { ctx.tuned.cutoff = cutoff; ctx.tuned.cutoff_source = "--cutoff override"; }
    if (!pinned) log << "WARNING: could not pin the benchmark thread to CPU " << s.bench_cpu << "\n";
    if (ctx.env.git_dirty && !quick)
        log << "WARNING: the working tree has uncommitted changes; results will not match commit "
            << ctx.env.git_commit << "\n";

    // ---- estimates --------------------------------------------------------
    std::vector<double> est;
    double est_total = 0.0;
    {
        std::ostringstream line;
        line << std::fixed << std::setprecision(1) << ctx.cal.blocked_flops_per_s / 1e9;
        log << "\nEstimated runtime (calibrated blocked rate " << line.str() << " GFLOP/s):\n";
    }
    for (const auto& id : ids) {
        const auto& info = *std::find_if(bench::all_experiments().begin(), bench::all_experiments().end(),
                                         [&](const bench::ExperimentInfo& e) { return e.id == id; });
        est.push_back(bench::estimate_seconds(ctx, id));
        est_total += est.back();
        log << "  " << id << "  " << std::left << std::setw(50) << info.title << std::right << std::setw(10)
            << bench::format_duration(est.back()) << "\n";
    }
    log << "  " << std::left << std::setw(54) << "total" << std::right << std::setw(10)
        << bench::format_duration(est_total) << "\n"
        << "  (estimates assume tile 64 / cutoff 128 until E1/E2 have run)\n\n";
    if (estimate_only) return 0;

    // ---- correctness gate -------------------------------------------------
    if (!bench::correctness_gate(log)) {
        log << "Correctness gate FAILED - no experiment was run.\n";
        return 1;
    }

    // ---- run --------------------------------------------------------------
    const std::filesystem::path dir(s.results_dir);
    {
        std::ofstream info(dir / "run_info.txt", std::ios::app);
        write_run_header(info, ctx, ids, pinned, priority);
    }
    const bool new_runtimes = !std::filesystem::exists(dir / "runtimes.csv");
    std::ofstream runtimes(dir / "runtimes.csv", std::ios::app);
    if (new_runtimes)
        runtimes << "finished_at,git_commit,mode,experiment,estimated_s,actual_s,ref_gflops_min,ref_gflops_max\n";

    std::vector<double> actual, ref_min, ref_max;
    const double run_start = bench::now_seconds();
    try {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& info = *std::find_if(bench::all_experiments().begin(), bench::all_experiments().end(),
                                             [&](const bench::ExperimentInfo& e) { return e.id == ids[i]; });
            log << "[" << ids[i] << "] " << info.title << "  (estimated " << bench::format_duration(est[i]) << ")\n";
            const double t0 = bench::now_seconds();
            ctx.exp_ref_min = ctx.exp_ref_max = 0.0;
            bench::run_experiment(ctx, ids[i]);
            actual.push_back(bench::now_seconds() - t0);
            log << "[" << ids[i] << "] finished in " << bench::format_duration(actual.back()) << " (estimated "
                << bench::format_duration(est[i]) << ")\n\n";
            ref_min.push_back(ctx.exp_ref_min);
            ref_max.push_back(ctx.exp_ref_max);
            runtimes << bench::timestamp() << "," << ctx.env.git_commit << "," << ctx.mode() << "," << ids[i] << ","
                     << est[i] << "," << actual.back() << ","
                     << (ref_max.back() > 0 ? std::to_string(ref_min.back()) : "") << ","
                     << (ref_max.back() > 0 ? std::to_string(ref_max.back()) : "") << std::endl;
        }
    } catch (const std::exception& e) {
        log << "ERROR: " << e.what() << "\n";
        return 1;
    }
    const double total = bench::now_seconds() - run_start;

    std::ofstream info(dir / "run_info.txt", std::ios::app);
    std::ostringstream summary;
    summary << "tuned_tile: " << ctx.tuned.tile << " (" << ctx.tuned.tile_source << ")\n"
            << "tuned_cutoff: " << ctx.tuned.cutoff << " (" << ctx.tuned.cutoff_source << ")\n"
            << "runtime per experiment (actual / estimated)   drift monitor range (GFLOP/s, % of baseline):\n";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        summary << "  " << ids[i] << "  " << std::setw(10) << bench::format_duration(actual[i]) << " / "
                << std::left << std::setw(10) << bench::format_duration(est[i]) << std::right;
        if (ref_max[i] > 0) {
            std::ostringstream r;
            r << std::fixed << std::setprecision(2) << ref_min[i] << " - " << ref_max[i] << "  ("
              << std::setprecision(1) << 100 * ref_min[i] / ctx.ref_baseline_gflops << "% - "
              << 100 * ref_max[i] / ctx.ref_baseline_gflops << "%)";
            summary << "   " << r.str();
        }
        summary << "\n";
    }
    summary << "  total " << bench::format_duration(total) << " / " << bench::format_duration(est_total) << "\n"
            << "==== run finished " << bench::timestamp() << " ====\n\n";
    info << summary.str();
    log << summary.str() << "Results in " << s.results_dir << "\n";
    return 0;
}
