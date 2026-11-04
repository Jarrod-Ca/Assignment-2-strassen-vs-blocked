#include "experiments.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "algo_blocked/blocked.hpp"
#include "algo_strassen/strassen.hpp"
#include "cache_pressure.hpp"
#include "common/compare.hpp"
#include "common/generators.hpp"
#include "common/rng.hpp"
#include "reference/naive.hpp"

namespace bench {

using mm::ConstMatrixView;
using mm::Matrix;
namespace strassen = mm::strassen;
namespace blocked = mm::blocked;

// ===========================================================================
// Small utilities
// ===========================================================================
std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
}

std::string format_duration(double seconds) {
    const long long s = static_cast<long long>(std::llround(seconds));
    std::ostringstream os;
    if (s >= 3600) os << s / 3600 << "h " << std::setw(2) << std::setfill('0') << (s % 3600) / 60 << "m";
    else if (s >= 60) os << s / 60 << "m " << std::setw(2) << std::setfill('0') << s % 60 << "s";
    else os << std::fixed << std::setprecision(1) << seconds << "s";
    return os.str();
}

namespace {

std::string fmt(double x) {
    std::ostringstream os;
    os << std::setprecision(10) << x;
    return os.str();
}

class CsvWriter {
public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header) : out_(path) {
        if (!out_) throw std::runtime_error("cannot open " + path);
        row(header);
    }
    void row(const std::vector<std::string>& cells) {
        for (std::size_t i = 0; i < cells.size(); ++i) out_ << (i ? "," : "") << cells[i];
        out_ << "\n";
        out_.flush();  // an interrupted run still leaves every completed row on disk
    }

private:
    std::ofstream out_;
};

// Human-readable duration with a sensible unit (for the console only; CSVs store seconds).
std::string format_time(double s) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    if (s < 1e-3) os << s * 1e6 << " us";
    else if (s < 1.0) os << s * 1e3 << " ms";
    else os << s << " s";
    return os.str();
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double cube(std::size_t n) { return static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n); }

// Inputs for size n: identical in every experiment that uses that n.
std::uint64_t seed_a(const Settings& s, std::size_t n) { return mm::derive_seed(s.base_seed, 2 * n); }
std::uint64_t seed_b(const Settings& s, std::size_t n) { return mm::derive_seed(s.base_seed, 2 * n + 1); }

// ===========================================================================
// Declarative experiment plans. Estimation and execution both read these, so
// the estimate always describes exactly what will run.
// ===========================================================================
constexpr std::size_t kTuned = 0;  // "use the tuned value"

struct VariantSpec {
    std::string algorithm;  // "blocked" | "strassen"
    std::string label;
    std::size_t tile = kTuned;
    std::size_t cutoff = kTuned;
    strassen::Padding padding = strassen::Padding::Minimal;
    std::size_t temp_pad = 0;
    std::size_t leaf_align = 4;    // Aligned padding only
    bool alloc_each_call = false;  // strassen::multiply (allocates) instead of a reused Plan
};

struct Config {
    std::string tag;
    std::size_t n = 0;
    std::size_t stride = 0;  // 0 = tightly packed
    std::size_t antagonist_mb = 0;
    int reps = 0;
    std::vector<VariantSpec> variants;
};

VariantSpec blocked_spec(std::size_t tile = kTuned, std::string label = "") {
    VariantSpec v;
    v.algorithm = "blocked";
    v.tile = tile;
    v.label = label.empty() ? (tile == kTuned ? "tuned" : "T=" + std::to_string(tile)) : label;
    return v;
}

VariantSpec strassen_spec(std::size_t cutoff = kTuned, strassen::Padding pad = strassen::Padding::Minimal,
                          std::size_t temp_pad = 0, std::string label = "") {
    VariantSpec v;
    v.algorithm = "strassen";
    v.cutoff = cutoff;
    v.padding = pad;
    v.temp_pad = temp_pad;
    if (label.empty()) {
        label = cutoff == kTuned ? "tuned" : "c=" + std::to_string(cutoff);
        if (pad == strassen::Padding::NextPow2) label += "+nextpow2";
        if (pad == strassen::Padding::Aligned) label += "+aligned";
        if (temp_pad) label += "+temp_pad=" + std::to_string(temp_pad);
    }
    v.label = label;
    return v;
}

bool keep(const Settings& s, std::size_t n) { return !(s.skip_large && n > 2048); }

std::vector<Config> plan_for(const Settings& s, const std::string& id) {
    std::vector<Config> plan;
    auto add = [&](Config c) {
        if (keep(s, c.n)) plan.push_back(std::move(c));
    };
    if (id == "E0") {
        for (std::size_t n : s.e0_sizes)
        {
            VariantSpec st = strassen_spec(s.e0_cutoff);
            st.tile = s.e0_tile;  // pre-tuning: fixed base tile, stated in the CSV's tile column
            add({"pilot", n, 0, 0, s.pilot_reps, {blocked_spec(s.e0_tile), st}});
        }
    } else if (id == "E1") {
        for (std::size_t n : s.e1_sizes) {
            Config c{"tile", n, 0, 0, s.reps, {}};
            if (n == s.e1_t1_size) c.variants.push_back(blocked_spec(1));
            for (std::size_t t : s.e1_tiles)
                if (t < n) c.variants.push_back(blocked_spec(t));
            c.variants.push_back(blocked_spec(n, "untiled"));
            add(c);
        }
    } else if (id == "E2") {
        for (std::size_t n : s.e2_sizes) {
            Config c{"cutoff", n, 0, 0, s.reps, {blocked_spec()}};
            for (std::size_t cut : s.e2_cutoffs) c.variants.push_back(strassen_spec(cut));
            add(c);
        }
    } else if (id == "E3") {
        for (std::size_t n : s.e3_sizes) add({"scaling", n, 0, 0, s.reps, {blocked_spec(), strassen_spec()}});
    } else if (id == "E4") {
        const std::vector<VariantSpec> v{blocked_spec(), strassen_spec(),
                                         strassen_spec(kTuned, strassen::Padding::NextPow2)};
        for (int k : s.e4_k) {
            const std::size_t p = std::size_t{1} << k;
            for (std::size_t n : {p - 1, p, p + 1}) add({"edge_k" + std::to_string(k), n, 0, 0, s.reps, v});
        }
        for (std::size_t n = s.e4_dense_from; n <= s.e4_dense_to; n += s.e4_dense_step)
            add({"dense", n, 0, 0, s.e4_dense_reps, v});
    } else if (id == "E5") {
        const std::vector<VariantSpec> v{blocked_spec(), strassen_spec(),
                                         strassen_spec(kTuned, strassen::Padding::Minimal, s.e5_pad)};
        for (std::size_t n : s.e5_sizes) {
            add({"packed", n, 0, 0, s.reps, v});
            add({"ld+" + std::to_string(s.e5_pad), n, n + s.e5_pad, 0, s.reps, v});
            add({"n+" + std::to_string(s.e5_pad), n + s.e5_pad, 0, 0, s.reps, v});
        }
    } else if (id == "E6") {
        // n outer, pressure level inner: with round-robin execution (see run_experiment)
        // every pressure level of one size is measured within seconds of the others.
        for (std::size_t n : s.e6_sizes)
            for (std::size_t mb : s.e6_mb)
                add({"pressure_" + std::to_string(mb) + "MB", n, 0, mb, s.reps, {blocked_spec(), strassen_spec()}});
    } else if (id == "E7") {
        for (std::size_t n : s.e7_sizes) {
            VariantSpec alloc = strassen_spec(kTuned, strassen::Padding::Minimal, 0, "tuned+alloc_each_call");
            alloc.alloc_each_call = true;
            add({"allocation", n, 0, 0, s.reps, {blocked_spec(), strassen_spec(kTuned, strassen::Padding::Minimal, 0, "tuned+plan_reused"), alloc}});
        }
    } else if (id == "E9") {
        std::vector<VariantSpec> v{blocked_spec(), strassen_spec()};
        for (std::size_t a : s.e9_aligns) {
            VariantSpec al = strassen_spec(kTuned, strassen::Padding::Aligned, 0, "tuned+aligned" + std::to_string(a));
            al.leaf_align = a;
            v.push_back(al);
        }
        for (const auto& r : s.e9_ranges)
            for (std::size_t n = r[0]; n <= r[1]; n += r[2])
                add({"align_" + std::to_string(r[0]), n, 0, 0, s.reps, v});
    } else if (id != "E8") {
        throw std::invalid_argument("unknown experiment " + id);
    }
    return plan;
}

std::size_t resolve_tile(const Context& ctx, std::size_t t) { return t == kTuned ? ctx.tuned.tile : t; }
std::size_t resolve_cutoff(const Context& ctx, std::size_t c) { return c == kTuned ? ctx.tuned.cutoff : c; }

strassen::Params strassen_params(const Context& ctx, const VariantSpec& v) {
    strassen::Params p;
    p.cutoff = resolve_cutoff(ctx, v.cutoff);
    p.padding = v.padding;
    p.temp_pad = v.temp_pad;
    p.leaf_align = v.leaf_align;
    p.base.tile = resolve_tile(ctx, v.tile);  // VariantSpec::tile is Strassen's base-case tile
    return p;
}

// ---------------------------------------------------------------------------
// Runtime model used only for the printed estimate.
// ---------------------------------------------------------------------------
double estimate_call(const Context& ctx, const VariantSpec& v, std::size_t n) {
    const double rate = ctx.cal.blocked_flops_per_s;
    if (v.algorithm == "blocked") {
        const std::size_t t = v.tile == kTuned ? 64 : v.tile;
        double factor = 1.0;
        if (t == 1) factor = 12.0;            // scalar inner loop
        else if (t >= n && n > 256) factor = 3.0;  // untiled: B streamed from far memory
        else if (t < 16) factor = 1.6;
        return factor * 2.0 * cube(n) / rate;
    }
    strassen::Params p;
    p.cutoff = v.cutoff == kTuned ? 128 : v.cutoff;
    p.padding = v.padding;
    p.temp_pad = v.temp_pad;
    p.leaf_align = v.leaf_align;
    const strassen::Stats st = strassen::predict(n, p);
    // Additions are memory-bound: assume ~8x the per-flop cost of the kernel.
    return (static_cast<double>(st.base_flops) + 8.0 * static_cast<double>(st.add_flops + st.copy_elems)) / rate;
}

constexpr double kReferenceSeconds = 0.05;  // drift-monitor workload length

bool interleaved(const std::string& id) { return id == "E5" || id == "E6"; }

double estimate_plan(const Context& ctx, const std::vector<Config>& plan, bool rounds) {
    double total = 0.0;
    for (const Config& c : plan) {
        const int setups = rounds ? c.reps : 1;  // round-robin re-creates the configuration every round
        double per_setup = kReferenceSeconds + 0.01 + 1e-8 * static_cast<double>(c.n) * c.n;
        if (c.antagonist_mb) per_setup += 0.05 + 3.0 * static_cast<double>(c.antagonist_mb) / 20000.0;
        total += setups * per_setup;
        for (const VariantSpec& v : c.variants) {
            const double t = estimate_call(ctx, v, c.n);
            total += setups * t + c.reps * std::max(t, ctx.s.min_sample_s);
        }
    }
    return total;
}

double estimate_e8(const Context& ctx) {
    const Settings& s = ctx.s;
    double total = 0.0;
    for (std::size_t n : s.e8_sizes) {
        if (!keep(s, n)) continue;
        double per = ctx.cal.dd_seconds_per_n3 * cube(n) + 2.0 * 2.0 * cube(n) / ctx.cal.blocked_flops_per_s;
        for (std::size_t c : s.e8_cutoffs) per += estimate_call(ctx, strassen_spec(c), n);
        total += 3.0 * s.e8_seeds * per;  // three input families
    }
    return total;
}

// ===========================================================================
// The generic timing runner.
// ===========================================================================
struct Live {
    VariantSpec spec;
    std::size_t tile = 0, cutoff = 0, padded_n = 0, depth = 0;
    double flops = 0.0;  // flops actually executed per call (padding included)
    Matrix C;
    std::unique_ptr<strassen::Plan> plan;
    std::function<void()> run;
    int inner = 1;
    double rel_diff = 0.0;
    std::vector<double> times;
};

const std::vector<std::string> kTimingHeader{
    "experiment", "tag",      "algorithm",     "variant",       "n",        "stride",     "padded_n",
    "depth",      "tile",     "cutoff",        "padding",       "temp_pad", "alloc",      "antagonist_mb",
    "antagonist_pinned",      "family",        "seed",          "rep",      "order",      "inner_calls",
    "seconds",    "eff_gflops", "actual_gflops", "checksum",    "rel_diff_vs_first", "ref_gflops"};

struct ConfigResult {
    Config cfg;
    std::vector<std::string> labels;          // "algorithm:variant"
    std::vector<double> medians;              // filled by finalize()
    std::vector<std::vector<double>> times;   // per-repetition seconds, in repetition order
    std::vector<double> ref_gflops;           // drift-monitor readings taken for this configuration
};

void finalize(ConfigResult& r) {
    r.medians.clear();
    for (const auto& t : r.times) r.medians.push_back(median(t));
}

void print_result(std::ostream& log, const ConfigResult& r, double baseline) {
    std::ostringstream head;
    head << "  " << std::left << std::setw(16) << r.cfg.tag << " n=" << r.cfg.n;
    if (r.cfg.stride) head << "  stride=" << r.cfg.stride;
    if (r.cfg.antagonist_mb) head << "  antagonist=" << r.cfg.antagonist_mb << " MB";
    if (!r.ref_gflops.empty()) {
        const double ref = median(r.ref_gflops);
        head << "   [ref " << std::fixed << std::setprecision(2) << ref << " GFLOP/s, " << std::showpos
             << std::setprecision(1) << 100.0 * (ref / baseline - 1.0) << "% vs run start]";
    }
    log << head.str() << "\n";
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
        const auto colon = r.labels[i].find(':');
        std::ostringstream line;  // format locally so no stream state leaks into the log
        line << "      " << std::left << std::setw(9) << r.labels[i].substr(0, colon) << std::setw(28)
             << r.labels[i].substr(colon + 1) << std::right << std::setw(12) << format_time(r.medians[i]) << "  "
             << std::fixed << std::setprecision(2) << std::setw(7)
             << 2.0 * cube(r.cfg.n) / r.medians[i] / 1e9 << " eff GFLOP/s";
        log << line.str() << "\n";
    }
}

// Runs one configuration: `cfg.reps` interleaved repetitions of every variant.
// `rep_offset` numbers the rows when round-robin execution calls this once per round.
ConfigResult run_config(Context& ctx, const std::string& exp, const Config& cfg, CsvWriter& csv, int rep_offset = 0) {
    std::ostream& log = *ctx.log;
    const std::size_t n = cfg.n;

    // Drift monitor first, with no antagonist running.
    const double ref_gflops = measure_reference(ctx);

    CachePressure pressure(cfg.antagonist_mb << 20, ctx.s.antagonist_cpu);
    if (pressure.active() && !pressure.pinned())
        log << "    WARNING: cache-pressure thread could not be pinned to CPU " << ctx.s.antagonist_cpu << "\n";

    const std::uint64_t sa = seed_a(ctx.s, n), sb = seed_b(ctx.s, n);
    const Matrix A = mm::make_uniform(n, n, sa, cfg.stride);
    const Matrix B = mm::make_uniform(n, n, sb, cfg.stride);

    std::vector<std::unique_ptr<Live>> lives;
    for (const VariantSpec& spec : cfg.variants) {
        auto L = std::make_unique<Live>();
        L->spec = spec;
        L->C = Matrix(n, n, cfg.stride);
        Live* l = L.get();
        // Make "tuned" labels self-describing and unique per algorithm, e.g.
        // "tuned_T=192" / "tuned_c=128+nextpow2".
        const bool tuned_label = spec.label.rfind("tuned", 0) == 0;
        if (spec.algorithm == "blocked") {
            l->tile = resolve_tile(ctx, spec.tile);
            if (tuned_label) l->spec.label = "tuned_T=" + std::to_string(l->tile) + spec.label.substr(5);
            l->padded_n = n;
            l->flops = 2.0 * cube(n);
            const blocked::Params bp{l->tile};
            l->run = [&A, &B, l, bp] { blocked::multiply(A, B, l->C, bp); };
        } else {
            const strassen::Params p = strassen_params(ctx, spec);
            const strassen::Stats st = strassen::predict(n, p);
            l->tile = p.base.tile;
            l->cutoff = p.cutoff;
            if (tuned_label) l->spec.label = "tuned_c=" + std::to_string(l->cutoff) + spec.label.substr(5);
            l->padded_n = st.padded_n;
            l->depth = st.depth;
            l->flops = static_cast<double>(st.total_flops());
            if (spec.alloc_each_call) {
                l->run = [&A, &B, l, p] { strassen::multiply(A, B, l->C, p); };
            } else {
                l->plan = std::make_unique<strassen::Plan>(n, p);  // all allocation happens here
                l->run = [&A, &B, l] { l->plan->execute(A, B, l->C); };
            }
        }
        lives.push_back(std::move(L));
    }

    // Warm-up (untimed for the record) also sizes the inner loop.
    for (auto& l : lives) {
        const double t0 = now_seconds();
        l->run();
        const double t = std::max(now_seconds() - t0, 1e-9);
        l->inner = static_cast<int>(std::min(10000.0, std::max(1.0, std::ceil(ctx.s.min_sample_s / t))));
    }
    for (auto& l : lives) l->rel_diff = mm::relative_error_fro(l->C, lives.front()->C);

    const std::size_t V = lives.size();
    for (int rep = 0; rep < cfg.reps; ++rep) {
        for (std::size_t k = 0; k < V; ++k) {
            Live& l = *lives[(k + static_cast<std::size_t>(rep)) % V];
            const double t0 = now_seconds();
            for (int i = 0; i < l.inner; ++i) l.run();
            const double seconds = (now_seconds() - t0) / l.inner;
            l.times.push_back(seconds);
            // --- outside the timed region from here ---
            const double sum = mm::checksum(l.C);
            csv.row({exp,
                     cfg.tag,
                     l.spec.algorithm,
                     l.spec.label,
                     std::to_string(n),
                     std::to_string(cfg.stride ? cfg.stride : n),
                     std::to_string(l.padded_n),
                     std::to_string(l.depth),
                     std::to_string(l.tile),
                     l.spec.algorithm == "strassen" ? std::to_string(l.cutoff) : "",
                     l.spec.algorithm == "strassen" ? strassen::padding_name(l.spec.padding) : "",
                     l.spec.algorithm == "strassen" ? std::to_string(l.spec.temp_pad) : "",
                     l.spec.algorithm == "strassen" ? (l.spec.alloc_each_call ? "each_call" : "plan") : "",
                     std::to_string(cfg.antagonist_mb),
                     pressure.active() ? (pressure.pinned() ? "1" : "0") : "",
                     "uniform",
                     std::to_string(sa),
                     std::to_string(rep_offset + rep),
                     std::to_string(k),
                     std::to_string(l.inner),
                     fmt(seconds),
                     fmt(2.0 * cube(n) / seconds / 1e9),
                     fmt(l.flops / seconds / 1e9),
                     fmt(sum),
                     fmt(l.rel_diff),
                     fmt(ref_gflops)});
        }
    }

    ConfigResult result{cfg, {}, {}, {}, {ref_gflops}};
    for (auto& l : lives) {
        result.labels.push_back(l->spec.algorithm + ":" + l->spec.label);
        result.times.push_back(l->times);
    }
    finalize(result);
    return result;
}

// Geometric-mean-of-normalised-medians selection: for every configuration,
// divide each candidate's median by the best median in that configuration;
// the winner has the smallest geometric mean of those ratios across all
// configurations. Candidates missing from any configuration are ignored.
std::string select_best(const std::vector<ConfigResult>& results, const std::string& prefix,
                        const std::vector<std::string>& exclude, std::ostream& log) {
    std::map<std::string, std::vector<double>> ratios;
    for (const auto& r : results) {
        double best = 1e300;
        for (std::size_t i = 0; i < r.labels.size(); ++i)
            if (r.labels[i].rfind(prefix, 0) == 0) best = std::min(best, r.medians[i]);
        for (std::size_t i = 0; i < r.labels.size(); ++i)
            if (r.labels[i].rfind(prefix, 0) == 0) ratios[r.labels[i]].push_back(r.medians[i] / best);
    }
    std::string winner;
    double winner_score = 1e300;
    log << "  selection (geometric mean of time / best-in-configuration, lower is better):\n";
    for (const auto& [label, v] : ratios) {
        if (std::find(exclude.begin(), exclude.end(), label) != exclude.end()) continue;
        if (v.size() != results.size()) continue;
        double logsum = 0.0;
        for (double x : v) logsum += std::log(x);
        const double score = std::exp(logsum / static_cast<double>(v.size()));
        std::ostringstream line;
        line << "      " << std::left << std::setw(30) << label << std::fixed << std::setprecision(4) << score;
        log << line.str() << "\n";
        if (score < winner_score) { winner_score = score; winner = label; }
    }
    return winner;
}

std::size_t parse_suffix_number(const std::string& label, const std::string& key) {
    const auto p = label.find(key);
    if (p == std::string::npos) throw std::runtime_error("cannot parse " + key + " from " + label);
    return static_cast<std::size_t>(std::stoull(label.substr(p + key.size())));
}

void require_tuned(const Context& ctx, const std::string& id) {
    const bool needs_tile = id != "E0" && id != "E1";
    const bool needs_cutoff = id != "E0" && id != "E1" && id != "E2" && id != "E8";
    if (needs_tile && ctx.tuned.tile == 0)
        throw std::runtime_error(id + " needs the tuned tile size: run E1 first, or pass --tile");
    if (needs_cutoff && ctx.tuned.cutoff == 0)
        throw std::runtime_error(id + " needs the tuned cutoff: run E2 first, or pass --cutoff");
}

// ===========================================================================
// E8 accuracy
// ===========================================================================
void run_e8(Context& ctx, const std::string& path) {
    std::ostream& log = *ctx.log;
    const Settings& s = ctx.s;
    CsvWriter csv(path, {"experiment", "family", "n", "seed", "algorithm", "variant", "tile", "cutoff", "padded_n",
                         "depth", "base_size", "rel_err_fro", "max_abs_err", "normwise_err",
                         "max_componentwise_ratio", "max_entry_rel_err", "gamma_n", "higham_factor"});
    const double u = std::ldexp(1.0, -53);
    const mm::Family families[] = {mm::Family::Uniform, mm::Family::WideRange, mm::Family::SmallInt};
    for (mm::Family fam : families) {
        mm::InputSpec spec;
        spec.family = fam;
        spec.max_exp = s.e8_max_exp;
        spec.int_bound = s.e8_int_bound;
        for (std::size_t n : s.e8_sizes) {
            if (!keep(s, n)) continue;
            for (int seed = 0; seed < s.e8_seeds; ++seed) {
                const std::uint64_t sa = mm::derive_seed(s.base_seed + 8, 2 * (n * 100 + seed));
                const std::uint64_t sb = mm::derive_seed(s.base_seed + 8, 2 * (n * 100 + seed) + 1);
                const Matrix A = mm::make_matrix(spec, n, n, sa), B = mm::make_matrix(spec, n, n, sb);
                Matrix ref(n, n), absA(n, n), absB(n, n), D(n, n), C(n, n);
                mm::reference::reference_multiply_dd(A, B, ref);
                mm::abs_into(A, absA);
                mm::abs_into(B, absB);
                blocked::multiply(absA, absB, D, {ctx.tuned.tile});
                const double norm_ab = mm::max_abs(A) * mm::max_abs(B);
                const double gamma_n = n * u / (1.0 - n * u);

                auto record = [&](const std::string& alg, const std::string& label, std::size_t cutoff,
                                  std::size_t padded, std::size_t depth, std::size_t base, double higham) {
                    const double max_abs = mm::max_abs_diff(C, ref);
                    csv.row({"E8", mm::family_name(fam), std::to_string(n), std::to_string(sa), alg, label,
                             std::to_string(ctx.tuned.tile), cutoff ? std::to_string(cutoff) : "",
                             std::to_string(padded), std::to_string(depth), std::to_string(base),
                             fmt(mm::relative_error_fro(C, ref)), fmt(max_abs),
                             fmt(norm_ab > 0 ? max_abs / norm_ab : 0.0), fmt(mm::max_componentwise_ratio(C, ref, D)),
                             fmt(mm::max_entrywise_relative_error(C, ref)), fmt(gamma_n),
                             higham >= 0.0 ? fmt(higham) : "n/a"});  // Higham's bound is Strassen-specific
                };
                blocked::multiply(A, B, C, {ctx.tuned.tile});
                record("blocked", "tuned_T=" + std::to_string(ctx.tuned.tile), 0, n, 0, n, -1.0);
                for (std::size_t cut : s.e8_cutoffs) {
                    strassen::Params p;
                    p.cutoff = cut;
                    p.base.tile = ctx.tuned.tile;
                    const strassen::Stats st = strassen::multiply(A, B, C, p);
                    // Higham's normwise bound factor: [(m/n0)^log2(12) (n0^2 + 5 n0) - 5m] u
                    const double m = static_cast<double>(st.padded_n), n0 = static_cast<double>(st.base_size);
                    const double higham =
                        st.depth == 0 ? n * u
                                      : (std::pow(m / n0, std::log2(12.0)) * (n0 * n0 + 5.0 * n0) - 5.0 * m) * u;
                    record("strassen", "c=" + std::to_string(cut), cut, st.padded_n, st.depth, st.base_size, higham);
                }
            }
            log << "  " << mm::family_name(fam) << " n=" << n << " done\n";
        }
    }
}

}  // namespace

// ===========================================================================
// Public entry points
// ===========================================================================
Settings quick_settings() {
    Settings s;
    s.quick = true;
    s.reps = 2;
    s.pilot_reps = 4;
    s.min_sample_s = 0.002;
    s.e0_sizes = {64};
    s.e0_tile = 16;
    s.e0_cutoff = 16;
    s.e1_sizes = {32, 100};
    s.e1_tiles = {8, 16, 32};
    s.e1_t1_size = 32;
    s.e2_sizes = {64, 128};
    s.e2_cutoffs = {8, 16, 32};
    s.e3_sizes = {32, 64, 100, 128};
    s.e4_k = {5, 6};
    s.e4_dense_from = 60;
    s.e4_dense_to = 70;
    s.e4_dense_step = 5;
    s.e4_dense_reps = 2;
    s.e5_sizes = {64};
    s.e6_sizes = {64, 100};
    s.e6_mb = {0, 1, 4};
    s.e7_sizes = {32, 64};
    s.e8_sizes = {32, 64};
    s.e8_cutoffs = {4, 16};
    s.e8_seeds = 1;
    s.e9_ranges = {{60, 70, 5}};
    return s;
}

const std::vector<ExperimentInfo>& all_experiments() {
    static const std::vector<ExperimentInfo> list{
        {"E0", "pilot: run-to-run variance at fixed parameters", "E0_pilot.csv"},
        {"E1", "tile-size sweep (blocked)  -> tuned tile", "E1_tile.csv"},
        {"E2", "cutoff sweep (Strassen)    -> tuned cutoff", "E2_cutoff.csv"},
        {"E3", "scaling and crossover", "E3_scaling.csv"},
        {"E4", "padding: nextpow2 vs minimal", "E4_padding.csv"},
        {"E5", "leading-dimension (stride) aliasing", "E5_stride.csv"},
        {"E6", "L3 cache pressure", "E6_cache_pressure.csv"},
        {"E7", "allocation cost: Plan reuse vs allocate per call", "E7_allocation.csv"},
        {"E8", "numerical accuracy (no timing)", "E8_accuracy.csv"},
        {"E9", "follow-up: leaf alignment (aligned vs minimal padding)", "E9_alignment.csv"},
    };
    return list;
}

Calibration calibrate() {
    Calibration cal;
    {
        const std::size_t n = 384;
        const Matrix A = mm::make_uniform(n, n, 1), B = mm::make_uniform(n, n, 2);
        Matrix C(n, n);
        blocked::multiply(A, B, C, {64});  // warm-up
        const double t0 = now_seconds();
        blocked::multiply(A, B, C, {64});
        cal.blocked_flops_per_s = 2.0 * cube(n) / std::max(now_seconds() - t0, 1e-9);
    }
    {
        const std::size_t n = 128;
        const Matrix A = mm::make_uniform(n, n, 3), B = mm::make_uniform(n, n, 4);
        Matrix C(n, n);
        const double t0 = now_seconds();
        mm::reference::reference_multiply_dd(A, B, C);
        cal.dd_seconds_per_n3 = std::max(now_seconds() - t0, 1e-9) / cube(n);
    }
    return cal;
}

double estimate_seconds(const Context& ctx, const std::string& id) {
    if (id == "E8") return estimate_e8(ctx);
    return estimate_plan(ctx, plan_for(ctx.s, id), interleaved(id));
}

namespace {
double time_reference(Context& ctx) {
    const std::size_t n = 384;
    if (ctx.ref_A.rows() != n) {
        ctx.ref_A = mm::make_uniform(n, n, mm::derive_seed(ctx.s.base_seed, 999001));
        ctx.ref_B = mm::make_uniform(n, n, mm::derive_seed(ctx.s.base_seed, 999002));
        ctx.ref_C = Matrix(n, n);
    }
    const blocked::Params p{64};
    blocked::multiply(ctx.ref_A, ctx.ref_B, ctx.ref_C, p);  // warm
    int calls = 0;
    const double t0 = now_seconds();
    double t = 0.0;
    do {
        blocked::multiply(ctx.ref_A, ctx.ref_B, ctx.ref_C, p);
        ++calls;
        t = now_seconds() - t0;
    } while (t < kReferenceSeconds);
    return calls * 2.0 * cube(n) / t / 1e9;
}
}  // namespace

void establish_reference_baseline(Context& ctx) {
    std::vector<double> readings;
    for (int i = 0; i < 8; ++i) readings.push_back(time_reference(ctx));  // ~0.4 s
    ctx.ref_baseline_gflops = median(std::vector<double>(readings.begin() + 3, readings.end()));
}

double measure_reference(Context& ctx) {
    const double gflops = time_reference(ctx);
    if (ctx.ref_baseline_gflops == 0.0) establish_reference_baseline(ctx);
    if (ctx.exp_ref_max == 0.0) ctx.exp_ref_min = ctx.exp_ref_max = gflops;
    ctx.exp_ref_min = std::min(ctx.exp_ref_min, gflops);
    ctx.exp_ref_max = std::max(ctx.exp_ref_max, gflops);
    const double dev = gflops / ctx.ref_baseline_gflops - 1.0;
    if (std::fabs(dev) > kDriftWarnFraction) {
        std::ostringstream w;
        w << std::fixed << std::setprecision(1) << "    WARNING: drift monitor reads " << gflops << " GFLOP/s, "
          << std::showpos << 100.0 * dev << "% vs run start - is something else using the machine?";
        *ctx.log << w.str() << "\n";
    }
    return gflops;
}

void run_experiment(Context& ctx, const std::string& id) {
    const auto& list = all_experiments();
    const auto it = std::find_if(list.begin(), list.end(), [&](const ExperimentInfo& e) { return e.id == id; });
    if (it == list.end()) throw std::invalid_argument("unknown experiment " + id);
    require_tuned(ctx, id);
    const std::string path = (std::filesystem::path(ctx.s.results_dir) / it->csv).string();

    if (id == "E8") {
        run_e8(ctx, path);
        return;
    }
    CsvWriter csv(path, kTimingHeader);
    std::ostream& log = *ctx.log;
    const std::vector<Config> plan = plan_for(ctx.s, id);
    std::vector<ConfigResult> results;
    if (!interleaved(id)) {
        for (const Config& cfg : plan) {
            results.push_back(run_config(ctx, id, cfg, csv));
            print_result(log, results.back(), ctx.ref_baseline_gflops);
        }
    } else {
        // Round-robin: E5 and E6 draw their conclusions from comparisons *between*
        // configurations (stride vs stride, pressure level vs pressure level), which
        // sequential execution would confound with slow drift. Each round runs one
        // repetition of every configuration; the order is rotated every round and
        // reversed on odd rounds, so no configuration is systematically early or late.
        int rounds = 0;
        for (const Config& c : plan) rounds = std::max(rounds, c.reps);
        const std::size_t P = plan.size();
        results.resize(P);
        log << "  round-robin: " << rounds << " rounds x " << P << " configurations\n";
        for (int r = 0; r < rounds; ++r) {
            for (std::size_t k = 0; k < P; ++k) {
                const std::size_t rot = (k + static_cast<std::size_t>(r)) % P;
                const std::size_t idx = r % 2 ? P - 1 - rot : rot;
                if (r >= plan[idx].reps) continue;
                Config one = plan[idx];
                one.reps = 1;
                ConfigResult cr = run_config(ctx, id, one, csv, r);
                ConfigResult& acc = results[idx];
                if (acc.labels.empty()) {
                    acc = cr;
                    acc.cfg = plan[idx];
                } else {
                    for (std::size_t i = 0; i < cr.times.size(); ++i)
                        acc.times[i].insert(acc.times[i].end(), cr.times[i].begin(), cr.times[i].end());
                    acc.ref_gflops.insert(acc.ref_gflops.end(), cr.ref_gflops.begin(), cr.ref_gflops.end());
                }
            }
            log << "  round " << r + 1 << "/" << rounds << " done\n";
        }
        for (auto& r : results) {
            finalize(r);
            print_result(log, r, ctx.ref_baseline_gflops);
        }
    }

    if (id == "E0") {
        // Variance summary: coefficient of variation per variant, and drift (first vs last fifth).
        log << "  pilot summary (coefficient of variation = stdev / mean):\n";
        for (const ConfigResult& r : results) {
            for (std::size_t i = 0; i < r.labels.size(); ++i) {
                const std::vector<double>& v = r.times[i];
                if (v.size() < 2) continue;
                double mean = 0.0;
                for (double x : v) mean += x;
                mean /= static_cast<double>(v.size());
                double var = 0.0;
                for (double x : v) var += (x - mean) * (x - mean);
                const double cv = std::sqrt(var / static_cast<double>(v.size() - 1)) / mean;
                const std::ptrdiff_t fifth = static_cast<std::ptrdiff_t>(std::max<std::size_t>(1, v.size() / 5));
                const double first = median(std::vector<double>(v.begin(), v.begin() + fifth));
                const double last = median(std::vector<double>(v.end() - fifth, v.end()));
                std::ostringstream out;
                out << "      " << std::left << std::setw(24) << (r.labels[i] + " n=" + std::to_string(r.cfg.n))
                    << " CV " << std::fixed << std::setprecision(2) << 100 * cv << "%   drift (last/first fifth) "
                    << std::setprecision(3) << last / first;
                log << out.str() << "\n";
            }
        }
    } else if (id == "E1") {
        std::vector<std::string> exclude{"blocked:T=1", "blocked:untiled"};
        const std::string best = select_best(results, "blocked:T=", exclude, log);
        ctx.tuned.tile = parse_suffix_number(best, "T=");
        ctx.tuned.tile_source = "E1 " + timestamp() + " commit " + ctx.env.git_commit;
        save_tuned(ctx.s.results_dir, ctx.tuned);
        log << "  -> tuned tile = " << ctx.tuned.tile << " (written to tuned_params.txt)\n";
    } else if (id == "E2") {
        const std::string best = select_best(results, "strassen:c=", {}, log);
        ctx.tuned.cutoff = parse_suffix_number(best, "c=");
        ctx.tuned.cutoff_source = "E2 " + timestamp() + " commit " + ctx.env.git_commit;
        save_tuned(ctx.s.results_dir, ctx.tuned);
        log << "  -> tuned cutoff = " << ctx.tuned.cutoff << " (written to tuned_params.txt)\n";
    }
}

bool correctness_gate(std::ostream& log) {
    int failures = 0, checks = 0;
    const std::size_t sizes[] = {1, 2, 17, 64, 100, 129, 256};
    for (std::size_t n : sizes) {
        const Matrix A = mm::make_small_int(n, n, 900 + n, 8), B = mm::make_small_int(n, n, 950 + n, 8);
        Matrix ref(n, n), C(n, n);
        mm::reference::naive_multiply(A, B, ref);
        for (std::size_t t : {7u, 16u, 64u}) {
            blocked::multiply(A, B, C, {t});
            ++checks;
            if (!mm::exactly_equal(C, ref)) { ++failures; log << "  FAIL blocked n=" << n << " T=" << t << "\n"; }
        }
        for (std::size_t c : {1u, 8u, 64u, 128u}) {
            if (c == 1 && n > 64) continue;
            for (auto pad : {strassen::Padding::Minimal, strassen::Padding::NextPow2, strassen::Padding::Aligned})
                for (std::size_t tp : {0u, 8u}) {
                    if (pad == strassen::Padding::Aligned && c < 4) continue;
                    strassen::Params p;
                    p.cutoff = c;
                    p.padding = pad;
                    p.temp_pad = tp;
                    p.base.tile = 16;
                    strassen::multiply(A, B, C, p);
                    ++checks;
                    if (!mm::exactly_equal(C, ref)) {
                        ++failures;
                        log << "  FAIL strassen n=" << n << " c=" << c << " " << strassen::padding_name(pad)
                            << " temp_pad=" << tp << "\n";
                    }
                }
        }
    }
    log << "correctness gate: " << checks - failures << "/" << checks << " exact matches against the naive reference\n";
    return failures == 0;
}

Tuned load_tuned(const std::string& results_dir) {
    Tuned t;
    std::ifstream in(std::filesystem::path(results_dir) / "tuned_params.txt");
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "tile") t.tile = static_cast<std::size_t>(std::stoull(v));
        else if (k == "cutoff") t.cutoff = static_cast<std::size_t>(std::stoull(v));
        else if (k == "tile_source") t.tile_source = v;
        else if (k == "cutoff_source") t.cutoff_source = v;
    }
    return t;
}

void save_tuned(const std::string& results_dir, const Tuned& t) {
    std::ofstream out(std::filesystem::path(results_dir) / "tuned_params.txt");
    out << "tile=" << t.tile << "\ntile_source=" << t.tile_source << "\ncutoff=" << t.cutoff
        << "\ncutoff_source=" << t.cutoff_source << "\n";
}

}  // namespace bench
