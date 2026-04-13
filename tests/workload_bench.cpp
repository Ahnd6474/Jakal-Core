#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CliOptions {
    bool smoke = false;
    bool host_only = false;
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--smoke") {
            options.smoke = true;
        } else if (arg == "--host-only") {
            options.host_only = true;
        }
    }
    return options;
}

double measure_ms(const auto& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

std::optional<jakal::WorkloadSpec> find_cpu_preset(const std::string& dataset_tag) {
    for (const auto& preset : jakal::cpu_deep_learning_exploration_presets()) {
        if (preset.workload.dataset_tag == dataset_tag) {
            return preset.workload;
        }
    }
    return std::nullopt;
}

std::optional<jakal::WorkloadSpec> find_canonical_preset(const std::string& dataset_tag) {
    for (const auto& preset : jakal::canonical_workload_presets()) {
        if (preset.workload.dataset_tag == dataset_tag) {
            return preset.workload;
        }
    }
    return std::nullopt;
}

struct BenchSample {
    double optimize_ms = 0.0;
    double execute_ms = 0.0;
    double total_runtime_us = 0.0;
    double total_reference_runtime_us = 0.0;
    std::size_t operation_count = 0u;
    std::size_t host_ops = 0u;
    std::size_t gpu_ops = 0u;
    std::size_t mixed_ops = 0u;
    bool loaded_from_cache = false;
};

std::optional<BenchSample> capture_sample(
    jakal::Runtime& runtime,
    const std::string& label,
    const jakal::WorkloadSpec& workload) {
    jakal::OptimizationReport report;
    const double optimize_ms = measure_ms([&]() {
        report = runtime.optimize(workload);
    });
    if (report.operations.empty()) {
        std::cerr << "benchmark failed: optimize returned no operations for " << label << '\n';
        return std::nullopt;
    }

    jakal::DirectExecutionReport execution;
    const double execute_ms = measure_ms([&]() {
        execution = runtime.execute(workload);
    });
    if (!execution.all_succeeded || execution.operations.empty()) {
        std::cerr << "benchmark failed: execute did not succeed for " << label << '\n';
        return std::nullopt;
    }

    BenchSample sample;
    sample.optimize_ms = optimize_ms;
    sample.execute_ms = execute_ms;
    sample.total_runtime_us = execution.total_runtime_us;
    sample.total_reference_runtime_us = execution.total_reference_runtime_us;
    sample.operation_count = execution.operations.size();
    sample.loaded_from_cache = report.loaded_from_cache;
    for (const auto& operation : execution.operations) {
        const bool host = operation.backend_name.find("host") != std::string::npos;
        const bool mixed = operation.backend_name.find("mixed") != std::string::npos;
        if (mixed) {
            ++sample.mixed_ops;
        } else if (host) {
            ++sample.host_ops;
        } else {
            ++sample.gpu_ops;
        }
    }
    return sample;
}

bool run_case(
    const std::string& label,
    const jakal::WorkloadSpec& workload,
    const bool host_only) {
    const auto cache_base = unique_temp_file("jakal-workload-bench", ".tsv");
    const auto make_options = [&](const bool summary_only, const bool trusted_validation) {
        jakal::RuntimeOptions options;
        options.cache_path = cache_base;
        options.execution_cache_path = cache_base;
        options.product.observability.persist_telemetry = false;
        if (summary_only) {
            options.product.performance.diagnostics_mode = jakal::RuntimeDiagnosticsMode::summary_only;
        }
        if (trusted_validation) {
            options.product.performance.direct_execution.enable_trusted_cached_validation = true;
            options.product.performance.direct_execution.trusted_verification_interval = 8u;
            options.product.performance.direct_execution.trusted_verification_sample_budget = 2u;
        }
        if (host_only) {
            options.enable_opencl_probe = false;
            options.enable_level_zero_probe = false;
            options.enable_vulkan_probe = false;
            options.enable_cuda_probe = false;
            options.enable_rocm_probe = false;
            options.enable_vulkan_status = false;
        }
        return options;
    };

    jakal::Runtime cold_runtime(make_options(false, false));
    const auto cold = capture_sample(cold_runtime, label + "-cold", workload);
    if (!cold.has_value()) {
        return false;
    }

    jakal::Runtime warm_runtime(make_options(false, false));
    if (!capture_sample(warm_runtime, label + "-warm-prime", workload).has_value()) {
        return false;
    }
    const auto warm = capture_sample(warm_runtime, label + "-warm", workload);
    if (!warm.has_value()) {
        return false;
    }

    jakal::Runtime summary_runtime(make_options(true, false));
    if (!capture_sample(summary_runtime, label + "-summary-prime", workload).has_value()) {
        return false;
    }
    const auto summary = capture_sample(summary_runtime, label + "-summary", workload);
    if (!summary.has_value()) {
        return false;
    }

    jakal::Runtime trusted_runtime(make_options(false, true));
    if (!capture_sample(trusted_runtime, label + "-trusted-prime", workload).has_value()) {
        return false;
    }
    const auto trusted = capture_sample(trusted_runtime, label + "-trusted", workload);
    if (!trusted.has_value()) {
        return false;
    }

    std::cout << label
              << " cold_opt_ms=" << cold->optimize_ms
              << " cold_exec_ms=" << cold->execute_ms
              << " warm_opt_ms=" << warm->optimize_ms
              << " warm_exec_ms=" << warm->execute_ms
              << " summary_exec_ms=" << summary->execute_ms
              << " trusted_exec_ms=" << trusted->execute_ms
              << " warm_opt_delta_ms=" << (warm->optimize_ms - cold->optimize_ms)
              << " warm_exec_delta_ms=" << (warm->execute_ms - cold->execute_ms)
              << " summary_exec_delta_ms=" << (summary->execute_ms - warm->execute_ms)
              << " trusted_exec_delta_ms=" << (trusted->execute_ms - warm->execute_ms)
              << " warm_ref_us=" << warm->total_reference_runtime_us
              << " trusted_ref_us=" << trusted->total_reference_runtime_us
              << " ops=" << warm->operation_count
              << " host_ops=" << warm->host_ops
              << " gpu_ops=" << warm->gpu_ops
              << " mixed_ops=" << warm->mixed_ops
              << " warm_cache=" << (warm->loaded_from_cache ? "hit" : "miss")
              << " summary_cache=" << (summary->loaded_from_cache ? "hit" : "miss")
              << " trusted_cache=" << (trusted->loaded_from_cache ? "hit" : "miss")
              << '\n';

    std::error_code ec;
    for (const auto* suffix : {
             "",
             ".bin",
             ".perf",
             ".perf.bin",
             ".perf.family",
             ".perf.family.bin",
             ".cpuhint",
             ".cpuhint.bin",
         }) {
        std::filesystem::remove(std::filesystem::path(cache_base.string() + suffix), ec);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_args(argc, argv);

    const auto decode = find_cpu_preset("llm-decode-token-lite");
    const auto prefill = find_cpu_preset("llm-prefill-context-lite");
    const auto vision = find_canonical_preset("ai-vision-inference-224");
    const auto upscale = find_canonical_preset("gaming-fsr-like-720p-to-1080p");
    if (!decode.has_value() || !prefill.has_value() || !vision.has_value() || !upscale.has_value()) {
        std::cerr << "benchmark failed: missing built-in workload presets\n";
        return 1;
    }

    std::vector<std::pair<std::string, jakal::WorkloadSpec>> cases = {
        {"decode", *decode},
        {"prefill", *prefill},
        {"vision", *vision},
        {"upscale", *upscale},
    };
    if (cli.smoke) {
        cases.resize(2u);
    }

    for (const auto& [label, workload] : cases) {
        if (!run_case(label, workload, cli.host_only)) {
            return 1;
        }
    }

    return 0;
}
