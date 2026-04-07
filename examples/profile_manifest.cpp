#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct CliOptions {
    std::filesystem::path manifest_path;
    std::string ollama_model;
    std::uint32_t passes = 3;
    bool host_only = false;
    bool level_zero_only = false;
    bool opencl_only = false;
    bool show_operations = false;
};

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void print_usage() {
    std::cout << "Usage: jakal_profile_manifest <path-to-gguf-or-onnx> | --ollama-model <model>"
              << " [--passes N] [--host-only] [--level-zero-only] [--opencl-only] [--show-ops]\n";
}

#if defined(_WIN32)
using PipeHandle = FILE*;
#define JAKAL_POPEN _popen
#define JAKAL_PCLOSE _pclose
#else
using PipeHandle = FILE*;
#define JAKAL_POPEN popen
#define JAKAL_PCLOSE pclose
#endif

std::string trim_copy(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1u);
}

std::string unquote_copy(const std::string& value) {
    if (value.size() >= 2u && value.front() == '"' && value.back() == '"') {
        return value.substr(1u, value.size() - 2u);
    }
    return value;
}

std::optional<std::filesystem::path> resolve_ollama_blob_path(const std::string& model_name) {
    const std::string command = "ollama show " + model_name + " --modelfile 2>NUL";
    PipeHandle pipe = JAKAL_POPEN(command.c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int exit_code = JAKAL_PCLOSE(pipe);
    if (exit_code != 0) {
        return std::nullopt;
    }

    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        line = trim_copy(line);
        if (!line.starts_with("FROM ")) {
            continue;
        }
        const auto path = std::filesystem::path(unquote_copy(trim_copy(line.substr(5u))));
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
}

std::optional<CliOptions> parse_args(const int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--passes") {
            if (index + 1 >= argc) {
                std::cerr << "--passes requires a value.\n";
                return std::nullopt;
            }
            options.passes = static_cast<std::uint32_t>(std::max(1, std::stoi(argv[++index])));
            continue;
        }
        if (arg == "--ollama-model") {
            if (index + 1 >= argc) {
                std::cerr << "--ollama-model requires a value.\n";
                return std::nullopt;
            }
            options.ollama_model = argv[++index];
            continue;
        }
        if (arg == "--host-only") {
            options.host_only = true;
            continue;
        }
        if (arg == "--level-zero-only") {
            options.level_zero_only = true;
            continue;
        }
        if (arg == "--opencl-only") {
            options.opencl_only = true;
            continue;
        }
        if (arg == "--show-ops") {
            options.show_operations = true;
            continue;
        }
        if (!arg.empty() && arg.front() == '-') {
            std::cerr << "Unknown option: " << arg << '\n';
            return std::nullopt;
        }
        if (!options.ollama_model.empty()) {
            std::cerr << "Choose either a manifest path or --ollama-model, not both.\n";
            return std::nullopt;
        }
        if (!options.manifest_path.empty()) {
            std::cerr << "Only one manifest path can be provided.\n";
            return std::nullopt;
        }
        options.manifest_path = arg;
    }

    const std::uint32_t mode_count =
        static_cast<std::uint32_t>(options.host_only) +
        static_cast<std::uint32_t>(options.level_zero_only) +
        static_cast<std::uint32_t>(options.opencl_only);
    if (mode_count > 1u) {
        std::cerr << "Choose at most one of --host-only, --level-zero-only, or --opencl-only.\n";
        return std::nullopt;
    }
    if (options.manifest_path.empty() && options.ollama_model.empty()) {
        print_usage();
        return std::nullopt;
    }
    if (!options.manifest_path.empty() && !options.ollama_model.empty()) {
        std::cerr << "Choose either a manifest path or --ollama-model.\n";
        return std::nullopt;
    }
    return options;
}

std::string summarize_backend_counts(const std::vector<jakal::OperationExecutionRecord>& operations) {
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& operation : operations) {
        ++counts[operation.backend_name.empty() ? std::string("unknown") : operation.backend_name];
    }
    std::ostringstream stream;
    bool first = true;
    for (const auto& [backend, count] : counts) {
        if (!first) {
            stream << ' ';
        }
        first = false;
        stream << backend << '=' << count;
    }
    return first ? std::string("none") : stream.str();
}

void print_pass_summary(const std::uint32_t pass, const jakal::ManagedExecutionReport& report) {
    std::cout << "pass=" << pass
              << " executed=" << (report.executed ? "yes" : "no")
              << " source=" << jakal::to_string(report.planning.strategy_source)
              << " strategy=" << jakal::to_string(report.planning.resolved_partition_strategy)
              << " confidence=" << std::fixed << std::setprecision(3) << report.planning.strategy_confidence
              << " risk=" << report.safety.planner_risk_score
              << " total_us=" << report.execution.total_runtime_us
              << " reference_us=" << report.execution.total_reference_runtime_us
              << " speedup=" << report.execution.speedup_vs_reference << "x"
              << " backends=" << summarize_backend_counts(report.execution.operations)
              << '\n';

    std::cout << "  safety requested=" << jakal::to_string(report.safety.requested_strategy)
              << " selected=" << jakal::to_string(report.safety.selected_strategy)
              << " final=" << jakal::to_string(report.safety.final_strategy)
              << " rollback=" << (report.safety.rolled_back_to_auto ? "yes" : "no")
              << " planner_auto=" << (report.safety.planner_forced_auto ? "yes" : "no")
              << " mem_block=" << (report.safety.blocked_by_memory ? "yes" : "no")
              << '\n';

    std::cout << "  memory peak_pressure=" << report.memory_preflight.peak_pressure_ratio
              << " spill=" << report.memory_preflight.predicted_spill_bytes
              << " reload=" << report.memory_preflight.predicted_reload_bytes
              << " layout_cache=" << report.asset_prefetch.total_layout_cache_bytes
              << " telemetry=" << report.telemetry_path.string()
              << '\n';
}

void print_operation_details(const jakal::ManagedExecutionReport& report) {
    for (const auto& operation : report.execution.operations) {
        std::cout << "    op=" << std::setw(24) << std::left << operation.operation_name
                  << " backend=" << std::setw(24) << operation.backend_name
                  << " runtime_us=" << std::fixed << std::setprecision(3) << operation.runtime_us
                  << " ref_us=" << operation.reference_runtime_us
                  << " speedup=" << operation.speedup_vs_reference << "x"
                  << " verified=" << (operation.verified ? "yes" : "no")
                  << " parts=" << operation.logical_partitions_used
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto parsed = parse_args(argc, argv);
    if (!parsed.has_value()) {
        return 1;
    }

    auto options = *parsed;
    if (!options.ollama_model.empty()) {
        const auto resolved = resolve_ollama_blob_path(options.ollama_model);
        if (!resolved.has_value()) {
            std::cerr << "Unable to resolve Ollama model: " << options.ollama_model << '\n';
            return 1;
        }
        options.manifest_path = *resolved;
    }
    if (!std::filesystem::exists(options.manifest_path)) {
        std::cerr << "Input path does not exist: " << options.manifest_path << '\n';
        return 1;
    }

    const auto manifest = jakal::load_workload_source(options.manifest_path);

    jakal::RuntimeOptions runtime_options;
    runtime_options.cache_path = unique_temp_file("jakal-manifest-plan", ".tsv");
    runtime_options.execution_cache_path = unique_temp_file("jakal-manifest-exec", ".tsv");
    runtime_options.product.observability.telemetry_path = unique_temp_file("jakal-manifest", ".telemetry.tsv");
    runtime_options.enable_host_probe = true;
    runtime_options.enable_opencl_probe = !options.host_only && !options.level_zero_only;
    runtime_options.enable_level_zero_probe = !options.host_only && !options.opencl_only;
    runtime_options.enable_cuda_probe = false;
    runtime_options.enable_rocm_probe = false;
    runtime_options.prefer_level_zero_over_opencl = true;

    if (options.opencl_only) {
        runtime_options.enable_level_zero_probe = false;
        runtime_options.enable_opencl_probe = true;
    }
    if (options.level_zero_only) {
        runtime_options.enable_opencl_probe = false;
        runtime_options.enable_level_zero_probe = true;
    }

    jakal::Runtime runtime(runtime_options);

    std::cout << "Manifest profile\n";
    if (!options.ollama_model.empty()) {
        std::cout << "  ollama_model=" << options.ollama_model << '\n';
    }
    std::cout << "  source=" << options.manifest_path.string() << '\n';
    std::cout << "  format=" << manifest.source_format
              << " imported=" << (manifest.imported ? "yes" : "no")
              << " has_graph=" << (manifest.has_graph ? "yes" : "no")
              << '\n';
    std::cout << "  workload=" << manifest.workload.name
              << " dataset=" << manifest.workload.dataset_tag
              << " phase=" << jakal::to_string(manifest.workload.phase)
              << " shape=" << manifest.workload.shape_bucket
              << " kind=" << jakal::to_string(manifest.workload.kind)
              << '\n';
    std::cout << "  assets=" << manifest.assets.size()
              << " ops=" << manifest.graph.operations.size()
              << " tensors=" << manifest.graph.tensors.size()
              << '\n';
    std::cout << "  devices";
    for (const auto& graph : runtime.devices()) {
        std::cout << " [" << graph.probe << ':' << graph.presentation_name << "]";
    }
    std::cout << '\n';

    for (std::uint32_t pass = 1; pass <= options.passes; ++pass) {
        const auto report = runtime.execute_manifest(options.manifest_path);
        print_pass_summary(pass, report);
        if (options.show_operations) {
            print_operation_details(report);
        }
    }

    return 0;
}
