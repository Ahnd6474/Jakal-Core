#include "gpu/execution.hpp"

#include "gpu/device.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace gpu {
namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr double kBytesPerGb = 1.0e9;
constexpr double kMicrosecondsPerSecond = 1.0e6;

struct ValidationResult {
    double reference_latency_us = 0.0;
    double candidate_latency_us = 0.0;
    double relative_error = 0.0;
};

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<std::string> split_csv(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    std::vector<std::string> fields;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (!field.empty()) {
            fields.push_back(field);
        }
    }
    return fields;
}

std::string join_csv(const std::vector<std::string>& values) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    return stream.str();
}

std::uint64_t clamp_u64(const std::uint64_t value, const std::uint64_t min_value, const std::uint64_t max_value) {
    return std::min(std::max(value, min_value), max_value);
}

std::uint32_t round_down_to_multiple(std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0) {
        return value;
    }
    value = std::max(value, multiple);
    return value - (value % multiple);
}

std::uint32_t round_up_to_multiple(std::uint32_t value, const std::uint32_t multiple) {
    if (multiple == 0) {
        return value;
    }
    const auto remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    return value + (multiple - remainder);
}

template <typename Func>
double measure_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

float quantize_value(const float value, const bool low_precision) {
    if (!low_precision) {
        return value;
    }
    return std::round(value * 1024.0f) / 1024.0f;
}

std::vector<float> make_pattern(const std::size_t count, const float phase) {
    std::vector<float> data(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float base = static_cast<float>((static_cast<std::uint32_t>(index * 17u + 23u) % 257u) - 128u);
        data[index] = (base / 97.0f) + (phase * 0.03125f);
    }
    return data;
}

double relative_l2_error(const std::vector<float>& reference, const std::vector<float>& candidate) {
    if (reference.empty() || reference.size() != candidate.size()) {
        return 0.0;
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double ref = static_cast<double>(reference[index]);
        const double diff = ref - static_cast<double>(candidate[index]);
        numerator += diff * diff;
        denominator += ref * ref;
    }

    if (denominator <= 1.0e-18) {
        return std::sqrt(numerator);
    }
    return std::sqrt(numerator / denominator);
}

double scalar_relative_error(const double reference, const double candidate) {
    const double denominator = std::max(std::abs(reference), 1.0e-9);
    return std::abs(reference - candidate) / denominator;
}

const HardwareGraph* find_graph(
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::string& uid) {
    const auto it = graph_lookup.find(uid);
    return it == graph_lookup.end() ? nullptr : it->second;
}

const HardwareObjectNode* find_first_node(
    const HardwareGraph& graph,
    const std::initializer_list<HardwareObjectRole> roles) {
    for (const auto role : roles) {
        for (const auto& node : graph.nodes) {
            if (node.role == role) {
                return &node;
            }
        }
    }
    return graph.nodes.empty() ? nullptr : &graph.nodes.front();
}

double host_link_gbps(const HardwareGraphSummary& summary) {
    const double link = std::max(summary.host_read_gbps, summary.host_write_gbps);
    if (link > 0.0) {
        return link;
    }
    if (summary.unified_address_space) {
        return 96.0;
    }
    if (summary.coherent_with_host) {
        return 48.0;
    }
    return 16.0;
}

double transfer_cost_us(const std::uint64_t bytes, const HardwareGraphSummary& summary) {
    if (bytes == 0) {
        return 0.0;
    }

    const double bandwidth = host_link_gbps(summary);
    const double transfer_us =
        (static_cast<double>(bytes) / (bandwidth * kBytesPerGb)) * kMicrosecondsPerSecond;
    return summary.average_transfer_cost_us + transfer_us;
}

double dispatch_cost_us(const HardwareGraphSummary& summary) {
    if (summary.average_dispatch_cost_us > 0.0) {
        return summary.average_dispatch_cost_us;
    }
    return summary.dispatch_latency_us;
}

double sync_cost_us(const HardwareGraphSummary& summary) {
    if (summary.synchronization_latency_us > 0.0) {
        return summary.synchronization_latency_us;
    }
    return summary.average_hierarchy_cost_us > 0.0 ? summary.average_hierarchy_cost_us : 4.0;
}

double estimate_device_gflops(
    const HardwareGraphSummary& summary,
    const OperationSpec& operation,
    const bool low_precision) {
    const double execution_objects = static_cast<double>(std::max(summary.execution_objects, 1u));
    const double lanes = static_cast<double>(std::max(summary.lanes_per_object, 1u));
    const double contexts = static_cast<double>(std::max(summary.resident_contexts, 1u));
    const double clock_ghz =
        static_cast<double>(summary.clock_mhz == 0 ? (summary.host_visible ? 2400u : 1000u) : summary.clock_mhz) /
        1000.0;
    double ops_per_cycle = execution_objects * lanes * std::log2(contexts + 1.0);
    if (operation.matrix_friendly) {
        ops_per_cycle += static_cast<double>(summary.matrix_units) * 32.0;
    }
    if (low_precision && (summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8)) {
        ops_per_cycle *= 1.35;
    }
    return std::max(1.0, ops_per_cycle * clock_ghz);
}

double expected_relative_error(const OperationSpec& operation, const ExecutionConfig& config) {
    double error = 0.0;
    if (config.use_low_precision) {
        error += operation.matrix_friendly ? 7.5e-4 : 4.0e-4;
    }
    if (config.strategy == ExecutionStrategy::sharded) {
        error += operation.reduction_like ? 3.0e-4 : 1.5e-4;
    }
    if (operation.reduction_like) {
        error += 1.0e-4;
    }
    return error;
}

std::string build_report_signature(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<OperationSpec>& operations) {
    std::ostringstream stream;
    stream << placement.signature << '|'
           << workload.name << '|'
           << to_string(workload.kind) << '|'
           << operations.size();
    for (const auto& operation : operations) {
        stream << '|' << operation.name << ':' << to_string(operation.op_class);
        for (const auto extent : operation.extents) {
            stream << ':' << extent;
        }
    }
    return stream.str();
}

std::string build_config_signature(const std::string& report_signature, const ExecutionConfig& config) {
    std::ostringstream stream;
    stream << report_signature << '|'
           << config.operation_name << '|'
           << to_string(config.strategy) << '|'
           << config.primary_device_uid << '|'
           << join_csv(config.participating_devices) << '|'
           << config.queue_depth << '|'
           << config.stages << '|'
           << config.tile_x << '|'
           << config.tile_y << '|'
           << config.tile_k << '|'
           << config.overlap_transfers << '|'
           << config.use_low_precision << '|'
           << std::fixed << std::setprecision(6) << config.target_error_tolerance;
    return stream.str();
}

std::uint32_t choose_queue_depth(const HardwareGraphSummary& summary) {
    if (summary.queue_slots == 0) {
        return 1;
    }
    return std::clamp(summary.queue_slots / 32u, 1u, 8u);
}

void configure_tiles(const OperationSpec& operation, const HardwareGraphSummary& summary, ExecutionConfig& config) {
    const std::uint32_t lanes = std::max(summary.lanes_per_object, 1u);
    const std::uint32_t vector_width = std::max(summary.native_vector_bits / 32u, 1u);

    switch (operation.op_class) {
    case OperationClass::matmul:
        config.tile_x = std::clamp(round_up_to_multiple(lanes * 8u, 16u), 32u, 128u);
        config.tile_y = config.tile_x;
        config.tile_k = std::clamp(round_up_to_multiple(vector_width * 8u, 8u), 16u, 128u);
        break;
    case OperationClass::convolution_2d:
    case OperationClass::resample_2d:
        config.tile_x = std::clamp(round_down_to_multiple(lanes * 4u, 8u), 16u, 64u);
        config.tile_y = std::clamp(round_down_to_multiple(lanes * 2u, 8u), 8u, 32u);
        config.tile_k = vector_width * 8u;
        break;
    case OperationClass::reduction:
    case OperationClass::elementwise_map:
    default:
        config.tile_x = std::clamp(round_up_to_multiple(lanes * 32u, 32u), 128u, 4096u);
        config.tile_y = 1;
        config.tile_k = vector_width * 4u;
        break;
    }
}

bool supports_low_precision(const HardwareGraphSummary& summary) {
    return summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8;
}

std::vector<ExecutionConfig> build_candidate_configs(
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const std::string& report_signature) {
    std::vector<ExecutionConfig> candidates;
    if (placement.allocations.empty()) {
        return candidates;
    }

    const auto& primary = placement.allocations.front();
    const auto* primary_graph = find_graph(graph_lookup, primary.device.uid);
    if (primary_graph == nullptr) {
        return candidates;
    }

    const auto primary_summary = summarize_graph(*primary_graph);

    auto add_candidate = [&](ExecutionConfig config) {
        config.operation_name = operation.name;
        config.target_error_tolerance = operation.max_relative_error;
        configure_tiles(operation, primary_summary, config);
        if (const auto* graph = find_graph(graph_lookup, config.primary_device_uid)) {
            const auto summary = summarize_graph(*graph);
            if (config.tile_x == 0 || config.tile_y == 0 || config.tile_k == 0) {
                configure_tiles(operation, summary, config);
            }
            if (config.queue_depth == 0) {
                config.queue_depth = choose_queue_depth(summary);
            }
            if (config.stages == 0) {
                config.stages = summary.supports_asynchronous_dispatch ? 2u : 1u;
            }
        }
        config.signature = build_config_signature(report_signature, config);
        const auto duplicate = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&](const ExecutionConfig& existing) {
                return existing.signature == config.signature;
            });
        if (duplicate == candidates.end()) {
            candidates.push_back(std::move(config));
        }
    };

    ExecutionConfig base;
    base.strategy = ExecutionStrategy::single_device;
    base.primary_device_uid = primary.device.uid;
    base.participating_devices = {primary.device.uid};
    base.queue_depth = choose_queue_depth(primary_summary);
    base.stages = primary_summary.supports_asynchronous_dispatch ? 2u : 1u;
    add_candidate(base);

    if (operation.streaming_friendly && primary_summary.supports_asynchronous_dispatch) {
        ExecutionConfig streaming = base;
        streaming.strategy = ExecutionStrategy::streaming;
        streaming.stages = std::max(3u, streaming.stages);
        streaming.overlap_transfers = true;
        add_candidate(streaming);
    }

    if (supports_low_precision(primary_summary) && operation.max_relative_error >= 4.0e-4) {
        ExecutionConfig low_precision = base;
        low_precision.use_low_precision = true;
        low_precision.overlap_transfers = primary_summary.supports_asynchronous_dispatch;
        low_precision.strategy = operation.parallelizable && primary_summary.supports_asynchronous_dispatch
                                     ? ExecutionStrategy::overlapped
                                     : ExecutionStrategy::single_device;
        add_candidate(low_precision);
    }

    if (operation.parallelizable && placement.allocations.size() > 1 && !workload.latency_sensitive) {
        ExecutionConfig sharded = base;
        sharded.strategy = ExecutionStrategy::sharded;
        sharded.participating_devices.clear();
        for (const auto& allocation : placement.allocations) {
            sharded.participating_devices.push_back(allocation.device.uid);
        }
        sharded.overlap_transfers = true;
        sharded.stages = std::max(2u, sharded.stages);
        add_candidate(sharded);
    }

    return candidates;
}

std::unordered_map<std::string, double> normalized_ratios(const ExecutionPlan& placement, const ExecutionConfig& config) {
    std::unordered_map<std::string, double> ratios;
    double total = 0.0;
    for (const auto& allocation : placement.allocations) {
        if (std::find(
                config.participating_devices.begin(),
                config.participating_devices.end(),
                allocation.device.uid) != config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
            total += allocation.ratio;
        }
    }

    if (total <= 0.0) {
        const double uniform = config.participating_devices.empty()
                                   ? 0.0
                                   : (1.0 / static_cast<double>(config.participating_devices.size()));
        for (const auto& uid : config.participating_devices) {
            ratios[uid] = uniform;
        }
        return ratios;
    }

    for (auto& [uid, ratio] : ratios) {
        ratio /= total;
    }
    return ratios;
}

double effective_edge_latency(const ExecutionEdge& edge) {
    return edge.predicted_latency_us * (1.0 - std::clamp(edge.overlap_ratio, 0.0, 0.95));
}

double finalize_graph_latency(ExecutionGraph& graph) {
    std::unordered_map<std::string, double> node_latency;
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<const ExecutionEdge*>> outgoing;
    node_latency.reserve(graph.nodes.size());
    indegree.reserve(graph.nodes.size());
    outgoing.reserve(graph.nodes.size());

    for (const auto& node : graph.nodes) {
        node_latency.emplace(node.id, node.predicted_latency_us);
        indegree.emplace(node.id, 0);
    }

    for (const auto& edge : graph.edges) {
        ++indegree[edge.target_id];
        outgoing[edge.source_id].push_back(&edge);
    }

    std::vector<std::string> frontier;
    frontier.reserve(graph.nodes.size());
    for (const auto& [node_id, degree] : indegree) {
        if (degree == 0) {
            frontier.push_back(node_id);
        }
    }

    std::unordered_map<std::string, double> longest;
    longest.reserve(graph.nodes.size());
    for (const auto& node_id : frontier) {
        longest[node_id] = node_latency[node_id];
    }

    for (std::size_t index = 0; index < frontier.size(); ++index) {
        const auto current = frontier[index];
        const double base = longest[current];

        if (const auto out_it = outgoing.find(current); out_it != outgoing.end()) {
            for (const auto* edge : out_it->second) {
                const double candidate = base + effective_edge_latency(*edge) + node_latency[edge->target_id];
                auto& target_distance = longest[edge->target_id];
                target_distance = std::max(target_distance, candidate);
                if (--indegree[edge->target_id] == 0) {
                    frontier.push_back(edge->target_id);
                }
            }
        }
    }

    double total = 0.0;
    for (const auto& [node_id, distance] : longest) {
        total = std::max(total, distance);
    }
    graph.predicted_latency_us = total;
    return total;
}

ExecutionGraph build_execution_graph(
    const std::string& report_signature,
    const OperationSpec& operation,
    ExecutionConfig& config,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    ExecutionGraph graph;
    graph.workload_signature = report_signature;
    graph.operation = operation;
    graph.participating_devices = config.participating_devices;
    config.mapped_structural_nodes.clear();

    const auto ratios = normalized_ratios(placement, config);

    auto add_node = [&](ExecutionNode node) {
        graph.nodes.push_back(std::move(node));
    };

    auto add_edge = [&](ExecutionEdge edge) {
        graph.edges.push_back(std::move(edge));
    };

    const std::string global_sink_id = operation.name + ".sink";
    const std::string global_sync_id = operation.name + ".sync";
    const bool needs_sync =
        config.participating_devices.size() > 1 || config.strategy != ExecutionStrategy::single_device;

    if (needs_sync) {
        add_node(ExecutionNode{
            global_sync_id,
            operation.name + "-sync",
            config.primary_device_uid,
            "",
            ExecutionNodeKind::synchronize,
            operation.output_bytes,
            0.0,
            0.0});
    }

    add_node(ExecutionNode{
        global_sink_id,
        operation.name + "-sink",
        config.primary_device_uid,
        "",
        ExecutionNodeKind::sink,
        operation.output_bytes,
        0.0,
        0.0});

    for (const auto& device_uid : config.participating_devices) {
        const auto* hardware = find_graph(graph_lookup, device_uid);
        if (hardware == nullptr) {
            continue;
        }

        const auto summary = summarize_graph(*hardware);
        const auto* control_node = find_first_node(*hardware, {HardwareObjectRole::queue, HardwareObjectRole::scheduler});
        const auto* compute_node = find_first_node(
            *hardware,
            {HardwareObjectRole::tile, HardwareObjectRole::cluster, HardwareObjectRole::pipeline});
        const auto* storage_node = find_first_node(
            *hardware,
            {HardwareObjectRole::scratchpad,
             HardwareObjectRole::cache,
             HardwareObjectRole::global_memory,
             HardwareObjectRole::host_memory});

        if (control_node != nullptr) {
            config.mapped_structural_nodes.push_back(control_node->id);
        }
        if (compute_node != nullptr) {
            config.mapped_structural_nodes.push_back(compute_node->id);
        }
        if (storage_node != nullptr) {
            config.mapped_structural_nodes.push_back(storage_node->id);
        }

        const double ratio = ratios.contains(device_uid) ? ratios.at(device_uid) : 1.0;
        const std::uint64_t input_share = static_cast<std::uint64_t>(std::ceil(static_cast<double>(operation.input_bytes) * ratio));
        const std::uint64_t output_share = operation.reduction_like
                                               ? operation.output_bytes
                                               : static_cast<std::uint64_t>(std::ceil(static_cast<double>(operation.output_bytes) * ratio));
        const double flops_share = operation.estimated_flops * ratio;
        const double compute_latency_us =
            (flops_share / (estimate_device_gflops(summary, operation, config.use_low_precision) * 1.0e9)) *
            kMicrosecondsPerSecond;

        const std::string prefix = operation.name + "." + device_uid;
        const std::string source_id = prefix + ".source";
        const std::string dispatch_id = prefix + ".dispatch";
        const std::string compute_id = prefix + ".compute";
        const std::string aggregate_id = prefix + ".aggregate";

        add_node(ExecutionNode{
            source_id,
            hardware->presentation_name + "-source",
            device_uid,
            storage_node == nullptr ? std::string() : storage_node->id,
            ExecutionNodeKind::source,
            input_share,
            0.0,
            0.0});

        add_node(ExecutionNode{
            dispatch_id,
            hardware->presentation_name + "-dispatch",
            device_uid,
            control_node == nullptr ? std::string() : control_node->id,
            ExecutionNodeKind::dispatch,
            0,
            0.0,
            dispatch_cost_us(summary)});

        add_node(ExecutionNode{
            compute_id,
            hardware->presentation_name + "-compute",
            device_uid,
            compute_node == nullptr ? std::string() : compute_node->id,
            ExecutionNodeKind::compute,
            output_share,
            flops_share,
            std::max(0.01, compute_latency_us)});

        add_edge(ExecutionEdge{
            source_id,
            dispatch_id,
            ExecutionEdgeKind::dataflow,
            true,
            input_share,
            transfer_cost_us(input_share, summary),
            config.overlap_transfers ? 0.35 : 0.0});

        add_edge(ExecutionEdge{
            dispatch_id,
            compute_id,
            ExecutionEdgeKind::control,
            true,
            0,
            dispatch_cost_us(summary),
            0.0});

        if (operation.reduction_like) {
            add_node(ExecutionNode{
                aggregate_id,
                hardware->presentation_name + "-aggregate",
                device_uid,
                compute_node == nullptr ? std::string() : compute_node->id,
                ExecutionNodeKind::aggregate,
                operation.output_bytes,
                static_cast<double>(operation.output_bytes),
                sync_cost_us(summary) * 0.5});

            add_edge(ExecutionEdge{
                compute_id,
                aggregate_id,
                ExecutionEdgeKind::aggregation,
                true,
                output_share,
                transfer_cost_us(output_share, summary) * 0.5,
                config.overlap_transfers ? 0.20 : 0.0});

            if (needs_sync) {
                add_edge(ExecutionEdge{
                    aggregate_id,
                    global_sync_id,
                    ExecutionEdgeKind::dependency,
                    true,
                    operation.output_bytes,
                    sync_cost_us(summary),
                    0.0});
            } else {
                add_edge(ExecutionEdge{
                    aggregate_id,
                    global_sink_id,
                    ExecutionEdgeKind::dataflow,
                    true,
                    operation.output_bytes,
                    transfer_cost_us(operation.output_bytes, summary),
                    0.0});
            }
        } else if (needs_sync) {
            add_edge(ExecutionEdge{
                compute_id,
                global_sync_id,
                ExecutionEdgeKind::dependency,
                true,
                output_share,
                transfer_cost_us(output_share, summary),
                config.overlap_transfers ? 0.25 : 0.0});
        } else {
            add_edge(ExecutionEdge{
                compute_id,
                global_sink_id,
                ExecutionEdgeKind::dataflow,
                true,
                output_share,
                transfer_cost_us(output_share, summary),
                config.overlap_transfers ? 0.25 : 0.0});
        }
    }

    if (needs_sync) {
        double sync_latency = 0.0;
        for (const auto& device_uid : config.participating_devices) {
            if (const auto* hardware = find_graph(graph_lookup, device_uid)) {
                sync_latency = std::max(sync_latency, sync_cost_us(summarize_graph(*hardware)));
            }
        }
        for (auto& node : graph.nodes) {
            if (node.id == global_sync_id) {
                node.predicted_latency_us = std::max(0.1, sync_latency);
                break;
            }
        }
        add_edge(ExecutionEdge{
            global_sync_id,
            global_sink_id,
            ExecutionEdgeKind::dataflow,
            true,
            operation.output_bytes,
            operation.output_bytes == 0 ? 0.0 : (sync_latency * 0.25),
            0.0});
    }

    graph.expected_relative_error = expected_relative_error(operation, config);
    config.signature = build_config_signature(report_signature, config);
    graph.signature = config.signature;
    finalize_graph_latency(graph);
    return graph;
}

ValidationResult validate_elementwise(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto a = make_pattern(count, 1.0f);
    const auto b = make_pattern(count, 2.0f);
    std::vector<float> reference(count, 0.0f);
    std::vector<float> candidate(count, 0.0f);

    const double reference_latency = measure_us([&]() {
        for (std::size_t index = 0; index < count; ++index) {
            reference[index] = (a[index] * 1.125f) + (b[index] * 0.25f) - 0.03125f;
        }
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 128u);
    const double candidate_latency = measure_us([&]() {
        for (std::size_t base = 0; base < count; base += tile) {
            const auto end = std::min(base + tile, count);
            for (std::size_t index = base; index < end; ++index) {
                const float lhs = quantize_value(a[index] * 1.125f, config.use_low_precision);
                const float rhs = quantize_value(b[index] * 0.25f, config.use_low_precision);
                candidate[index] = quantize_value(lhs + rhs - 0.03125f, config.use_low_precision);
            }
        }
    });

    return ValidationResult{
        reference_latency,
        candidate_latency,
        relative_l2_error(reference, candidate)};
}

ValidationResult validate_reduction(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto count = static_cast<std::size_t>(operation.extents.at(0));
    const auto data = make_pattern(count, 3.0f);
    double reference_value = 0.0;
    double candidate_value = 0.0;

    const double reference_latency = measure_us([&]() {
        double total = 0.0;
        for (const auto value : data) {
            total += value;
        }
        reference_value = total;
    });

    const std::size_t tile = std::max<std::size_t>(config.tile_x, 256u);
    const double candidate_latency = measure_us([&]() {
        float total = 0.0f;
        for (std::size_t base = 0; base < count; base += tile) {
            const auto end = std::min(base + tile, count);
            float partial = 0.0f;
            for (std::size_t index = base; index < end; ++index) {
                partial = quantize_value(partial + data[index], config.use_low_precision);
            }
            total = quantize_value(total + partial, config.use_low_precision);
        }
        candidate_value = total;
    });

    return ValidationResult{
        reference_latency,
        candidate_latency,
        scalar_relative_error(reference_value, candidate_value)};
}

ValidationResult validate_matmul(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto m = static_cast<std::size_t>(operation.extents.at(0));
    const auto n = static_cast<std::size_t>(operation.extents.at(1));
    const auto k = static_cast<std::size_t>(operation.extents.at(2));
    const auto a = make_pattern(m * k, 4.0f);
    const auto b = make_pattern(k * n, 5.0f);
    std::vector<float> reference(m * n, 0.0f);
    std::vector<float> candidate(m * n, 0.0f);

    const double reference_latency = measure_us([&]() {
        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                float acc = 0.0f;
                for (std::size_t inner = 0; inner < k; ++inner) {
                    acc += a[row * k + inner] * b[inner * n + col];
                }
                reference[row * n + col] = acc;
            }
        }
    });

    const std::size_t tile_m = std::max<std::size_t>(config.tile_x, 16u);
    const std::size_t tile_n = std::max<std::size_t>(config.tile_y, 16u);
    const std::size_t tile_k = std::max<std::size_t>(config.tile_k, 8u);
    const double candidate_latency = measure_us([&]() {
        for (std::size_t row_base = 0; row_base < m; row_base += tile_m) {
            for (std::size_t col_base = 0; col_base < n; col_base += tile_n) {
                for (std::size_t inner_base = 0; inner_base < k; inner_base += tile_k) {
                    const auto row_end = std::min(row_base + tile_m, m);
                    const auto col_end = std::min(col_base + tile_n, n);
                    const auto inner_end = std::min(inner_base + tile_k, k);
                    for (std::size_t row = row_base; row < row_end; ++row) {
                        for (std::size_t col = col_base; col < col_end; ++col) {
                            float acc = inner_base == 0 ? 0.0f : candidate[row * n + col];
                            for (std::size_t inner = inner_base; inner < inner_end; ++inner) {
                                const float lhs = quantize_value(a[row * k + inner], config.use_low_precision);
                                const float rhs = quantize_value(b[inner * n + col], config.use_low_precision);
                                acc = quantize_value(acc + (lhs * rhs), config.use_low_precision);
                            }
                            candidate[row * n + col] = acc;
                        }
                    }
                }
            }
        }
    });

    return ValidationResult{
        reference_latency,
        candidate_latency,
        relative_l2_error(reference, candidate)};
}

ValidationResult validate_convolution(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto height = static_cast<std::size_t>(operation.extents.at(0));
    const auto width = static_cast<std::size_t>(operation.extents.at(1));
    const auto input = make_pattern(height * width, 6.0f);
    const std::array<float, 9> kernel{0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};
    std::vector<float> reference((height - 2) * (width - 2), 0.0f);
    std::vector<float> candidate((height - 2) * (width - 2), 0.0f);

    const double reference_latency = measure_us([&]() {
        for (std::size_t y = 1; y + 1 < height; ++y) {
            for (std::size_t x = 1; x + 1 < width; ++x) {
                float acc = 0.0f;
                for (std::size_t ky = 0; ky < 3; ++ky) {
                    for (std::size_t kx = 0; kx < 3; ++kx) {
                        acc += input[(y + ky - 1) * width + (x + kx - 1)] * kernel[ky * 3 + kx];
                    }
                }
                reference[(y - 1) * (width - 2) + (x - 1)] = acc;
            }
        }
    });

    const std::size_t tile_y = std::max<std::size_t>(config.tile_y, 8u);
    const std::size_t tile_x = std::max<std::size_t>(config.tile_x, 8u);
    const double candidate_latency = measure_us([&]() {
        for (std::size_t y_base = 1; y_base + 1 < height; y_base += tile_y) {
            for (std::size_t x_base = 1; x_base + 1 < width; x_base += tile_x) {
                const auto y_end = std::min(y_base + tile_y, height - 1);
                const auto x_end = std::min(x_base + tile_x, width - 1);
                for (std::size_t y = y_base; y < y_end; ++y) {
                    for (std::size_t x = x_base; x < x_end; ++x) {
                        float acc = 0.0f;
                        for (std::size_t ky = 0; ky < 3; ++ky) {
                            for (std::size_t kx = 0; kx < 3; ++kx) {
                                const float value = quantize_value(
                                    input[(y + ky - 1) * width + (x + kx - 1)],
                                    config.use_low_precision);
                                acc = quantize_value(acc + (value * kernel[ky * 3 + kx]), config.use_low_precision);
                            }
                        }
                        candidate[(y - 1) * (width - 2) + (x - 1)] = acc;
                    }
                }
            }
        }
    });

    return ValidationResult{
        reference_latency,
        candidate_latency,
        relative_l2_error(reference, candidate)};
}

ValidationResult validate_resample(const OperationSpec& operation, const ExecutionConfig& config) {
    const auto src_h = static_cast<std::size_t>(operation.extents.at(0));
    const auto src_w = static_cast<std::size_t>(operation.extents.at(1));
    const auto dst_h = static_cast<std::size_t>(operation.extents.at(2));
    const auto dst_w = static_cast<std::size_t>(operation.extents.at(3));
    const auto input = make_pattern(src_h * src_w, 7.0f);
    std::vector<float> reference(dst_h * dst_w, 0.0f);
    std::vector<float> candidate(dst_h * dst_w, 0.0f);

    auto bilinear = [&](std::vector<float>& output, const bool low_precision) {
        for (std::size_t y = 0; y < dst_h; ++y) {
            const float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
            const auto y0 = static_cast<std::size_t>(std::clamp(std::floor(src_y), 0.0, static_cast<double>(src_h - 1)));
            const auto y1 = std::min(y0 + 1, src_h - 1);
            const float wy = static_cast<float>(src_y - std::floor(src_y));
            for (std::size_t x = 0; x < dst_w; ++x) {
                const float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
                const auto x0 = static_cast<std::size_t>(std::clamp(std::floor(src_x), 0.0, static_cast<double>(src_w - 1)));
                const auto x1 = std::min(x0 + 1, src_w - 1);
                const float wx = static_cast<float>(src_x - std::floor(src_x));
                const float v00 = quantize_value(input[y0 * src_w + x0], low_precision);
                const float v01 = quantize_value(input[y0 * src_w + x1], low_precision);
                const float v10 = quantize_value(input[y1 * src_w + x0], low_precision);
                const float v11 = quantize_value(input[y1 * src_w + x1], low_precision);
                const float top = quantize_value(v00 + ((v01 - v00) * wx), low_precision);
                const float bottom = quantize_value(v10 + ((v11 - v10) * wx), low_precision);
                output[y * dst_w + x] = quantize_value(top + ((bottom - top) * wy), low_precision);
            }
        }
    };

    const double reference_latency = measure_us([&]() {
        bilinear(reference, false);
    });
    const double candidate_latency = measure_us([&]() {
        bilinear(candidate, config.use_low_precision);
    });

    return ValidationResult{
        reference_latency,
        candidate_latency,
        relative_l2_error(reference, candidate)};
}

BenchmarkRecord benchmark_operation(
    const OperationSpec& operation,
    const ExecutionConfig& config,
    const ExecutionGraph& graph,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup) {
    ValidationResult validation;
    switch (operation.op_class) {
    case OperationClass::elementwise_map:
        validation = validate_elementwise(operation, config);
        break;
    case OperationClass::reduction:
        validation = validate_reduction(operation, config);
        break;
    case OperationClass::matmul:
        validation = validate_matmul(operation, config);
        break;
    case OperationClass::convolution_2d:
        validation = validate_convolution(operation, config);
        break;
    case OperationClass::resample_2d:
    default:
        validation = validate_resample(operation, config);
        break;
    }

    bool executable_on_host = config.participating_devices.size() == 1;
    if (executable_on_host) {
        if (const auto* hardware = find_graph(graph_lookup, config.primary_device_uid)) {
            executable_on_host = hardware->probe == "host";
        }
    }

    const double effective_latency = executable_on_host ? validation.candidate_latency_us : graph.predicted_latency_us;
    BenchmarkRecord record;
    record.operation_name = operation.name;
    record.config_signature = config.signature;
    record.reference_latency_us = validation.reference_latency_us;
    record.validation_latency_us = validation.candidate_latency_us;
    record.predicted_latency_us = graph.predicted_latency_us;
    record.effective_latency_us = std::max(0.01, effective_latency);
    record.speedup_vs_reference = validation.reference_latency_us / record.effective_latency_us;
    record.relative_error = validation.relative_error;
    record.accuracy_within_tolerance = validation.relative_error <= (config.target_error_tolerance + 1.0e-12);
    record.simulated = !executable_on_host;
    return record;
}

ExecutionStrategy parse_strategy(const std::string_view value) {
    if (value == "sharded") {
        return ExecutionStrategy::sharded;
    }
    if (value == "streaming") {
        return ExecutionStrategy::streaming;
    }
    if (value == "overlapped") {
        return ExecutionStrategy::overlapped;
    }
    return ExecutionStrategy::single_device;
}

OperationOptimizationResult optimize_operation(
    const std::string& report_signature,
    const OperationSpec& operation,
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::unordered_map<std::string, const HardwareGraph*>& graph_lookup,
    const ExecutionConfig* cached_config) {
    std::vector<ExecutionConfig> candidates;
    if (cached_config != nullptr) {
        candidates.push_back(*cached_config);
    } else {
        candidates = build_candidate_configs(operation, workload, placement, graph_lookup, report_signature);
    }

    if (candidates.empty()) {
        return {};
    }

    double best_objective = std::numeric_limits<double>::max();
    OperationOptimizationResult best;

    for (auto candidate : candidates) {
        auto graph = build_execution_graph(report_signature, operation, candidate, placement, graph_lookup);
        auto benchmark = benchmark_operation(operation, candidate, graph, graph_lookup);

        double objective = graph.predicted_latency_us;
        if (!benchmark.simulated) {
            objective = std::min(objective, benchmark.validation_latency_us);
        }
        if (!benchmark.accuracy_within_tolerance) {
            const double tolerance = std::max(candidate.target_error_tolerance, 1.0e-9);
            objective *= 10.0 + (benchmark.relative_error / tolerance);
        }

        if (objective < best_objective) {
            best_objective = objective;
            best.operation = operation;
            best.config = std::move(candidate);
            best.graph = std::move(graph);
            best.benchmark = std::move(benchmark);
        }
    }

    return best;
}

}  // namespace

std::string to_string(const OperationClass op_class) {
    switch (op_class) {
    case OperationClass::elementwise_map:
        return "elementwise_map";
    case OperationClass::reduction:
        return "reduction";
    case OperationClass::matmul:
        return "matmul";
    case OperationClass::convolution_2d:
        return "convolution_2d";
    case OperationClass::resample_2d:
    default:
        return "resample_2d";
    }
}

std::string to_string(const ExecutionNodeKind kind) {
    switch (kind) {
    case ExecutionNodeKind::source:
        return "source";
    case ExecutionNodeKind::dispatch:
        return "dispatch";
    case ExecutionNodeKind::compute:
        return "compute";
    case ExecutionNodeKind::aggregate:
        return "aggregate";
    case ExecutionNodeKind::sink:
        return "sink";
    case ExecutionNodeKind::synchronize:
    default:
        return "synchronize";
    }
}

std::string to_string(const ExecutionEdgeKind kind) {
    switch (kind) {
    case ExecutionEdgeKind::dataflow:
        return "dataflow";
    case ExecutionEdgeKind::control:
        return "control";
    case ExecutionEdgeKind::dependency:
        return "dependency";
    case ExecutionEdgeKind::aggregation:
    default:
        return "aggregation";
    }
}

std::string to_string(const ExecutionStrategy strategy) {
    switch (strategy) {
    case ExecutionStrategy::single_device:
        return "single_device";
    case ExecutionStrategy::sharded:
        return "sharded";
    case ExecutionStrategy::streaming:
        return "streaming";
    case ExecutionStrategy::overlapped:
    default:
        return "overlapped";
    }
}

std::vector<OperationSpec> default_operation_suite(const WorkloadSpec& workload) {
    const std::uint64_t working_set =
        workload.working_set_bytes == 0 ? (32ull * kMiB) : workload.working_set_bytes;
    const std::uint64_t sample_bytes = clamp_u64(working_set / 12ull, 2ull * kMiB, 16ull * kMiB);
    const std::uint64_t vector_count = std::max<std::uint64_t>(sample_bytes / sizeof(float), 64ull * 1024ull);

    const auto matmul_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 12.0),
            32.0,
            96.0)),
        16u);
    const auto conv_side = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 8.0),
            32.0,
            80.0)),
        8u);
    const auto resample_src = round_down_to_multiple(
        static_cast<std::uint32_t>(std::clamp(
            std::sqrt(static_cast<double>(sample_bytes) / 4.0),
            64.0,
            160.0)),
        16u);
    const auto resample_dst = std::max<std::uint32_t>(96u, (resample_src * 3u) / 2u);

    const OperationSpec elementwise{
        "elementwise-map",
        OperationClass::elementwise_map,
        {vector_count},
        vector_count * sizeof(float) * 2ull,
        vector_count * sizeof(float),
        0,
        static_cast<double>(vector_count) * 3.0,
        5.0e-4,
        true,
        false,
        true,
        false};

    const OperationSpec reduction{
        "reduction-sum",
        OperationClass::reduction,
        {vector_count},
        vector_count * sizeof(float),
        sizeof(float),
        32ull * kKiB,
        static_cast<double>(vector_count),
        1.0e-3,
        true,
        true,
        false,
        false};

    const OperationSpec matmul{
        "blocked-matmul",
        OperationClass::matmul,
        {matmul_side, matmul_side, matmul_side},
        2ull * matmul_side * matmul_side * sizeof(float),
        1ull * matmul_side * matmul_side * sizeof(float),
        0,
        2.0 * static_cast<double>(matmul_side) * static_cast<double>(matmul_side) * static_cast<double>(matmul_side),
        2.0e-3,
        true,
        false,
        false,
        true};

    const OperationSpec convolution{
        "conv3x3",
        OperationClass::convolution_2d,
        {conv_side, conv_side},
        1ull * conv_side * conv_side * sizeof(float),
        1ull * (conv_side - 2u) * (conv_side - 2u) * sizeof(float),
        9ull * sizeof(float),
        18.0 * static_cast<double>(conv_side - 2u) * static_cast<double>(conv_side - 2u),
        2.0e-3,
        true,
        false,
        true,
        false};

    const OperationSpec resample{
        "bilinear-resample",
        OperationClass::resample_2d,
        {resample_src, resample_src, resample_dst, resample_dst},
        1ull * resample_src * resample_src * sizeof(float),
        1ull * resample_dst * resample_dst * sizeof(float),
        0,
        8.0 * static_cast<double>(resample_dst) * static_cast<double>(resample_dst),
        1.5e-3,
        true,
        false,
        true,
        false};

    switch (workload.kind) {
    case WorkloadKind::image:
        return {elementwise, convolution, resample};
    case WorkloadKind::inference:
        return {elementwise, reduction, matmul, convolution};
    case WorkloadKind::tensor:
        return {elementwise, reduction, matmul, convolution, resample};
    case WorkloadKind::custom:
    default:
        return {elementwise, reduction, matmul};
    }
}

ExecutionOptimizer::ExecutionOptimizer(std::filesystem::path cache_path)
    : cache_path_(std::move(cache_path)) {}

std::filesystem::path ExecutionOptimizer::default_cache_path() {
    try {
        return std::filesystem::temp_directory_path() / "gpu_runtime_execution_cache.tsv";
    } catch (const std::exception&) {
        return std::filesystem::path("gpu_runtime_execution_cache.tsv");
    }
}

OptimizationReport ExecutionOptimizer::optimize(
    const WorkloadSpec& workload,
    const ExecutionPlan& placement,
    const std::vector<HardwareGraph>& graphs) {
    load_cache();

    OptimizationReport report;
    report.placement = placement;

    const auto operations = default_operation_suite(workload);
    report.signature = build_report_signature(workload, placement, operations);

    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    graph_lookup.reserve(graphs.size());
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }

    std::unordered_map<std::string, ExecutionConfig> cached_by_operation;
    bool fully_cached = false;
    if (const auto cache_it = cache_.find(report.signature); cache_it != cache_.end()) {
        fully_cached = true;
        for (const auto& cached : cache_it->second) {
            if (!graph_lookup.contains(cached.config.primary_device_uid)) {
                fully_cached = false;
                break;
            }
            cached_by_operation.emplace(cached.operation_name, cached.config);
        }
        for (const auto& operation : operations) {
            if (!cached_by_operation.contains(operation.name)) {
                fully_cached = false;
                break;
            }
        }
    }

    std::vector<CachedConfig> persisted_configs;
    persisted_configs.reserve(operations.size());

    for (const auto& operation : operations) {
        const auto cached_it = cached_by_operation.find(operation.name);
        const ExecutionConfig* cached_config = cached_it == cached_by_operation.end() ? nullptr : &cached_it->second;
        auto result = optimize_operation(report.signature, operation, workload, placement, graph_lookup, cached_config);
        if (result.config.signature.empty()) {
            continue;
        }
        persisted_configs.push_back(CachedConfig{
            operation.name,
            result.config});
        report.operations.push_back(std::move(result));
    }

    cache_[report.signature] = std::move(persisted_configs);
    persist_cache();
    report.loaded_from_cache = fully_cached;
    return report;
}

void ExecutionOptimizer::load_cache() {
    if (cache_loaded_) {
        return;
    }
    cache_loaded_ = true;

    std::ifstream input(cache_path_);
    if (!input.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const auto fields = split_tab(line);
        if (fields.size() != 14) {
            continue;
        }

        try {
            ExecutionConfig config;
            config.operation_name = fields[1];
            config.strategy = parse_strategy(fields[2]);
            config.primary_device_uid = fields[3];
            config.participating_devices = split_csv(fields[4]);
            config.mapped_structural_nodes = split_csv(fields[5]);
            config.queue_depth = static_cast<std::uint32_t>(std::stoul(fields[6]));
            config.stages = static_cast<std::uint32_t>(std::stoul(fields[7]));
            config.tile_x = static_cast<std::uint32_t>(std::stoul(fields[8]));
            config.tile_y = static_cast<std::uint32_t>(std::stoul(fields[9]));
            config.tile_k = static_cast<std::uint32_t>(std::stoul(fields[10]));
            config.overlap_transfers = std::stoi(fields[11]) != 0;
            config.use_low_precision = std::stoi(fields[12]) != 0;
            config.target_error_tolerance = std::stod(fields[13]);
            cache_[fields[0]].push_back(CachedConfig{config.operation_name, std::move(config)});
        } catch (const std::exception&) {
            continue;
        }
    }
}

void ExecutionOptimizer::persist_cache() const {
    const auto parent = cache_path_.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream output(cache_path_, std::ios::trunc);
    if (!output.is_open()) {
        return;
    }

    output << "# signature\toperation\tstrategy\tprimary_device\tparticipating_devices\tmapped_nodes\tqueue_depth\tstages\ttile_x\ttile_y\ttile_k\toverlap\tlow_precision\ttolerance\n";
    for (const auto& [signature, configs] : cache_) {
        for (const auto& cached : configs) {
            output << signature << '\t'
                   << cached.operation_name << '\t'
                   << to_string(cached.config.strategy) << '\t'
                   << cached.config.primary_device_uid << '\t'
                   << join_csv(cached.config.participating_devices) << '\t'
                   << join_csv(cached.config.mapped_structural_nodes) << '\t'
                   << cached.config.queue_depth << '\t'
                   << cached.config.stages << '\t'
                   << cached.config.tile_x << '\t'
                   << cached.config.tile_y << '\t'
                   << cached.config.tile_k << '\t'
                   << (cached.config.overlap_transfers ? 1 : 0) << '\t'
                   << (cached.config.use_low_precision ? 1 : 0) << '\t'
                   << cached.config.target_error_tolerance << '\n';
        }
    }
}

}  // namespace gpu
