#include "jakal/executors/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace jakal::executors {

namespace {

double log_bucket(const double value) {
    return value <= 0.0 ? 0.0 : std::log2(value + 1.0);
}

double inference_bias(const OperationOptimizationResult& operation, const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);
    const double execution_width =
        static_cast<double>(std::max(summary.execution_objects, 1u)) *
        static_cast<double>(std::max(summary.lanes_per_object, 1u));
    const double clock_scale = static_cast<double>(std::max(summary.clock_mhz, 1u)) / 1000.0;
    double bias = 1.0 + std::min(log_bucket((execution_width * clock_scale) / 16.0) * 0.10, 0.40);

    if (operation.operation.matrix_friendly || operation.operation.op_class == OperationClass::matmul) {
        bias += std::min(log_bucket(static_cast<double>(summary.matrix_units) + 1.0) * 0.12, 0.55);
        bias += std::min(log_bucket(static_cast<double>(summary.cache_bytes) / (256.0 * 1024.0)) * 0.08, 0.30);
        bias += std::min(log_bucket(static_cast<double>(summary.local_scratch_bytes) / (32.0 * 1024.0)) * 0.08, 0.24);
        if (operation.config.use_low_precision) {
            bias += (summary.supports_fp16 || summary.supports_bf16 || summary.supports_int8) ? 0.10 : -0.08;
        }
    } else if (operation.operation.op_class == OperationClass::reduction) {
        bias += std::min(log_bucket(static_cast<double>(summary.cache_bytes) / (128.0 * 1024.0)) * 0.07, 0.24);
    } else if (operation.operation.streaming_friendly || operation.operation.op_class == OperationClass::resample_2d) {
        const double transfer_gbps = std::max(summary.host_read_gbps, summary.host_write_gbps);
        bias += std::min(log_bucket(transfer_gbps / 16.0) * 0.08, 0.25);
    }

    if (summary.supports_asynchronous_dispatch && operation.config.overlap_transfers) {
        bias += 0.08;
    }
    if ((operation.operation.input_bytes + operation.operation.output_bytes) >= (16ull * 1024ull * 1024ull) &&
        !summary.coherent_with_host &&
        !summary.unified_address_space) {
        bias -= 0.08;
    }
    if (graph.probe == "host" && (operation.operation.matrix_friendly || operation.operation.op_class == OperationClass::matmul)) {
        bias -= 0.10;
    }

    return std::clamp(bias, 0.45, 2.50);
}

}  // namespace

std::vector<DeviceAssignment> DefaultIntraDeviceScheduler::make_assignments(
    const OptimizationReport& optimization,
    const OperationOptimizationResult& operation,
    const std::vector<HardwareGraph>& graphs,
    const std::size_t total_items) const {
    std::unordered_map<std::string, const HardwareGraph*> graph_lookup;
    for (const auto& graph : graphs) {
        graph_lookup.emplace(graph.uid, &graph);
    }

    std::unordered_map<std::string, double> ratios;
    for (const auto& allocation : optimization.placement.allocations) {
        if (std::find(
                operation.config.participating_devices.begin(),
                operation.config.participating_devices.end(),
                allocation.device.uid) != operation.config.participating_devices.end()) {
            ratios[allocation.device.uid] = allocation.ratio;
        }
    }

    if (ratios.empty()) {
        for (const auto& uid : operation.config.participating_devices) {
            ratios[uid] = 1.0;
        }
    }

    std::unordered_map<std::string, double> weighted_ratios;
    double total_ratio = 0.0;
    for (const auto& uid : operation.config.participating_devices) {
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }
        const double base_ratio = ratios.contains(uid) ? ratios.at(uid) : 1.0;
        const double weighted_ratio = base_ratio * inference_bias(operation, *graph_it->second);
        weighted_ratios[uid] = weighted_ratio;
        total_ratio += weighted_ratio;
    }

    if (total_ratio <= 0.0) {
        total_ratio = static_cast<double>(std::max<std::size_t>(operation.config.participating_devices.size(), 1u));
        for (const auto& uid : operation.config.participating_devices) {
            weighted_ratios[uid] = 1.0;
        }
    }

    std::vector<DeviceAssignment> assignments;
    assignments.reserve(
        operation.config.participating_devices.size() *
        static_cast<std::size_t>(std::max(operation.config.logical_partitions, 1u)));
    std::size_t consumed = 0;

    for (std::size_t index = 0; index < operation.config.participating_devices.size(); ++index) {
        const auto& uid = operation.config.participating_devices[index];
        const auto graph_it = graph_lookup.find(uid);
        if (graph_it == graph_lookup.end()) {
            continue;
        }

        const double ratio = weighted_ratios.contains(uid) ? weighted_ratios.at(uid) / total_ratio : (1.0 / total_ratio);
        std::size_t count = 0;
        if (index + 1 == operation.config.participating_devices.size()) {
            count = total_items - consumed;
        } else {
            count = static_cast<std::size_t>(std::llround(static_cast<double>(total_items) * ratio));
            count = std::min(count, total_items - consumed);
        }

        const auto partitions = std::max(operation.config.logical_partitions, 1u);
        std::size_t local_consumed = 0;
        for (std::uint32_t partition = 0; partition < partitions; ++partition) {
            std::size_t partition_count = 0;
            if (partition + 1 == partitions) {
                partition_count = count - local_consumed;
            } else {
                partition_count = static_cast<std::size_t>(
                    std::llround(static_cast<double>(count) / static_cast<double>(partitions)));
                partition_count = std::min(partition_count, count - local_consumed);
            }

            assignments.push_back(DeviceAssignment{
                graph_it->second,
                ratio / static_cast<double>(partitions),
                {consumed + local_consumed, partition_count},
                partition,
                partitions});
            local_consumed += partition_count;
        }

        consumed += count;
    }

    if (!assignments.empty() && consumed < total_items) {
        assignments.back().shard.count += total_items - consumed;
    }

    assignments.erase(
        std::remove_if(assignments.begin(), assignments.end(), [](const DeviceAssignment& assignment) {
            return assignment.graph == nullptr || assignment.shard.count == 0;
        }),
        assignments.end());

    return assignments;
}

}  // namespace jakal::executors

