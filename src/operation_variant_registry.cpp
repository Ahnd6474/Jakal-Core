#include "jakal/operation_variant_registry.hpp"

#include <algorithm>
#include <utility>

namespace jakal {
namespace {

bool is_applicable(const OperationVariantSpec& spec, const OperationVariantRequest& request) {
    if (!spec.applicable_classes.empty() &&
        std::find(spec.applicable_classes.begin(), spec.applicable_classes.end(), request.operation.op_class) ==
            spec.applicable_classes.end()) {
        return false;
    }
    if (spec.requires_parallelizable && !request.operation.parallelizable) {
        return false;
    }
    if (spec.requires_streaming_friendly && !request.operation.streaming_friendly) {
        return false;
    }
    if (spec.requires_low_precision_tolerance && !request.tolerates_low_precision) {
        return false;
    }
    if (spec.requires_multiple_devices && request.placement_device_count < 2u) {
        return false;
    }
    if (spec.requires_multiple_accelerators && request.accelerator_device_count < 2u) {
        return false;
    }
    if (spec.latency_only && !request.workload.latency_sensitive) {
        return false;
    }
    if (spec.exclude_when_latency_sensitive && request.workload.latency_sensitive) {
        return false;
    }
    if (spec.scope == OperationVariantScope::placement_sharded && request.placement_device_count < 2u) {
        return false;
    }
    if (spec.scope == OperationVariantScope::placement_sharded && !request.allow_placement_sharding) {
        return false;
    }
    if (spec.scope == OperationVariantScope::accelerator_sharded && request.accelerator_device_count < 2u) {
        return false;
    }
    if (spec.scope == OperationVariantScope::accelerator_sharded && !request.allow_accelerator_sharding) {
        return false;
    }
    return true;
}

OperationVariantRegistry make_builtin_registry() {
    OperationVariantRegistry registry;

    auto spec = [](std::string id,
                   std::string description,
                   const OperationVariantScope scope,
                   const ExecutionStrategy strategy,
                   const int sort_order) {
        OperationVariantSpec value;
        value.id = std::move(id);
        value.description = std::move(description);
        value.scope = scope;
        value.strategy = strategy;
        value.sort_order = sort_order;
        return value;
    };

    auto balanced = spec(
        "single-device-balanced",
        "Baseline per-device execution with device-local defaults.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::single_device,
        10);
    balanced.applicable_classes = {
        OperationClass::elementwise_map,
        OperationClass::reduction,
        OperationClass::matmul,
        OperationClass::convolution_2d,
        OperationClass::resample_2d};
    balanced.forced_logical_partitions = 1u;
    balanced.activation_bias = 0.15;
    registry.register_variant(std::move(balanced));

    auto host_critical = spec(
        "host-critical-path",
        "Low-overhead host-friendly path for dispatch-bound stages.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::single_device,
        15);
    host_critical.applicable_classes = {OperationClass::elementwise_map, OperationClass::reduction};
    host_critical.forced_queue_depth = 1u;
    host_critical.forced_stages = 1u;
    host_critical.forced_logical_partitions = 1u;
    host_critical.activation_bias = 0.38;
    registry.register_variant(std::move(host_critical));

    auto latency = spec(
        "single-device-latency",
        "Latency-oriented single-device execution with shallow queues.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::single_device,
        20);
    latency.applicable_classes = {
        OperationClass::reduction,
        OperationClass::matmul,
        OperationClass::convolution_2d,
        OperationClass::resample_2d};
    latency.latency_only = true;
    latency.forced_queue_depth = 1u;
    latency.forced_stages = 1u;
    latency.forced_logical_partitions = 1u;
    latency.activation_bias = 0.45;
    registry.register_variant(std::move(latency));

    auto streaming = spec(
        "streaming-pipeline",
        "Streaming-oriented execution for resample-style operations.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::streaming,
        30);
    streaming.applicable_classes = {OperationClass::resample_2d};
    streaming.overlap_transfers = true;
    streaming.requires_streaming_friendly = true;
    streaming.min_stages = 2u;
    streaming.activation_bias = 0.30;
    registry.register_variant(std::move(streaming));

    auto throughput_overlap = spec(
        "throughput-overlap",
        "Overlap-oriented execution that favors sustained throughput.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::overlapped,
        40);
    throughput_overlap.applicable_classes = {
        OperationClass::reduction,
        OperationClass::matmul,
        OperationClass::convolution_2d};
    throughput_overlap.overlap_transfers = true;
    throughput_overlap.requires_parallelizable = true;
    throughput_overlap.min_queue_depth = 2u;
    throughput_overlap.min_stages = 2u;
    throughput_overlap.activation_bias = 0.28;
    registry.register_variant(std::move(throughput_overlap));

    auto cooperative_split = spec(
        "cooperative-split",
        "Split work across host and accelerator with low sharding overhead.",
        OperationVariantScope::placement_sharded,
        ExecutionStrategy::sharded,
        50);
    cooperative_split.applicable_classes = {
        OperationClass::reduction,
        OperationClass::matmul,
        OperationClass::convolution_2d};
    cooperative_split.requires_parallelizable = true;
    cooperative_split.requires_multiple_devices = true;
    cooperative_split.forced_queue_depth = 1u;
    cooperative_split.forced_stages = 1u;
    cooperative_split.forced_logical_partitions = 2u;
    cooperative_split.activation_bias = 0.48;
    registry.register_variant(std::move(cooperative_split));

    auto low_precision = spec(
        "low-precision-overlap",
        "Low-precision overlap path for accuracy-tolerant operations.",
        OperationVariantScope::per_allocation,
        ExecutionStrategy::overlapped,
        40);
    low_precision.applicable_classes = {
        OperationClass::elementwise_map,
        OperationClass::matmul,
        OperationClass::convolution_2d,
        OperationClass::resample_2d};
    low_precision.overlap_transfers = true;
    low_precision.use_low_precision = true;
    low_precision.requires_low_precision_tolerance = true;
    low_precision.min_queue_depth = 2u;
    low_precision.min_stages = 1u;
    low_precision.activation_bias = 0.35;
    registry.register_variant(std::move(low_precision));

    auto placement = spec(
        "placement-sharded",
        "Shard the operation across all planned devices.",
        OperationVariantScope::placement_sharded,
        ExecutionStrategy::sharded,
        60);
    placement.applicable_classes = {
        OperationClass::reduction,
        OperationClass::matmul,
        OperationClass::convolution_2d};
    placement.overlap_transfers = true;
    placement.requires_parallelizable = true;
    placement.requires_multiple_devices = true;
    placement.exclude_when_latency_sensitive = true;
    placement.min_queue_depth = 2u;
    placement.min_stages = 2u;
    placement.activation_bias = 0.55;
    registry.register_variant(std::move(placement));

    auto ddp = spec(
        "accelerator-ddp",
        "Shard the operation across accelerator devices only.",
        OperationVariantScope::accelerator_sharded,
        ExecutionStrategy::sharded,
        70);
    ddp.applicable_classes = {OperationClass::matmul, OperationClass::convolution_2d};
    ddp.overlap_transfers = true;
    ddp.requires_parallelizable = true;
    ddp.requires_multiple_devices = true;
    ddp.requires_multiple_accelerators = true;
    ddp.forced_logical_partitions = 1u;
    ddp.min_queue_depth = 2u;
    ddp.min_stages = 2u;
    ddp.activation_bias = 0.70;
    registry.register_variant(std::move(ddp));

    return registry;
}

}  // namespace

void OperationVariantRegistry::register_variant(OperationVariantSpec spec) {
    variants_.push_back(std::move(spec));
}

std::vector<OperationVariantSpec> OperationVariantRegistry::resolve(const OperationVariantRequest& request) const {
    std::vector<OperationVariantSpec> resolved;
    resolved.reserve(variants_.size());

    for (const auto& variant : variants_) {
        if (is_applicable(variant, request)) {
            resolved.push_back(variant);
        }
    }

    std::sort(resolved.begin(), resolved.end(), [](const OperationVariantSpec& left, const OperationVariantSpec& right) {
        if (left.sort_order != right.sort_order) {
            return left.sort_order < right.sort_order;
        }
        return left.id < right.id;
    });

    return resolved;
}

const OperationVariantRegistry& OperationVariantRegistry::builtin() {
    static const OperationVariantRegistry registry = make_builtin_registry();
    return registry;
}

}  // namespace jakal
