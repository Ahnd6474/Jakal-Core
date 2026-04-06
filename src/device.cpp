#include "gpu/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <unordered_map>

namespace gpu {
namespace {

bool is_execution_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::tile;
}

bool is_lane_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::lane;
}

bool is_pipeline_node(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::compute && node.role == HardwareObjectRole::pipeline;
}

bool is_latency_candidate(const HardwareObjectNode& node) {
    return node.domain == HardwareObjectDomain::transfer || node.domain == HardwareObjectDomain::control;
}

template <typename T>
T min_positive(const T current, const T candidate) {
    if (candidate <= static_cast<T>(0)) {
        return current;
    }
    if (current <= static_cast<T>(0)) {
        return candidate;
    }
    return std::min(current, candidate);
}

}  // namespace

std::string to_string(const HardwareObjectDomain domain) {
    switch (domain) {
    case HardwareObjectDomain::compute:
        return "compute";
    case HardwareObjectDomain::storage:
        return "storage";
    case HardwareObjectDomain::transfer:
        return "transfer";
    case HardwareObjectDomain::control:
    default:
        return "control";
    }
}

std::string to_string(const HardwareObjectRole role) {
    switch (role) {
    case HardwareObjectRole::root:
        return "root";
    case HardwareObjectRole::cluster:
        return "cluster";
    case HardwareObjectRole::tile:
        return "tile";
    case HardwareObjectRole::lane:
        return "lane";
    case HardwareObjectRole::pipeline:
        return "pipeline";
    case HardwareObjectRole::global_memory:
        return "global_memory";
    case HardwareObjectRole::host_memory:
        return "host_memory";
    case HardwareObjectRole::cache:
        return "cache";
    case HardwareObjectRole::scratchpad:
        return "scratchpad";
    case HardwareObjectRole::transfer_link:
        return "transfer_link";
    case HardwareObjectRole::queue:
        return "queue";
    case HardwareObjectRole::scheduler:
        return "scheduler";
    case HardwareObjectRole::router:
    default:
        return "router";
    }
}

std::string to_string(const HardwareObjectResolution resolution) {
    switch (resolution) {
    case HardwareObjectResolution::coarse:
        return "coarse";
    case HardwareObjectResolution::medium:
        return "medium";
    case HardwareObjectResolution::aggressive:
    default:
        return "aggressive";
    }
}

std::string to_string(const GraphEdgeSemantics semantics) {
    switch (semantics) {
    case GraphEdgeSemantics::contains:
        return "contains";
    case GraphEdgeSemantics::controls:
        return "controls";
    case GraphEdgeSemantics::dispatches:
        return "dispatches";
    case GraphEdgeSemantics::transfers_to:
        return "transfers_to";
    case GraphEdgeSemantics::reads_from:
        return "reads_from";
    case GraphEdgeSemantics::writes_to:
        return "writes_to";
    case GraphEdgeSemantics::feeds:
        return "feeds";
    case GraphEdgeSemantics::synchronizes_with:
    default:
        return "synchronizes_with";
    }
}

HardwareGraphSummary summarize_graph(const HardwareGraph& graph) {
    HardwareGraphSummary summary;
    bool has_numa_domain = false;

    std::unordered_map<std::string, std::uint32_t> lanes_per_parent;
    for (const auto& node : graph.nodes) {
        if (is_lane_node(node) && !node.parent_id.empty()) {
            ++lanes_per_parent[node.parent_id];
        }
    }

    std::uint32_t fallback_compute_nodes = 0;

    for (const auto& node : graph.nodes) {
        if (node.domain == HardwareObjectDomain::compute) {
            summary.clock_mhz = std::max(summary.clock_mhz, node.compute.clock_mhz);
            summary.native_vector_bits = std::max(summary.native_vector_bits, node.compute.native_vector_bits);
            summary.supports_fp64 = summary.supports_fp64 || node.compute.supports_fp64;
            summary.supports_fp32 = summary.supports_fp32 || node.compute.supports_fp32;
            summary.supports_fp16 = summary.supports_fp16 || node.compute.supports_fp16;
            summary.supports_bf16 = summary.supports_bf16 || node.compute.supports_bf16;
            summary.supports_int8 = summary.supports_int8 || node.compute.supports_int8;

            if (is_execution_node(node)) {
                ++summary.execution_objects;
                summary.resident_contexts += std::max(node.compute.resident_contexts, 1u);
                summary.matrix_units += node.compute.matrix_engines;
                summary.lanes_per_object = std::max(
                    summary.lanes_per_object,
                    std::max(lanes_per_parent[node.id], node.compute.execution_width));
            } else if (node.role == HardwareObjectRole::cluster) {
                ++fallback_compute_nodes;
            } else if (is_pipeline_node(node) && node.compute.matrix_engines > 0) {
                summary.matrix_units += node.compute.matrix_engines;
            }
        }

        if (node.domain == HardwareObjectDomain::storage) {
            summary.coherent_with_host = summary.coherent_with_host || node.storage.coherent_with_host;
            summary.unified_address_space = summary.unified_address_space || node.storage.unified_address_space;

            if (node.role == HardwareObjectRole::global_memory) {
                summary.addressable_bytes += node.storage.capacity_bytes;
                summary.directly_attached_bytes += node.storage.directly_attached_bytes;
                summary.shared_host_bytes += node.storage.shared_host_bytes;
            } else if (node.role == HardwareObjectRole::host_memory) {
                if (summary.addressable_bytes == 0) {
                    summary.addressable_bytes += node.storage.capacity_bytes;
                    summary.shared_host_bytes += node.storage.shared_host_bytes;
                }
            } else if (node.role == HardwareObjectRole::scratchpad) {
                summary.local_scratch_bytes += node.storage.capacity_bytes;
            } else if (node.role == HardwareObjectRole::cache) {
                summary.cache_bytes += node.storage.capacity_bytes;
            }
        }

        if (node.domain == HardwareObjectDomain::transfer) {
            summary.host_read_gbps = std::max(summary.host_read_gbps, node.transfer.read_bandwidth_gbps);
            summary.host_write_gbps = std::max(summary.host_write_gbps, node.transfer.write_bandwidth_gbps);
        }

        if (is_latency_candidate(node)) {
            summary.dispatch_latency_us = min_positive(summary.dispatch_latency_us, node.transfer.dispatch_latency_us);
            summary.synchronization_latency_us =
                min_positive(summary.synchronization_latency_us, node.transfer.synchronization_latency_us);
        }

        if (node.domain == HardwareObjectDomain::control) {
            summary.queue_slots = std::max(summary.queue_slots, node.control.queue_slots);
            if (!has_numa_domain) {
                summary.numa_domain = node.control.numa_domain;
                has_numa_domain = true;
            } else {
                summary.numa_domain = std::min(summary.numa_domain, node.control.numa_domain);
            }
            summary.host_visible = summary.host_visible || node.control.host_visible;
            summary.supports_asynchronous_dispatch =
                summary.supports_asynchronous_dispatch || node.control.supports_asynchronous_dispatch;
        }
    }

    if (summary.execution_objects == 0 && fallback_compute_nodes > 0) {
        summary.execution_objects = fallback_compute_nodes;
    }

    if (summary.lanes_per_object == 0 && summary.native_vector_bits > 0) {
        summary.lanes_per_object = std::max(1u, summary.native_vector_bits / 32u);
    }

    if (summary.dispatch_latency_us <= 0.0) {
        summary.dispatch_latency_us = summary.host_visible ? 2.0 : 24.0;
    }
    if (summary.synchronization_latency_us <= 0.0) {
        summary.synchronization_latency_us = summary.host_visible ? 1.0 : 16.0;
    }

    return summary;
}

std::string structural_fingerprint(const HardwareGraph& graph) {
    const auto summary = summarize_graph(graph);

    std::ostringstream stream;
    stream << summary.execution_objects << ':'
           << summary.lanes_per_object << ':'
           << summary.resident_contexts << ':'
           << summary.matrix_units << ':'
           << summary.clock_mhz << ':'
           << summary.queue_slots << ':'
           << summary.numa_domain << ':'
           << summary.native_vector_bits << ':'
           << summary.addressable_bytes << ':'
           << summary.directly_attached_bytes << ':'
           << summary.shared_host_bytes << ':'
           << summary.local_scratch_bytes << ':'
           << summary.cache_bytes << ':'
           << summary.host_read_gbps << ':'
           << summary.host_write_gbps << ':'
           << summary.dispatch_latency_us << ':'
           << summary.synchronization_latency_us << ':'
           << summary.coherent_with_host << ':'
           << summary.unified_address_space << ':'
           << summary.host_visible << ':'
           << summary.supports_asynchronous_dispatch << ':'
           << summary.supports_fp64 << ':'
           << summary.supports_fp32 << ':'
           << summary.supports_fp16 << ':'
           << summary.supports_bf16 << ':'
           << summary.supports_int8 << ':'
           << graph.nodes.size() << ':'
           << graph.edges.size();
    return stream.str();
}

}  // namespace gpu
