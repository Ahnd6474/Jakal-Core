#include "gpu/runtime.hpp"

#include <algorithm>
#include <utility>

namespace gpu {

Runtime::Runtime(RuntimeOptions options)
    : options_(std::move(options)),
      planner_(options_.cache_path.empty() ? Planner::default_cache_path() : options_.cache_path),
      execution_optimizer_(
          options_.execution_cache_path.empty()
              ? ExecutionOptimizer::default_cache_path()
              : options_.execution_cache_path) {
    if (options_.enable_host_probe) {
        probes_.push_back(make_host_probe());
    }
    if (options_.enable_opencl_probe) {
        probes_.push_back(make_opencl_probe());
    }
    refresh_hardware();
}

void Runtime::refresh_hardware() {
    devices_.clear();

    for (auto& probe : probes_) {
        if (!probe->available()) {
            continue;
        }

        for (auto& graph : probe->discover_hardware()) {
            if (should_include_descriptor(graph)) {
                devices_.push_back(std::move(graph));
            }
        }
    }

    std::sort(devices_.begin(), devices_.end(), [](const HardwareGraph& left, const HardwareGraph& right) {
        const auto left_summary = summarize_graph(left);
        const auto right_summary = summarize_graph(right);

        if (left_summary.execution_objects != right_summary.execution_objects) {
            return left_summary.execution_objects > right_summary.execution_objects;
        }
        if (left_summary.addressable_bytes != right_summary.addressable_bytes) {
            return left_summary.addressable_bytes > right_summary.addressable_bytes;
        }
        if (left_summary.host_read_gbps != right_summary.host_read_gbps) {
            return left_summary.host_read_gbps > right_summary.host_read_gbps;
        }
        return structural_fingerprint(left) < structural_fingerprint(right);
    });
}

const std::vector<HardwareGraph>& Runtime::devices() const {
    return devices_;
}

ExecutionPlan Runtime::plan(const WorkloadSpec& workload) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    return planner_.build_plan(workload, devices_);
}

OptimizationReport Runtime::optimize(const WorkloadSpec& workload) {
    if (devices_.empty()) {
        refresh_hardware();
    }

    const auto placement = planner_.build_plan(workload, devices_);
    return execution_optimizer_.optimize(workload, placement, devices_);
}

bool Runtime::should_include_descriptor(const HardwareGraph& candidate) const {
    return std::none_of(devices_.begin(), devices_.end(), [&](const HardwareGraph& existing) {
        if (existing.uid == candidate.uid) {
            return true;
        }

        const bool same_name = existing.presentation_name == candidate.presentation_name;
        const bool same_probe_shape = structural_fingerprint(existing) == structural_fingerprint(candidate);
        return same_name && same_probe_shape;
    });
}

}  // namespace gpu
