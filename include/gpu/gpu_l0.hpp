#pragma once

#include "gpu/device.hpp"
#include "gpu/execution.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {

enum class GpuVendorFamily {
    intel,
    amd,
    nvidia,
    unknown
};

enum class GpuBackendKind {
    opencl,
    level_zero,
    vulkan_compute,
    cuda,
    rocm
};

struct GpuL0Capabilities {
    bool adapter_available = false;
    bool direct_command_submission = false;
    bool explicit_memory_import = false;
    bool timeline_synchronization = false;
    bool binary_module_loading = false;
    bool kernel_specialization = false;
    bool subgroup_matrix = false;
    bool unified_memory = false;
    bool persistent_kernel_cache = false;
    bool asynchronous_dispatch = false;
    std::uint32_t preferred_queue_batch = 1;
    std::uint32_t preferred_workgroup = 1;
    double estimated_launch_latency_us = 0.0;
    double estimated_host_read_gbps = 0.0;
    double estimated_host_write_gbps = 0.0;
};

struct GpuL0Binding {
    std::string device_uid;
    std::string graph_fingerprint;
    std::string adapter_id;
    std::string presentation_name;
    GpuVendorFamily vendor = GpuVendorFamily::unknown;
    GpuBackendKind backend = GpuBackendKind::opencl;
    double suitability_score = 0.0;
    GpuL0Capabilities capabilities;
};

struct GpuL0WorkloadTraits {
    OperationClass op_class = OperationClass::elementwise_map;
    std::vector<std::uint64_t> extents;
    std::uint64_t bytes = 0;
    double estimated_flops = 0.0;
    bool latency_sensitive = false;
    bool matrix_friendly = false;
    bool streaming_friendly = false;
};

struct GpuL0LaunchTuning {
    std::uint32_t workgroup_x = 1;
    std::uint32_t workgroup_y = 1;
    std::uint32_t workgroup_z = 1;
    std::uint32_t queue_batch = 1;
    std::uint32_t staging_window = 1;
    bool prefer_vectorized_io = false;
    bool prefer_persistent_modules = true;
    bool prefer_shared_host_staging = false;
};

class IGpuL0Adapter {
public:
    virtual ~IGpuL0Adapter() = default;

    [[nodiscard]] virtual std::string id() const = 0;
    [[nodiscard]] virtual GpuVendorFamily vendor() const = 0;
    [[nodiscard]] virtual GpuBackendKind backend_kind() const = 0;
    [[nodiscard]] virtual bool available() const = 0;
    [[nodiscard]] virtual bool matches(const HardwareGraph& graph) const = 0;
    [[nodiscard]] virtual GpuL0Binding describe(const HardwareGraph& graph) const = 0;
    [[nodiscard]] virtual GpuL0LaunchTuning suggest_tuning(
        const HardwareGraph& graph,
        const GpuL0WorkloadTraits& workload) const = 0;
};

[[nodiscard]] std::string to_string(GpuVendorFamily vendor);
[[nodiscard]] std::string to_string(GpuBackendKind backend);
[[nodiscard]] std::vector<std::unique_ptr<IGpuL0Adapter>> make_default_gpu_l0_adapters();

}  // namespace gpu
