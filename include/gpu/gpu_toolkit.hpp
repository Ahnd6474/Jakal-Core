#pragma once

#include "gpu/gpu_l0.hpp"

#include <optional>
#include <string>
#include <vector>

namespace gpu {

struct GpuToolkitVariant {
    GpuL0Binding binding;
    GpuL0LaunchTuning tuning;
    double toolkit_score = 0.0;
    bool executable = false;
    std::string rationale;
};

struct GpuToolkitIndexEntry {
    std::string device_uid;
    std::string graph_fingerprint;
    std::vector<GpuToolkitVariant> variants;
};

class GpuToolkit {
public:
    GpuToolkit();
    explicit GpuToolkit(std::vector<std::unique_ptr<IGpuL0Adapter>> adapters);

    [[nodiscard]] std::vector<GpuToolkitVariant> rank_variants(
        const HardwareGraph& graph,
        const GpuL0WorkloadTraits& workload = {}) const;

    [[nodiscard]] std::optional<GpuToolkitVariant> select_best(
        const HardwareGraph& graph,
        const GpuL0WorkloadTraits& workload = {}) const;

    [[nodiscard]] std::vector<GpuToolkitIndexEntry> build_index(
        const std::vector<HardwareGraph>& graphs,
        const GpuL0WorkloadTraits& workload = {}) const;

private:
    std::vector<std::unique_ptr<IGpuL0Adapter>> adapters_;
};

}  // namespace gpu
