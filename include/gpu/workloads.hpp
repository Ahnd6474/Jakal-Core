#pragma once

#include "gpu/planner.hpp"

#include <string>
#include <vector>

namespace gpu {

struct CanonicalWorkloadPreset {
    WorkloadSpec workload;
    std::string description;
    std::string baseline_label;
};

[[nodiscard]] std::vector<CanonicalWorkloadPreset> canonical_workload_presets();

}  // namespace gpu
