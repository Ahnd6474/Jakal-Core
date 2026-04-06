#pragma once

#include "gpu/backend.hpp"
#include "gpu/planner.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace gpu {

struct RuntimeOptions {
    bool enable_host_probe = true;
    bool enable_opencl_probe = true;
    std::filesystem::path cache_path;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});

    void refresh_hardware();

    [[nodiscard]] const std::vector<HardwareGraph>& devices() const;
    [[nodiscard]] ExecutionPlan plan(const WorkloadSpec& workload);

private:
    [[nodiscard]] bool should_include_descriptor(const HardwareGraph& candidate) const;

    RuntimeOptions options_;
    Planner planner_;
    std::vector<std::unique_ptr<IDeviceProbe>> probes_;
    std::vector<HardwareGraph> devices_;
};

}  // namespace gpu
