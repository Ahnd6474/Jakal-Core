#pragma once

#include "gpu/executors/interfaces.hpp"

namespace gpu::executors {

class DefaultIntraDeviceScheduler final : public IIntraDeviceScheduler {
public:
    [[nodiscard]] std::vector<DeviceAssignment> make_assignments(
        const OptimizationReport& optimization,
        const OperationOptimizationResult& operation,
        const std::vector<HardwareGraph>& graphs,
        std::size_t total_items) const override;
};

}  // namespace gpu::executors
