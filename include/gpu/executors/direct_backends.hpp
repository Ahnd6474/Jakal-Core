#pragma once

#include "gpu/executors/interfaces.hpp"

#include <memory>
#include <vector>

namespace gpu::executors {

std::unique_ptr<IKernelBackend> make_host_kernel_backend();
std::unique_ptr<IKernelBackend> make_opencl_kernel_backend();

class DirectDeviceExecutor final : public IDeviceExecutor {
public:
    explicit DirectDeviceExecutor(std::vector<std::unique_ptr<IKernelBackend>> backends);

    [[nodiscard]] bool matches(const HardwareGraph& graph) const override;
    [[nodiscard]] std::string name() const override;

    BackendRunResult dispatch(
        const DeviceAssignment& assignment,
        const OperationOptimizationResult& operation,
        const OperationData& data) const override;

private:
    std::vector<std::unique_ptr<IKernelBackend>> backends_;
};

}  // namespace gpu::executors
