#pragma once

#include "gpu/executors/interfaces.hpp"
#include "gpu/gpu_l0.hpp"

#include <memory>

namespace gpu::executors {

std::unique_ptr<IKernelBackend> make_native_gpu_kernel_backend(GpuBackendKind backend);

}  // namespace gpu::executors
