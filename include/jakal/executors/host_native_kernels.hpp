#pragma once

#include "jakal/device.hpp"

#include <cstdint>
#include <span>

namespace jakal::executors {

bool try_run_host_native_low_precision_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    bool prefer_fp16,
    bool prefer_bf16,
    std::span<float> output);

bool try_run_host_native_bf16_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t depth,
    std::span<float> output);

}  // namespace jakal::executors
