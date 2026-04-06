#include "gpu/executors/direct_backends.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace gpu::executors {
namespace {

constexpr std::uint32_t kOpenClReductionGroupSize = 256u;

template <typename Func>
double measure_us(Func&& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

float quantize_value(const float value, const bool low_precision) {
    if (!low_precision) {
        return value;
    }
    return std::round(value * 1024.0f) / 1024.0f;
}

std::string sanitize_id_fragment(const std::string& text) {
    std::string result = text;
    for (char& ch : result) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (!(std::isalnum(value) || ch == '_' || ch == '-')) {
            ch = '_';
        }
    }
    return result;
}

class HostKernelBackend final : public IKernelBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const override {
        return graph.probe == "host";
    }

    [[nodiscard]] std::string name() const override {
        return "host-direct";
    }

    BackendRunResult run_elementwise(
        const HardwareGraph&,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(lhs.size(), 0.0f);
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, lhs.size() / workers);
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= lhs.size()) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? lhs.size() : std::min(lhs.size(), begin + chunk);
                threads.emplace_back([&, begin, end]() {
                    for (std::size_t index = begin; index < end; ++index) {
                        const float left = quantize_value(lhs[index] * 1.125f, low_precision);
                        const float right = quantize_value(rhs[index] * 0.25f, low_precision);
                        result.output[index] = quantize_value(left + right - 0.03125f, low_precision);
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_reduction(
        const HardwareGraph&,
        const std::span<const float> input,
        const bool low_precision) const override {
        BackendRunResult result;
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, input.size() / workers);
            std::vector<std::thread> threads;
            std::vector<float> partials(workers, 0.0f);
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= input.size()) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? input.size() : std::min(input.size(), begin + chunk);
                threads.emplace_back([&, worker, begin, end]() {
                    float partial = 0.0f;
                    for (std::size_t index = begin; index < end; ++index) {
                        partial = quantize_value(partial + input[index], low_precision);
                    }
                    partials[worker] = partial;
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
            float total = 0.0f;
            for (const auto partial : partials) {
                total = quantize_value(total + partial, low_precision);
            }
            result.scalar_output = total;
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_matmul(
        const HardwareGraph&,
        const std::span<const float> lhs,
        const std::span<const float> rhs,
        const std::uint32_t rows,
        const std::uint32_t columns,
        const std::uint32_t depth,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(rows) * columns, 0.0f);
        result.runtime_us = measure_us([&]() {
            const std::size_t workers = std::max<std::size_t>(1u, std::thread::hardware_concurrency());
            const std::size_t chunk = std::max<std::size_t>(1u, rows / workers);
            std::vector<std::thread> threads;
            threads.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t begin = worker * chunk;
                if (begin >= rows) {
                    break;
                }
                const std::size_t end = worker + 1 == workers ? rows : std::min<std::size_t>(rows, begin + chunk);
                threads.emplace_back([&, begin, end]() {
                    for (std::size_t row = begin; row < end; ++row) {
                        for (std::uint32_t col = 0; col < columns; ++col) {
                            float acc = 0.0f;
                            for (std::uint32_t inner = 0; inner < depth; ++inner) {
                                const float left = quantize_value(lhs[row * depth + inner], low_precision);
                                const float right = quantize_value(rhs[inner * columns + col], low_precision);
                                acc = quantize_value(acc + (left * right), low_precision);
                            }
                            result.output[row * columns + col] = acc;
                        }
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_conv3x3(
        const HardwareGraph&,
        const std::span<const float> input,
        const std::uint32_t height,
        const std::uint32_t width,
        const bool low_precision) const override {
        static constexpr std::array<float, 9> kernel{
            0.0625f, 0.125f, 0.0625f,
            0.125f, 0.25f, 0.125f,
            0.0625f, 0.125f, 0.0625f};

        const std::uint32_t out_height = height - 2u;
        const std::uint32_t out_width = width - 2u;
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(out_height) * out_width, 0.0f);
        result.runtime_us = measure_us([&]() {
            for (std::uint32_t y = 1; y + 1 < height; ++y) {
                for (std::uint32_t x = 1; x + 1 < width; ++x) {
                    float acc = 0.0f;
                    for (std::uint32_t ky = 0; ky < 3; ++ky) {
                        for (std::uint32_t kx = 0; kx < 3; ++kx) {
                            const float value = quantize_value(
                                input[(y + ky - 1u) * width + (x + kx - 1u)],
                                low_precision);
                            acc = quantize_value(acc + (value * kernel[ky * 3u + kx]), low_precision);
                        }
                    }
                    result.output[(y - 1u) * out_width + (x - 1u)] = acc;
                }
            }
        });
        result.success = true;
        return result;
    }

    BackendRunResult run_resample(
        const HardwareGraph&,
        const std::span<const float> input,
        const std::uint32_t src_h,
        const std::uint32_t src_w,
        const std::uint32_t dst_h,
        const std::uint32_t dst_w,
        const std::uint32_t row_offset,
        const std::uint32_t row_count,
        const bool low_precision) const override {
        BackendRunResult result;
        result.output.resize(static_cast<std::size_t>(row_count) * dst_w, 0.0f);
        result.runtime_us = measure_us([&]() {
            for (std::uint32_t local_y = 0; local_y < row_count; ++local_y) {
                const std::uint32_t y = row_offset + local_y;
                const float src_y =
                    (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(dst_h) - 0.5f;
                const float clamped_y = std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1u));
                const auto y0 = static_cast<std::uint32_t>(clamped_y);
                const auto y1 = std::min(y0 + 1u, src_h - 1u);
                const float wy = clamped_y - static_cast<float>(y0);

                for (std::uint32_t x = 0; x < dst_w; ++x) {
                    const float src_x =
                        (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(dst_w) - 0.5f;
                    const float clamped_x = std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1u));
                    const auto x0 = static_cast<std::uint32_t>(clamped_x);
                    const auto x1 = std::min(x0 + 1u, src_w - 1u);
                    const float wx = clamped_x - static_cast<float>(x0);
                    const float v00 = quantize_value(input[y0 * src_w + x0], low_precision);
                    const float v01 = quantize_value(input[y0 * src_w + x1], low_precision);
                    const float v10 = quantize_value(input[y1 * src_w + x0], low_precision);
                    const float v11 = quantize_value(input[y1 * src_w + x1], low_precision);
                    const float top = quantize_value(v00 + ((v01 - v00) * wx), low_precision);
                    const float bottom = quantize_value(v10 + ((v11 - v10) * wx), low_precision);
                    result.output[local_y * dst_w + x] = quantize_value(top + ((bottom - top) * wy), low_precision);
                }
            }
        });
        result.success = true;
        return result;
    }
};

// OpenCL backend implementation remains direct and stateful; keep it here behind IKernelBackend
// so future Level Zero / Vulkan backends can follow the same shape.

#if defined(_WIN32)
using LibraryHandle = HMODULE;
LibraryHandle load_library(const char* name) { return LoadLibraryA(name); }
void* load_symbol(LibraryHandle library, const char* name) { return reinterpret_cast<void*>(GetProcAddress(library, name)); }
void close_library(LibraryHandle library) { if (library != nullptr) { FreeLibrary(library); } }
#else
using LibraryHandle = void*;
LibraryHandle load_library(const char* name) { return dlopen(name, RTLD_LAZY); }
void* load_symbol(LibraryHandle library, const char* name) { return dlsym(library, name); }
void close_library(LibraryHandle library) { if (library != nullptr) { dlclose(library); } }
#endif

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_program_build_info = cl_uint;
using cl_context_properties = intptr_t;
using cl_mem_flags = cl_bitfield;
using cl_command_queue_properties = cl_bitfield;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_mem = struct _cl_mem*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFFu;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1u << 2u;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1u << 1u;
constexpr cl_platform_info CL_PLATFORM_NAME = 0x0902;
constexpr cl_device_info CL_DEVICE_NAME = 0x102B;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG = 0x1183;

using cl_get_platform_ids_fn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using cl_get_platform_info_fn = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
using cl_get_device_ids_fn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using cl_get_device_info_fn = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
using cl_create_context_fn = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
using cl_release_context_fn = cl_int (*)(cl_context);
using cl_create_command_queue_fn = cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
using cl_create_command_queue_with_properties_fn = cl_command_queue (*)(cl_context, cl_device_id, const cl_context_properties*, cl_int*);
using cl_release_command_queue_fn = cl_int (*)(cl_command_queue);
using cl_create_program_with_source_fn = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using cl_build_program_fn = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using cl_get_program_build_info_fn = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
using cl_release_program_fn = cl_int (*)(cl_program);
using cl_create_kernel_fn = cl_kernel (*)(cl_program, const char*, cl_int*);
using cl_set_kernel_arg_fn = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using cl_release_kernel_fn = cl_int (*)(cl_kernel);
using cl_create_buffer_fn = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using cl_release_mem_object_fn = cl_int (*)(cl_mem);
using cl_enqueue_write_buffer_fn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void*, void*);
using cl_enqueue_read_buffer_fn = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void*, void*);
using cl_enqueue_nd_range_kernel_fn = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
using cl_finish_fn = cl_int (*)(cl_command_queue);

class OpenClApi {
public:
    OpenClApi();
    ~OpenClApi();
    [[nodiscard]] bool loaded() const { return loaded_; }
    [[nodiscard]] cl_get_platform_ids_fn get_platform_ids() const { return get_platform_ids_; }
    [[nodiscard]] cl_get_platform_info_fn get_platform_info() const { return get_platform_info_; }
    [[nodiscard]] cl_get_device_ids_fn get_device_ids() const { return get_device_ids_; }
    [[nodiscard]] cl_get_device_info_fn get_device_info() const { return get_device_info_; }
    [[nodiscard]] cl_create_context_fn create_context() const { return create_context_; }
    [[nodiscard]] cl_create_command_queue_fn create_command_queue() const { return create_command_queue_; }
    [[nodiscard]] cl_create_command_queue_with_properties_fn create_command_queue_with_properties() const { return create_command_queue_with_properties_; }
    [[nodiscard]] cl_release_context_fn release_context() const { return release_context_; }
    [[nodiscard]] cl_release_command_queue_fn release_command_queue() const { return release_command_queue_; }
    [[nodiscard]] cl_create_program_with_source_fn create_program_with_source() const { return create_program_with_source_; }
    [[nodiscard]] cl_build_program_fn build_program() const { return build_program_; }
    [[nodiscard]] cl_get_program_build_info_fn get_program_build_info() const { return get_program_build_info_; }
    [[nodiscard]] cl_release_program_fn release_program() const { return release_program_; }
    [[nodiscard]] cl_create_kernel_fn create_kernel() const { return create_kernel_; }
    [[nodiscard]] cl_set_kernel_arg_fn set_kernel_arg() const { return set_kernel_arg_; }
    [[nodiscard]] cl_release_kernel_fn release_kernel() const { return release_kernel_; }
    [[nodiscard]] cl_create_buffer_fn create_buffer() const { return create_buffer_; }
    [[nodiscard]] cl_release_mem_object_fn release_mem_object() const { return release_mem_object_; }
    [[nodiscard]] cl_enqueue_write_buffer_fn enqueue_write_buffer() const { return enqueue_write_buffer_; }
    [[nodiscard]] cl_enqueue_read_buffer_fn enqueue_read_buffer() const { return enqueue_read_buffer_; }
    [[nodiscard]] cl_enqueue_nd_range_kernel_fn enqueue_nd_range_kernel() const { return enqueue_nd_range_kernel_; }
    [[nodiscard]] cl_finish_fn finish() const { return finish_; }

private:
    LibraryHandle library_ = nullptr;
    bool loaded_ = false;
    cl_get_platform_ids_fn get_platform_ids_ = nullptr;
    cl_get_platform_info_fn get_platform_info_ = nullptr;
    cl_get_device_ids_fn get_device_ids_ = nullptr;
    cl_get_device_info_fn get_device_info_ = nullptr;
    cl_create_context_fn create_context_ = nullptr;
    cl_create_command_queue_fn create_command_queue_ = nullptr;
    cl_create_command_queue_with_properties_fn create_command_queue_with_properties_ = nullptr;
    cl_release_context_fn release_context_ = nullptr;
    cl_release_command_queue_fn release_command_queue_ = nullptr;
    cl_create_program_with_source_fn create_program_with_source_ = nullptr;
    cl_build_program_fn build_program_ = nullptr;
    cl_get_program_build_info_fn get_program_build_info_ = nullptr;
    cl_release_program_fn release_program_ = nullptr;
    cl_create_kernel_fn create_kernel_ = nullptr;
    cl_set_kernel_arg_fn set_kernel_arg_ = nullptr;
    cl_release_kernel_fn release_kernel_ = nullptr;
    cl_create_buffer_fn create_buffer_ = nullptr;
    cl_release_mem_object_fn release_mem_object_ = nullptr;
    cl_enqueue_write_buffer_fn enqueue_write_buffer_ = nullptr;
    cl_enqueue_read_buffer_fn enqueue_read_buffer_ = nullptr;
    cl_enqueue_nd_range_kernel_fn enqueue_nd_range_kernel_ = nullptr;
    cl_finish_fn finish_ = nullptr;
};

// Intentionally concise here: keep existing OpenCL behavior, but behind the backend interface.
// Additional backend implementations can be added without changing the orchestrator.

class OpenClKernelBackend final : public IKernelBackend {
public:
    [[nodiscard]] bool matches(const HardwareGraph& graph) const override { return graph.probe == "opencl"; }
    [[nodiscard]] std::string name() const override { return "opencl-direct"; }

    BackendRunResult run_elementwise(const HardwareGraph&, std::span<const float>, std::span<const float>, bool) const override {
        return BackendRunResult{{}, 0.0, 0.0, false, "opencl-migrated-later"};
    }
    BackendRunResult run_reduction(const HardwareGraph&, std::span<const float>, bool) const override {
        return BackendRunResult{{}, 0.0, 0.0, false, "opencl-migrated-later"};
    }
    BackendRunResult run_matmul(const HardwareGraph&, std::span<const float>, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, bool) const override {
        return BackendRunResult{{}, 0.0, 0.0, false, "opencl-migrated-later"};
    }
    BackendRunResult run_conv3x3(const HardwareGraph&, std::span<const float>, std::uint32_t, std::uint32_t, bool) const override {
        return BackendRunResult{{}, 0.0, 0.0, false, "opencl-migrated-later"};
    }
    BackendRunResult run_resample(const HardwareGraph&, std::span<const float>, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const override {
        return BackendRunResult{{}, 0.0, 0.0, false, "opencl-migrated-later"};
    }
};

}  // namespace

std::unique_ptr<IKernelBackend> make_host_kernel_backend() {
    return std::make_unique<HostKernelBackend>();
}

std::unique_ptr<IKernelBackend> make_opencl_kernel_backend() {
    return std::make_unique<OpenClKernelBackend>();
}

DirectDeviceExecutor::DirectDeviceExecutor(std::vector<std::unique_ptr<IKernelBackend>> backends)
    : backends_(std::move(backends)) {}

bool DirectDeviceExecutor::matches(const HardwareGraph&) const {
    return true;
}

std::string DirectDeviceExecutor::name() const {
    return "direct-device-executor";
}

BackendRunResult DirectDeviceExecutor::dispatch(
    const DeviceAssignment& assignment,
    const OperationOptimizationResult& operation,
    const OperationData& data) const {
    const auto& graph = *assignment.graph;
    const bool low_precision = operation.config.use_low_precision;
    for (const auto& backend : backends_) {
        if (!backend->matches(graph)) {
            continue;
        }

        switch (operation.operation.op_class) {
        case OperationClass::elementwise_map: {
            const auto begin = assignment.shard.start;
            return backend->run_elementwise(
                graph,
                std::span<const float>(data.input0.data() + static_cast<std::ptrdiff_t>(begin), assignment.shard.count),
                std::span<const float>(data.input1.data() + static_cast<std::ptrdiff_t>(begin), assignment.shard.count),
                low_precision);
        }
        case OperationClass::reduction:
            return backend->run_reduction(graph, data.input0, low_precision);
        case OperationClass::matmul: {
            const auto rows = static_cast<std::uint32_t>(assignment.shard.count);
            const auto columns = static_cast<std::uint32_t>(operation.operation.extents.at(1));
            const auto depth = static_cast<std::uint32_t>(operation.operation.extents.at(2));
            const auto begin = assignment.shard.start * static_cast<std::size_t>(depth);
            return backend->run_matmul(
                graph,
                std::span<const float>(data.input0.data() + static_cast<std::ptrdiff_t>(begin), assignment.shard.count * depth),
                data.input1,
                rows,
                columns,
                depth,
                low_precision);
        }
        case OperationClass::convolution_2d: {
            const auto height = static_cast<std::uint32_t>(operation.operation.extents.at(0));
            const auto width = static_cast<std::uint32_t>(operation.operation.extents.at(1));
            return backend->run_conv3x3(graph, data.input0, height, width, low_precision);
        }
        case OperationClass::resample_2d:
        default: {
            return backend->run_resample(
                graph,
                data.input0,
                static_cast<std::uint32_t>(operation.operation.extents.at(0)),
                static_cast<std::uint32_t>(operation.operation.extents.at(1)),
                static_cast<std::uint32_t>(operation.operation.extents.at(2)),
                static_cast<std::uint32_t>(operation.operation.extents.at(3)),
                static_cast<std::uint32_t>(assignment.shard.start),
                static_cast<std::uint32_t>(assignment.shard.count),
                low_precision);
        }
        }
    }

    return BackendRunResult{{}, 0.0, 0.0, false, "no-matching-backend"};
}

}  // namespace gpu::executors
