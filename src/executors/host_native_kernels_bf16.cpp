#include "jakal/executors/host_native_kernels.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace jakal::executors {
namespace {

std::uint32_t float_to_bits(const float value) {
    return std::bit_cast<std::uint32_t>(value);
}

float bits_to_float(const std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

float quantize_bf16_scalar(const float value) {
    if (!std::isfinite(value)) {
        return value;
    }

    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t lsb = (bits >> 16u) & 1u;
    const std::uint32_t rounded = bits + 0x7fffu + lsb;
    return bits_to_float(rounded & 0xffff0000u);
}

float horizontal_sum_512(const __m512 value) {
    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), value);
    float sum = 0.0f;
    for (const auto lane : lanes) {
        sum += lane;
    }
    return sum;
}

void run_bf16_matmul_rows(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    for (std::size_t row = row_begin; row < row_end; ++row) {
        const auto lhs_base = static_cast<std::size_t>(row) * depth;
        for (std::uint32_t col = 0; col < columns; ++col) {
            __m512 acc = _mm512_setzero_ps();
            std::uint32_t inner = 0;
            for (; inner + 32u <= depth; inner += 32u) {
                const __m512 lhs_lo = _mm512_loadu_ps(lhs.data() + lhs_base + inner);
                const __m512 lhs_hi = _mm512_loadu_ps(lhs.data() + lhs_base + inner + 16u);
                const __m512i rhs_lo_indices = _mm512_setr_epi32(
                    static_cast<int>((inner + 0u) * columns + col),
                    static_cast<int>((inner + 1u) * columns + col),
                    static_cast<int>((inner + 2u) * columns + col),
                    static_cast<int>((inner + 3u) * columns + col),
                    static_cast<int>((inner + 4u) * columns + col),
                    static_cast<int>((inner + 5u) * columns + col),
                    static_cast<int>((inner + 6u) * columns + col),
                    static_cast<int>((inner + 7u) * columns + col),
                    static_cast<int>((inner + 8u) * columns + col),
                    static_cast<int>((inner + 9u) * columns + col),
                    static_cast<int>((inner + 10u) * columns + col),
                    static_cast<int>((inner + 11u) * columns + col),
                    static_cast<int>((inner + 12u) * columns + col),
                    static_cast<int>((inner + 13u) * columns + col),
                    static_cast<int>((inner + 14u) * columns + col),
                    static_cast<int>((inner + 15u) * columns + col));
                const __m512i rhs_hi_indices = _mm512_setr_epi32(
                    static_cast<int>((inner + 16u) * columns + col),
                    static_cast<int>((inner + 17u) * columns + col),
                    static_cast<int>((inner + 18u) * columns + col),
                    static_cast<int>((inner + 19u) * columns + col),
                    static_cast<int>((inner + 20u) * columns + col),
                    static_cast<int>((inner + 21u) * columns + col),
                    static_cast<int>((inner + 22u) * columns + col),
                    static_cast<int>((inner + 23u) * columns + col),
                    static_cast<int>((inner + 24u) * columns + col),
                    static_cast<int>((inner + 25u) * columns + col),
                    static_cast<int>((inner + 26u) * columns + col),
                    static_cast<int>((inner + 27u) * columns + col),
                    static_cast<int>((inner + 28u) * columns + col),
                    static_cast<int>((inner + 29u) * columns + col),
                    static_cast<int>((inner + 30u) * columns + col),
                    static_cast<int>((inner + 31u) * columns + col));
                const __m512 rhs_lo = _mm512_i32gather_ps(rhs_lo_indices, rhs.data(), 4);
                const __m512 rhs_hi = _mm512_i32gather_ps(rhs_hi_indices, rhs.data(), 4);
                const __m512bh lhs_bf16 = _mm512_cvtne2ps_pbh(lhs_hi, lhs_lo);
                const __m512bh rhs_bf16 = _mm512_cvtne2ps_pbh(rhs_hi, rhs_lo);
                acc = _mm512_dpbf16_ps(acc, lhs_bf16, rhs_bf16);
            }

            float scalar_acc = quantize_bf16_scalar(horizontal_sum_512(acc));
            for (; inner < depth; ++inner) {
                const float left = quantize_bf16_scalar(lhs[lhs_base + inner]);
                const float right = quantize_bf16_scalar(rhs[static_cast<std::size_t>(inner) * columns + col]);
                scalar_acc = quantize_bf16_scalar(scalar_acc + (left * right));
            }
            output[static_cast<std::size_t>(row) * columns + col] = scalar_acc;
        }
    }
}

}  // namespace

bool try_run_host_native_bf16_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    std::span<float> output) {
#if defined(_M_X64) || defined(__x86_64__)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || !summary.supports_bf16 || summary.native_vector_bits < 512u || depth < 32u) {
        return false;
    }

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
        threads.emplace_back([=, &lhs, &rhs, &output]() {
            run_bf16_matmul_rows(lhs, rhs, columns, depth, begin, end, output);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return true;
#else
    (void)graph;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)columns;
    (void)depth;
    (void)output;
    return false;
#endif
}

}  // namespace jakal::executors
