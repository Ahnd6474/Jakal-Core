#include "jakal/executors/host_native_kernels.hpp"

#include <algorithm>
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

float quantize_fp16_scalar(const float value) {
    if (!std::isfinite(value)) {
        return value;
    }

    const std::uint32_t bits = float_to_bits(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    std::uint16_t half = 0;

    if (exponent <= 0) {
        if (exponent < -10) {
            half = static_cast<std::uint16_t>(sign);
        } else {
            mantissa |= 0x800000u;
            const auto shift = static_cast<std::uint32_t>(14 - exponent);
            std::uint32_t rounded = mantissa >> shift;
            if (((mantissa >> (shift - 1u)) & 1u) != 0u) {
                rounded += 1u;
            }
            half = static_cast<std::uint16_t>(sign | (rounded & 0x03ffu));
        }
    } else if (exponent >= 31) {
        half = static_cast<std::uint16_t>(sign | 0x7c00u);
    } else {
        mantissa += 0x1000u;
        if ((mantissa & 0x00800000u) != 0u) {
            mantissa = 0;
            ++exponent;
        }
        if (exponent >= 31) {
            half = static_cast<std::uint16_t>(sign | 0x7c00u);
        } else {
            half = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) |
                                              ((mantissa >> 13u) & 0x03ffu));
        }
    }

    const std::uint32_t half_sign = (static_cast<std::uint32_t>(half & 0x8000u)) << 16u;
    const std::uint32_t half_exponent = (half >> 10u) & 0x1fu;
    const std::uint32_t half_mantissa = half & 0x03ffu;
    std::uint32_t restored = 0;
    if (half_exponent == 0u) {
        if (half_mantissa == 0u) {
            restored = half_sign;
        } else {
            std::uint32_t mantissa_norm = half_mantissa;
            int exp = -14;
            while ((mantissa_norm & 0x0400u) == 0u) {
                mantissa_norm <<= 1u;
                --exp;
            }
            mantissa_norm &= 0x03ffu;
            restored = half_sign | (static_cast<std::uint32_t>(exp + 127) << 23u) | (mantissa_norm << 13u);
        }
    } else if (half_exponent == 0x1fu) {
        restored = half_sign | 0x7f800000u | (half_mantissa << 13u);
    } else {
        restored = half_sign | ((half_exponent + 112u) << 23u) | (half_mantissa << 13u);
    }
    return bits_to_float(restored);
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

#if defined(__AVX2__) || defined(_MSC_VER)
float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void run_fp16_matmul_rows(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    constexpr int rounding = _MM_FROUND_TO_NEAREST_INT;
    for (std::size_t row = row_begin; row < row_end; ++row) {
        const auto lhs_base = static_cast<std::size_t>(row) * depth;
        for (std::uint32_t col = 0; col < columns; ++col) {
            __m256 acc = _mm256_setzero_ps();
            std::uint32_t inner = 0;
            for (; inner + 8u <= depth; inner += 8u) {
                const __m256 lhs_vec = _mm256_loadu_ps(lhs.data() + lhs_base + inner);
                const __m256i rhs_indices = _mm256_setr_epi32(
                    static_cast<int>((inner + 0u) * columns + col),
                    static_cast<int>((inner + 1u) * columns + col),
                    static_cast<int>((inner + 2u) * columns + col),
                    static_cast<int>((inner + 3u) * columns + col),
                    static_cast<int>((inner + 4u) * columns + col),
                    static_cast<int>((inner + 5u) * columns + col),
                    static_cast<int>((inner + 6u) * columns + col),
                    static_cast<int>((inner + 7u) * columns + col));
                const __m256 rhs_vec = _mm256_i32gather_ps(rhs.data(), rhs_indices, 4);
                const __m128i lhs_half = _mm256_cvtps_ph(lhs_vec, rounding);
                const __m128i rhs_half = _mm256_cvtps_ph(rhs_vec, rounding);
                const __m256 lhs_quant = _mm256_cvtph_ps(lhs_half);
                const __m256 rhs_quant = _mm256_cvtph_ps(rhs_half);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(lhs_quant, rhs_quant));
            }

            float scalar_acc = horizontal_sum(acc);
            for (; inner < depth; ++inner) {
                const float left = quantize_fp16_scalar(lhs[lhs_base + inner]);
                const float right = quantize_fp16_scalar(rhs[static_cast<std::size_t>(inner) * columns + col]);
                scalar_acc = quantize_fp16_scalar(scalar_acc + (left * right));
            }
            output[static_cast<std::size_t>(row) * columns + col] = scalar_acc;
        }
    }
}
#endif

bool try_run_fp16_native_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    std::span<float> output) {
#if defined(__AVX2__) || defined(_MSC_VER)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || !summary.supports_fp16 || summary.native_vector_bits < 256u || depth < 8u) {
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
            run_fp16_matmul_rows(lhs, rhs, columns, depth, begin, end, output);
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

}  // namespace

bool try_run_host_native_low_precision_matmul(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool prefer_fp16,
    const bool prefer_bf16,
    std::span<float> output) {
    if (prefer_bf16 &&
        try_run_host_native_bf16_matmul(graph, lhs, rhs, rows, columns, depth, output)) {
        return true;
    }
    if (prefer_fp16 &&
        try_run_fp16_native_matmul(graph, lhs, rhs, rows, columns, depth, output)) {
        return true;
    }
    return false;
}

}  // namespace jakal::executors
