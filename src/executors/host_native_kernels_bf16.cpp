#include "jakal/executors/host_native_kernels.hpp"
#include "jakal/executors/host_thread_pool.hpp"
#include "jakal/execution.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
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
    const bool rhs_transposed,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    for (std::size_t row = row_begin; row < row_end; ++row) {
        const auto lhs_base = static_cast<std::size_t>(row) * depth;
        for (std::uint32_t col = 0; col < columns; ++col) {
            __m512 acc = _mm512_setzero_ps();
            std::uint32_t inner = 0;
            const auto rhs_base = static_cast<std::size_t>(col) * depth;
            for (; inner + 32u <= depth; inner += 32u) {
                const __m512 lhs_lo = _mm512_loadu_ps(lhs.data() + lhs_base + inner);
                const __m512 lhs_hi = _mm512_loadu_ps(lhs.data() + lhs_base + inner + 16u);
                const __m512 rhs_lo = rhs_transposed
                    ? _mm512_loadu_ps(rhs.data() + rhs_base + inner)
                    : _mm512_i32gather_ps(
                          _mm512_setr_epi32(
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
                              static_cast<int>((inner + 15u) * columns + col)),
                          rhs.data(),
                          4);
                const __m512 rhs_hi = rhs_transposed
                    ? _mm512_loadu_ps(rhs.data() + rhs_base + inner + 16u)
                    : _mm512_i32gather_ps(
                          _mm512_setr_epi32(
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
                              static_cast<int>((inner + 31u) * columns + col)),
                          rhs.data(),
                          4);
                const __m512bh lhs_bf16 = _mm512_cvtne2ps_pbh(lhs_hi, lhs_lo);
                const __m512bh rhs_bf16 = _mm512_cvtne2ps_pbh(rhs_hi, rhs_lo);
                acc = _mm512_dpbf16_ps(acc, lhs_bf16, rhs_bf16);
            }

            float scalar_acc = quantize_bf16_scalar(horizontal_sum_512(acc));
            for (; inner < depth; ++inner) {
                const float left = quantize_bf16_scalar(lhs[lhs_base + inner]);
                const float right = quantize_bf16_scalar(
                    rhs_transposed
                        ? rhs[rhs_base + inner]
                        : rhs[static_cast<std::size_t>(inner) * columns + col]);
                scalar_acc = quantize_bf16_scalar(scalar_acc + (left * right));
            }
            output[static_cast<std::size_t>(row) * columns + col] = scalar_acc;
        }
    }
}

void pack_rhs_panel(
    std::span<const float> rhs,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const std::uint32_t k_begin,
    const std::uint32_t k_count,
    const std::uint32_t col_begin,
    const std::uint32_t col_count,
    const bool rhs_transposed,
    std::span<float> packed_rhs) {
    for (std::uint32_t k = 0u; k < k_count; ++k) {
        for (std::uint32_t n = 0u; n < col_count; ++n) {
            const auto rhs_index = rhs_transposed
                ? static_cast<std::size_t>(col_begin + n) * depth + (k_begin + k)
                : static_cast<std::size_t>(k_begin + k) * columns + (col_begin + n);
            packed_rhs[static_cast<std::size_t>(k) * col_count + n] = rhs[rhs_index];
        }
    }
}

std::uint32_t effective_tile_hint(
    const std::uint32_t requested,
    const std::uint32_t fallback,
    const std::uint32_t multiple,
    const std::uint32_t minimum) {
    const auto value = requested == 0u ? fallback : requested;
    return std::max(minimum, ((std::max(value, minimum) + multiple - 1u) / multiple) * multiple);
}

std::uint32_t fit_tile_to_pack_budget(
    const std::uint32_t tile_k,
    std::uint32_t tile_n,
    const std::uint64_t pack_budget_bytes) {
    if (pack_budget_bytes == 0u || tile_k == 0u || tile_n == 0u) {
        return tile_n;
    }
    while ((static_cast<std::uint64_t>(tile_k) * tile_n * sizeof(float)) > pack_budget_bytes && tile_n > 8u) {
        tile_n = std::max<std::uint32_t>(8u, tile_n - 8u);
    }
    return tile_n;
}

std::size_t choose_row_chunk(
    const OperationSpec& operation,
    const std::uint32_t rows,
    const std::size_t fallback) {
    if (operation.cpu_single_thread_cutoff > 0u) {
        const auto work_items =
            static_cast<std::uint64_t>(rows) *
            std::max<std::uint64_t>(1ull, operation.extents.size() > 1u ? operation.extents[1] : 1ull) *
            std::max<std::uint64_t>(1ull, operation.extents.size() > 2u ? operation.extents[2] : 1ull);
        if (work_items < static_cast<std::uint64_t>(operation.cpu_single_thread_cutoff)) {
            return static_cast<std::size_t>(rows);
        }
    }
    return std::max<std::size_t>(1u, operation.cpu_parallel_chunk == 0u ? fallback : operation.cpu_parallel_chunk);
}

}  // namespace

bool try_run_host_native_bf16_matmul(
    const HardwareGraph& graph,
    const OperationSpec& operation,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t parallel_chunk_rows,
    std::span<float> output) {
#if defined(_M_X64) || defined(__x86_64__)
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" || !summary.supports_bf16 || summary.native_vector_bits < 512u || depth < 32u) {
        return false;
    }

    const auto effective_tile_m_value = effective_tile_hint(operation.cpu_tile_m, 32u, 4u, 4u);
    auto effective_tile_n_value = effective_tile_hint(operation.cpu_tile_n, 32u, 8u, 8u);
    const auto effective_tile_k_value = effective_tile_hint(operation.cpu_tile_k, 64u, 16u, 16u);
    effective_tile_n_value = fit_tile_to_pack_budget(
        effective_tile_k_value,
        effective_tile_n_value,
        operation.cpu_pack_budget_bytes);
    if (rows <= 2u && rhs_transposed) {
        run_bf16_matmul_rows(lhs, rhs, columns, depth, true, 0u, rows, output);
        return true;
    }
    const auto run_blocked = [&](const std::size_t begin, const std::size_t end) {
        const auto row_start = static_cast<std::uint32_t>(begin);
        const auto row_limit = static_cast<std::uint32_t>(end);
        for (std::uint32_t col_block = 0u; col_block < columns; col_block += effective_tile_n_value) {
            const auto col_count = std::min(effective_tile_n_value, columns - col_block);
            std::vector<float> packed_rhs(static_cast<std::size_t>(effective_tile_k_value) * col_count, 0.0f);
            for (std::uint32_t k_block = 0u; k_block < depth; k_block += effective_tile_k_value) {
                const auto k_count = std::min(effective_tile_k_value, depth - k_block);
                pack_rhs_panel(
                    rhs,
                    columns,
                    depth,
                    k_block,
                    k_count,
                    col_block,
                    col_count,
                    rhs_transposed,
                    std::span<float>(packed_rhs.data(), static_cast<std::size_t>(k_count) * col_count));
                for (std::uint32_t row_block = row_start; row_block < row_limit; row_block += effective_tile_m_value) {
                    const auto row_end = std::min(row_limit, row_block + effective_tile_m_value);
                    for (std::uint32_t row = row_block; row < row_end; ++row) {
                        const auto lhs_base = static_cast<std::size_t>(row) * depth;
                        for (std::uint32_t col = 0u; col < col_count; ++col) {
                            const auto out_index = static_cast<std::size_t>(row) * columns + col_block + col;
                            float scalar_acc = output[out_index];
                            std::uint32_t inner = 0u;
                            for (; inner + 32u <= k_count; inner += 32u) {
                                const __m512 lhs_lo = _mm512_loadu_ps(lhs.data() + lhs_base + k_block + inner);
                                const __m512 lhs_hi = _mm512_loadu_ps(lhs.data() + lhs_base + k_block + inner + 16u);
                                alignas(64) float rhs_lo_lanes[16]{};
                                alignas(64) float rhs_hi_lanes[16]{};
                                for (std::uint32_t lane = 0u; lane < 16u; ++lane) {
                                    rhs_lo_lanes[lane] =
                                        packed_rhs[static_cast<std::size_t>(inner + lane) * col_count + col];
                                    rhs_hi_lanes[lane] =
                                        packed_rhs[static_cast<std::size_t>(inner + 16u + lane) * col_count + col];
                                }
                                const __m512 rhs_lo = _mm512_load_ps(rhs_lo_lanes);
                                const __m512 rhs_hi = _mm512_load_ps(rhs_hi_lanes);
                                const __m512bh lhs_bf16 = _mm512_cvtne2ps_pbh(lhs_hi, lhs_lo);
                                const __m512bh rhs_bf16 = _mm512_cvtne2ps_pbh(rhs_hi, rhs_lo);
                                scalar_acc = quantize_bf16_scalar(
                                    scalar_acc + horizontal_sum_512(_mm512_dpbf16_ps(_mm512_setzero_ps(), lhs_bf16, rhs_bf16)));
                            }
                            for (; inner < k_count; ++inner) {
                                scalar_acc = quantize_bf16_scalar(
                                    scalar_acc +
                                    (quantize_bf16_scalar(lhs[lhs_base + k_block + inner]) *
                                     quantize_bf16_scalar(
                                         packed_rhs[static_cast<std::size_t>(inner) * col_count + col])));
                            }
                            output[out_index] = scalar_acc;
                        }
                    }
                }
            }
        }
    };

    const auto min_chunk = choose_row_chunk(operation, rows, parallel_chunk_rows == 0u ? 4u : parallel_chunk_rows);
    if (min_chunk >= rows) {
        run_blocked(0u, rows);
    } else {
        HostThreadPool::instance().parallel_for(rows, min_chunk, run_blocked);
    }
    return true;
#else
    (void)graph;
    (void)operation;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)columns;
    (void)depth;
    (void)rhs_transposed;
    (void)parallel_chunk_rows;
    (void)output;
    return false;
#endif
}

}  // namespace jakal::executors
