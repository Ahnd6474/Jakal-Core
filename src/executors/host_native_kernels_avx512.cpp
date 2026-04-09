#include "jakal/executors/host_native_kernels.hpp"
#include "jakal/executors/host_thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <immintrin.h>

namespace jakal::executors {
namespace {

template <typename Func>
void run_parallel_rows(const std::uint32_t rows, const std::size_t min_chunk, Func&& func) {
    HostThreadPool::instance().parallel_for(rows, min_chunk, std::forward<Func>(func));
}

constexpr std::uint32_t kHostMatmulAvx512Mr = 8u;
constexpr std::uint32_t kHostMatmulAvx512Nr = 16u;

std::uint32_t effective_tile_hint(
    const std::uint32_t requested,
    const std::uint32_t fallback,
    const std::uint32_t multiple,
    const std::uint32_t minimum) {
    const auto value = requested == 0u ? fallback : requested;
    return std::max(minimum, ((std::max(value, minimum) + multiple - 1u) / multiple) * multiple);
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

void microkernel_fp32_8x16_avx512(
    std::span<const float> lhs,
    const std::uint32_t depth,
    std::span<const float> packed_rhs,
    const std::uint32_t k_count,
    const std::uint32_t k_begin,
    const std::uint32_t row,
    const std::uint32_t col_count,
    const std::uint32_t col_offset,
    const std::uint32_t rows_in_tile,
    const std::uint32_t cols_in_tile,
    std::span<float> output,
    const std::uint32_t columns) {
    __m512 acc[kHostMatmulAvx512Mr];
    for (std::uint32_t m = 0u; m < kHostMatmulAvx512Mr; ++m) {
        if (m < rows_in_tile) {
            if (cols_in_tile == kHostMatmulAvx512Nr) {
                acc[m] = _mm512_loadu_ps(
                    output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset);
            } else {
                alignas(64) float partial[kHostMatmulAvx512Nr]{};
                for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                    partial[n] =
                        output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n];
                }
                acc[m] = _mm512_load_ps(partial);
            }
        } else {
            acc[m] = _mm512_setzero_ps();
        }
    }

    for (std::uint32_t k = 0u; k < k_count; ++k) {
        alignas(64) float rhs_lane[kHostMatmulAvx512Nr]{};
        const float* rhs_ptr = packed_rhs.data() + static_cast<std::size_t>(k) * col_count;
        for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
            rhs_lane[n] = rhs_ptr[n];
        }
        const __m512 rhs_vec = _mm512_load_ps(rhs_lane);
        for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
            const float lhs_scalar =
                lhs[(static_cast<std::size_t>(row + m) * depth) + k_begin + k];
            acc[m] = _mm512_add_ps(acc[m], _mm512_mul_ps(_mm512_set1_ps(lhs_scalar), rhs_vec));
        }
    }

    for (std::uint32_t m = 0u; m < rows_in_tile; ++m) {
        if (cols_in_tile == kHostMatmulAvx512Nr) {
            _mm512_storeu_ps(
                output.data() +
                    (static_cast<std::size_t>(row + m) * columns) +
                    col_offset,
                acc[m]);
        } else {
            alignas(64) float partial[kHostMatmulAvx512Nr]{};
            _mm512_store_ps(partial, acc[m]);
            for (std::uint32_t n = 0u; n < cols_in_tile; ++n) {
                output[(static_cast<std::size_t>(row + m) * columns) + col_offset + n] =
                    partial[n];
            }
        }
    }
}

void run_fp32_blocked_matmul_rows_avx512(
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k,
    const std::size_t row_begin,
    const std::size_t row_end,
    std::span<float> output) {
    (void)rows;
    const auto row_start = static_cast<std::uint32_t>(row_begin);
    const auto row_limit = static_cast<std::uint32_t>(row_end);
    for (std::uint32_t col_block = 0u; col_block < columns; col_block += tile_n) {
        const auto col_count = std::min(tile_n, columns - col_block);
        std::vector<float> packed_rhs(static_cast<std::size_t>(tile_k) * col_count, 0.0f);
        for (std::uint32_t k_block = 0u; k_block < depth; k_block += tile_k) {
            const auto k_count = std::min(tile_k, depth - k_block);
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

            for (std::uint32_t row_block = row_start; row_block < row_limit; row_block += tile_m) {
                const auto row_block_end = std::min(row_limit, row_block + tile_m);
                for (std::uint32_t row = row_block; row < row_block_end; row += kHostMatmulAvx512Mr) {
                    const auto rows_in_tile = std::min(kHostMatmulAvx512Mr, row_block_end - row);
                    for (std::uint32_t n = 0u; n < col_count; n += kHostMatmulAvx512Nr) {
                        const auto cols_in_tile = std::min(kHostMatmulAvx512Nr, col_count - n);
                        microkernel_fp32_8x16_avx512(
                            lhs,
                            depth,
                            std::span<const float>(
                                packed_rhs.data() + n,
                                static_cast<std::size_t>(k_count) * col_count - n),
                            k_count,
                            k_block,
                            row,
                            col_count,
                            col_block + n,
                            rows_in_tile,
                            cols_in_tile,
                            output,
                            columns);
                    }
                }
            }
        }
    }
}

}  // namespace

bool host_native_avx512_available() {
    return true;
}

bool try_run_host_native_matmul_avx512(
    const HardwareGraph& graph,
    std::span<const float> lhs,
    std::span<const float> rhs,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint32_t depth,
    const bool rhs_transposed,
    const std::uint32_t tile_m,
    const std::uint32_t tile_n,
    const std::uint32_t tile_k,
    const std::uint32_t parallel_chunk_rows,
    std::span<float> output) {
    const auto summary = summarize_graph(graph);
    if (graph.probe != "host" ||
        summary.native_vector_bits < 512u ||
        columns < kHostMatmulAvx512Nr ||
        depth < 16u) {
        return false;
    }

    const auto avx512_tile_m = effective_tile_hint(tile_m, 64u, kHostMatmulAvx512Mr, kHostMatmulAvx512Mr);
    const auto avx512_tile_n = effective_tile_hint(tile_n, 96u, kHostMatmulAvx512Nr, kHostMatmulAvx512Nr);
    const auto avx512_tile_k = effective_tile_hint(tile_k, 128u, 16u, 16u);
    const auto min_chunk = std::max<std::size_t>(1u, parallel_chunk_rows == 0u ? 4u : parallel_chunk_rows);
    run_parallel_rows(rows, min_chunk, [&](const std::size_t begin, const std::size_t end) {
        run_fp32_blocked_matmul_rows_avx512(
            lhs,
            rhs,
            rows,
            columns,
            depth,
            rhs_transposed,
            avx512_tile_m,
            avx512_tile_n,
            avx512_tile_k,
            begin,
            end,
            output);
    });
    return true;
}

}  // namespace jakal::executors
