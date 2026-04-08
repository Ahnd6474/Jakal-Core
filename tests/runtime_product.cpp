#include "jakal/runtime.hpp"
#include "jakal/workloads.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(nonce) + extension);
}

void write_manifest(
    const std::filesystem::path& path,
    const std::uint64_t working_set_bytes,
    const std::uint64_t tensor_bytes,
    const bool include_graph = true,
    const std::filesystem::path& weight_asset_path = {}) {
    std::ofstream output(path, std::ios::trunc);
    output << "[workload]\n";
    output << "name=manifest-exec\n";
    output << "kind=inference\n";
    output << "dataset_tag=manifest-exec-lite\n";
    output << "phase=decode\n";
    output << "working_set_bytes=" << working_set_bytes << "\n";
    output << "host_exchange_bytes=1048576\n";
    output << "estimated_flops=12000000\n";
    output << "batch_size=1\n";
    output << "latency_sensitive=true\n";
    output << "prefer_unified_memory=true\n";
    output << "matrix_friendly=false\n\n";

    if (!weight_asset_path.empty()) {
        output << "[asset]\n";
        output << "id=weights-shard\n";
        output << "path=" << weight_asset_path.string() << "\n";
        output << "tensor_ids=weights\n";
        output << "preload_required=true\n";
        output << "persistent=true\n";
        output << "host_visible=true\n\n";
    }

    if (!include_graph) {
        return;
    }

    output << "[tensor]\n";
    output << "id=input\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "consumers=normalize\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=weights\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "consumers=normalize\n";
    output << "persistent=true\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=hidden\n";
    output << "bytes=" << tensor_bytes << "\n";
    output << "producer=normalize\n";
    output << "consumers=score\n\n";

    output << "[tensor]\n";
    output << "id=score-out\n";
    output << "bytes=4\n";
    output << "producer=score\n\n";

    output << "[operation]\n";
    output << "name=normalize\n";
    output << "class=elementwise_map\n";
    output << "extents=4096\n";
    output << "input_bytes=" << tensor_bytes * 2ull << "\n";
    output << "output_bytes=" << tensor_bytes << "\n";
    output << "estimated_flops=8192\n";
    output << "parallelizable=true\n";
    output << "streaming_friendly=true\n";
    output << "inputs=input,weights\n";
    output << "outputs=hidden\n\n";

    output << "[operation]\n";
    output << "name=score\n";
    output << "class=reduction\n";
    output << "extents=4096\n";
    output << "input_bytes=" << tensor_bytes << "\n";
    output << "output_bytes=4\n";
    output << "estimated_flops=4096\n";
    output << "parallelizable=true\n";
    output << "reduction_like=true\n";
    output << "inputs=hidden\n";
    output << "outputs=score-out\n";
}

void write_spatial_manifest(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_path) {
    std::ofstream output(path, std::ios::trunc);
    output << "[workload]\n";
    output << "name=manifest-spatial\n";
    output << "kind=gaming\n";
    output << "dataset_tag=manifest-spatial-lite\n";
    output << "phase=decode\n";
    output << "working_set_bytes=16777216\n";
    output << "host_exchange_bytes=2097152\n";
    output << "estimated_flops=48000000\n";
    output << "batch_size=1\n";
    output << "latency_sensitive=true\n";
    output << "prefer_unified_memory=true\n\n";

    output << "[asset]\n";
    output << "id=frame-history\n";
    output << "path=" << asset_path.string() << "\n";
    output << "tensor_ids=frame\n";
    output << "preload_required=true\n";
    output << "persistent=true\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=frame\n";
    output << "bytes=4096\n";
    output << "persistent=true\n";
    output << "host_visible=true\n";
    output << "consumers=spatial-conv,spatial-upsample\n\n";

    output << "[tensor]\n";
    output << "id=conv-out\n";
    output << "bytes=3600\n";
    output << "producer=spatial-conv\n\n";

    output << "[tensor]\n";
    output << "id=upscale-out\n";
    output << "bytes=9216\n";
    output << "producer=spatial-upsample\n\n";

    output << "[operation]\n";
    output << "name=spatial-conv\n";
    output << "class=convolution_2d\n";
    output << "extents=32,32\n";
    output << "input_bytes=4096\n";
    output << "output_bytes=3600\n";
    output << "estimated_flops=129600\n";
    output << "parallelizable=true\n";
    output << "inputs=frame\n";
    output << "outputs=conv-out\n\n";

    output << "[operation]\n";
    output << "name=spatial-upsample\n";
    output << "class=resample_2d\n";
    output << "extents=32,32,48,48\n";
    output << "input_bytes=4096\n";
    output << "output_bytes=9216\n";
    output << "estimated_flops=36864\n";
    output << "parallelizable=true\n";
    output << "streaming_friendly=true\n";
    output << "inputs=frame\n";
    output << "outputs=upscale-out\n";
}

void write_matmul_manifest(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_path) {
    std::ofstream output(path, std::ios::trunc);
    output << "[workload]\n";
    output << "name=manifest-matmul\n";
    output << "kind=inference\n";
    output << "dataset_tag=manifest-matmul-lite\n";
    output << "phase=prefill\n";
    output << "working_set_bytes=16777216\n";
    output << "host_exchange_bytes=1048576\n";
    output << "estimated_flops=65536\n";
    output << "batch_size=1\n";
    output << "latency_sensitive=true\n";
    output << "matrix_friendly=true\n\n";

    output << "[asset]\n";
    output << "id=matmul-weights\n";
    output << "path=" << asset_path.string() << "\n";
    output << "tensor_ids=weights\n";
    output << "preload_required=true\n";
    output << "persistent=true\n";
    output << "host_visible=true\n\n";

    output << "[tensor]\n";
    output << "id=input\n";
    output << "bytes=4096\n";
    output << "host_visible=true\n";
    output << "consumers=patch-proj\n\n";

    output << "[tensor]\n";
    output << "id=weights\n";
    output << "bytes=4096\n";
    output << "persistent=true\n";
    output << "host_visible=true\n";
    output << "consumers=patch-proj\n\n";

    output << "[tensor]\n";
    output << "id=proj-out\n";
    output << "bytes=4096\n";
    output << "producer=patch-proj\n\n";

    output << "[operation]\n";
    output << "name=patch-proj\n";
    output << "class=matmul\n";
    output << "extents=32,32,32\n";
    output << "input_bytes=8192\n";
    output << "output_bytes=4096\n";
    output << "estimated_flops=65536\n";
    output << "parallelizable=true\n";
    output << "matrix_friendly=true\n";
    output << "inputs=input,weights\n";
    output << "outputs=proj-out\n";
}

}  // namespace

int main() {
    try {
        const auto manifest_path = unique_temp_file("runtime-product-graph", ".workload");
        const auto runtime_manifest_path = unique_temp_file("runtime-product-runtime", ".workload");
        const auto blocked_manifest_path = unique_temp_file("runtime-product-blocked", ".workload");
        const auto missing_asset_manifest_path = unique_temp_file("runtime-product-missing-asset", ".workload");
        const auto spatial_manifest_path = unique_temp_file("runtime-product-spatial", ".workload");
        const auto matmul_manifest_path = unique_temp_file("runtime-product-matmul", ".workload");
        const auto weight_asset_path = unique_temp_file("runtime-product-weights", ".bin");
        const auto missing_weight_asset_path = unique_temp_file("runtime-product-weights-missing", ".bin");
        const auto cache_path = unique_temp_file("runtime-product-cache", ".tsv");

        {
            std::ofstream weights(weight_asset_path, std::ios::binary | std::ios::trunc);
            std::string payload(16ull * 1024ull, '\x5a');
            weights.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        }

        write_manifest(manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, true, weight_asset_path);
        write_manifest(runtime_manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, false);
        write_manifest(blocked_manifest_path, 1ull << 50u, 4ull * 1024ull * 1024ull * 1024ull, true);
        write_manifest(missing_asset_manifest_path, 8ull * 1024ull * 1024ull, 16ull * 1024ull, true, missing_weight_asset_path);
        write_spatial_manifest(spatial_manifest_path, weight_asset_path);
        write_matmul_manifest(matmul_manifest_path, weight_asset_path);

        const auto manifest = jakal::load_workload_manifest(manifest_path);
        if (!manifest.has_graph || manifest.graph.operations.size() != 2u || manifest.graph.tensors.size() != 4u) {
            std::cerr << "manifest graph parsing failed\n";
            return 1;
        }
        if (manifest.assets.size() != 1u || manifest.assets.front().bytes != 16ull * 1024ull) {
            std::cerr << "manifest asset parsing failed\n";
            return 1;
        }

        jakal::RuntimeOptions options;
        options.enable_host_probe = true;
        options.enable_opencl_probe = false;
        options.enable_level_zero_probe = false;
        options.enable_cuda_probe = false;
        options.enable_rocm_probe = false;
        options.cache_path = cache_path;
        options.product.observability.telemetry_path = unique_temp_file("runtime-product-managed", ".telemetry.tsv");

        jakal::HardwareGraph synthetic_level_zero;
        synthetic_level_zero.uid = "synthetic:level-zero";
        synthetic_level_zero.probe = "level-zero";
        synthetic_level_zero.presentation_name = "Intel Synthetic GPU";
        synthetic_level_zero.driver_version = "1.2.3";
        synthetic_level_zero.runtime_version = "1.2.3";
        synthetic_level_zero.compiler_version = "ocloc-1";
        const auto level_zero_tag = jakal::runtime_backend_cache_tag_for_graph(synthetic_level_zero);
        synthetic_level_zero.driver_version = "1.2.4";
        const auto changed_driver_tag = jakal::runtime_backend_cache_tag_for_graph(synthetic_level_zero);
        synthetic_level_zero.driver_version = "1.2.3";
        synthetic_level_zero.compiler_version = "ocloc-2";
        const auto changed_compiler_tag = jakal::runtime_backend_cache_tag_for_graph(synthetic_level_zero);
        if (level_zero_tag == changed_driver_tag || level_zero_tag == changed_compiler_tag) {
            std::cerr << "backend cache tag did not react to version changes\n";
            return 1;
        }

        jakal::Runtime runtime(options);
        const auto optimized = runtime.optimize(manifest.workload, manifest.graph);
        if (optimized.operations.empty()) {
            std::cerr << "custom graph optimize produced no operations\n";
            return 1;
        }
        const auto manifest_managed = runtime.execute_manifest(manifest_path);
        if (!manifest_managed.executed) {
            std::cerr << "managed execute did not run custom manifest path\n";
            return 1;
        }
        if (manifest_managed.execution.optimization.operations.size() != manifest.graph.operations.size()) {
            std::cerr << "managed execute optimized unexpected custom graph operation count\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.entries.empty() || manifest_managed.asset_prefetch.total_prefetch_bytes != 16ull * 1024ull) {
            std::cerr << "managed execute did not produce asset prefetch plan\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.total_host_io_bytes != 16ull * 1024ull ||
            manifest_managed.asset_prefetch.total_host_to_device_bytes != 0u) {
            std::cerr << "managed execute asset queue accounting mismatch\n";
            return 1;
        }
        if (manifest_managed.asset_prefetch.entries.front().queue_hint != "host_io" ||
            manifest_managed.asset_prefetch.entries.front().target_residency != "auto") {
            std::cerr << "managed execute asset queue hint mismatch\n";
            return 1;
        }
        if (!manifest_managed.kernel_coverage.all_supported) {
            std::cerr << "host-only managed path should have full kernel coverage\n";
            return 1;
        }
        if (manifest_managed.planning.resolved_partition_strategy != manifest_managed.execution.optimization.partition_strategy ||
            manifest_managed.safety.planner_confidence < 0.0 ||
            manifest_managed.safety.planner_strategy_source == jakal::PlanStrategySource::exploration) {
            std::cerr << "managed execute planner diagnostics mismatch\n";
            return 1;
        }
        if (manifest_managed.residency_sequence.actions.empty()) {
            std::cerr << "managed execute did not emit residency sequence\n";
            return 1;
        }
        if (manifest_managed.residency_sequence.indexed_tensors.empty() ||
            manifest_managed.residency_sequence.indexed_devices.empty() ||
            manifest_managed.residency_sequence.indexed_operations.empty()) {
            std::cerr << "managed execute did not index residency sequence metadata\n";
            return 1;
        }
        for (const auto& action : manifest_managed.residency_sequence.actions) {
            if (action.tensor_index >= manifest_managed.residency_sequence.indexed_tensors.size() ||
                manifest_managed.residency_sequence.indexed_tensors[action.tensor_index] != action.tensor_id ||
                action.device_index >= manifest_managed.residency_sequence.indexed_devices.size() ||
                manifest_managed.residency_sequence.indexed_devices[action.device_index] != action.device_uid ||
                action.operation_index >= manifest_managed.residency_sequence.indexed_operations.size() ||
                manifest_managed.residency_sequence.indexed_operations[action.operation_index] !=
                    action.trigger_operation_name) {
                std::cerr << "managed execute emitted invalid residency action indices\n";
                return 1;
            }
        }
        if (!manifest_managed.executed ||
            manifest_managed.execution.total_predicted_transfer_runtime_us <= 0.0 ||
            manifest_managed.execution.transfer_overlap_ratio < 0.0 ||
            manifest_managed.execution.transfer_overlap_ratio > 1.0) {
            std::cerr << "managed execute did not capture transfer overlap metrics\n";
            return 1;
        }

        const auto runtime_managed = runtime.execute_manifest(runtime_manifest_path);
        if (!runtime_managed.executed) {
            std::cerr << "managed execute did not run spec-only manifest path\n";
            return 1;
        }
        if (runtime_managed.execution.optimization.operations.empty()) {
            std::cerr << "spec-only manifest execute produced no optimized operations\n";
            return 1;
        }

        jakal::RuntimeOptions blocked_options = options;
        blocked_options.product.memory.max_pressure_ratio = 0.01;
        blocked_options.product.memory.enforce_preflight = true;
        blocked_options.product.observability.telemetry_path = unique_temp_file("runtime-product-blocked", ".telemetry.tsv");
        jakal::Runtime blocked_runtime(blocked_options);
        const auto blocked = blocked_runtime.execute_manifest(blocked_manifest_path);
        if (blocked.executed || !blocked.safety.blocked_by_memory) {
            std::cerr << "memory safety gate did not block oversized workload\n";
            return 1;
        }
        if (!std::filesystem::exists(blocked.telemetry_path)) {
            std::cerr << "blocked managed path did not emit telemetry\n";
            return 1;
        }
        if (blocked.memory_preflight.devices.empty()) {
            std::cerr << "memory preflight did not capture device reservations\n";
            return 1;
        }
        if (blocked.residency_sequence.actions.empty()) {
            std::cerr << "blocked managed path did not emit residency sequence\n";
            return 1;
        }
        if (blocked.residency_sequence.indexed_tensors.empty() ||
            blocked.residency_sequence.indexed_devices.empty() ||
            blocked.residency_sequence.indexed_operations.empty()) {
            std::cerr << "blocked managed path did not index residency metadata\n";
            return 1;
        }
        if (blocked.memory_preflight.predicted_spill_bytes != blocked.residency_sequence.spill_bytes ||
            blocked.memory_preflight.predicted_reload_bytes != blocked.residency_sequence.reload_bytes ||
            blocked.memory_preflight.forced_spill_count != blocked.residency_sequence.forced_spill_count) {
            std::cerr << "memory preflight and residency sequence drifted\n";
            return 1;
        }
        if (blocked.memory_preflight.predicted_spill_bytes == 0u && blocked.memory_preflight.forced_spill_count == 0u) {
            std::cerr << "blocked managed path did not predict spill pressure\n";
            return 1;
        }

        const auto missing_asset = runtime.execute_manifest(missing_asset_manifest_path);
        if (missing_asset.executed || !missing_asset.asset_prefetch.missing_required_assets) {
            std::cerr << "missing required asset was not blocked\n";
            return 1;
        }

        const auto spatial = runtime.execute_manifest(spatial_manifest_path);
        if (!spatial.executed) {
            std::cerr << "spatial manifest did not execute\n";
            return 1;
        }
        if (spatial.asset_prefetch.total_layout_cache_bytes == 0u) {
            std::cerr << "spatial manifest did not emit layout cache bytes\n";
            return 1;
        }
        const auto has_conv_cache = std::any_of(
            spatial.asset_prefetch.entries.begin(),
            spatial.asset_prefetch.entries.end(),
            [](const jakal::AssetPrefetchEntry& entry) {
                return entry.derived_cache && entry.materialization_kind == "cpu-conv-patch9";
            });
        const auto has_resample_cache = std::any_of(
            spatial.asset_prefetch.entries.begin(),
            spatial.asset_prefetch.entries.end(),
            [](const jakal::AssetPrefetchEntry& entry) {
                return entry.derived_cache && entry.materialization_kind == "cpu-resample-packed6";
            });
        if (!has_conv_cache || !has_resample_cache) {
            std::cerr << "spatial manifest missing lowered layout cache entries\n";
            return 1;
        }
        const auto persisted_conv_cache = std::find_if(
            spatial.asset_prefetch.entries.begin(),
            spatial.asset_prefetch.entries.end(),
            [](const jakal::AssetPrefetchEntry& entry) {
                return entry.derived_cache && entry.materialization_kind == "cpu-conv-patch9" &&
                       entry.path.extension() == ".jpkd" && entry.exists_on_disk;
            });
        if (persisted_conv_cache == spatial.asset_prefetch.entries.end() ||
            !std::filesystem::exists(persisted_conv_cache->path)) {
            std::cerr << "spatial manifest did not persist conv cache blob\n";
            return 1;
        }

        const auto matmul_first = runtime.execute_manifest(matmul_manifest_path);
        const auto first_blob_it = std::find_if(
            matmul_first.asset_prefetch.entries.begin(),
            matmul_first.asset_prefetch.entries.end(),
            [](const jakal::AssetPrefetchEntry& entry) {
                return entry.derived_cache && entry.materialization_kind == "cpu-packed-rhs" &&
                       entry.path.extension() == ".jpkd" && entry.exists_on_disk;
            });
        if (first_blob_it == matmul_first.asset_prefetch.entries.end() ||
            !std::filesystem::exists(first_blob_it->path)) {
            std::cerr << "matmul manifest did not persist packed rhs blob\n";
            return 1;
        }
        const auto first_blob_time = std::filesystem::last_write_time(first_blob_it->path);

        const auto matmul_second = runtime.execute_manifest(matmul_manifest_path);
        const auto second_blob_it = std::find_if(
            matmul_second.asset_prefetch.entries.begin(),
            matmul_second.asset_prefetch.entries.end(),
            [](const jakal::AssetPrefetchEntry& entry) {
                return entry.derived_cache && entry.materialization_kind == "cpu-packed-rhs" &&
                       entry.path.extension() == ".jpkd" && entry.exists_on_disk;
            });
        if (second_blob_it == matmul_second.asset_prefetch.entries.end() ||
            second_blob_it->path != first_blob_it->path) {
            std::cerr << "matmul manifest did not reuse packed rhs blob path\n";
            return 1;
        }

        std::ifstream telemetry(manifest_managed.telemetry_path);
        std::string telemetry_header;
        std::string telemetry_row;
        std::getline(telemetry, telemetry_header);
        std::getline(telemetry, telemetry_row);
        if (telemetry_header.find("transfer_us") == std::string::npos ||
            telemetry_header.find("overlapped_transfer_us") == std::string::npos ||
            telemetry_header.find("transfer_overlap_ratio") == std::string::npos ||
            telemetry_row.empty()) {
            std::cerr << "runtime telemetry missing transfer overlap columns\n";
            return 1;
        }
        const auto second_blob_time = std::filesystem::last_write_time(second_blob_it->path);
        if (second_blob_time != first_blob_time) {
            std::cerr << "matmul manifest rewrote packed rhs blob instead of reusing it\n";
            return 1;
        }

        std::error_code ec;
        std::filesystem::remove(manifest_path, ec);
        std::filesystem::remove(runtime_manifest_path, ec);
        std::filesystem::remove(blocked_manifest_path, ec);
        std::filesystem::remove(missing_asset_manifest_path, ec);
        std::filesystem::remove(spatial_manifest_path, ec);
        std::filesystem::remove(matmul_manifest_path, ec);
        std::filesystem::remove(weight_asset_path, ec);
        std::filesystem::remove(cache_path, ec);
        const auto packed_root = cache_path.parent_path() / (cache_path.stem().string() + "-packed-layouts");
        std::filesystem::remove_all(packed_root, ec);
        std::filesystem::remove(manifest_managed.telemetry_path, ec);
        std::filesystem::remove(runtime_managed.telemetry_path, ec);
        std::filesystem::remove(blocked.telemetry_path, ec);
        std::filesystem::remove(missing_asset.telemetry_path, ec);
        std::filesystem::remove(spatial.telemetry_path, ec);
        std::filesystem::remove(matmul_first.telemetry_path, ec);
        std::filesystem::remove(matmul_second.telemetry_path, ec);

        std::cout << "runtime product path ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << '\n';
        return 1;
    }
}
