#include "jakal/device.hpp"
#include "jakal/execution.hpp"
#include "jakal/planner.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path unique_temp_file(const std::string& stem) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "-" + std::to_string(nonce) + ".tsv");
}

jakal::HardwareGraph make_manual_host_graph() {
    jakal::HardwareGraph graph;
    graph.uid = "host:test:0";
    graph.probe = "host";
    graph.presentation_name = "Test CPU";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = false;
    graph.nodes.push_back({"core-cluster", "core-cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 16;
    graph.nodes.back().compute.clock_mhz = 4200;
    graph.nodes.back().compute.supports_fp16 = true;
    graph.nodes.push_back({"host-memory", "host-memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::host_memory});
    graph.nodes.back().storage.capacity_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = true;
    graph.nodes.back().storage.coherent_with_host = true;
    graph.nodes.back().storage.shared_host_bytes = graph.nodes.back().storage.capacity_bytes;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "core-cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "core-cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 12.0});
    graph.edges.push_back({"host-memory", "core-cluster", jakal::GraphEdgeSemantics::feeds, true, 1.0, 64.0, 8.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

jakal::HardwareGraph make_manual_gpu_graph() {
    jakal::HardwareGraph graph;
    graph.uid = "gpu:test:0";
    graph.probe = "opencl";
    graph.presentation_name = "Intel Arc A770";

    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = true;
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 128;
    graph.nodes.back().compute.clock_mhz = 1800;
    graph.nodes.back().compute.matrix_engines = 16;
    graph.nodes.back().compute.supports_fp16 = true;
    graph.nodes.back().compute.supports_int8 = true;
    graph.nodes.push_back({"memory", "memory", "root", jakal::HardwareObjectDomain::storage, jakal::HardwareObjectRole::global_memory});
    graph.nodes.back().storage.capacity_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    graph.nodes.back().storage.unified_address_space = true;
    graph.nodes.back().storage.coherent_with_host = true;
    graph.nodes.back().storage.shared_host_bytes = graph.nodes.back().storage.capacity_bytes;
    graph.nodes.push_back({"host-link", "host-link", "root", jakal::HardwareObjectDomain::transfer, jakal::HardwareObjectRole::transfer_link});
    graph.nodes.back().transfer.read_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.write_bandwidth_gbps = 128.0;
    graph.nodes.back().transfer.dispatch_latency_us = 6.0;
    graph.nodes.back().transfer.synchronization_latency_us = 5.0;

    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "memory", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "host-link", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 6.0});
    graph.edges.push_back({"host-link", "memory", jakal::GraphEdgeSemantics::transfers_to, true, 1.0, 128.0, 5.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

const jakal::OperationOptimizationResult* find_operation(
    const jakal::OptimizationReport& report,
    const std::string& name) {
    const auto it = std::find_if(report.operations.begin(), report.operations.end(), [&](const auto& result) {
        return result.operation.name == name;
    });
    return it == report.operations.end() ? nullptr : &*it;
}

jakal::WorkloadSpec make_workload(const jakal::PartitionStrategy strategy) {
    return {
        "llm-decode-token-lite",
        jakal::WorkloadKind::inference,
        "llm-decode-token-lite",
        640ull * 1024ull * 1024ull,
        12ull * 1024ull * 1024ull,
        3.8e10,
        1,
        true,
        true,
        true,
        strategy};
}

bool verify_role_split(jakal::Planner& planner, jakal::ExecutionOptimizer& optimizer, const std::vector<jakal::HardwareGraph>& graphs) {
    const auto workload = make_workload(jakal::PartitionStrategy::role_split);
    const auto report = optimizer.optimize(workload, planner.build_plan(workload, graphs), graphs);
    const auto* qkv = find_operation(report, "decode-qkv");
    const auto* kv = find_operation(report, "kv-append");
    const auto* sample = find_operation(report, "decode-sample");
    return qkv != nullptr &&
           kv != nullptr &&
           sample != nullptr &&
           qkv->config.primary_device_uid == "gpu:test:0" &&
           qkv->config.participating_devices.size() == 1u &&
           kv->config.primary_device_uid == "host:test:0" &&
           sample->config.primary_device_uid == "host:test:0";
}

bool verify_projection_sharded(
    jakal::Planner& planner,
    jakal::ExecutionOptimizer& optimizer,
    const std::vector<jakal::HardwareGraph>& graphs) {
    const auto workload = make_workload(jakal::PartitionStrategy::projection_sharded);
    const auto report = optimizer.optimize(workload, planner.build_plan(workload, graphs), graphs);
    const auto* qkv = find_operation(report, "decode-qkv");
    const auto* context = find_operation(report, "decode-context");
    const auto* kv = find_operation(report, "kv-append");
    return qkv != nullptr &&
           context != nullptr &&
           kv != nullptr &&
           qkv->config.strategy == jakal::ExecutionStrategy::sharded &&
           qkv->config.participating_devices.size() == 2u &&
           qkv->config.logical_partitions == 4u &&
           context->config.strategy == jakal::ExecutionStrategy::sharded &&
           kv->config.primary_device_uid == "host:test:0";
}

bool verify_tpu_like(jakal::Planner& planner, jakal::ExecutionOptimizer& optimizer, const std::vector<jakal::HardwareGraph>& graphs) {
    const auto workload = make_workload(jakal::PartitionStrategy::tpu_like);
    const auto plan = planner.build_plan(workload, graphs);
    if (plan.allocations.size() != 2u ||
        plan.allocations.front().device.uid != "host:test:0" ||
        plan.allocations.back().device.uid != "gpu:test:0" ||
        plan.allocations.front().ratio >= plan.allocations.back().ratio) {
        return false;
    }

    const auto report = optimizer.optimize(workload, plan, graphs);
    const auto* qkv = find_operation(report, "decode-qkv");
    const auto* reduce = find_operation(report, "decode-score-reduce");
    const auto* kv = find_operation(report, "kv-append");
    return qkv != nullptr &&
           reduce != nullptr &&
           kv != nullptr &&
           qkv->config.primary_device_uid == "gpu:test:0" &&
           qkv->config.strategy == jakal::ExecutionStrategy::overlapped &&
           qkv->config.logical_partitions == 4u &&
           qkv->config.queue_depth >= 3u &&
           qkv->config.stages >= 3u &&
           reduce->config.primary_device_uid == "gpu:test:0" &&
           reduce->config.strategy == jakal::ExecutionStrategy::overlapped &&
           kv->config.primary_device_uid == "host:test:0";
}

}  // namespace

int main() {
    const auto plan_cache = unique_temp_file("partition-plan-test");
    const auto exec_cache = unique_temp_file("partition-exec-test");

    jakal::Planner planner(plan_cache);
    jakal::ExecutionOptimizer optimizer(exec_cache);
    const std::vector<jakal::HardwareGraph> graphs{
        make_manual_host_graph(),
        make_manual_gpu_graph()};

    const bool ok =
        verify_role_split(planner, optimizer, graphs) &&
        verify_projection_sharded(planner, optimizer, graphs) &&
        verify_tpu_like(planner, optimizer, graphs);

    std::error_code ec;
    std::filesystem::remove(plan_cache, ec);
    std::filesystem::remove(exec_cache, ec);
    std::filesystem::remove(exec_cache.string() + ".perf", ec);

    if (!ok) {
        std::cerr << "partition strategy verification failed\n";
        return 1;
    }

    std::cout << "partition strategies ok\n";
    return 0;
}
