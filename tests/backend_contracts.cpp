#include "jakal/runtime.hpp"
#include "jakal/executors/direct_backends.hpp"
#include "jakal/executors/native_gpu_backend.hpp"

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace {

jakal::HardwareGraph make_graph(const std::string& probe, const std::string& uid) {
    jakal::HardwareGraph graph;
    graph.probe = probe;
    graph.uid = uid;
    graph.presentation_name = probe + "-device";
    return graph;
}

jakal::HardwareGraph make_async_graph(
    const std::string& probe,
    const std::string& uid,
    const bool async_dispatch) {
    auto graph = make_graph(probe, uid);
    graph.nodes.push_back({"root", "root", "", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::root});
    graph.nodes.push_back({"queue", "queue", "root", jakal::HardwareObjectDomain::control, jakal::HardwareObjectRole::queue});
    graph.nodes.back().control.supports_asynchronous_dispatch = async_dispatch;
    graph.nodes.push_back({"cluster", "cluster", "root", jakal::HardwareObjectDomain::compute, jakal::HardwareObjectRole::cluster});
    graph.nodes.back().compute.execution_width = 64;
    graph.edges.push_back({"root", "queue", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"root", "cluster", jakal::GraphEdgeSemantics::contains, true});
    graph.edges.push_back({"queue", "cluster", jakal::GraphEdgeSemantics::dispatches, true, 1.0, 0.0, 4.0});
    jakal::materialize_graph_costs(graph);
    return graph;
}

bool expect_backend_contract(
    const jakal::HardwareGraph& graph,
    const std::string& expected_backend_name) {
    if (jakal::runtime_backend_name_for_graph(graph) != expected_backend_name) {
        std::cerr << "unexpected backend name for probe " << graph.probe << '\n';
        return false;
    }

    constexpr std::array<jakal::OperationClass, 5> kAllOperationClasses = {
        jakal::OperationClass::elementwise_map,
        jakal::OperationClass::reduction,
        jakal::OperationClass::matmul,
        jakal::OperationClass::convolution_2d,
        jakal::OperationClass::resample_2d,
    };
    for (const auto op_class : kAllOperationClasses) {
        std::string reason;
        if (!jakal::runtime_backend_supports_operation(graph, op_class, &reason)) {
            std::cerr << "backend contract rejected op for probe " << graph.probe
                      << " reason=" << reason << '\n';
            return false;
        }
    }

    return true;
}

bool expect_native_backend_contract(
    const jakal::JakalBackendKind backend_kind,
    const jakal::HardwareGraph& graph,
    const std::string& expected_name) {
    auto backend = jakal::executors::make_native_gpu_kernel_backend(backend_kind);
    if (!backend) {
        std::cerr << "native backend factory returned null for " << graph.probe << '\n';
        return false;
    }
    if (backend->name() != expected_name) {
        std::cerr << "unexpected native backend name for " << graph.probe << '\n';
        return false;
    }
    if (!backend->matches(graph)) {
        std::cerr << "native backend did not match graph for " << graph.probe << '\n';
        return false;
    }
    return true;
}

bool expect_async_dispatch_contracts() {
    const auto host_graph = make_async_graph("host", "host-async-off", false);
    const auto level_zero_graph = make_async_graph("level-zero", "gpu-async-on", true);

    auto host_backend = jakal::executors::make_host_kernel_backend();
    if (!host_backend || host_backend->supports_async_dispatch(host_graph)) {
        std::cerr << "host backend should not advertise async dispatch\n";
        return false;
    }

    auto native_backend = jakal::executors::make_native_gpu_kernel_backend(jakal::JakalBackendKind::level_zero);
    if (!native_backend || !native_backend->supports_async_dispatch(level_zero_graph)) {
        std::cerr << "level-zero backend should advertise async dispatch\n";
        return false;
    }

    return true;
}

bool expect_persistent_direct_backend_reuse() {
    auto graph = make_async_graph("vulkan", "gpu-vulkan-warm", true);
    graph.presentation_name = "vulkan-device";

    auto backend = jakal::executors::make_vulkan_kernel_backend();
    if (!backend || !backend->matches(graph)) {
        std::cerr << "vulkan backend did not match test graph\n";
        return false;
    }

    jakal::OperationSpec operation;
    operation.name = "warm-matmul";
    operation.op_class = jakal::OperationClass::matmul;
    operation.extents = {32, 32, 32};
    operation.matrix_friendly = true;

    std::vector<float> lhs(32u * 32u, 1.0f);
    std::vector<float> rhs(32u * 32u, 0.5f);
    const auto first = backend->run_matmul(graph, operation, lhs, rhs, 32u, 32u, 32u, true);
    const auto second = backend->run_matmul(graph, operation, lhs, rhs, 32u, 32u, 32u, true);
    if (!first.success || !second.success) {
        std::cerr << "vulkan warm-cache matmul failed\n";
        return false;
    }
    if (!(second.submit_runtime_us < first.submit_runtime_us) ||
        !(second.runtime_us < first.runtime_us)) {
        std::cerr << "expected warm-cache backend reuse to reduce runtime\n";
        return false;
    }
    return true;
}

bool expect_persistent_native_backend_reuse() {
    auto graph = make_async_graph("level-zero", "gpu-native-warm", true);
    graph.presentation_name = "level-zero-device";

    auto backend = jakal::executors::make_native_gpu_kernel_backend(jakal::JakalBackendKind::level_zero);
    if (!backend || !backend->matches(graph)) {
        std::cerr << "level-zero backend did not match test graph\n";
        return false;
    }

    jakal::OperationSpec operation;
    operation.name = "native-warm-matmul";
    operation.op_class = jakal::OperationClass::matmul;
    operation.extents = {64, 64, 64};
    operation.matrix_friendly = true;

    std::vector<float> lhs(64u * 64u, 1.0f);
    std::vector<float> rhs(64u * 64u, 0.5f);
    const auto first = backend->run_matmul(graph, operation, lhs, rhs, 64u, 64u, 64u, true);
    const auto second = backend->run_matmul(graph, operation, lhs, rhs, 64u, 64u, 64u, true);
    if (!first.success || !second.success) {
        std::cerr << "native warm-cache matmul failed\n";
        return false;
    }
    if (!(second.submit_runtime_us < first.submit_runtime_us) ||
        !(second.runtime_us < first.runtime_us) ||
        !(second.synchronize_runtime_us <= first.synchronize_runtime_us)) {
        std::cerr << "expected native warm-cache reuse to reduce runtime\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const auto opencl = make_graph("opencl", "gpu-opencl");
    const auto level_zero = make_graph("level-zero", "gpu-level-zero");
    const auto cuda = make_graph("cuda", "gpu-cuda");
    const auto rocm = make_graph("rocm", "gpu-rocm");

    if (!expect_backend_contract(opencl, "opencl-direct") ||
        !expect_backend_contract(level_zero, "level-zero-native") ||
        !expect_backend_contract(cuda, "cuda-native") ||
        !expect_backend_contract(rocm, "rocm-native")) {
        return 1;
    }

    if (!expect_native_backend_contract(jakal::JakalBackendKind::level_zero, level_zero, "level-zero-native") ||
        !expect_native_backend_contract(jakal::JakalBackendKind::cuda, cuda, "cuda-native") ||
        !expect_native_backend_contract(jakal::JakalBackendKind::rocm, rocm, "rocm-native")) {
        return 1;
    }

    if (!expect_async_dispatch_contracts()) {
        return 1;
    }
    if (!expect_persistent_direct_backend_reuse()) {
        return 1;
    }
    if (!expect_persistent_native_backend_reuse()) {
        return 1;
    }

    std::string reason;
    const auto unknown = make_graph("mystery", "gpu-unknown");
    if (jakal::runtime_backend_supports_operation(unknown, jakal::OperationClass::matmul, &reason) ||
        reason.empty()) {
        std::cerr << "unknown backend should fail kernel coverage with a reason\n";
        return 1;
    }

    std::cout << "backend contracts ok\n";
    return 0;
}
