#include "gpu/runtime.hpp"

#include <iomanip>
#include <iostream>

int main() {
    gpu::Runtime runtime;

    std::cout << "Discovered hardware graphs\n";
    for (const auto& graph : runtime.devices()) {
        const auto summary = gpu::summarize_graph(graph);
        std::cout
            << "\n[" << graph.ordinal << "] " << graph.presentation_name
            << " | probe=" << graph.probe
            << " | execution_objects=" << summary.execution_objects
            << " | lanes/object=" << summary.lanes_per_object
            << " | memory=" << (summary.addressable_bytes / (1024ull * 1024ull)) << " MiB"
            << " | nodes=" << graph.nodes.size()
            << " | edges=" << graph.edges.size()
            << '\n';

        std::cout << "  Nodes\n";
        for (const auto& node : graph.nodes) {
            std::cout
                << "    " << std::setw(24) << std::left << node.label
                << " domain=" << gpu::to_string(node.domain)
                << " role=" << gpu::to_string(node.role)
                << " resolution=" << gpu::to_string(node.resolution)
                << '\n';
        }

        std::cout << "  Edges\n";
        for (const auto& edge : graph.edges) {
            std::cout
                << "    " << edge.source_id
                << " -> " << edge.target_id
                << " semantics=" << gpu::to_string(edge.semantics)
                << " weight=" << edge.weight
                << " bw=" << edge.bandwidth_gbps
                << "GB/s latency=" << edge.latency_us
                << "us\n";
        }
    }

    const gpu::WorkloadSpec workload{
        "sample-inference",
        gpu::WorkloadKind::inference,
        4ull * 1024ull * 1024ull * 1024ull,
        512ull * 1024ull * 1024ull,
        30.0e12,
        16,
        false,
        false,
        true};

    const auto plan = runtime.plan(workload);
    std::cout << "\nPlan signature: " << plan.signature << '\n';
    std::cout << "Loaded from cache: " << (plan.loaded_from_cache ? "yes" : "no") << '\n';

    for (const auto& allocation : plan.allocations) {
        std::cout
            << "  " << std::setw(24) << std::left << allocation.device.presentation_name
            << " ratio=" << std::fixed << std::setprecision(3) << allocation.ratio
            << " score=" << allocation.score
            << '\n';
    }

    return 0;
}
