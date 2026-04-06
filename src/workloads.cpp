#include "gpu/workloads.hpp"

namespace gpu {

std::vector<CanonicalWorkloadPreset> canonical_workload_presets() {
    return {
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "gaming-upscale-1080p",
                WorkloadKind::gaming,
                "gaming-fsr-like-720p-to-1080p",
                768ull * 1024ull * 1024ull,
                96ull * 1024ull * 1024ull,
                1.2e12,
                1,
                true,
                true,
                false},
            "Realtime render reconstruction and post-processing chain from 1280x720 to 1920x1080.",
            "single-path frame pipeline"},
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "ai-vision-inference-lite",
                WorkloadKind::inference,
                "ai-vision-inference-224",
                1024ull * 1024ull * 1024ull,
                128ull * 1024ull * 1024ull,
                4.5e12,
                8,
                false,
                false,
                true},
            "Vision encoder style inference chain with convolution stem, projection, attention-like GEMMs, and MLP blocks.",
            "single-device reference kernels"},
        CanonicalWorkloadPreset{
            WorkloadSpec{
                "ai-train-step-lite",
                WorkloadKind::training,
                "ai-transformer-train-step-lite",
                1536ull * 1024ull * 1024ull,
                256ull * 1024ull * 1024ull,
                7.5e12,
                16,
                false,
                false,
                true},
            "Compact training-step surrogate with forward, reduction, gradient GEMMs, and optimizer-style updates.",
            "single-device reference kernels"}}; 
}

}  // namespace gpu
