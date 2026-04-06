#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_runtime gpu_runtime_t;

typedef struct gpu_device_info {
    char uid[128];
    char probe[32];
    char presentation_name[128];
    unsigned int ordinal;
    unsigned int execution_objects;
    unsigned int lanes_per_object;
    unsigned int resident_contexts;
    unsigned int matrix_units;
    unsigned int clock_mhz;
    unsigned int queue_slots;
    unsigned int numa_domain;
    unsigned int native_vector_bits;
    unsigned long long addressable_memory_bytes;
    unsigned long long directly_attached_memory_bytes;
    unsigned long long shared_host_memory_bytes;
    unsigned long long local_scratch_bytes;
    unsigned long long cache_bytes;
    double host_read_gbps;
    double host_write_gbps;
    double dispatch_latency_us;
    double synchronization_latency_us;
    unsigned long long node_count;
    unsigned long long edge_count;
    int coherent_with_host;
    int unified_address_space;
    int host_visible;
    int supports_asynchronous_dispatch;
    int supports_fp64;
    int supports_fp32;
    int supports_fp16;
    int supports_bf16;
    int supports_int8;
} gpu_device_info;

typedef struct gpu_graph_node_info {
    char id[128];
    char label[128];
    char parent_id[128];
    char domain[32];
    char role[32];
    char resolution[32];
    unsigned int ordinal;
    unsigned int execution_width;
    unsigned int resident_contexts;
    unsigned int matrix_engines;
    unsigned int clock_mhz;
    unsigned int native_vector_bits;
    unsigned int queue_slots;
    unsigned int numa_domain;
    unsigned long long capacity_bytes;
    unsigned long long directly_attached_bytes;
    unsigned long long shared_host_bytes;
    double read_bandwidth_gbps;
    double write_bandwidth_gbps;
    double dispatch_latency_us;
    double synchronization_latency_us;
    int coherent_with_host;
    int unified_address_space;
    int host_visible;
    int supports_asynchronous_dispatch;
    int supports_fp64;
    int supports_fp32;
    int supports_fp16;
    int supports_bf16;
    int supports_int8;
} gpu_graph_node_info;

typedef struct gpu_graph_edge_info {
    char source_id[128];
    char target_id[128];
    char semantics[32];
    int directed;
    double weight;
    double bandwidth_gbps;
    double latency_us;
} gpu_graph_edge_info;

typedef struct gpu_workload_spec {
    const char* name;
    const char* kind;
    unsigned long long working_set_bytes;
    unsigned long long host_exchange_bytes;
    double estimated_flops;
    unsigned int batch_size;
    int latency_sensitive;
    int prefer_unified_memory;
    int matrix_friendly;
} gpu_workload_spec;

typedef struct gpu_plan_entry {
    gpu_device_info device;
    double ratio;
    double score;
} gpu_plan_entry;

gpu_runtime_t* gpu_runtime_create(void);
void gpu_runtime_destroy(gpu_runtime_t* runtime);
int gpu_runtime_refresh(gpu_runtime_t* runtime);
size_t gpu_runtime_device_count(const gpu_runtime_t* runtime);
int gpu_runtime_get_device(const gpu_runtime_t* runtime, size_t index, gpu_device_info* out_device);
size_t gpu_runtime_graph_node_count(const gpu_runtime_t* runtime, size_t device_index);
int gpu_runtime_get_graph_node(
    const gpu_runtime_t* runtime,
    size_t device_index,
    size_t node_index,
    gpu_graph_node_info* out_node);
size_t gpu_runtime_graph_edge_count(const gpu_runtime_t* runtime, size_t device_index);
int gpu_runtime_get_graph_edge(
    const gpu_runtime_t* runtime,
    size_t device_index,
    size_t edge_index,
    gpu_graph_edge_info* out_edge);
int gpu_runtime_plan(
    gpu_runtime_t* runtime,
    const gpu_workload_spec* workload,
    gpu_plan_entry* entries,
    size_t capacity,
    size_t* out_count,
    int* out_loaded_from_cache);

#ifdef __cplusplus
}
#endif
