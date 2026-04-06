# GPU

This repository is an early runtime skeleton for heterogeneous compute that models hardware as an execution graph rather than as a vendor label or a fixed device class.

The target is a global optimizer for AI training, inference, and general compute workloads that can use whatever resources are present on the machine:

- multiple vendors at the same time
- CPU as a first-class execution domain
- graph-based placement, scheduling, memory mapping, and dataflow optimization
- framework integration points for Torch, TensorFlow, and similar systems

## Current status

This is still an early runtime skeleton, not a finished execution platform.

Implemented now:

- C++20 runtime core
- host hardware probe
- OpenCL-based graph probe as one discovery path
- hierarchical hardware graph model
- directed, weighted execution and transfer edges
- planner that partitions work from graph-derived summaries
- execution-graph builder that maps global placements into signal-flow graphs
- operation-family optimization across the discovered device set
- validation microbenchmarks with latency and accuracy recording
- execution-setting cache persistence
- plan cache persistence
- minimal C ABI with graph node and edge inspection
- example and test targets

Not implemented yet:

- real kernel execution
- tensor memory model
- CUDA probe / execution path
- Level Zero / OpenVINO / oneDNN probe and execution path
- inter-device transfer scheduler
- direct Torch / TensorFlow bridge

## Core idea

The runtime should not optimize from strings like "CPU", "GPU", "TPU", "NVIDIA", or "Intel".

Instead, each discovered hardware target is normalized into a graph:

- node = one hardware object
- edge = one structural or execution relation
- direction = explicit when execution or transfer is directional
- weight = available when bandwidth, latency, or control cost is known

The graph is built from hardware-readable information first. If a detail is not directly readable, it should not be invented casually.

Edge `weight` is treated as an estimated execution cost in microseconds for a small canonical operation on that relation. Bandwidth and latency remain explicit on the edge, and `weight` is materialized from them with lightweight hierarchy propagation.

## Graph model

### Medium resolution

The default graph separates:

- `ComputeDomain`
- `StorageDomain`
- `TransferDomain`
- `ControlDomain`

It also expresses repeated structure hierarchically:

- root
- cluster
- tile

This is the main level for:

- placement
- scheduling
- memory mapping
- dataflow optimization

### Aggressive resolution

The graph model also supports finer objects:

- lane
- pipeline
- scratchpad

These are only instantiated when the probe can read enough hardware information to justify them. The current implementation mixes medium resolution with selective aggressive nodes so the graph stays useful without exploding the search space.

Bank, router, and deeper pipeline decomposition are intentionally left as future selective refinements. They should only be expanded where measured bottlenecks justify the extra complexity.

When aggressive nodes are present, their edge costs are propagated upward so parent `contains`, `controls`, and `dispatches` relations change as the lower-level graph changes.

## Architecture

```text
include/gpu/
  backend.hpp      hardware probe interface
  c_api.h          C ABI
  device.hpp       graph nodes, edges, summaries
  planner.hpp      global partition planner
  runtime.hpp      runtime entry point

src/
  backends/
    cpu_backend.cpp
    opencl_backend.cpp
```

## Planner model

The planner now works on graph-derived summaries, not on flat device labels.

Current scoring uses:

- execution object count
- lanes per execution object
- resident contexts
- matrix capability
- memory hierarchy capacity
- host exchange cost
- dispatch and synchronization latency
- unified/coherent memory affinity
- graph richness bonus for mapping-visible structure

Inputs from the workload side:

- working set size
- host exchange size
- estimated flops
- batch size
- latency sensitivity
- unified memory preference
- matrix-friendly flag

This is still heuristic, but it is aligned with the intended architecture: the optimizer reasons from graph structure and measured-readable hardware properties.

## Execution graph and optimization

The runtime now has two graph layers:

- structural graph: what hardware objects exist and how they relate
- execution graph: how data, dispatch, compute, aggregation, and synchronization flow across the placed hardware set

The execution graph is built from the structural graph plus the global placement plan. It is not device-class-specific: one optimization pass can pick one device, shard across several, or overlap stages when the graph suggests it is useful.

Current operation families:

- elementwise map
- reduction
- blocked matmul
- 3x3 convolution
- bilinear resample

For each operation family, the optimizer:

1. generates a small candidate set of execution settings
2. builds an execution graph for each candidate
3. predicts latency from graph structure and edge costs
4. runs a lightweight validation benchmark
5. records latency and relative error
6. keeps the fastest candidate that stays inside tolerance

The chosen execution settings are persisted so later runs can rebuild the same execution graph without re-searching from scratch.

Validation is still lightweight. The benchmark path is a correctness and shape-validation loop, not a full backend execution engine.

## Low-spec track and learning cache

The optimizer now always captures a lightweight system profile and can switch into a low-spec track automatically.

Inputs reflected in the surrogate cost:

- cold vs warm execution
- battery and battery-saver state
- available memory and paging risk
- sustained slowdown estimated from prior runs
- one-time device initialization amortization

When the machine is constrained, the candidate generator becomes more conservative:

- shallower queue depth
- fewer pipeline stages
- stronger bias toward streaming
- reduced bias toward multi-device sharding when paging risk is high

The optimizer also keeps a lightweight shape-bucket performance cache:

- key = graph set, environment bucket, operation family, shape bucket, execution config
- value = running averages for latency, prediction scale, error, and system penalty

This cache is used by default and grows only with the distinct shapes and environment buckets actually observed. It is meant to stay lightweight while making repeated runs better over time.

## Build

```powershell
cmake -S . -B build
cmake --build build
```

Example:

```powershell
.\build\gpu_inspect.exe
```

## Next priorities

1. Add more graph probes that do not leak vendor-specific concepts into the runtime core.
2. Add at least one real execution path.
3. Add explicit memory residency and transfer planning on top of the graph.
4. Replace more heuristic estimates with measured calibration.
5. Add direct framework bridges.
