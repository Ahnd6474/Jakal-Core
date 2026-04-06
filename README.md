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
- plan cache persistence
- minimal C ABI with graph node and edge inspection
- example and smoke test targets

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
