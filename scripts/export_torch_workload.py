from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return sanitized or "value"


def flatten_tensor_sources(value: Any) -> list[torch.fx.Node]:
    if isinstance(value, torch.fx.Node):
        return [value]
    if isinstance(value, (list, tuple)):
        nodes: list[torch.fx.Node] = []
        for item in value:
            nodes.extend(flatten_tensor_sources(item))
        return nodes
    return []


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float64: "f64",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.int16: "i16",
        torch.int8: "i8",
        torch.uint8: "u8",
        torch.bool: "u8",
    }
    return mapping.get(dtype, "f32")


def tensor_shape(meta: Any) -> list[int]:
    shape = getattr(meta, "shape", None)
    if shape is None:
        return []
    return [int(dim) for dim in shape]


@dataclass
class SourceMetadata:
    name: str
    dataset_tag: str
    kind: str
    batch_size: int
    latency_sensitive: bool
    prefer_unified_memory: bool
    matrix_friendly: bool
    description: str
    shape_bucket: str
    phase: str = "unknown"
    entry: str = "forward"


@dataclass
class ValueRecord:
    identifier: str
    shape: list[int]
    dtype: str
    initializer: bool = False
    persistent: bool = False
    temporary: bool = False
    host_visible: bool = False
    producer: str = ""
    consumers: list[str] | None = None


@dataclass
class NodeRecord:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    shape: list[int]
    matrix_friendly: bool = False
    reduction_like: bool = False


@dataclass
class PresetDefinition:
    metadata: SourceMetadata
    build_model: Callable[[], nn.Module]
    build_inputs: Callable[[], tuple[torch.Tensor, ...]]


class FinanceFactorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.exposure = nn.Linear(256, 512)
        self.activation = nn.GELU()
        self.scenario = nn.Linear(512, 128)
        self.probability = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        projected = self.exposure(normalized)
        activated = self.activation(projected)
        scenario_scores = self.scenario(activated)
        distribution = self.probability(scenario_scores)
        return distribution.sum(dim=-1, keepdim=True)


class SignalFilterbankModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analysis = nn.Linear(1024, 512, bias=False)
        self.activation = nn.ReLU()
        self.synthesis = nn.Linear(512, 256, bias=False)
        self.gate = nn.Linear(1024, 256)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        spectrum = self.analysis(samples)
        filtered = self.activation(spectrum)
        bank = self.synthesis(filtered)
        gains = torch.sigmoid(self.gate(samples))
        weighted = bank * gains
        return weighted.sum(dim=-1, keepdim=True)


class GraphRankingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seed = nn.Linear(128, 128)
        self.activation = nn.ReLU()
        self.refine = nn.Linear(128, 128)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        seed_scores = self.seed(state)
        activated = self.activation(seed_scores)
        refined = self.refine(activated)
        merged = refined + seed_scores
        probabilities = torch.sigmoid(merged)
        return probabilities.mean(dim=-1, keepdim=True)


class ScientificSolverStepModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_proj = nn.Linear(512, 512)
        self.activation = nn.Tanh()
        self.correction = nn.Linear(512, 512)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        projected = self.state_proj(state)
        activated = self.activation(projected)
        correction = self.correction(activated)
        next_state = correction + state
        return next_state.sum(dim=-1, keepdim=True)


PRESETS: "OrderedDict[str, PresetDefinition]" = OrderedDict(
    {
        "finance-factor-risk-lite": PresetDefinition(
            metadata=SourceMetadata(
                name="finance-factor-risk-lite",
                dataset_tag="finance-factor-risk-lite",
                kind="tensor",
                batch_size=64,
                latency_sensitive=False,
                prefer_unified_memory=False,
                matrix_friendly=True,
                description="Tabular factor-risk style scoring with normalization, scenario projection, and portfolio reduction.",
                shape_bucket="b64-f256-s128",
            ),
            build_model=FinanceFactorModel,
            build_inputs=lambda: (torch.randn(64, 256),),
        ),
        "signal-filterbank-lite": PresetDefinition(
            metadata=SourceMetadata(
                name="signal-filterbank-lite",
                dataset_tag="signal-filterbank-lite",
                kind="tensor",
                batch_size=32,
                latency_sensitive=True,
                prefer_unified_memory=True,
                matrix_friendly=True,
                description="Signal-processing style filter-bank projection with gain shaping and energy accumulation.",
                shape_bucket="b32-f1024-c256",
            ),
            build_model=SignalFilterbankModel,
            build_inputs=lambda: (torch.randn(32, 1024),),
        ),
        "graph-ranking-lite": PresetDefinition(
            metadata=SourceMetadata(
                name="graph-ranking-lite",
                dataset_tag="graph-ranking-lite",
                kind="custom",
                batch_size=128,
                latency_sensitive=True,
                prefer_unified_memory=False,
                matrix_friendly=True,
                description="Graph-ranking surrogate with seed propagation, residual refinement, and score collapse.",
                shape_bucket="b128-v128",
            ),
            build_model=GraphRankingModel,
            build_inputs=lambda: (torch.randn(128, 128),),
        ),
        "scientific-solver-step-lite": PresetDefinition(
            metadata=SourceMetadata(
                name="scientific-solver-step-lite",
                dataset_tag="scientific-solver-step-lite",
                kind="tensor",
                batch_size=64,
                latency_sensitive=False,
                prefer_unified_memory=False,
                matrix_friendly=True,
                description="Scientific residual-iteration surrogate with projection, nonlinear update, and convergence reduction.",
                shape_bucket="b64-s512",
            ),
            build_model=ScientificSolverStepModel,
            build_inputs=lambda: (torch.randn(64, 512),),
        ),
    }
)


def parameter_value_id(name: str) -> str:
    return f"param_{sanitize_identifier(name)}"


def node_output_value_id(name: str) -> str:
    return sanitize_identifier(name)


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def ensure_shape_metadata(graph_module: torch.fx.GraphModule, sample_inputs: tuple[torch.Tensor, ...]) -> None:
    ShapeProp(graph_module).propagate(*sample_inputs)


def map_function_op_type(target: Any) -> str:
    lowered = str(target).lower()
    if "softmax" in lowered:
        return "Softmax"
    if "gelu" in lowered:
        return "GELU"
    if "relu" in lowered:
        return "ReLU"
    if "sigmoid" in lowered:
        return "Sigmoid"
    if "tanh" in lowered:
        return "Tanh"
    if "mean" in lowered:
        return "ReduceMean"
    if "sum" in lowered:
        return "ReduceSum"
    if "mul" in lowered:
        return "Mul"
    if "add" in lowered:
        return "Add"
    raise NotImplementedError(f"Unsupported function op: {target}")


def map_module_op_type(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        return "Linear"
    if isinstance(module, nn.LayerNorm):
        return "LayerNormalization"
    if isinstance(module, nn.GELU):
        return "GELU"
    if isinstance(module, nn.ReLU):
        return "ReLU"
    if isinstance(module, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(module, nn.Tanh):
        return "Tanh"
    if isinstance(module, nn.Softmax):
        return "Softmax"
    raise NotImplementedError(f"Unsupported module op: {module.__class__.__name__}")


def write_workload(
    output_path: Path,
    metadata: SourceMetadata,
    values: list[ValueRecord],
    nodes: list[NodeRecord],
) -> None:
    lines: list[str] = []

    def emit_section(name: str, fields: list[str]) -> None:
        lines.append(f"[{name}]")
        lines.extend(field for field in fields if field)
        lines.append("")

    emit_section(
        "source",
        [
            "format=pytorch_export",
            f"name={metadata.name}",
            f"kind={metadata.kind}",
            f"dataset_tag={metadata.dataset_tag}",
            f"phase={metadata.phase}",
            f"entry={metadata.entry}",
            f"batch_size={metadata.batch_size}",
            f"latency_sensitive={'true' if metadata.latency_sensitive else 'false'}",
            f"prefer_unified_memory={'true' if metadata.prefer_unified_memory else 'false'}",
            f"matrix_friendly={'true' if metadata.matrix_friendly else 'false'}",
            f"shape_bucket={metadata.shape_bucket}",
        ],
    )

    for value in values:
        emit_section(
            "value",
            [
                f"id={value.identifier}",
                f"shape={','.join(str(dim) for dim in value.shape)}" if value.shape else "",
                f"dtype={value.dtype}",
                f"producer={value.producer}" if value.producer else "",
                f"consumers={','.join(value.consumers or [])}" if value.consumers else "",
                f"initializer={'true' if value.initializer else 'false'}" if value.initializer else "",
                f"persistent={'true' if value.persistent else 'false'}" if value.persistent else "",
                f"temporary={'true' if value.temporary else 'false'}" if value.temporary else "",
                f"host_visible={'true' if value.host_visible else 'false'}" if value.host_visible else "",
            ],
        )

    for node in nodes:
        emit_section(
            "node",
            [
                f"name={node.name}",
                f"op_type={node.op_type}",
                f"inputs={','.join(node.inputs)}",
                f"outputs={','.join(node.outputs)}",
                f"shape={','.join(str(dim) for dim in node.shape)}" if node.shape else "",
                "matrix_friendly=true" if node.matrix_friendly else "",
                "reduction_like=true" if node.reduction_like else "",
            ],
        )

    rendered = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered.strip() + "\n", encoding="utf-8")


def export_preset(preset_name: str, output_path: Path) -> Path:
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS)
        raise KeyError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    preset = PRESETS[preset_name]
    torch.manual_seed(0)
    model = preset.build_model().eval()
    sample_inputs = tuple(tensor.detach().clone() for tensor in preset.build_inputs())
    graph_module = torch.fx.symbolic_trace(model)
    ensure_shape_metadata(graph_module, sample_inputs)

    modules = dict(graph_module.named_modules())
    named_parameters = dict(graph_module.named_parameters())
    named_buffers = dict(graph_module.named_buffers())

    values_by_id: OrderedDict[str, ValueRecord] = OrderedDict()
    consumers_by_value: dict[str, list[str]] = {}
    node_to_value_id: dict[str, str] = {}

    def add_value(record: ValueRecord) -> None:
        if record.identifier not in values_by_id:
            values_by_id[record.identifier] = record

    def ensure_parameter(name: str) -> str:
        identifier = parameter_value_id(name)
        if identifier in values_by_id:
            return identifier

        tensor = named_parameters.get(name)
        if tensor is None:
            tensor = named_buffers.get(name)
        if tensor is None:
            raise KeyError(f"Missing parameter or buffer '{name}' in traced module.")

        add_value(
            ValueRecord(
                identifier=identifier,
                shape=[int(dim) for dim in tensor.shape],
                dtype=dtype_name(tensor.dtype),
                initializer=True,
                persistent=True,
                host_visible=True,
                consumers=[],
            )
        )
        return identifier

    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    for index, node in enumerate(placeholders):
        tensor = sample_inputs[index]
        value_id = node_output_value_id(node.name)
        node_to_value_id[node.name] = value_id
        add_value(
            ValueRecord(
                identifier=value_id,
                shape=[int(dim) for dim in tensor.shape],
                dtype=dtype_name(tensor.dtype),
                host_visible=True,
                consumers=[],
            )
        )

    nodes: list[NodeRecord] = []
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            ensure_parameter(str(node.target))
            node_to_value_id[node.name] = parameter_value_id(str(node.target))
            continue

        input_ids = [node_to_value_id[source.name] for source in flatten_tensor_sources(node.args)]
        op_type = ""
        extra_inputs: list[str] = []

        if node.op == "call_module":
            module = modules[str(node.target)]
            op_type = map_module_op_type(module)
            if isinstance(module, nn.Linear):
                extra_inputs.append(ensure_parameter(f"{node.target}.weight"))
                if module.bias is not None:
                    extra_inputs.append(ensure_parameter(f"{node.target}.bias"))
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    extra_inputs.append(ensure_parameter(f"{node.target}.weight"))
                if module.bias is not None:
                    extra_inputs.append(ensure_parameter(f"{node.target}.bias"))
        elif node.op in {"call_function", "call_method"}:
            op_type = map_function_op_type(node.target)
        else:
            raise NotImplementedError(f"Unsupported FX node op '{node.op}'")

        input_ids.extend(extra_inputs)
        input_ids = dedupe_preserve_order(input_ids)
        output_value_id = node_output_value_id(node.name)
        node_to_value_id[node.name] = output_value_id

        tensor_meta = node.meta.get("tensor_meta")
        output_shape = tensor_shape(tensor_meta)
        output_dtype = dtype_name(getattr(tensor_meta, "dtype", torch.float32))
        add_value(
            ValueRecord(
                identifier=output_value_id,
                shape=output_shape,
                dtype=output_dtype,
                producer=sanitize_identifier(node.name),
                consumers=[],
            )
        )

        current_node = NodeRecord(
            name=sanitize_identifier(node.name),
            op_type=op_type,
            inputs=input_ids,
            outputs=[output_value_id],
            shape=output_shape,
            matrix_friendly=op_type in {"Linear"},
            reduction_like=op_type in {"Softmax", "ReduceSum", "ReduceMean"},
        )
        nodes.append(current_node)

        for input_id in current_node.inputs:
            consumers_by_value.setdefault(input_id, []).append(current_node.name)

    for value_id, consumers in consumers_by_value.items():
        if value_id in values_by_id:
            values_by_id[value_id].consumers = dedupe_preserve_order(consumers)

    write_workload(output_path, preset.metadata, list(values_by_id.values()), nodes)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export built-in PyTorch presets into Jakal .workload manifests.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Built-in PyTorch preset to export.")
    parser.add_argument("--output", type=Path, help="Output .workload path.")
    parser.add_argument("--list-presets", action="store_true", help="List the available presets and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"{name}\t{preset.metadata.description}")
        return 0

    if args.preset is None or args.output is None:
        raise SystemExit("--preset and --output are required unless --list-presets is used.")

    exported = export_preset(args.preset, args.output)
    print(f"exported={exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
