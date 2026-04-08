from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from subprocess import CompletedProcess, run

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_torch_workload


PASS_TOTAL_RE = re.compile(r"total_us=([0-9]+(?:\.[0-9]+)?)")
BACKENDS_RE = re.compile(r"backends=([^\r\n]+)")
STRATEGY_RE = re.compile(r"strategy=([A-Za-z0-9_\-]+)")
EXECUTED_RE = re.compile(r"executed=(yes|no)")

PROFILE_STATE_DEFAULTS: dict[str, dict[str, float]] = {
    "default": {},
    "host-latency": {
        "queue_depth_raw": -0.8,
        "stage_raw": -0.6,
        "tile_raw": -0.1,
        "overlap_raw": -0.7,
        "partition_raw": -1.2,
        "precision_raw": -0.4,
        "single_device_logit": 1.6,
        "sharded_logit": -1.5,
        "streaming_logit": -0.2,
        "overlapped_logit": 0.0,
    },
    "hybrid-balanced": {
        "queue_depth_raw": 0.0,
        "stage_raw": 0.1,
        "tile_raw": 0.3,
        "overlap_raw": -0.1,
        "partition_raw": -0.5,
        "precision_raw": 0.1,
        "single_device_logit": 0.9,
        "sharded_logit": -0.2,
        "streaming_logit": 0.2,
        "overlapped_logit": 0.3,
    },
    "accelerator-throughput": {
        "queue_depth_raw": 0.7,
        "stage_raw": 0.6,
        "tile_raw": 0.9,
        "overlap_raw": 0.5,
        "partition_raw": 0.5,
        "precision_raw": 0.5,
        "single_device_logit": 0.1,
        "sharded_logit": 0.8,
        "streaming_logit": 0.1,
        "overlapped_logit": 0.8,
    },
    "cooperative-split": {
        "queue_depth_raw": -0.1,
        "stage_raw": -0.2,
        "tile_raw": 0.5,
        "overlap_raw": -0.2,
        "partition_raw": 0.2,
        "precision_raw": 0.2,
        "single_device_logit": 0.4,
        "sharded_logit": 0.7,
        "streaming_logit": 0.0,
        "overlapped_logit": 0.2,
    },
}

TUNABLE_KEYS: tuple[str, ...] = (
    "partition_raw",
    "tile_raw",
    "overlap_raw",
    "queue_depth_raw",
    "stage_raw",
    "single_device_logit",
    "sharded_logit",
)

TUNABLE_DELTAS: dict[str, float] = {
    "partition_raw": 0.45,
    "tile_raw": 0.35,
    "overlap_raw": 0.35,
    "queue_depth_raw": 0.35,
    "stage_raw": 0.35,
    "single_device_logit": 0.45,
    "sharded_logit": 0.45,
}


@dataclass(frozen=True)
class Candidate:
    label: str
    partition_strategy: str | None = None
    tuning_profile: str | None = None
    graph_rewrite_level: int | None = None
    graph_passes: int | None = None
    state_overrides: tuple[str, ...] = ()

    def command_flags(self) -> list[str]:
        flags: list[str] = []
        if self.partition_strategy is not None:
            flags.extend(["--partition-strategy", self.partition_strategy])
        if self.tuning_profile is not None:
            flags.extend(["--tuning-profile", self.tuning_profile])
        if self.graph_rewrite_level is not None:
            flags.extend(["--graph-rewrite-level", str(self.graph_rewrite_level)])
        if self.graph_passes is not None:
            flags.extend(["--graph-passes", str(self.graph_passes)])
        for override in self.state_overrides:
            flags.extend(["--state", override])
        return flags

    def state_dict(self) -> dict[str, float]:
        state = dict(PROFILE_STATE_DEFAULTS.get(self.tuning_profile or "default", {}))
        for assignment in self.state_overrides:
            key, value_text = assignment.split("=", 1)
            state[key] = float(value_text)
        return state

    def key(self) -> tuple[object, ...]:
        return (
            self.partition_strategy,
            self.tuning_profile,
            self.graph_rewrite_level,
            self.graph_passes,
            self.state_overrides,
        )


@dataclass
class StrategyResult:
    workload: str
    manifest: str
    candidate_label: str
    status: str
    passes: int
    mode: str
    average_total_us: float | None = None
    min_total_us: float | None = None
    max_total_us: float | None = None
    backend_summary: str = ""
    resolved_strategy: str = ""
    executed: bool = False
    flags: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def repo_root() -> Path:
    return SCRIPT_DIR.parent


def default_build_dir() -> Path:
    return repo_root() / "build" / "Debug"


def shell_join(command: list[str]) -> str:
    rendered: list[str] = []
    for part in command:
        if re.search(r"\s", part):
            rendered.append(f'"{part}"')
        else:
            rendered.append(part)
    return " ".join(rendered)


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing executable: {path}")


def run_command(command: list[str], cwd: Path) -> CompletedProcess[str]:
    return run(command, cwd=cwd, text=True, capture_output=True, check=False)


def export_presets_to_temp(preset_names: list[str]) -> list[tuple[str, Path]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="jakal-search-"))
    manifests: list[tuple[str, Path]] = []
    for preset_name in preset_names:
        output_path = temp_dir / f"{preset_name}.workload"
        export_torch_workload.export_preset(preset_name, output_path)
        manifests.append((preset_name, output_path))
    return manifests


def parse_result(
    workload_name: str,
    manifest_path: Path,
    mode: str,
    candidate: Candidate,
    process: CompletedProcess[str],
    passes: int,
) -> StrategyResult:
    totals = [float(match.group(1)) for match in PASS_TOTAL_RE.finditer(process.stdout)]
    backend_match = BACKENDS_RE.findall(process.stdout)
    strategy_match = STRATEGY_RE.findall(process.stdout)
    executed_match = EXECUTED_RE.findall(process.stdout)

    result = StrategyResult(
        workload=workload_name,
        manifest=str(manifest_path),
        candidate_label=candidate.label,
        status="ok" if process.returncode == 0 and totals else "failed",
        passes=passes,
        mode=mode,
        flags=candidate.command_flags(),
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )
    if totals:
        result.average_total_us = statistics.fmean(totals)
        result.min_total_us = min(totals)
        result.max_total_us = max(totals)
    if backend_match:
        result.backend_summary = backend_match[-1].strip()
    if strategy_match:
        result.resolved_strategy = strategy_match[-1].strip()
    if executed_match:
        result.executed = executed_match[-1].strip() == "yes"
    return result


def mode_flags(mode: str) -> list[str]:
    if mode == "auto":
        return []
    if mode == "host":
        return ["--host-only"]
    if mode == "level-zero":
        return ["--level-zero-only"]
    if mode == "opencl":
        return ["--opencl-only"]
    raise ValueError(f"Unsupported mode: {mode}")


def candidate(
    label: str,
    partition_strategy: str | None = None,
    tuning_profile: str | None = None,
    graph_rewrite_level: int | None = None,
    graph_passes: int | None = None,
    state_overrides: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        label=label,
        partition_strategy=partition_strategy,
        tuning_profile=tuning_profile,
        graph_rewrite_level=graph_rewrite_level,
        graph_passes=graph_passes,
        state_overrides=state_overrides,
    )


def full_candidates() -> list[Candidate]:
    return dedupe_candidates(
        [
            candidate("baseline-auto"),
            candidate("auto-default-r1", "auto_balanced", "default", 1, 0),
            candidate("auto-hybrid-r2", "auto_balanced", "hybrid-balanced", 2, 2),
            candidate("auto-hybrid-deep", "auto_balanced", "hybrid-balanced", 2, 4),
            candidate("auto-coop-r2", "auto_balanced", "cooperative-split", 2, 2),
            candidate("auto-coop-lowpass", "auto_balanced", "cooperative-split", 2, 0),
            candidate("auto-host-r1", "auto_balanced", "host-latency", 1, 0),
            candidate("role-hybrid-r2", "role_split", "hybrid-balanced", 2, 2),
            candidate("role-coop-r2", "role_split", "cooperative-split", 2, 2),
            candidate("role-host-r1", "role_split", "host-latency", 1, 0),
            candidate("reduce-accel-r2", "reduce_on_gpu", "accelerator-throughput", 2, 4),
            candidate("reduce-coop-r2", "reduce_on_gpu", "cooperative-split", 2, 2),
            candidate("proj-accel-r2", "projection_sharded", "accelerator-throughput", 2, 4),
            candidate("proj-coop-r2", "projection_sharded", "cooperative-split", 2, 2),
            candidate("tpu-accel-r2", "tpu_like", "accelerator-throughput", 2, 4),
            candidate("blind-accel-r2", "blind_sharded", "accelerator-throughput", 2, 4),
            candidate(
                "proj-coop-nudge",
                "projection_sharded",
                "cooperative-split",
                2,
                2,
                ("partition_raw=0.45", "sharded_logit=0.95", "tile_raw=0.60"),
            ),
            candidate(
                "role-host-nudge",
                "role_split",
                "host-latency",
                1,
                0,
                ("single_device_logit=1.80", "partition_raw=-1.10"),
            ),
        ]
    )


def focused_candidates() -> list[Candidate]:
    return dedupe_candidates(
        [
            candidate("auto-hybrid-deep", "auto_balanced", "hybrid-balanced", 2, 4),
            candidate("auto-coop-r2", "auto_balanced", "cooperative-split", 2, 2),
            candidate("reduce-accel-r2", "reduce_on_gpu", "accelerator-throughput", 2, 4),
            candidate(
                "reduce-accel-host-lean",
                "reduce_on_gpu",
                "accelerator-throughput",
                2,
                4,
                ("single_device_logit=-0.35",),
            ),
            candidate("proj-accel-r2", "projection_sharded", "accelerator-throughput", 2, 4),
            candidate(
                "proj-accel-balanced-shard",
                "projection_sharded",
                "accelerator-throughput",
                2,
                4,
                ("sharded_logit=0.35",),
            ),
            candidate("proj-coop-r2", "projection_sharded", "cooperative-split", 2, 2),
            candidate("role-accel-r2", "role_split", "accelerator-throughput", 2, 4),
        ]
    )


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: list[Candidate] = []
    seen: set[tuple[object, ...]] = set()
    for entry in candidates:
        key = entry.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def format_override(key: str, value: float) -> str:
    return f"{key}={value:.2f}"


def refine_candidates(seed: Candidate) -> list[Candidate]:
    state = seed.state_dict()
    neighbors: list[Candidate] = []
    rewrite_level = 1 if seed.graph_rewrite_level == 2 else 2
    neighbors.append(
        Candidate(
            label=f"{seed.label}+rw{rewrite_level}",
            partition_strategy=seed.partition_strategy,
            tuning_profile=seed.tuning_profile,
            graph_rewrite_level=rewrite_level,
            graph_passes=seed.graph_passes,
            state_overrides=seed.state_overrides,
        )
    )

    base_graph_passes = seed.graph_passes if seed.graph_passes is not None else 2
    for new_passes in sorted({0, 2, 4, max(0, base_graph_passes - 2), min(4, base_graph_passes + 2)}):
        if new_passes == base_graph_passes:
            continue
        neighbors.append(
            Candidate(
                label=f"{seed.label}+gp{new_passes}",
                partition_strategy=seed.partition_strategy,
                tuning_profile=seed.tuning_profile,
                graph_rewrite_level=seed.graph_rewrite_level,
                graph_passes=new_passes,
                state_overrides=seed.state_overrides,
            )
        )

    for key in TUNABLE_KEYS:
        base_value = state.get(key, 0.0)
        delta = TUNABLE_DELTAS[key]
        for direction, suffix in ((-1.0, "lo"), (1.0, "hi")):
            updated = dict(state)
            updated[key] = base_value + (delta * direction)
            overrides = tuple(format_override(name, updated[name]) for name in sorted(updated))
            neighbors.append(
                Candidate(
                    label=f"{seed.label}+{key.split('_')[0]}-{suffix}",
                    partition_strategy=seed.partition_strategy,
                    tuning_profile=seed.tuning_profile,
                    graph_rewrite_level=seed.graph_rewrite_level,
                    graph_passes=seed.graph_passes,
                    state_overrides=overrides,
                )
            )

    if seed.partition_strategy != "projection_sharded":
        neighbors.append(
            Candidate(
                label=f"{seed.label}+proj",
                partition_strategy="projection_sharded",
                tuning_profile=seed.tuning_profile,
                graph_rewrite_level=seed.graph_rewrite_level,
                graph_passes=seed.graph_passes,
                state_overrides=seed.state_overrides,
            )
        )
    if seed.partition_strategy != "role_split":
        neighbors.append(
            Candidate(
                label=f"{seed.label}+role",
                partition_strategy="role_split",
                tuning_profile=seed.tuning_profile,
                graph_rewrite_level=seed.graph_rewrite_level,
                graph_passes=seed.graph_passes,
                state_overrides=seed.state_overrides,
            )
        )
    return dedupe_candidates(neighbors)


def rank_results(results: list[StrategyResult]) -> list[StrategyResult]:
    def sort_key(result: StrategyResult) -> tuple[float, int, int]:
        if result.average_total_us is None:
            return (float("inf"), 1, 1)
        mixed_bonus = 0 if ("host" in result.backend_summary and "level-zero" in result.backend_summary) else 1
        execute_penalty = 0 if result.executed else 1
        return (result.average_total_us, mixed_bonus, execute_penalty)

    return sorted(results, key=sort_key)


def run_candidate(
    profile_exe: Path,
    manifest_path: Path,
    workload_name: str,
    mode: str,
    candidate_config: Candidate,
    passes: int,
) -> StrategyResult:
    command = [str(profile_exe), str(manifest_path), "--passes", str(passes)]
    command.extend(mode_flags(mode))
    command.extend(candidate_config.command_flags())
    process = run_command(command, repo_root())
    return parse_result(workload_name, manifest_path, mode, candidate_config, process, passes)


def workload_entries_from_args(args: argparse.Namespace) -> list[tuple[str, Path]]:
    preset_names = list(args.torch_preset)
    if args.all_torch_presets:
        preset_names = list(export_torch_workload.PRESETS.keys())

    manifest_entries: list[tuple[str, Path]] = []
    if preset_names:
        manifest_entries.extend(export_presets_to_temp(preset_names))
    for manifest in args.manifest:
        path = Path(manifest).resolve()
        manifest_entries.append((path.stem, path))
    return manifest_entries


def print_ranked(workload_name: str, ranked: list[StrategyResult], limit: int) -> None:
    print(f"workload={workload_name}")
    print("rank\tavg_total_us\tbackends\tcandidate\tresolved_strategy\tflags")
    for index, result in enumerate(ranked[:limit], start=1):
        command_flags = " ".join(result.flags) if result.flags else "(default)"
        average = f"{result.average_total_us:.3f}" if result.average_total_us is not None else "n/a"
        print(
            "\t".join(
                [
                    str(index),
                    average,
                    result.backend_summary or "n/a",
                    result.candidate_label,
                    result.resolved_strategy or "n/a",
                    command_flags,
                ]
            )
        )


def print_best_summary(best_results: list[StrategyResult], profile_exe: Path, mode: str) -> None:
    print("best-workloads")
    print("workload\tavg_total_us\tbackends\tcandidate\tresolved_strategy\trepro")
    for result in best_results:
        repro = shell_join(
            [str(profile_exe), result.manifest, "--passes", str(result.passes)]
            + mode_flags(mode)
            + result.flags
        )
        average = f"{result.average_total_us:.3f}" if result.average_total_us is not None else "n/a"
        print(
            "\t".join(
                [
                    result.workload,
                    average,
                    result.backend_summary or "n/a",
                    result.candidate_label,
                    result.resolved_strategy or "n/a",
                    repro,
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for high-performing Jakal optimization strategies across partition heuristics and execution tuning."
    )
    parser.add_argument(
        "--torch-preset",
        action="append",
        default=[],
        choices=list(export_torch_workload.PRESETS.keys()),
        help="Built-in PyTorch preset to export and search. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all-torch-presets",
        action="store_true",
        help="Search every built-in PyTorch preset.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Existing manifest path (.workload, .onnx, .gguf) to search. Can be specified multiple times.",
    )
    parser.add_argument(
        "--search-space",
        choices=("full", "focused"),
        default="full",
        help="Candidate search-space size. 'focused' keeps only the strongest mixed CPU/GPU strategies.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "host", "level-zero", "opencl"),
        default="auto",
        help="Execution mode for the search. 'auto' keeps both host and accelerator visible.",
    )
    parser.add_argument(
        "--initial-passes",
        type=int,
        default=1,
        help="Warm search pass count for the initial candidate sweep.",
    )
    parser.add_argument(
        "--refine-passes",
        type=int,
        default=2,
        help="Pass count for the refinement search around the top candidates.",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=2,
        help="How many initial candidates per workload to refine.",
    )
    parser.add_argument(
        "--summary-top-k",
        type=int,
        default=5,
        help="How many ranked candidates per workload to print.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir(),
        help="Directory containing jakal_profile_manifest.exe.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON file containing the full search traces.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = workload_entries_from_args(args)
    if not manifests:
        raise SystemExit("No workloads selected. Use --torch-preset, --all-torch-presets, or --manifest.")

    profile_exe = args.build_dir / "jakal_profile_manifest.exe"
    ensure_executable(profile_exe)

    base = full_candidates() if args.search_space == "full" else focused_candidates()
    all_results: list[StrategyResult] = []
    best_results: list[StrategyResult] = []

    for workload_name, manifest_path in manifests:
        initial_results = [
            run_candidate(profile_exe, manifest_path, workload_name, args.mode, entry, max(1, args.initial_passes))
            for entry in base
        ]
        all_results.extend(initial_results)
        ranked_initial = rank_results(initial_results)

        refinement_results: list[StrategyResult] = []
        refinement_pool: list[Candidate] = []
        if args.refine_top_k > 0:
            refine_inputs = ranked_initial[: args.refine_top_k]
            for seed_result in refine_inputs:
                seed_candidate = next(entry for entry in base if entry.label == seed_result.candidate_label)
                refinement_pool.extend(refine_candidates(seed_candidate))
            refinement_pool = dedupe_candidates(refinement_pool)
            refinement_results = [
                run_candidate(profile_exe, manifest_path, workload_name, args.mode, entry, max(1, args.refine_passes))
                for entry in refinement_pool
            ]
        all_results.extend(refinement_results)

        ranked = rank_results(initial_results + refinement_results)
        print_ranked(workload_name, ranked, max(1, args.summary_top_k))
        best_results.append(ranked[0])

    print_best_summary(best_results, profile_exe, args.mode)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "initial_passes": args.initial_passes,
                    "refine_passes": args.refine_passes,
                    "refine_top_k": args.refine_top_k,
                    "results": [asdict(result) for result in all_results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
