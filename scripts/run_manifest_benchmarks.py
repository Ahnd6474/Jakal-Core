from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import CompletedProcess, run

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_torch_workload


PASS_TOTAL_RE = re.compile(r"total_us=([0-9]+(?:\.[0-9]+)?)")
BACKENDS_RE = re.compile(r"backends=([^\r\n]+)")
DIRECTML_OP_RE = re.compile(r"executed_ops=([0-9]+)\s+skipped_ops=([0-9]+)")


@dataclass
class BenchmarkResult:
    workload: str
    manifest: str
    mode: str
    status: str
    passes: int
    average_total_us: float | None = None
    min_total_us: float | None = None
    max_total_us: float | None = None
    backend_summary: str = ""
    executed_ops: int | None = None
    skipped_ops: int | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def repo_root() -> Path:
    return SCRIPT_DIR.parent


def default_build_dir() -> Path:
    return repo_root() / "build" / "Debug"


def parse_result(
    workload_name: str,
    manifest_path: Path,
    mode: str,
    process: CompletedProcess[str],
    passes: int,
) -> BenchmarkResult:
    totals = [float(match.group(1)) for match in PASS_TOTAL_RE.finditer(process.stdout)]
    backend_match = BACKENDS_RE.findall(process.stdout)
    directml_match = DIRECTML_OP_RE.findall(process.stdout)

    result = BenchmarkResult(
        workload=workload_name,
        manifest=str(manifest_path),
        mode=mode,
        status="ok" if process.returncode == 0 and totals else "failed",
        passes=passes,
        stdout=process.stdout,
        stderr=process.stderr,
        returncode=process.returncode,
    )
    if totals:
        result.average_total_us = statistics.fmean(totals)
        result.min_total_us = min(totals)
        result.max_total_us = max(totals)
    if backend_match:
        result.backend_summary = backend_match[-1].strip()
    if directml_match:
        executed_ops, skipped_ops = directml_match[-1]
        result.executed_ops = int(executed_ops)
        result.skipped_ops = int(skipped_ops)
    return result


def run_command(command: list[str], cwd: Path) -> CompletedProcess[str]:
    return run(command, cwd=cwd, text=True, capture_output=True, check=False)


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing executable: {path}")


def export_presets_to_temp(preset_names: list[str]) -> list[tuple[str, Path]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="jakal-bench-"))
    manifests: list[tuple[str, Path]] = []
    for preset_name in preset_names:
        output_path = temp_dir / f"{preset_name}.workload"
        export_torch_workload.export_preset(preset_name, output_path)
        manifests.append((preset_name, output_path))
    return manifests


def build_commands(
    profile_exe: Path,
    directml_exe: Path,
    manifest_path: Path,
    passes: int,
    include_opencl: bool,
) -> list[tuple[str, list[str]]]:
    commands = [
        (
            "host",
            [str(profile_exe), str(manifest_path), "--passes", str(passes), "--host-only"],
        ),
        (
            "level-zero",
            [str(profile_exe), str(manifest_path), "--passes", str(passes), "--level-zero-only"],
        ),
    ]
    if include_opencl:
        commands.append(
            (
                "opencl",
                [str(profile_exe), str(manifest_path), "--passes", str(passes), "--opencl-only"],
            )
        )
    if directml_exe.exists():
        commands.append(
            (
                "directml-standalone",
                [str(directml_exe), str(manifest_path), "--passes", str(passes)],
            )
        )
    return commands


def print_summary(results: list[BenchmarkResult]) -> None:
    print("workload\tmode\tstatus\tavg_total_us\tmin_total_us\tmax_total_us\tbackends\texecuted_ops\tskipped_ops")
    for result in results:
        print(
            "\t".join(
                [
                    result.workload,
                    result.mode,
                    result.status,
                    f"{result.average_total_us:.3f}" if result.average_total_us is not None else "n/a",
                    f"{result.min_total_us:.3f}" if result.min_total_us is not None else "n/a",
                    f"{result.max_total_us:.3f}" if result.max_total_us is not None else "n/a",
                    result.backend_summary or "n/a",
                    str(result.executed_ops) if result.executed_ops is not None else "n/a",
                    str(result.skipped_ops) if result.skipped_ops is not None else "n/a",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Jakal manifests across host, Level Zero, OpenCL, and DirectML.")
    parser.add_argument(
        "--torch-preset",
        action="append",
        default=[],
        choices=list(export_torch_workload.PRESETS.keys()),
        help="Built-in PyTorch preset to export and benchmark. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all-torch-presets",
        action="store_true",
        help="Benchmark every built-in PyTorch preset.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Existing manifest path (.workload, .onnx, .gguf) to benchmark. Can be specified multiple times.",
    )
    parser.add_argument("--passes", type=int, default=3, help="Number of passes per backend.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir(),
        help="Directory containing jakal_profile_manifest.exe and jakal_directml_manifest_bench.exe.",
    )
    parser.add_argument("--include-opencl", action="store_true", help="Include the OpenCL-only benchmark path.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path for full benchmark results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    preset_names = list(args.torch_preset)
    if args.all_torch_presets:
        preset_names = list(export_torch_workload.PRESETS.keys())

    manifest_entries: list[tuple[str, Path]] = []
    if preset_names:
        manifest_entries.extend(export_presets_to_temp(preset_names))
    for manifest in args.manifest:
        path = Path(manifest).resolve()
        manifest_entries.append((path.stem, path))

    if not manifest_entries:
        raise SystemExit("No workloads selected. Use --torch-preset, --all-torch-presets, or --manifest.")

    profile_exe = args.build_dir / "jakal_profile_manifest.exe"
    directml_exe = args.build_dir / "jakal_directml_manifest_bench.exe"
    ensure_executable(profile_exe)

    results: list[BenchmarkResult] = []
    cwd = repo_root()
    for workload_name, manifest_path in manifest_entries:
        for mode, command in build_commands(profile_exe, directml_exe, manifest_path, args.passes, args.include_opencl):
            process = run_command(command, cwd)
            results.append(parse_result(workload_name, manifest_path, mode, process, args.passes))

    print_summary(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
