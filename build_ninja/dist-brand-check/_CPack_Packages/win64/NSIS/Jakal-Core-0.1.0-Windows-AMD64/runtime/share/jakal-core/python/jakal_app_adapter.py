import argparse
import json

from jakal_runtime import Runtime, default_host_only_options, make_workload


def main() -> int:
    parser = argparse.ArgumentParser(description="Jakal runtime app adapter")
    parser.add_argument("command", choices=["doctor", "paths", "optimize-smoke", "run-manifest"])
    parser.add_argument("value", nargs="?")
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()

    runtime = Runtime(default_host_only_options() if args.host_only else None)
    if args.command == "doctor":
        print(json.dumps({"paths": runtime.paths(), "backends": runtime.backend_statuses()}, indent=2))
        return 0
    if args.command == "paths":
        print(json.dumps(runtime.paths(), indent=2))
        return 0
    if args.command == "optimize-smoke":
        workload = make_workload(
            name="python-smoke",
            kind="tensor",
            dataset_tag="python-smoke-lite",
            phase="prefill",
            shape_bucket="b1-lite",
            working_set_bytes=8 * 1024 * 1024,
            host_exchange_bytes=1024 * 1024,
            estimated_flops=1.0e7,
        )
        print(json.dumps(runtime.optimize(workload), indent=2))
        return 0
    if args.command == "run-manifest":
        if not args.value:
            raise SystemExit("run-manifest requires a manifest path")
        print(json.dumps(runtime.execute_manifest(args.value), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
