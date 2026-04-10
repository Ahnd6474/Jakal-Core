import ctypes
import os
from pathlib import Path


def _default_library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    candidates = []
    env = os.environ.get("JAKAL_RUNTIME_DLL")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            root / "build_ninja" / "jakal_runtime.dll",
            root / "build_ninja" / "Debug" / "jakal_runtime.dll",
            root / "build" / "Debug" / "jakal_runtime.dll",
            root / "build" / "jakal_runtime.dll",
            root / "build_ninja" / "libjakal_runtime.so",
            root / "build" / "libjakal_runtime.so",
        ]
    )
    return candidates


def load_runtime_library() -> ctypes.CDLL:
    for candidate in _default_library_candidates():
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    raise FileNotFoundError("Unable to locate jakal_runtime shared library. Set JAKAL_RUNTIME_DLL.")


class RuntimeOptions(ctypes.Structure):
    _fields_ = [
        ("enable_host_probe", ctypes.c_int),
        ("enable_opencl_probe", ctypes.c_int),
        ("enable_level_zero_probe", ctypes.c_int),
        ("enable_vulkan_probe", ctypes.c_int),
        ("enable_vulkan_status", ctypes.c_int),
        ("enable_cuda_probe", ctypes.c_int),
        ("enable_rocm_probe", ctypes.c_int),
        ("prefer_level_zero_over_opencl", ctypes.c_int),
        ("eager_hardware_refresh", ctypes.c_int),
        ("install_root", ctypes.c_char_p),
        ("cache_path", ctypes.c_char_p),
        ("execution_cache_path", ctypes.c_char_p),
        ("telemetry_path", ctypes.c_char_p),
    ]


class WorkloadSpec(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("kind", ctypes.c_char_p),
        ("dataset_tag", ctypes.c_char_p),
        ("phase", ctypes.c_char_p),
        ("shape_bucket", ctypes.c_char_p),
        ("working_set_bytes", ctypes.c_ulonglong),
        ("host_exchange_bytes", ctypes.c_ulonglong),
        ("estimated_flops", ctypes.c_double),
        ("batch_size", ctypes.c_uint),
        ("latency_sensitive", ctypes.c_int),
        ("prefer_unified_memory", ctypes.c_int),
        ("matrix_friendly", ctypes.c_int),
    ]


class OptimizationInfo(ctypes.Structure):
    _fields_ = [
        ("signature", ctypes.c_char * 128),
        ("workload_kind", ctypes.c_char * 32),
        ("dataset_tag", ctypes.c_char * 128),
        ("operation_count", ctypes.c_ulonglong),
        ("tensor_count", ctypes.c_ulonglong),
        ("dependency_count", ctypes.c_ulonglong),
        ("readiness_score", ctypes.c_double),
        ("stability_score", ctypes.c_double),
        ("sustained_slowdown", ctypes.c_double),
        ("loaded_from_cache", ctypes.c_int),
    ]


class OperationOptimizationInfo(ctypes.Structure):
    _fields_ = [
        ("operation_name", ctypes.c_char * 128),
        ("strategy", ctypes.c_char * 32),
        ("primary_device_uid", ctypes.c_char * 128),
        ("logical_partitions", ctypes.c_uint),
        ("participating_device_count", ctypes.c_uint),
        ("predicted_latency_us", ctypes.c_double),
        ("predicted_speedup_vs_reference", ctypes.c_double),
        ("predicted_transfer_latency_us", ctypes.c_double),
        ("predicted_memory_pressure", ctypes.c_double),
        ("peak_resident_bytes", ctypes.c_ulonglong),
        ("target_error_tolerance", ctypes.c_double),
        ("use_low_precision", ctypes.c_int),
    ]


class ExecutionInfo(ctypes.Structure):
    _fields_ = [
        ("signature", ctypes.c_char * 128),
        ("operation_count", ctypes.c_ulonglong),
        ("total_runtime_us", ctypes.c_double),
        ("total_reference_runtime_us", ctypes.c_double),
        ("speedup_vs_reference", ctypes.c_double),
        ("all_succeeded", ctypes.c_int),
    ]


class ExecutionOperationInfo(ctypes.Structure):
    _fields_ = [
        ("operation_name", ctypes.c_char * 128),
        ("backend_name", ctypes.c_char * 64),
        ("requested_gpu_vendor", ctypes.c_char * 32),
        ("requested_gpu_backend", ctypes.c_char * 32),
        ("runtime_us", ctypes.c_double),
        ("reference_runtime_us", ctypes.c_double),
        ("speedup_vs_reference", ctypes.c_double),
        ("relative_error", ctypes.c_double),
        ("verified", ctypes.c_int),
        ("used_host", ctypes.c_int),
        ("used_opencl", ctypes.c_int),
        ("used_multiple_devices", ctypes.c_int),
        ("logical_partitions_used", ctypes.c_uint),
    ]


class RuntimePaths(ctypes.Structure):
    _fields_ = [
        ("install_root", ctypes.c_char * 260),
        ("writable_root", ctypes.c_char * 260),
        ("config_dir", ctypes.c_char * 260),
        ("cache_dir", ctypes.c_char * 260),
        ("logs_dir", ctypes.c_char * 260),
        ("telemetry_path", ctypes.c_char * 260),
        ("planner_cache_path", ctypes.c_char * 260),
        ("execution_cache_path", ctypes.c_char * 260),
        ("python_dir", ctypes.c_char * 260),
    ]


class BackendStatus(ctypes.Structure):
    _fields_ = [
        ("backend_name", ctypes.c_char * 32),
        ("device_uid", ctypes.c_char * 128),
        ("code", ctypes.c_char * 32),
        ("detail", ctypes.c_char * 256),
        ("enabled", ctypes.c_int),
        ("available", ctypes.c_int),
        ("direct_execution", ctypes.c_int),
        ("modeled_fallback", ctypes.c_int),
    ]


class Runtime:
    def __init__(self, options: RuntimeOptions | None = None):
        self._lib = load_runtime_library()
        self._lib.jakal_core_runtime_create.restype = ctypes.c_void_p
        self._lib.jakal_core_runtime_create_with_options.restype = ctypes.c_void_p
        self._lib.jakal_core_runtime_backend_status_count.restype = ctypes.c_size_t
        self._lib.jakal_core_runtime_device_count.restype = ctypes.c_size_t
        handle = (
            self._lib.jakal_core_runtime_create()
            if options is None
            else self._lib.jakal_core_runtime_create_with_options(ctypes.byref(options))
        )
        if not handle:
            raise RuntimeError("Failed to create Jakal runtime")
        self._handle = ctypes.c_void_p(handle)

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.jakal_core_runtime_destroy(self._handle)
            self._handle = None

    def last_error(self) -> str:
        buffer = ctypes.create_string_buffer(1024)
        if self._lib.jakal_core_runtime_get_last_error(self._handle, buffer, len(buffer)) != 0:
            return ""
        return buffer.value.decode()

    def paths(self) -> dict[str, str]:
        paths = RuntimePaths()
        if self._lib.jakal_core_runtime_get_install_paths(self._handle, ctypes.byref(paths)) != 0:
            raise RuntimeError("Failed to query install paths")
        return {name: bytes(getattr(paths, name)).split(b"\0", 1)[0].decode() for name, _ in paths._fields_}

    def backend_statuses(self) -> list[dict[str, str | int]]:
        count = self._lib.jakal_core_runtime_backend_status_count(self._handle)
        statuses = []
        for index in range(count):
            item = BackendStatus()
            if self._lib.jakal_core_runtime_get_backend_status(self._handle, index, ctypes.byref(item)) == 0:
                statuses.append(
                    {
                        "backend_name": bytes(item.backend_name).split(b"\0", 1)[0].decode(),
                        "device_uid": bytes(item.device_uid).split(b"\0", 1)[0].decode(),
                        "code": bytes(item.code).split(b"\0", 1)[0].decode(),
                        "detail": bytes(item.detail).split(b"\0", 1)[0].decode(),
                        "enabled": int(item.enabled),
                        "available": int(item.available),
                        "direct_execution": int(item.direct_execution),
                        "modeled_fallback": int(item.modeled_fallback),
                    }
                )
        return statuses

    def optimize(self, workload: WorkloadSpec) -> dict:
        optimization = OptimizationInfo()
        operations = (OperationOptimizationInfo * 64)()
        count = ctypes.c_size_t()
        rc = self._lib.jakal_core_runtime_optimize(
            self._handle, ctypes.byref(workload), ctypes.byref(optimization), operations, 64, ctypes.byref(count)
        )
        if rc != 0:
            raise RuntimeError(self.last_error() or f"optimize failed: {rc}")
        return {
            "signature": bytes(optimization.signature).split(b"\0", 1)[0].decode(),
            "operation_count": int(optimization.operation_count),
            "dataset_tag": bytes(optimization.dataset_tag).split(b"\0", 1)[0].decode(),
            "operations": [
                {
                    "operation_name": bytes(operations[i].operation_name).split(b"\0", 1)[0].decode(),
                    "primary_device_uid": bytes(operations[i].primary_device_uid).split(b"\0", 1)[0].decode(),
                    "strategy": bytes(operations[i].strategy).split(b"\0", 1)[0].decode(),
                }
                for i in range(count.value)
            ],
        }

    def execute_manifest(self, manifest_path: str) -> dict:
        execution = ExecutionInfo()
        operations = (ExecutionOperationInfo * 64)()
        count = ctypes.c_size_t()
        rc = self._lib.jakal_core_runtime_execute_manifest(
            self._handle,
            manifest_path.encode(),
            ctypes.byref(execution),
            operations,
            64,
            ctypes.byref(count),
        )
        if rc != 0:
            raise RuntimeError(self.last_error() or f"execute_manifest failed: {rc}")
        return {
            "signature": bytes(execution.signature).split(b"\0", 1)[0].decode(),
            "operation_count": int(execution.operation_count),
            "total_runtime_us": float(execution.total_runtime_us),
            "all_succeeded": bool(execution.all_succeeded),
            "operations": [
                {
                    "operation_name": bytes(operations[i].operation_name).split(b"\0", 1)[0].decode(),
                    "backend_name": bytes(operations[i].backend_name).split(b"\0", 1)[0].decode(),
                    "runtime_us": float(operations[i].runtime_us),
                }
                for i in range(count.value)
            ],
        }


def default_host_only_options() -> RuntimeOptions:
    return RuntimeOptions(1, 0, 0, 0, 0, 0, 0, 1, 1, None, None, None, None)


def make_workload(
    *,
    name: str,
    kind: str,
    dataset_tag: str,
    phase: str,
    shape_bucket: str,
    working_set_bytes: int,
    host_exchange_bytes: int,
    estimated_flops: float,
    batch_size: int = 1,
    latency_sensitive: bool = False,
    prefer_unified_memory: bool = False,
    matrix_friendly: bool = True,
) -> WorkloadSpec:
    return WorkloadSpec(
        name.encode(),
        kind.encode(),
        dataset_tag.encode(),
        phase.encode(),
        shape_bucket.encode(),
        working_set_bytes,
        host_exchange_bytes,
        estimated_flops,
        batch_size,
        int(latency_sensitive),
        int(prefer_unified_memory),
        int(matrix_friendly),
    )
