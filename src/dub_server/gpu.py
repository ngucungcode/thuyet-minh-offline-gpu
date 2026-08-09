"""NVIDIA GPU preflight checks used by the offline worker.

The checks deliberately import optional GPU libraries at runtime so this
module remains unit-testable on hosts without CUDA or an NVIDIA driver.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Sequence

MIN_DRIVER = (570, 26)
MIN_COMPUTE_CAPABILITY = (7, 0)
MIN_VRAM_MIB = 6144
SUPPORTED_CUDA_ARCHITECTURES = frozenset(
    {(7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0)}
)
SUPPORTED_CT2_COMPUTE_TYPES = frozenset({"float16", "int8_float16"})
GPU_SUPPORT_SUPPORTED = "supported"
GPU_SUPPORT_MAINTENANCE_LIMITED = "maintenance-limited"
GPU_SUPPORT_EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class NvidiaGpu:
    uuid: str
    name: str
    driver_version: str
    memory_total_mib: int
    compute_capability: str


@dataclass(frozen=True)
class CudaDeviceIdentity:
    name: str
    memory_total_mib: int
    compute_capability: str
    uuid: str | None = None


@dataclass(frozen=True)
class ComponentStatus:
    available: bool
    version: str | None = None
    detail: str | None = None
    compute_types: tuple[str, ...] = ()
    device: CudaDeviceIdentity | None = None


@dataclass(frozen=True)
class GpuPreflightReport:
    ready: bool
    enforced: bool
    checked_at: str
    minimum_driver: str
    minimum_compute_capability: str
    supported_cuda_architectures: tuple[str, ...]
    minimum_vram_mib: int
    selected_gpu_uuid: str | None
    gpus: tuple[NvidiaGpu, ...]
    torch: ComponentStatus
    ctranslate2: ComponentStatus
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Provide the small Pydantic-compatible surface used by worker.py."""

        del mode
        return asdict(self)


class GpuPreflightError(RuntimeError):
    def __init__(self, report: GpuPreflightReport) -> None:
        self.report = report
        super().__init__("; ".join(report.errors) or "GPU preflight failed")


def gpu_support_tier(gpu: NvidiaGpu) -> str:
    """Return the release support tier for the selected logical CUDA device."""

    capability = _version_tuple(gpu.compute_capability)
    if capability == (7, 0):
        return GPU_SUPPORT_MAINTENANCE_LIMITED
    normalized_name = "".join(gpu.name.lower().split())
    if "cmp170hx" in normalized_name:
        return GPU_SUPPORT_EXPERIMENTAL
    return GPU_SUPPORT_SUPPORTED


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
ComponentProbe = Callable[[], ComponentStatus]


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for part in value.strip().split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _normalize_nvidia_gpu_uuid(value: object | None) -> str | None:
    """Return CUDA UUID values in the canonical nvidia-smi ``GPU-...`` form."""

    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.casefold().startswith("gpu-"):
        return f"GPU-{normalized[4:]}"
    return f"GPU-{normalized}"


def _parse_nvidia_smi(output: str) -> tuple[NvidiaGpu, ...]:
    parsed: list[NvidiaGpu] = []
    for row in csv.reader(StringIO(output)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 5:
            raise ValueError(f"Unexpected nvidia-smi row with {len(row)} fields")
        uuid, name, driver, memory_mib, capability = (item.strip() for item in row)
        parsed.append(
            NvidiaGpu(
                uuid=uuid,
                name=name,
                driver_version=driver,
                memory_total_mib=int(memory_mib),
                compute_capability=capability,
            )
        )
    return tuple(parsed)


def probe_torch_cuda() -> ComponentStatus:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return ComponentStatus(False, getattr(torch, "__version__", None), "torch.cuda is unavailable")

        # A real matrix multiplication plus synchronization proves that a CUDA
        # kernel executes; merely importing a CUDA wheel is not sufficient.
        with torch.inference_mode():
            left = torch.ones((256, 256), device="cuda", dtype=torch.float16)
            result = left @ left
            torch.cuda.synchronize()
            checksum = float(result[0, 0].item())
        if checksum != 256.0:
            return ComponentStatus(False, str(torch.__version__), f"unexpected CUDA checksum {checksum}")
        runtime = getattr(getattr(torch, "version", None), "cuda", None)
        properties = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        device = CudaDeviceIdentity(
            name=str(properties.name),
            memory_total_mib=int(properties.total_memory) // (1024 * 1024),
            compute_capability=f"{major}.{minor}",
            uuid=_normalize_nvidia_gpu_uuid(getattr(properties, "uuid", None)),
        )
        return ComponentStatus(
            True,
            str(torch.__version__),
            f"CUDA {runtime}; {device.name}; matmul checksum={checksum:g}",
            device=device,
        )
    except Exception as error:  # optional native imports can fail in several ways
        return ComponentStatus(False, detail=f"{type(error).__name__}: {error}")


def probe_ctranslate2_cuda() -> ComponentStatus:
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        compute_types = tuple(sorted(ctranslate2.get_supported_compute_types("cuda")))
        usable = bool(SUPPORTED_CT2_COMPUTE_TYPES.intersection(compute_types))
        detail = None if usable else "float16 or int8_float16 is not supported"
        return ComponentStatus(usable, str(ctranslate2.__version__), detail, compute_types)
    except Exception as error:
        return ComponentStatus(False, detail=f"{type(error).__name__}: {error}")


def inspect_gpu(
    *,
    require_gpu: bool = True,
    command_runner: CommandRunner = _run_command,
    torch_probe: ComponentProbe = probe_torch_cuda,
    ctranslate2_probe: ComponentProbe = probe_ctranslate2_cuda,
    minimum_driver: tuple[int, int] = MIN_DRIVER,
    minimum_compute_capability: tuple[int, int] = MIN_COMPUTE_CAPABILITY,
    minimum_vram_mib: int = MIN_VRAM_MIB,
    expected_gpu_uuid: str | None = None,
    expected_cuda_architecture: str | None = None,
) -> GpuPreflightReport:
    """Inspect GPU readiness and optionally fail closed for production workers."""

    failures: list[str] = []
    warnings: list[str] = []
    gpus: tuple[NvidiaGpu, ...] = ()
    try:
        result = command_runner(
            (
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            )
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            failures.append(f"nvidia-smi failed: {detail}")
        else:
            gpus = _parse_nvidia_smi(result.stdout)
            if not gpus:
                failures.append("nvidia-smi reported no visible GPU")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        failures.append(f"nvidia-smi failed: {type(error).__name__}: {error}")

    torch_status = torch_probe()
    if not torch_status.available:
        failures.append(f"PyTorch CUDA check failed: {torch_status.detail or 'unavailable'}")
    elif torch_status.device is not None and gpus:
        identity = torch_status.device
        selected_index: int | None = None
        identity_uuid = _normalize_nvidia_gpu_uuid(identity.uuid)
        if identity_uuid:
            selected_index = next(
                (
                    index
                    for index, gpu in enumerate(gpus)
                    if gpu.uuid.casefold() == identity_uuid.casefold()
                ),
                None,
            )
        if selected_index is None:
            matching_indices = [
                index
                for index, gpu in enumerate(gpus)
                if gpu.name.casefold() == identity.name.casefold()
                and gpu.compute_capability == identity.compute_capability
            ]
            if matching_indices:
                selected_index = min(
                    matching_indices,
                    key=lambda index: abs(
                        gpus[index].memory_total_mib - identity.memory_total_mib
                    ),
                )
        if selected_index is None:
            failures.append(
                "PyTorch logical CUDA device 0 cannot be matched to nvidia-smi"
            )
        else:
            selected_from_smi = gpus[selected_index]
            selected = NvidiaGpu(
                uuid=selected_from_smi.uuid,
                name=identity.name,
                driver_version=selected_from_smi.driver_version,
                memory_total_mib=identity.memory_total_mib,
                compute_capability=identity.compute_capability,
            )
            gpus = (
                selected,
                *(gpu for index, gpu in enumerate(gpus) if index != selected_index),
            )

    configured_uuid = (expected_gpu_uuid or "").strip()
    if configured_uuid and gpus and gpus[0].uuid.casefold() != configured_uuid.casefold():
        failures.append(
            "logical CUDA device 0 UUID "
            f"{gpus[0].uuid} does not match installed runtime UUID {configured_uuid}"
        )
    configured_architecture = (expected_cuda_architecture or "").strip().lower()
    if configured_architecture:
        if not re.fullmatch(r"sm_[0-9]{2,3}", configured_architecture):
            failures.append(
                "configured installed runtime CUDA architecture is invalid: "
                f"{configured_architecture}"
            )
        elif gpus:
            selected_architecture = f"sm_{gpus[0].compute_capability.replace('.', '')}"
            if selected_architecture != configured_architecture:
                failures.append(
                    "logical CUDA device 0 architecture "
                    f"{selected_architecture} does not match installed runtime "
                    f"{configured_architecture}"
                )

    logical_gpu_zero_eligible = False
    for index, gpu in enumerate(gpus):
        gpu_failures: list[str] = []
        if _version_tuple(gpu.driver_version) < minimum_driver:
            gpu_failures.append(f"driver {gpu.driver_version} < {minimum_driver[0]}.{minimum_driver[1]}")
        capability = _version_tuple(gpu.compute_capability)
        if capability not in SUPPORTED_CUDA_ARCHITECTURES:
            gpu_failures.append(
                f"compute capability {gpu.compute_capability} is not in the supported "
                "CUDA architecture matrix"
            )
        elif capability < minimum_compute_capability:
            gpu_failures.append(
                f"compute capability {gpu.compute_capability} < "
                f"{minimum_compute_capability[0]}.{minimum_compute_capability[1]}"
            )
        if gpu.memory_total_mib < minimum_vram_mib:
            gpu_failures.append(f"VRAM {gpu.memory_total_mib} MiB < {minimum_vram_mib} MiB")
        if gpu_failures:
            device_label = (
                "logical CUDA device 0"
                if index == 0
                else f"non-selected NVIDIA device {index}"
            )
            warnings.append(
                f"{device_label}: {gpu.name} ({gpu.uuid}): "
                + ", ".join(gpu_failures)
            )
        elif index == 0:
            logical_gpu_zero_eligible = True
    if gpus and not logical_gpu_zero_eligible:
        failures.append(
            "no visible GPU satisfies the requirements for logical CUDA device 0; "
            "expose one supported GPU as logical device 0 before starting the worker"
        )
    elif gpus:
        selected_tier = gpu_support_tier(gpus[0])
        if selected_tier == GPU_SUPPORT_MAINTENANCE_LIMITED:
            warnings.append("logical CUDA device 0 uses maintenance-limited Volta sm_70")
        if selected_tier == GPU_SUPPORT_EXPERIMENTAL:
            warnings.append("logical CUDA device 0 uses experimental CMP 170HX support")

    ctranslate2_status = ctranslate2_probe()
    if not ctranslate2_status.available:
        failures.append(f"CTranslate2 CUDA check failed: {ctranslate2_status.detail or 'unavailable'}")

    ready = not failures
    if not require_gpu and failures:
        warnings.extend(failures)
        failures.clear()

    report = GpuPreflightReport(
        ready=ready,
        enforced=require_gpu,
        checked_at=datetime.now(UTC).isoformat(),
        minimum_driver=f"{minimum_driver[0]}.{minimum_driver[1]}",
        minimum_compute_capability=f"{minimum_compute_capability[0]}.{minimum_compute_capability[1]}",
        supported_cuda_architectures=tuple(
            f"sm_{major}{minor}"
            for major, minor in sorted(SUPPORTED_CUDA_ARCHITECTURES)
        ),
        minimum_vram_mib=minimum_vram_mib,
        selected_gpu_uuid=gpus[0].uuid if gpus else None,
        gpus=gpus,
        torch=torch_status,
        ctranslate2=ctranslate2_status,
        errors=tuple(failures),
        warnings=tuple(warnings),
    )
    if require_gpu and not ready:
        raise GpuPreflightError(report)
    return report


def write_gpu_report(
    path: Path,
    report: GpuPreflightReport,
    *,
    now: datetime | None = None,
) -> None:
    """Atomically publish worker-owned GPU readiness to shared state."""

    payload = report.model_dump(mode="json")
    payload["worker_heartbeat_at"] = (now or datetime.now(UTC)).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_gpu_report(
    path: Path,
    *,
    max_age_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read worker readiness without invoking CUDA from the API process."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("ready"), bool):
            raise ValueError("invalid GPU report shape")
        heartbeat_raw = payload.get("worker_heartbeat_at")
        if not isinstance(heartbeat_raw, str):
            raise ValueError("missing worker heartbeat")
        heartbeat = datetime.fromisoformat(heartbeat_raw)
        if heartbeat.tzinfo is None:
            raise ValueError("worker heartbeat has no timezone")
        age = max(0.0, ((now or datetime.now(UTC)) - heartbeat).total_seconds())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "ready": False,
            "enforced": True,
            "errors": [],
            "warnings": [
                f"Không đọc được báo cáo GPU của worker: {type(error).__name__}"
            ],
        }
    if age > max_age_seconds:
        payload["ready"] = False
        warnings = payload.get("warnings")
        normalized = list(warnings) if isinstance(warnings, list) else []
        normalized.append(f"Heartbeat GPU đã cũ {age:.1f} giây")
        payload["warnings"] = normalized
    payload["heartbeat_age_seconds"] = round(age, 3)
    return payload


__all__ = [
    "ComponentStatus",
    "CudaDeviceIdentity",
    "GpuPreflightError",
    "GpuPreflightReport",
    "GPU_SUPPORT_EXPERIMENTAL",
    "GPU_SUPPORT_MAINTENANCE_LIMITED",
    "GPU_SUPPORT_SUPPORTED",
    "NvidiaGpu",
    "gpu_support_tier",
    "SUPPORTED_CUDA_ARCHITECTURES",
    "inspect_gpu",
    "read_gpu_report",
    "probe_ctranslate2_cuda",
    "probe_torch_cuda",
    "write_gpu_report",
]
