from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dub_server.gpu import (
    ComponentStatus,
    CudaDeviceIdentity,
    GpuPreflightError,
    inspect_gpu,
    read_gpu_report,
    write_gpu_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, stderr)


def successful_torch() -> ComponentStatus:
    return ComponentStatus(True, "2.8.0", "CUDA 12.8; Mock GPU; matmul checksum=256")


def successful_ctranslate2() -> ComponentStatus:
    return ComponentStatus(True, "4.8.1", compute_types=("float16", "int8_float16"))


def test_gpu_preflight_accepts_eligible_gpu() -> None:
    report = inspect_gpu(
        command_runner=lambda _: completed("GPU-test, NVIDIA RTX Test, 570.26, 16384, 8.9\n"),
        torch_probe=successful_torch,
        ctranslate2_probe=successful_ctranslate2,
    )

    assert report.ready is True
    assert report.gpus[0].memory_total_mib == 16384
    assert report.ctranslate2.compute_types == ("float16", "int8_float16")
    assert report.model_dump()["gpus"][0]["uuid"] == "GPU-test"


def test_gpu_preflight_binds_native_runtime_to_expected_uuid_and_architecture() -> None:
    report = inspect_gpu(
        command_runner=lambda _: completed(
            "GPU-selected, NVIDIA RTX Test, 570.26, 24576, 8.6\n"
        ),
        torch_probe=successful_torch,
        ctranslate2_probe=successful_ctranslate2,
        expected_gpu_uuid="GPU-selected",
        expected_cuda_architecture="sm_86",
    )

    assert report.ready is True


@pytest.mark.parametrize(
    ("expected_uuid", "expected_architecture", "message"),
    [
        ("GPU-other", "sm_86", "does not match installed runtime UUID"),
        ("GPU-selected", "sm_80", "does not match installed runtime sm_80"),
    ],
)
def test_gpu_preflight_rejects_runtime_gpu_binding_mismatch(
    expected_uuid: str,
    expected_architecture: str,
    message: str,
) -> None:
    with pytest.raises(GpuPreflightError) as captured:
        inspect_gpu(
            command_runner=lambda _: completed(
                "GPU-selected, NVIDIA RTX Test, 570.26, 24576, 8.6\n"
            ),
            torch_probe=successful_torch,
            ctranslate2_probe=successful_ctranslate2,
            expected_gpu_uuid=expected_uuid,
            expected_cuda_architecture=expected_architecture,
        )

    assert any(message in item for item in captured.value.report.errors)


@pytest.mark.parametrize(
    "compute_capability",
    ("7.0", "7.5", "8.0", "8.6", "8.9", "9.0"),
)
def test_gpu_preflight_accepts_only_the_release_cuda_architectures(
    compute_capability: str,
) -> None:
    report = inspect_gpu(
        command_runner=lambda _: completed(
            f"GPU-logical-0, NVIDIA Test GPU, 570.26, 6144, {compute_capability}\n"
        ),
        torch_probe=successful_torch,
        ctranslate2_probe=successful_ctranslate2,
    )

    assert report.ready is True
    assert report.minimum_vram_mib == 6144
    assert report.gpus[0].compute_capability == compute_capability


@pytest.mark.parametrize(
    "compute_capability",
    ("6.0", "6.1", "8.7", "9.1", "10.0"),
)
def test_gpu_preflight_rejects_pascal_and_unknown_cuda_architectures(
    compute_capability: str,
) -> None:
    with pytest.raises(GpuPreflightError) as captured:
        inspect_gpu(
            command_runner=lambda _: completed(
                f"GPU-logical-0, NVIDIA Test GPU, 570.26, 24576, {compute_capability}\n"
            ),
            torch_probe=successful_torch,
            ctranslate2_probe=successful_ctranslate2,
        )

    assert captured.value.report.ready is False
    assert any(
        compute_capability in warning
        for warning in captured.value.report.warnings
    )


def test_gpu_preflight_rejects_one_mib_below_the_release_vram_floor() -> None:
    with pytest.raises(GpuPreflightError) as captured:
        inspect_gpu(
            command_runner=lambda _: completed(
                "GPU-logical-0, NVIDIA Test GPU, 570.26, 6143, 8.6\n"
            ),
            torch_probe=successful_torch,
            ctranslate2_probe=successful_ctranslate2,
        )

    assert captured.value.report.minimum_vram_mib == 6144
    assert any("6143" in warning for warning in captured.value.report.warnings)


def test_gpu_preflight_fails_when_logical_gpu_zero_is_unsupported() -> None:
    rows = (
        "GPU-logical-0, NVIDIA Pascal GPU, 570.26, 24576, 6.1\n"
        "GPU-logical-1, NVIDIA Ampere GPU, 570.26, 24576, 8.6\n"
    )

    with pytest.raises(GpuPreflightError) as captured:
        inspect_gpu(
            command_runner=lambda _: completed(rows),
            torch_probe=successful_torch,
            ctranslate2_probe=successful_ctranslate2,
        )

    assert captured.value.report.ready is False
    assert captured.value.report.gpus[0].uuid == "GPU-logical-0"
    assert any(
        "GPU-logical-0" in warning
        for warning in captured.value.report.warnings
    )


def test_gpu_preflight_uses_supported_logical_gpu_zero_despite_other_gpus() -> None:
    rows = (
        "GPU-logical-0, NVIDIA Ampere GPU, 570.26, 6144, 8.0\n"
        "GPU-logical-1, NVIDIA Pascal GPU, 570.26, 24576, 6.1\n"
    )

    report = inspect_gpu(
        command_runner=lambda _: completed(rows),
        torch_probe=successful_torch,
        ctranslate2_probe=successful_ctranslate2,
    )

    assert report.ready is True
    assert report.gpus[0].uuid == "GPU-logical-0"
    assert any("GPU-logical-1" in warning for warning in report.warnings)


def test_gpu_preflight_uses_pytorch_identity_after_cuda_device_remap() -> None:
    rows = (
        "GPU-physical-0, NVIDIA Large GPU, 570.26, 24576, 8.6\n"
        "GPU-physical-1, NVIDIA Small GPU, 570.26, 7168, 8.0\n"
    )
    selected_by_torch = ComponentStatus(
        True,
        "2.8.0",
        "CUDA 12.8; NVIDIA Small GPU; matmul checksum=256",
        device=CudaDeviceIdentity(
            uuid="GPU-physical-1",
            name="NVIDIA Small GPU",
            memory_total_mib=7000,
            compute_capability="8.0",
        ),
    )

    report = inspect_gpu(
        command_runner=lambda _: completed(rows),
        torch_probe=lambda: selected_by_torch,
        ctranslate2_probe=successful_ctranslate2,
    )

    assert report.ready is True
    assert report.selected_gpu_uuid == "GPU-physical-1"
    assert report.gpus[0].uuid == "GPU-physical-1"
    assert report.gpus[0].memory_total_mib == 7000


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("GPU-test, NVIDIA MX250, 570.26, 2048, 6.1\n", "no visible GPU"),
        ("GPU-test, NVIDIA RTX Test, 569.99, 16384, 8.9\n", "no visible GPU"),
    ],
)
def test_gpu_preflight_rejects_unsupported_hardware(row: str, expected: str) -> None:
    with pytest.raises(GpuPreflightError, match=expected) as captured:
        inspect_gpu(
            command_runner=lambda _: completed(row),
            torch_probe=successful_torch,
            ctranslate2_probe=successful_ctranslate2,
        )

    assert captured.value.report.ready is False
    assert captured.value.report.warnings


def test_gpu_preflight_requires_torch_cuda_kernel() -> None:
    with pytest.raises(GpuPreflightError, match="PyTorch CUDA"):
        inspect_gpu(
            command_runner=lambda _: completed("GPU-test, NVIDIA RTX Test, 570.26, 16384, 8.9\n"),
            torch_probe=lambda: ComponentStatus(False, "2.8.0", "torch.cuda is unavailable"),
            ctranslate2_probe=successful_ctranslate2,
        )


def test_gpu_preflight_requires_ctranslate2_cuda_compute_type() -> None:
    with pytest.raises(GpuPreflightError, match="CTranslate2 CUDA"):
        inspect_gpu(
            command_runner=lambda _: completed("GPU-test, NVIDIA RTX Test, 570.26, 16384, 8.9\n"),
            torch_probe=successful_torch,
            ctranslate2_probe=lambda: ComponentStatus(False, "4.8.1", "float32 only", ("float32",)),
        )


def test_non_enforcing_mode_is_safe_for_cpu_only_unit_tests() -> None:
    report = inspect_gpu(
        require_gpu=False,
        command_runner=lambda _: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
        torch_probe=lambda: ComponentStatus(False, detail="torch is not installed"),
        ctranslate2_probe=lambda: ComponentStatus(False, detail="ctranslate2 is not installed"),
    )

    assert report.ready is False
    assert report.errors == ()
    assert any("nvidia-smi" in warning for warning in report.warnings)


def test_worker_gpu_report_is_atomic_and_stale_heartbeat_fails_closed(
    tmp_path: Path,
) -> None:
    checked = datetime(2026, 7, 31, tzinfo=UTC)
    report = inspect_gpu(
        command_runner=lambda _: completed(
            "GPU-test, NVIDIA RTX Test, 570.26, 16384, 8.9\n"
        ),
        torch_probe=successful_torch,
        ctranslate2_probe=successful_ctranslate2,
    )
    path = tmp_path / "state" / "gpu-health.json"
    write_gpu_report(path, report, now=checked)

    fresh = read_gpu_report(
        path,
        max_age_seconds=60,
        now=checked + timedelta(seconds=30),
    )
    stale = read_gpu_report(
        path,
        max_age_seconds=60,
        now=checked + timedelta(seconds=61),
    )

    assert fresh["ready"] is True
    assert fresh["heartbeat_age_seconds"] == 30.0
    assert stale["ready"] is False
    assert "Heartbeat GPU" in stale["warnings"][-1]


def test_locked_catalog_has_multiple_offline_choices_for_every_stage() -> None:
    catalog = json.loads((PROJECT_ROOT / "config" / "models.lock.json").read_text(encoding="utf-8"))
    models = catalog["models"]
    stage_counts = Counter(model["stage"] for model in models)

    assert catalog["runtime_downloads_allowed"] is False
    assert stage_counts["asr"] >= 2
    assert stage_counts["mt"] >= 2
    assert stage_counts["tts"] >= 2
    assert len({model["id"] for model in models}) == len(models)
    for model in models:
        assert model["installed"] is False
        assert model["revision"]
        assert model["license"]
        assert re.fullmatch(r"[0-9a-f]{64}", model["sha256"])


def test_compose_keeps_worker_offline_and_management_ports_local() -> None:
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "network_mode: none" in compose
    assert "capabilities: [gpu]" in compose
    assert "profiles: [models]" in compose
    assert compose.count('127.0.0.1:${') >= 3
    assert ":latest" not in compose
    assert "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04@sha256:" in dockerfile
