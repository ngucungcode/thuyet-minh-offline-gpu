from __future__ import annotations

import json
import threading

import pytest

import dub_server.worker as worker_module
from dub_server.config import Settings
from dub_server.gpu import ComponentStatus, GpuPreflightReport, NvidiaGpu
from dub_server.state import JobStage, JobStatus, StateStore
from dub_server.worker import process_next_phase4_job, process_next_transcript_job


def _ready_offline(store: StateStore, *, job_id: str = "job-1") -> str:
    job = store.create_job(
        "release-1",
        {"rights_confirmed": True},
        job_id=job_id,
    )
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(
        job.id,
        JobStatus.READY_OFFLINE,
        stage=JobStage.SUBTITLE,
        details={"transcript_source": "asr"},
    )
    return job.id


def _ready_gpu_report(*, vram_mib: int) -> GpuPreflightReport:
    component = ComponentStatus(available=True, version="test")
    return GpuPreflightReport(
        ready=True,
        enforced=True,
        checked_at="2026-08-09T00:00:00Z",
        minimum_driver="570.26",
        minimum_compute_capability="7.0",
        supported_cuda_architectures=("7.0",),
        minimum_vram_mib=6144,
        selected_gpu_uuid="GPU-test",
        gpus=(
            NvidiaGpu(
                uuid="GPU-test",
                name="NVIDIA test",
                driver_version="570.26",
                memory_total_mib=vram_mib,
                compute_capability="7.0",
            ),
        ),
        torch=component,
        ctranslate2=component,
        errors=(),
        warnings=(),
    )


def test_worker_forwards_installed_cuda_toolkit_to_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    report = _ready_gpu_report(vram_mib=24_576)

    def fake_inspect_gpu(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(worker_module, "inspect_gpu", fake_inspect_gpu)
    settings = Settings(
        selected_gpu_uuid="GPU-test",
        selected_cuda_architecture="sm_86",
        selected_cuda_toolkit_version="12.6",
    )

    assert worker_module._inspect_configured_gpu(settings) is report
    assert captured["expected_cuda_toolkit_version"] == "12.6"


def test_worker_cpu_mode_turns_gpu_failures_into_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    unavailable = ComponentStatus(available=False, detail="CUDA unavailable")
    report = GpuPreflightReport(
        ready=False,
        enforced=False,
        checked_at="2026-08-19T00:00:00Z",
        minimum_driver="570.26",
        minimum_compute_capability="7.0",
        supported_cuda_architectures=("sm_75",),
        minimum_vram_mib=6144,
        selected_gpu_uuid="GPU-mx250",
        gpus=(
            NvidiaGpu(
                uuid="GPU-mx250",
                name="NVIDIA GeForce MX250",
                driver_version="511.69",
                memory_total_mib=2048,
                compute_capability="6.1",
            ),
        ),
        torch=unavailable,
        ctranslate2=unavailable,
        errors=(),
        warnings=("unsupported GPU",),
    )

    def fake_inspect_gpu(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(worker_module, "inspect_gpu", fake_inspect_gpu)
    result = worker_module._inspect_configured_gpu(Settings(compute_mode="cpu"))

    assert captured["require_gpu"] is False
    assert result.ready is True
    assert result.enforced is False
    assert result.gpus == ()
    assert result.warnings == ("unsupported GPU",)


def test_worker_cpu_mode_skips_vram_model_guard(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job_id = _ready_offline(store)
    settings = Settings(
        compute_mode="cpu",
        models_lock_path=tmp_path / "missing-models.lock.json",
        models_dir=tmp_path / "models",
    )

    assert worker_module._ensure_selected_models_fit_vram(
        store,
        store.get_job(job_id),
        settings=settings,
        report=_ready_gpu_report(vram_mib=2048),
        stages=("asr",),
    )


@pytest.mark.asyncio
async def test_worker_dispatches_one_ready_transcript_job(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job_id = _ready_offline(store)
    calls: list[str] = []

    class Stage:
        async def run(self, selected_job_id: str):
            calls.append(selected_job_id)
            return store.update_status(
                selected_job_id,
                JobStatus.TRANSCRIBING,
                expected_status=JobStatus.READY_OFFLINE,
                stage=JobStage.ASR,
            )

    worked = await process_next_transcript_job(
        store,
        Stage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
    )

    assert worked is True
    assert calls == [job_id]
    assert store.get_job(job_id).status is JobStatus.TRANSCRIBING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript_source",
    ["asr", "subtitle"],
    ids=["direct-asr", "preferred-subtitle-fallback"],
)
async def test_worker_rechecks_model_vram_before_loading_the_model(
    tmp_path,
    transcript_source: str,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "subtitle_mode": "prefer",
            "models": {"asr": "asr-too-large"},
        },
        job_id="job-vram",
    )
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(
        job.id,
        JobStatus.READY_OFFLINE,
        stage=JobStage.SUBTITLE,
        details={"transcript_source": transcript_source},
    )
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "asr-too-large",
                        "stage": "asr",
                        "backend": "test",
                        "license": "MIT",
                        "minimum_vram_mib": 8192,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        database_path=tmp_path / "unused.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
    )

    class ForbiddenStage:
        async def run(self, _job_id: str):
            raise AssertionError("Model exceeding VRAM must not be loaded")

    worked = await process_next_transcript_job(
        store,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
        settings=settings,
        gpu_report=_ready_gpu_report(vram_mib=6144),
    )

    failed = store.get_job(job.id)
    assert worked is True
    assert failed.status is JobStatus.FAILED
    assert failed.error_code == "model_vram_insufficient"
    assert failed.retryable is False
    assert failed.details["required_vram_mib"] == 8192
    assert failed.details["available_vram_mib"] == 6144


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "catalog_failure",
    [False, True],
    ids=["vram-guard", "catalog-guard"],
)
async def test_worker_preserves_cancellation_that_races_model_guard_failure(
    tmp_path,
    monkeypatch,
    catalog_failure: bool,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "models": {"asr": "asr-too-large"},
        },
        job_id=f"job-race-{catalog_failure}",
    )
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.update_status(
        job.id,
        JobStatus.READY_OFFLINE,
        stage=JobStage.SUBTITLE,
        details={"transcript_source": "asr"},
    )
    lock_path = tmp_path / "models.lock.json"
    if catalog_failure:
        lock_path.write_text("{not-json", encoding="utf-8")
    else:
        lock_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": [
                        {
                            "id": "asr-too-large",
                            "stage": "asr",
                            "backend": "test",
                            "license": "MIT",
                            "minimum_vram_mib": 8192,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    settings = Settings(
        database_path=tmp_path / "unused.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
    )
    original_update_status = store.update_status
    race_won = False

    def update_status_with_cancel(job_id, status, **kwargs):
        nonlocal race_won
        if status is JobStatus.FAILED and not race_won:
            race_won = True
            cancelling = store.request_cancel(job_id)
            original_update_status(
                job_id,
                JobStatus.CANCELLING,
                expected_status=JobStatus.CANCELLING,
                details={**cancelling.details, "offline_cancel_pending": True},
                cancel_requested=True,
            )
        return original_update_status(job_id, status, **kwargs)

    monkeypatch.setattr(store, "update_status", update_status_with_cancel)

    class ForbiddenStage:
        async def run(self, _job_id: str):
            raise AssertionError("A cancellation race must prevent model loading")

    worked = await process_next_transcript_job(
        store,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
        settings=settings,
        gpu_report=_ready_gpu_report(vram_mib=6144),
    )

    cancelling = store.get_job(job.id)
    assert worked is True
    assert race_won is True
    assert cancelling.status is JobStatus.CANCELLING
    assert cancelling.cancel_requested is True
    assert cancelling.error_code is None


@pytest.mark.asyncio
async def test_worker_finalizes_offline_cancel_only_after_stage_returns(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job_id = _ready_offline(store)

    class Stage:
        async def run(self, selected_job_id: str):
            store.update_status(
                selected_job_id,
                JobStatus.TRANSCRIBING,
                expected_status=JobStatus.READY_OFFLINE,
                stage=JobStage.ASR,
            )
            cancelling = store.request_cancel(selected_job_id)
            return store.update_status(
                selected_job_id,
                JobStatus.CANCELLING,
                expected_status=JobStatus.CANCELLING,
                details={**cancelling.details, "offline_cancel_pending": True},
                cancel_requested=True,
            )

    await process_next_transcript_job(
        store,
        Stage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
    )

    assert store.get_job(job_id).status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_restarted_worker_reconciles_pending_offline_cancel(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job_id = _ready_offline(store)
    cancelling = store.request_cancel(job_id)
    store.update_status(
        job_id,
        JobStatus.CANCELLING,
        expected_status=JobStatus.CANCELLING,
        details={**cancelling.details, "offline_cancel_pending": True},
        cancel_requested=True,
    )

    class ForbiddenStage:
        async def run(self, _job_id: str):
            raise AssertionError("Reconciliation must not start inference")

    worked = await process_next_transcript_job(
        store,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
    )

    assert worked is True
    assert store.get_job(job_id).status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_worker_ignores_acquisition_cancellation_and_honors_shutdown(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(job.id, JobStatus.DOWNLOADING)
    store.request_cancel(job.id)

    class ForbiddenStage:
        async def run(self, _job_id: str):
            raise AssertionError("No transcript job should run")

    assert await process_next_transcript_job(
        store,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
    ) is False

    second_root = tmp_path / "second"
    second = StateStore(second_root / "jobs.sqlite3")
    _ready_offline(second)
    stopped = threading.Event()
    stopped.set()
    assert await process_next_transcript_job(
        second,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=stopped,
    ) is False


@pytest.mark.asyncio
async def test_worker_dispatches_and_safely_finalizes_phase4(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job("release-1", {"rights_confirmed": True, "models": {}})
    store.update_status(
        job.id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        force=True,
    )
    calls: list[str] = []

    class Stage:
        async def run(self, selected_job_id: str):
            calls.append(selected_job_id)
            store.update_status(
                selected_job_id,
                JobStatus.SEPARATING,
                expected_status=JobStatus.READY_TTS,
                stage=JobStage.SEPARATION,
            )
            store.request_cancel(selected_job_id)
            return store.get_job(selected_job_id)

    worked = await process_next_phase4_job(
        store,
        Stage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
    )

    assert worked is True
    assert calls == [job.id]
    assert store.get_job(job.id).status is JobStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase4_status",
    [JobStatus.SYNTHESIZING, JobStatus.TIMING],
    ids=["synthesizing", "timing"],
)
async def test_worker_checks_translation_vram_for_natural_timing_rewrite(
    tmp_path,
    phase4_status: JobStatus,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "timing_profile": "natural",
            "models": {
                "translation": "mt-too-large",
                "tts": "tts-small",
            },
        },
    )
    store.update_status(
        job.id,
        phase4_status,
        stage=(
            JobStage.TTS
            if phase4_status is JobStatus.SYNTHESIZING
            else JobStage.TIMING
        ),
        progress_permille=800,
        force=True,
    )
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "tts-small",
                        "stage": "tts",
                        "backend": "test",
                        "license": "MIT",
                        "minimum_vram_mib": 4096,
                    },
                    {
                        "id": "mt-too-large",
                        "stage": "mt",
                        "backend": "test",
                        "license": "MIT",
                        "minimum_vram_mib": 8192,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        database_path=tmp_path / "unused.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
    )

    class ForbiddenStage:
        async def run(self, _job_id: str):
            raise AssertionError("Oversized rewrite model must not be loaded")

    worked = await process_next_phase4_job(
        store,
        ForbiddenStage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
        settings=settings,
        gpu_report=_ready_gpu_report(vram_mib=6144),
    )

    failed = store.get_job(job.id)
    assert worked is True
    assert failed.status is JobStatus.FAILED
    assert failed.error_code == "model_vram_insufficient"
    assert failed.retryable is False
    assert failed.details["required_model_id"] == "mt-too-large"
    assert failed.details["required_vram_mib"] == 8192
    assert failed.details["available_vram_mib"] == 6144


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timing_profile", "rewrite_max_attempts"),
    [("strict", 3), ("natural", 0)],
    ids=["strict-profile", "rewrite-disabled"],
)
async def test_worker_skips_translation_vram_when_timing_rewrite_is_disabled(
    tmp_path,
    timing_profile: str,
    rewrite_max_attempts: int,
) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "timing_profile": timing_profile,
            "models": {
                "translation": "mt-too-large",
                "tts": "tts-small",
            },
        },
    )
    store.update_status(
        job.id,
        JobStatus.TIMING,
        stage=JobStage.TIMING,
        progress_permille=850,
        force=True,
    )
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "tts-small",
                        "stage": "tts",
                        "backend": "test",
                        "license": "MIT",
                        "minimum_vram_mib": 4096,
                    },
                    {
                        "id": "mt-too-large",
                        "stage": "mt",
                        "backend": "test",
                        "license": "MIT",
                        "minimum_vram_mib": 8192,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        database_path=tmp_path / "unused.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
        timing_rewrite_max_attempts=rewrite_max_attempts,
    )
    calls: list[str] = []

    class Stage:
        async def run(self, selected_job_id: str):
            calls.append(selected_job_id)
            return store.get_job(selected_job_id)

    worked = await process_next_phase4_job(
        store,
        Stage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
        settings=settings,
        gpu_report=_ready_gpu_report(vram_mib=6144),
    )

    assert worked is True
    assert calls == [job.id]
    assert store.get_job(job.id).status is JobStatus.TIMING


@pytest.mark.asyncio
async def test_worker_does_not_apply_model_vram_floor_after_inference(tmp_path) -> None:
    store = StateStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "models": {
                "separation": "separation-heavy",
                "tts": "tts-heavy",
            },
        },
    )
    store.update_status(
        job.id,
        JobStatus.MIXING,
        stage=JobStage.MIX,
        progress_permille=900,
        force=True,
    )
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text("{not-json", encoding="utf-8")
    settings = Settings(
        database_path=tmp_path / "unused.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
    )
    calls: list[str] = []

    class Stage:
        async def run(self, selected_job_id: str):
            calls.append(selected_job_id)
            return store.get_job(selected_job_id)

    worked = await process_next_phase4_job(
        store,
        Stage(),  # type: ignore[arg-type]
        shutdown=threading.Event(),
        settings=settings,
        gpu_report=_ready_gpu_report(vram_mib=6144),
    )

    assert worked is True
    assert calls == [job.id]
    assert store.get_job(job.id).status is JobStatus.MIXING
