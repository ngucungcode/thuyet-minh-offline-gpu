from __future__ import annotations

import threading

import pytest

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
