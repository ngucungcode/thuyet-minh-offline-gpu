"""Offline GPU worker for checkpointed ASR, translation, and dubbing stages."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import threading
import time
from contextlib import suppress
from typing import Any

from .asr import FasterWhisperRecognizer
from .config import Settings, get_settings
from .gpu import GpuPreflightError, GpuPreflightReport, inspect_gpu, write_gpu_report
from .offline import install_offline_network_guard
from .phase4_stage import Phase4Stage, build_phase4_stage
from .state import JobStatus, StateStore
from .transcription_stage import TranscriptionStage
from .translation_stage import TranslationStage


_TRANSCRIPT_QUEUE = (
    JobStatus.READY_OFFLINE,
    JobStatus.TRANSCRIBING,
    JobStatus.SUBTITLE_SELECTED,
    JobStatus.CANCELLING,
)

_TRANSLATION_QUEUE = (
    JobStatus.READY_TRANSLATION,
    JobStatus.TRANSLATING,
)

_PHASE4_QUEUE = (
    JobStatus.READY_TTS,
    JobStatus.SEPARATING,
    JobStatus.SYNTHESIZING,
    JobStatus.TIMING,
    JobStatus.MIXING,
    JobStatus.MUXING,
    JobStatus.VERIFYING,
)


def _print_event(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_transcription_stage(
    settings: Settings,
    store: StateStore,
    shutdown: threading.Event,
) -> TranscriptionStage:
    """Build the production stage without any network-enabled dependency."""

    return TranscriptionStage(
        models_lock_path=settings.models_lock_path,
        models_dir=settings.models_dir,
        jobs_dir=settings.jobs_dir,
        default_asr_model_id=settings.default_asr_model_id,
        compute_type=settings.asr_compute_type,
        store=store,
        recognizer_factory=lambda: FasterWhisperRecognizer(
            language_confidence_threshold=(
                settings.asr_language_confidence_threshold
            )
        ),
        shutdown_requested=shutdown.is_set,
    )


def build_translation_stage(
    settings: Settings,
    store: StateStore,
    shutdown: threading.Event,
) -> TranslationStage:
    """Build the local llama.cpp stage with no model provisioning path."""

    return TranslationStage(
        models_lock_path=settings.models_lock_path,
        models_dir=settings.models_dir,
        jobs_dir=settings.jobs_dir,
        default_translation_model_id=settings.default_translation_model_id,
        store=store,
        llama_server_binary=settings.llama_server_binary,
        llama_server_port=settings.llama_server_port,
        llama_context_size=settings.llama_context_size,
        llama_max_output_tokens=settings.llama_max_output_tokens,
        llama_startup_timeout_seconds=settings.llama_startup_timeout_seconds,
        llama_request_timeout_seconds=settings.llama_request_timeout_seconds,
        shutdown_requested=shutdown.is_set,
    )


async def process_next_transcript_job(
    store: StateStore,
    stage: TranscriptionStage,
    *,
    shutdown: threading.Event,
) -> bool:
    """Process at most one job and return whether queue work was observed."""

    records = store.list_jobs(_TRANSCRIPT_QUEUE, limit=100)
    records = [
        item
        for item in records
        if item.status is not JobStatus.CANCELLING
        or bool(item.details.get("offline_cancel_pending"))
    ]
    if not records:
        return False
    record = records[0]
    if shutdown.is_set():
        return False
    if record.status is JobStatus.CANCELLING:
        # A restarted worker has no live native handle. If the API marked this
        # as an offline cancellation, finalizing is now safe and idempotent.
        if record.details.get("offline_cancel_pending"):
            store.finalize_cancel(record.id)
            _print_event("job.cancelled", job_id=record.id)
        return True

    _print_event(
        "transcription.started",
        job_id=record.id,
        status=record.status.value,
        transcript_source=record.details.get("transcript_source"),
    )
    result = await stage.run(record.id)
    current = store.get_job(record.id)
    if current.status is JobStatus.CANCELLING:
        # stage.run has returned, therefore decoder/model cleanup and the
        # recognizer's finally block have completed before this transition.
        current = store.finalize_cancel(record.id)
    _print_event(
        "transcription.finished",
        job_id=current.id,
        status=current.status.value,
        error_code=current.error_code,
    )
    return True


async def process_next_translation_job(
    store: StateStore,
    stage: TranslationStage,
    *,
    shutdown: threading.Event,
) -> bool:
    """Process at most one ready/resumable translation job."""

    records = store.list_jobs(_TRANSLATION_QUEUE, limit=100)
    if not records or shutdown.is_set():
        return False
    record = records[0]
    _print_event(
        "translation.started",
        job_id=record.id,
        status=record.status.value,
        model_id=(record.spec.get("models") or {}).get("translation"),
    )
    await stage.run(record.id)
    current = store.get_job(record.id)
    if current.status is JobStatus.CANCELLING:
        # TranslationStage has closed llama-server in its finally block.
        current = store.finalize_cancel(record.id)
    _print_event(
        "translation.finished",
        job_id=current.id,
        status=current.status.value,
        error_code=current.error_code,
    )
    return True


async def process_next_phase4_job(
    store: StateStore,
    stage: Phase4Stage,
    *,
    shutdown: threading.Event,
) -> bool:
    """Process at most one ready or checkpointed Phase 4 job."""

    records = store.list_jobs(_PHASE4_QUEUE, limit=100)
    if not records or shutdown.is_set():
        return False
    record = records[0]
    _print_event(
        "phase4.started",
        job_id=record.id,
        status=record.status.value,
        separation_model_id=(record.spec.get("models") or {}).get("separation"),
        tts_model_id=(record.spec.get("models") or {}).get("tts"),
    )
    await stage.run(record.id)
    current = store.get_job(record.id)
    if current.status is JobStatus.CANCELLING:
        # All native/subprocess handles are released by Phase4Stage finally
        # blocks before run() returns.
        current = store.finalize_cancel(record.id)
    _print_event(
        "phase4.finished",
        job_id=current.id,
        status=current.status.value,
        error_code=current.error_code,
    )
    return True


async def _heartbeat_loop(
    settings: Settings,
    report: GpuPreflightReport,
    shutdown: threading.Event,
) -> None:
    interval = min(max(settings.gpu_report_max_age_seconds / 3.0, 1.0), 10.0)
    while not shutdown.is_set():
        write_gpu_report(settings.gpu_report_path, report)
        await asyncio.sleep(interval)


async def _worker_loop(
    settings: Settings,
    report: GpuPreflightReport,
    *,
    shutdown: threading.Event,
    poll_seconds: float,
    gpu_recheck_seconds: float,
) -> None:
    store = StateStore(settings.database_path)
    transcription_stage = build_transcription_stage(settings, store, shutdown)
    translation_stage = build_translation_stage(settings, store, shutdown)
    phase4_stage = build_phase4_stage(settings, store, shutdown)
    heartbeat = asyncio.create_task(
        _heartbeat_loop(settings, report, shutdown),
        name="gpu-heartbeat",
    )
    next_gpu_check = time.monotonic() + max(gpu_recheck_seconds, 5.0)
    try:
        while not shutdown.is_set():
            worked = await process_next_transcript_job(
                store,
                transcription_stage,
                shutdown=shutdown,
            )
            if not worked:
                worked = await process_next_translation_job(
                    store,
                    translation_stage,
                    shutdown=shutdown,
                )
            if not worked:
                worked = await process_next_phase4_job(
                    store,
                    phase4_stage,
                    shutdown=shutdown,
                )
            if shutdown.is_set():
                break
            if time.monotonic() >= next_gpu_check:
                try:
                    report = await asyncio.to_thread(inspect_gpu, require_gpu=True)
                except GpuPreflightError as error:
                    write_gpu_report(settings.gpu_report_path, error.report)
                    _print_event(
                        "gpu.failed",
                        report=error.report.model_dump(mode="json"),
                    )
                    raise
                write_gpu_report(settings.gpu_report_path, report)
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
                heartbeat = asyncio.create_task(
                    _heartbeat_loop(settings, report, shutdown),
                    name="gpu-heartbeat",
                )
                next_gpu_check = time.monotonic() + max(
                    gpu_recheck_seconds,
                    5.0,
                )
            if not worked:
                await asyncio.sleep(max(poll_seconds, 0.1))
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        write_gpu_report(settings.gpu_report_path, report)


def _install_signal_handlers(shutdown: threading.Event) -> None:
    def stop(_signum: int, _frame: object) -> None:
        shutdown.set()

    for signal_name in (signal.SIGTERM, signal.SIGINT):
        with suppress(ValueError, OSError):
            signal.signal(signal_name, stop)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline dubbing worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run only the GPU startup check and exit",
    )
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--gpu-recheck-seconds", type=float, default=30.0)
    args = parser.parse_args()

    settings = get_settings()
    try:
        report = inspect_gpu(require_gpu=True)
    except GpuPreflightError as error:
        report = error.report
        write_gpu_report(settings.gpu_report_path, report)
        _print_event("gpu.failed", report=report.model_dump(mode="json"))
        raise SystemExit(2) from error
    write_gpu_report(settings.gpu_report_path, report)
    _print_event("gpu.ready", report=report.model_dump(mode="json"))
    if args.once:
        return

    # No model provisioning, telemetry, API client, or acquisition adapter is
    # imported after this point. Docker adds network_mode:none; native mode is
    # additionally fail-closed for Python DNS/TCP through this audit hook.
    install_offline_network_guard(
        allowed_loopback_ports={settings.llama_server_port}
    )
    shutdown = threading.Event()
    _install_signal_handlers(shutdown)
    poll_seconds = (
        args.poll_seconds
        if args.poll_seconds is not None
        else settings.offline_worker_poll_seconds
    )
    try:
        asyncio.run(
            _worker_loop(
                settings,
                report,
                shutdown=shutdown,
                poll_seconds=max(poll_seconds, 0.1),
                gpu_recheck_seconds=max(args.gpu_recheck_seconds, 5.0),
            )
        )
    except GpuPreflightError as error:
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
