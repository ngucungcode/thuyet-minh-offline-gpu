from __future__ import annotations

import asyncio
import hashlib
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from dub_server.audio_mix_export import ExportedMedia
from dub_server.audio_separation import (
    AudioSeparationCancelled,
    AudioSeparationMetrics,
    AudioSeparationResult,
)
from dub_server.llama_translation import LlamaTranslationError
from dub_server.model_registry import VerifiedModel
from dub_server.narration import (
    NarrationError,
    SynthesizedNarration,
    TTS_SILENCE_TRIM_VERSION,
)
from dub_server.phase4_stage import Phase4Stage, _next_block_progress
from dub_server.state import JobStage, JobStatus, StateStore
from dub_server.timing import (
    FittedNarrationBlock,
    TimingError,
    TimingProfile,
    TimingQuality,
)
from dub_server.translation_artifact import (
    TranslationResult,
    TranslationSegment,
    load_translation_artifact,
    write_translation_artifact,
)


MODEL_SHA = "a" * 64
SOURCE_SHA = "b" * 64


def test_block_progress_coalesces_large_jobs_without_losing_completion() -> None:
    last_persisted = 735
    emitted: list[int] = []
    for completed in range(1, 10_001):
        mapped = _next_block_progress(
            completed=completed,
            total=10_000,
            range_start=735,
            range_size=115,
            last_persisted=last_persisted,
        )
        if mapped is not None:
            emitted.append(mapped)
            last_persisted = mapped

    assert emitted == sorted(set(emitted))
    assert len(emitted) == 115
    assert emitted[-1] == 850


def test_adaptive_rewrite_plan_rejects_audio_at_minimum_duration() -> None:
    with pytest.raises(TimingError) as caught:
        Phase4Stage._adaptive_timing_rewrite_plan(
            available_us=80_000,
            maximum_total_speed=1.2,
            previous_text="Rất ngắn",
            previous_target_us=120_000,
            previous_observed_us=120_000,
            adaptive_attempt=1,
        )

    assert caught.value.code == "timing_semantic_budget_impossible"
    assert caught.value.retryable is False


def test_adaptive_rewrite_target_stays_below_measured_duration() -> None:
    target_us, _max_words = Phase4Stage._adaptive_timing_rewrite_plan(
        available_us=200_000,
        maximum_total_speed=1.2,
        previous_text="Rất ngắn",
        previous_target_us=300_000,
        previous_observed_us=120_001,
        adaptive_attempt=1,
    )

    assert target_us == 120_000
    assert target_us < 120_001


def _write_wav(
    path: Path,
    *,
    sample_rate: int,
    frame_count: int,
    channels: int = 1,
    sample_value: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        sample = int(sample_value).to_bytes(2, "little", signed=True)
        stream.writeframes(sample * frame_count * channels)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_job(
    tmp_path: Path, *, timing_profile: str | None = None
) -> tuple[StateStore, str, Path]:
    store = StateStore(tmp_path / "state" / "jobs.sqlite3")
    source = tmp_path / "incoming" / "movie.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"local movie fixture")
    spec: dict[str, Any] = {
        "rights_confirmed": True,
        "models": {
            "separation": "separation-test",
            "tts": "tts-test",
        },
    }
    if timing_profile is not None:
        spec["timing_profile"] = timing_profile
    job = store.create_job(
        "release-1",
        spec,
    )
    translation = write_translation_artifact(
        tmp_path / "jobs" / job.id / "translated-transcript.json",
        TranslationResult(
            source_language="en",
            target_language="vi",
            duration_us=2_000_000,
            source_transcript_sha256=SOURCE_SHA,
            model_id="mt-test",
            segments=(
                TranslationSegment(0, 1_000_000, "One", "Một"),
                TranslationSegment(1_000_000, 2_000_000, "Two", "Hai"),
            ),
        ),
    )
    store.update_status(
        job.id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        details={
            "source_media_path": str(source),
            "source_transcript_sha256": SOURCE_SHA,
            "translated_transcript_path": str(translation.path),
            "translated_transcript_sha256": translation.sha256,
        },
        force=True,
    )
    return store, job.id, source


def _ready_elastic_gap_job(tmp_path: Path) -> tuple[StateStore, str, Path]:
    store = StateStore(tmp_path / "state" / "jobs.sqlite3")
    source = tmp_path / "incoming" / "movie.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"local movie fixture")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
            "timing_profile": "natural",
            "models": {
                "separation": "separation-test",
                "tts": "tts-test",
            },
        },
    )
    translation = write_translation_artifact(
        tmp_path / "jobs" / job.id / "translated-transcript.json",
        TranslationResult(
            source_language="en",
            target_language="vi",
            duration_us=3_000_000,
            source_transcript_sha256=SOURCE_SHA,
            model_id="mt-test",
            segments=(
                TranslationSegment(1_000_000, 2_000_000, "One", "Một"),
            ),
        ),
    )
    store.update_status(
        job.id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        details={
            "source_media_path": str(source),
            "source_transcript_sha256": SOURCE_SHA,
            "translated_transcript_path": str(translation.path),
            "translated_transcript_sha256": translation.sha256,
        },
        force=True,
    )
    return store, job.id, source


def _exhausted_v2_rewrite(
    *,
    ordinal: int,
    source_text: str,
    original_text: str,
    rewritten_text: str,
    observed_duration_us: int,
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "text": rewritten_text,
        "text_sha256": hashlib.sha256(rewritten_text.encode()).hexdigest(),
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "original_translation_sha256": hashlib.sha256(
            original_text.encode()
        ).hexdigest(),
        "attempt": 3,
        "adaptive_attempt": 3,
        "legacy_attempt_count": 0,
        "available_duration_us": 666_666,
        "target_duration_us": 500_000,
        "max_words": 2,
        "previous_text_sha256": hashlib.sha256(
            original_text.encode()
        ).hexdigest(),
        "previous_target_duration_us": 600_000,
        "previous_observed_duration_us": observed_duration_us,
        "observed_duration_us": observed_duration_us,
        "model_id": "mt-test",
        "model_tree_sha256": MODEL_SHA,
        "prompt_version": "timing-rewrite-v2",
        "accepted": False,
        "history": [],
    }


def _replace_tts_block_text(
    payload: dict[str, Any], *, ordinal: int, text: str
) -> None:
    blocks = [dict(item) for item in payload["blocks"]]
    blocks[ordinal]["text"] = text
    blocks[ordinal]["tts_normalized_text"] = text
    blocks[ordinal]["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    payload["blocks"] = blocks


class FakeSeparator:
    def __init__(self, *, duration_us: int = 2_000_000) -> None:
        self.calls = 0
        self.duration_us = duration_us

    async def separate(self, source: Path, output: Path, **kwargs: Any):
        self.calls += 1
        callback = kwargs.get("on_progress")
        if callback is not None:
            from dub_server.audio_separation import SeparationProgress

            callback(SeparationProgress("separating", 500, "Đang tách"))
        frame_count = round(self.duration_us * 48_000 / 1_000_000)
        _write_wav(
            output,
            sample_rate=48_000,
            frame_count=frame_count,
            channels=2,
        )
        return AudioSeparationResult(
            accompaniment_path=output,
            checksum_sha256=_sha(output),
            model_id="separation-test",
            model_tree_sha256=MODEL_SHA,
            backend_name="fake-tiger",
            metrics=AudioSeparationMetrics(
                elapsed_ms=10,
                duration_us=self.duration_us,
                real_time_factor=0.005,
                source_bytes=source.stat().st_size,
                output_bytes=output.stat().st_size,
                sample_rate=48_000,
                channels=2,
                sample_width_bytes=2,
                frame_count=frame_count,
                backend={"offline": True},
            ),
        )


class FakeSynthesizer:
    backend = "fake-tts"

    def __init__(
        self, *, fail_text: str | None = None, duration_us: int = 800_000
    ) -> None:
        self.fail_text = fail_text
        self.duration_us = duration_us
        self.calls: list[str] = []
        self.closed = False

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        speed: float = 1.0,
        **_kwargs: Any,
    ) -> SynthesizedNarration:
        self.calls.append(text)
        if self.fail_text == text:
            raise NarrationError(
                "tts_failed",
                "TTS fixture thất bại",
                retryable=True,
            )
        frame_count = round(self.duration_us * 24_000 / 1_000_000 / speed)
        _write_wav(
            output_path,
            sample_rate=24_000,
            frame_count=frame_count,
            sample_value=2_000,
        )
        return SynthesizedNarration(
            path=output_path,
            text=text,
            sample_rate=24_000,
            channels=1,
            sample_width_bytes=2,
            frame_count=frame_count,
            duration_us=round(frame_count * 1_000_000 / 24_000),
            native_speed=speed,
            backend=self.backend,
        )

    async def close(self) -> None:
        self.closed = True


class RewriteAwareSynthesizer(FakeSynthesizer):
    def __init__(
        self,
        durations_by_text: dict[str, int],
        *,
        default_us: int,
        fail_text: str | None = None,
    ) -> None:
        super().__init__(duration_us=default_us, fail_text=fail_text)
        self._durations_by_text = dict(durations_by_text)
        self._default_us = default_us

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesizedNarration:
        self.duration_us = self._durations_by_text.get(text, self._default_us)
        return await super().synthesize(
            text,
            output_path,
            speed=speed,
            **kwargs,
        )


class NormalizingRewriteSynthesizer(RewriteAwareSynthesizer):
    async def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> SynthesizedNarration:
        result = await super().synthesize(
            text,
            output_path,
            speed=speed,
            **kwargs,
        )
        return SynthesizedNarration(
            path=result.path,
            text=f"đã chuẩn hóa: {result.text}",
            sample_rate=result.sample_rate,
            channels=result.channels,
            sample_width_bytes=result.sample_width_bytes,
            frame_count=result.frame_count,
            duration_us=result.duration_us,
            native_speed=result.native_speed,
            backend=result.backend,
        )


class FakeTimingRewriter:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[str, int, str]] = []
        self.rewrite_contexts: list[dict[str, Any]] = []
        self.start_count = 0
        self.close_count = 0
        self.abort_count = 0

    def start(self) -> None:
        self.start_count += 1

    def rewrite_for_duration(
        self,
        source_text: str,
        prior_target_text: str,
        observed_duration_us: int,
        target_duration_us: int,
        max_output_words: int,
        *,
        source_language: str,
        target_language: str = "vi",
        canonical_vi: str | None = None,
        adaptive_attempt: int = 1,
        semantic_recovery: bool = False,
        context_before_vi: str | None = None,
        context_after_vi: str | None = None,
        rejected_candidate_hashes: tuple[str, ...] = (),
        **_kwargs: Any,
    ) -> str:
        self.calls.append((source_text, target_duration_us, source_language))
        context = {
                "source_text": source_text,
                "canonical_vi": canonical_vi,
                "prior_target_text": prior_target_text,
                "observed_duration_us": observed_duration_us,
                "target_duration_us": target_duration_us,
                "max_output_words": max_output_words,
                "source_language": source_language,
                "target_language": target_language,
                "adaptive_attempt": adaptive_attempt,
            }
        if semantic_recovery:
            context.update(
                {
                    "semantic_recovery": True,
                    "context_before_vi": context_before_vi,
                    "context_after_vi": context_after_vi,
                    "rejected_candidate_hashes": rejected_candidate_hashes,
                }
            )
        self.rewrite_contexts.append(context)
        return self._outputs.pop(0)

    def close(self) -> None:
        self.close_count += 1

    def abort(self) -> None:
        self.abort_count += 1


class BlockingTimingRewriter(FakeTimingRewriter):
    def __init__(self, *, fail_close: bool = False) -> None:
        super().__init__(["must not complete"])
        self.request_started = threading.Event()
        self.request_aborted = threading.Event()
        self.fail_close = fail_close

    def rewrite_for_duration(
        self,
        source_text: str,
        prior_target_text: str,
        observed_duration_us: int,
        target_duration_us: int,
        max_output_words: int,
        *,
        source_language: str,
        target_language: str = "vi",
        canonical_vi: str | None = None,
        adaptive_attempt: int = 1,
        **_kwargs: Any,
    ) -> str:
        self.calls.append((source_text, target_duration_us, source_language))
        self.rewrite_contexts.append(
            {
                "source_text": source_text,
                "canonical_vi": canonical_vi,
                "prior_target_text": prior_target_text,
                "observed_duration_us": observed_duration_us,
                "target_duration_us": target_duration_us,
                "max_output_words": max_output_words,
                "source_language": source_language,
                "target_language": target_language,
                "adaptive_attempt": adaptive_attempt,
            }
        )
        self.request_started.set()
        if not self.request_aborted.wait(5.0):
            raise AssertionError("timing rewrite request was not aborted")
        raise OSError("llama-server was aborted")

    def abort(self) -> None:
        super().abort()
        self.request_aborted.set()

    def close(self) -> None:
        super().close()
        if self.fail_close:
            raise OSError("cleanup failed after abort")


class FailingTimingRewriter(FakeTimingRewriter):
    def rewrite_for_duration(
        self,
        source_text: str,
        prior_target_text: str,
        observed_duration_us: int,
        target_duration_us: int,
        max_output_words: int,
        *,
        source_language: str,
        target_language: str = "vi",
        canonical_vi: str | None = None,
        adaptive_attempt: int = 1,
        **_kwargs: Any,
    ) -> str:
        self.calls.append((source_text, target_duration_us, source_language))
        self.rewrite_contexts.append(
            {
                "source_text": source_text,
                "canonical_vi": canonical_vi,
                "prior_target_text": prior_target_text,
                "observed_duration_us": observed_duration_us,
                "target_duration_us": target_duration_us,
                "max_output_words": max_output_words,
                "source_language": source_language,
                "target_language": target_language,
                "adaptive_attempt": adaptive_attempt,
            }
        )
        raise LlamaTranslationError(
            "request_failed",
            "llama-server tạm thời không phản hồi",
            retryable=True,
        )


class CrashingTimingRewriter(FakeTimingRewriter):
    def rewrite_for_duration(self, *args: Any, **kwargs: Any) -> str:
        super().rewrite_for_duration(*args, **kwargs)
        raise RuntimeError("simulated process death after durable ledger")


class InvalidThenFailTimingRewriter(FakeTimingRewriter):
    def rewrite_for_duration(self, *args: Any, **kwargs: Any) -> str:
        try:
            super().rewrite_for_duration(*args, **kwargs)
        except IndexError:
            pass
        if len(self.calls) == 1:
            raise LlamaTranslationError(
                "invalid_output",
                "Ứng viên đầu không hợp lệ",
                retryable=True,
            )
        raise LlamaTranslationError(
            "request_failed",
            "llama-server tạm thời không phản hồi",
            retryable=True,
        )


class InvalidOutputTimingRewriter(FakeTimingRewriter):
    def rewrite_for_duration(
        self,
        source_text: str,
        prior_target_text: str,
        observed_duration_us: int,
        target_duration_us: int,
        max_output_words: int,
        *,
        source_language: str,
        target_language: str = "vi",
        canonical_vi: str | None = None,
        adaptive_attempt: int = 1,
        **_kwargs: Any,
    ) -> str:
        self.calls.append((source_text, target_duration_us, source_language))
        self.rewrite_contexts.append(
            {
                "source_text": source_text,
                "canonical_vi": canonical_vi,
                "prior_target_text": prior_target_text,
                "observed_duration_us": observed_duration_us,
                "target_duration_us": target_duration_us,
                "max_output_words": max_output_words,
                "source_language": source_language,
                "target_language": target_language,
                "adaptive_attempt": adaptive_attempt,
            }
        )
        raise LlamaTranslationError(
            "invalid_output",
            "Model trả thêm phần giải thích",
            retryable=True,
        )


class PaddedSilenceSynthesizer(FakeSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.speeds: list[float] = []

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        speed: float = 1.0,
        **_kwargs: Any,
    ) -> SynthesizedNarration:
        self.calls.append(text)
        self.speeds.append(speed)
        sample_rate = 24_000
        leading_frames = round(2_400 / speed)
        speech_frames = round(19_200 / speed)
        trailing_frames = round(2_400 / speed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(b"\0\0" * leading_frames)
            stream.writeframes((2_000).to_bytes(2, "little", signed=True) * speech_frames)
            stream.writeframes(b"\0\0" * trailing_frames)
        frame_count = leading_frames + speech_frames + trailing_frames
        return SynthesizedNarration(
            path=output_path,
            text=text,
            sample_rate=sample_rate,
            channels=1,
            sample_width_bytes=2,
            frame_count=frame_count,
            duration_us=round(frame_count * 1_000_000 / sample_rate),
            native_speed=speed,
            backend=self.backend,
        )


class FakeTimingFitter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.slots: list[tuple[int, int]] = []
        self.maximum_speeds: list[float | None] = []

    async def fit(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_us: int,
        end_us: int,
        text: str,
        native_speed: float,
        **kwargs: Any,
    ) -> FittedNarrationBlock:
        self.calls.append(text)
        self.slots.append((start_us, end_us))
        self.maximum_speeds.append(kwargs.get("maximum_total_speed"))
        frames = round(end_us * 48_000 / 1_000_000) - round(
            start_us * 48_000 / 1_000_000
        )
        _write_wav(output_path, sample_rate=48_000, frame_count=frames)
        return FittedNarrationBlock(
            path=output_path,
            start_us=start_us,
            end_us=end_us,
            text=text,
            source_duration_us=888_875,
            target_frame_count=frames,
            output_frame_count=frames,
            native_speed=native_speed,
            atempo_speed=0.9,
            total_speed=native_speed * 0.9,
            padded_frame_count=0,
            quality=TimingQuality.NORMAL,
        )


class FakeExporter:
    def __init__(self, *, duration_us: int | None = None) -> None:
        self.calls = 0
        self.duration_us = duration_us

    async def export(
        self,
        _source: Path,
        _accompaniment: Path,
        _narration: Path,
        output: Path,
        *,
        expected_duration_us: int,
        **kwargs: Any,
    ) -> ExportedMedia:
        self.calls += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"verified one-video-one-aac fixture")
        callback = kwargs.get("on_progress")
        if callback is not None:
            from dub_server.audio_mix_export import ExportProgress

            callback(ExportProgress(expected_duration_us, expected_duration_us, 1.0))
        return ExportedMedia(
            path=output,
            duration_us=self.duration_us or expected_duration_us,
            video_start_us=0,
            audio_start_us=0,
            audio_codec="aac",
            size_bytes=output.stat().st_size,
        )


def _model_resolver(
    _lock: Path,
    models_dir: Path,
    model_id: str,
    stage: str,
) -> VerifiedModel:
    path = models_dir / model_id
    path.mkdir(parents=True, exist_ok=True)
    return VerifiedModel(
        entry={"id": model_id, "stage": stage, "backend": "fake"},
        path=path,
        tree_sha256=MODEL_SHA,
    )


def _stage(
    tmp_path: Path,
    store: StateStore,
    *,
    separator: FakeSeparator,
    synthesizer: FakeSynthesizer,
    fitter: FakeTimingFitter,
    exporter: FakeExporter,
    rewriter: FakeTimingRewriter | None = None,
    rewrite_max_attempts: int = 3,
) -> Phase4Stage:
    return Phase4Stage(
        models_lock_path=tmp_path / "models.lock.json",
        models_dir=tmp_path / "models",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
        default_separation_model_id="separation-test",
        default_tts_model_id="tts-test",
        default_translation_model_id="mt-test",
        tts_support_model_id=None,
        store=store,
        separator_factory=lambda _model: separator,  # type: ignore[arg-type]
        synthesizer_factory=lambda _model, _support: synthesizer,
        timing_rewriter_factory=(
            (lambda _model: rewriter) if rewriter is not None else None
        ),
        timing_rewrite_max_attempts=rewrite_max_attempts,
        timing_fitter_factory=lambda: fitter,  # type: ignore[arg-type]
        exporter_factory=lambda: exporter,  # type: ignore[arg-type]
        model_resolver=_model_resolver,
    )


async def _seed_exhausted_single_block_rewrite(
    tmp_path: Path,
    store: StateStore,
    job_id: str,
    *,
    rewritten_text: str = "Hai cũ",
) -> None:
    initial = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Một": 1_500_000, "Hai": 3_200_000},
            default_us=3_200_000,
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)
    assert initial.error_code == "timing_rewrite_required"
    assert initial.details["timing_failure"]["failure_kind"] in {
        "single_window_capacity",
        "elastic_postvalidation",
    }
    payload = dict(store.get_checkpoint(job_id, JobStage.TTS).payload)
    _replace_tts_block_text(payload, ordinal=1, text=rewritten_text)
    payload["timing_rewrites"] = [
        _exhausted_v2_rewrite(
            ordinal=1,
            source_text="Two",
            original_text="Hai",
            rewritten_text=rewritten_text,
            observed_duration_us=3_200_000,
        )
    ]
    store.save_checkpoint(job_id, JobStage.TTS, payload)
    store.update_status(
        job_id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        force=True,
    )


def test_phase4_checkpoint_identity_includes_timing_profile() -> None:
    assert Phase4Stage._checkpoint_profile_matches({}, TimingProfile.STRICT)
    assert not Phase4Stage._checkpoint_profile_matches({}, TimingProfile.NATURAL)
    assert Phase4Stage._checkpoint_profile_matches(
        {"timing_profile": "natural"}, TimingProfile.NATURAL
    )
    assert not Phase4Stage._checkpoint_profile_matches(
        {"timing_profile": "strict"}, TimingProfile.NATURAL
    )
    assert Phase4Stage._checkpoint_trim_matches({}, TimingProfile.STRICT)
    assert not Phase4Stage._checkpoint_trim_matches({}, TimingProfile.NATURAL)
    assert Phase4Stage._checkpoint_trim_matches(
        {"silence_trim_version": TTS_SILENCE_TRIM_VERSION}, TimingProfile.NATURAL
    )
    assert not Phase4Stage._checkpoint_trim_matches(
        {"silence_trim_version": "obsolete-trim-v0"}, TimingProfile.STRICT
    )


@pytest.mark.asyncio
async def test_phase4_completes_with_exact_artifacts_and_track_contract(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    separator = FakeSeparator()
    synthesizer = FakeSynthesizer()
    fitter = FakeTimingFitter()
    exporter = FakeExporter()

    result = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=exporter,
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED, (
        result.error_code,
        result.error_message,
        result.details.get("timing_failure"),
    )
    assert result.progress_permille == 1000
    assert result.result is not None
    assert result.result["video_track_count"] == 1
    assert result.result["audio_track_count"] == 1
    assert result.result["audio_codec"] == "aac"
    assert result.result["original_dialogue_removed"] is True
    assert result.result["music_and_effects_preserved"] is True
    assert Path(result.result["video_path"]).is_file()
    assert Path(result.result["srt_path"]).read_text(encoding="utf-8").startswith(
        "1\n00:00:00,000 --> 00:00:01,000\nMột"
    )
    assert store.get_checkpoint(job_id, JobStage.SEPARATION).payload["completed"]
    tts_checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    timing_checkpoint = store.get_checkpoint(job_id, JobStage.TIMING).payload
    assert tts_checkpoint["completed"]
    assert timing_checkpoint["completed"]
    # A job created before natural timing remains strict and migrates its
    # checkpoint identity without changing timestamps.
    assert tts_checkpoint["timing_profile"] == "strict"
    assert timing_checkpoint["timing_profile"] == "strict"
    assert store.get_checkpoint(job_id, JobStage.EXPORT).payload["audio_codec"] == "aac"
    assert separator.calls == 1
    assert synthesizer.closed is True
    assert fitter.calls == ["Một", "Hai"]
    assert fitter.slots == [(0, 1_000_000), (1_000_000, 2_000_000)]
    assert fitter.maximum_speeds == [None, None]
    assert exporter.calls == 1


@pytest.mark.asyncio
async def test_phase4_natural_profile_keeps_native_tts_and_adjusts_slots(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = FakeSynthesizer()
    fitter = FakeTimingFitter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=FakeExporter(),
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    assert synthesizer.calls == ["Một", "Hai"]
    assert fitter.slots == [(100_000, 900_000), (1_100_000, 1_900_000)]
    assert fitter.maximum_speeds == [1.2, 1.2]
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_profile"
    ] == "natural"
    assert store.get_checkpoint(job_id, JobStage.TIMING).payload[
        "timing_profile"
    ] == "natural"
    assert result.result is not None
    assert result.result["timing_profile"] == "natural"
    assert Path(result.result["srt_path"]).read_text(encoding="utf-8").startswith(
        "1\n00:00:00,100 --> 00:00:00,900\nMột"
    )


@pytest.mark.asyncio
async def test_phase4_strict_profile_computes_native_speed_after_silence_trim(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="strict")
    synthesizer = PaddedSilenceSynthesizer()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    assert synthesizer.speeds == [1.0, 0.9, 1.0, 0.9]
    tts_payload = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert tts_payload["silence_trim_version"] == TTS_SILENCE_TRIM_VERSION
    assert [block["native_speed"] for block in tts_payload["blocks"]] == [0.9, 0.9]


@pytest.mark.asyncio
async def test_phase4_natural_profile_plans_from_trimmed_tts_duration(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = PaddedSilenceSynthesizer()
    fitter = FakeTimingFitter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=FakeExporter(),
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    tts_payload = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert tts_payload["silence_trim_version"] == TTS_SILENCE_TRIM_VERSION
    assert [block["frame_count"] for block in tts_payload["blocks"]] == [
        20_640,
        20_640,
    ]
    assert [block["duration_us"] for block in tts_payload["blocks"]] == [
        860_000,
        860_000,
    ]
    assert fitter.slots == [(70_000, 930_000), (1_070_000, 1_930_000)]
    timing_payload = store.get_checkpoint(job_id, JobStage.TIMING).payload
    assert timing_payload["silence_trim_version"] == TTS_SILENCE_TRIM_VERSION


@pytest.mark.asyncio
async def test_phase4_natural_profile_fails_with_actionable_rewrite_details(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    fitter = FakeTimingFitter()
    exporter = FakeExporter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(duration_us=3_200_000),
        fitter=fitter,
        exporter=exporter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_rewrite_required"
    assert result.retryable is False
    assert result.error_message is not None and "rút gọn" in result.error_message
    failure = result.details["timing_failure"]
    assert failure["profile"] == "natural"
    assert failure["ordinal"] == failure["failure_ordinal"] == 0
    assert failure["required_duration_us"] == 2_666_667
    assert failure["available_duration_us"] == 1_800_000
    assert failure["maximum_total_speed"] == 1.2
    assert failure["failure_kind"] == "single_window_capacity"
    assert failure["critical_group_start_ordinal"] == 0
    assert failure["critical_group_end_ordinal"] == 0
    assert failure["schedule_deficit_us"] == 866_667
    assert fitter.calls == []
    assert exporter.calls == 0


@pytest.mark.asyncio
async def test_phase4_elastic_silent_slack_avoids_rewrite_and_tts_rerun(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_elastic_gap_job(tmp_path)
    synthesizer = FakeSynthesizer(duration_us=3_200_000)
    rewriter = FakeTimingRewriter(["không được gọi"])
    fitter = FakeTimingFitter()
    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(duration_us=3_000_000),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=FakeExporter(),
        rewriter=rewriter,
    )

    result = await stage.run(job_id)

    assert result.status is JobStatus.COMPLETED, (
        result.error_code,
        result.error_message,
        result.details.get("timing_failure"),
    )
    assert synthesizer.calls == ["Một"]
    assert rewriter.calls == []
    assert fitter.calls == ["Một"]
    timing = store.get_checkpoint(job_id, JobStage.TIMING).payload
    assert timing["completed"] is True
    assert timing["planner_policy"] == "natural-silent-slack-v2"
    job = store.get_job(job_id)
    artifact = load_translation_artifact(
        Path(str(job.details["translated_transcript_path"])),
        expected_sha256=str(job.details["translated_transcript_sha256"]),
    )
    tts_model = _model_resolver(
        tmp_path / "models.lock.json",
        tmp_path / "models",
        "tts-test",
        "tts",
    )
    assert stage._valid_timing_checkpoint(
        timing,
        translation=artifact,
        model=tts_model,
        block_count=1,
        timing_profile=TimingProfile.NATURAL,
        planner_policy="natural-silent-slack-v2",
    ) is not None
    assert stage._valid_timing_checkpoint(
        timing,
        translation=artifact,
        model=tts_model,
        block_count=1,
        timing_profile=TimingProfile.NATURAL,
        planner_policy="natural-base-v1",
    ) is None
    warnings = [
        item
        for item in result.details["warnings"]
        if item["code"] == "timing_silent_slack_used"
    ]
    assert len(warnings) == 1
    stage._warn_timing_silent_slack_used(job_id)
    assert sum(
        item["code"] == "timing_silent_slack_used"
        for item in store.get_job(job_id).details["warnings"]
    ) == 1


@pytest.mark.asyncio
async def test_phase4_exhausted_failure_owner_rewrites_only_group_predecessor(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    first_synthesizer = RewriteAwareSynthesizer(
        {"Một": 1_600_000, "Hai": 1_600_000},
        default_us=1_600_000,
    )
    initial = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=first_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert initial.status is JobStatus.FAILED
    assert initial.error_code == "timing_rewrite_required"
    payload = dict(store.get_checkpoint(job_id, JobStage.TTS).payload)
    _replace_tts_block_text(payload, ordinal=1, text="Hai cũ")
    payload["timing_rewrites"] = [
        _exhausted_v2_rewrite(
            ordinal=1,
            source_text="Two",
            original_text="Hai",
            rewritten_text="Hai cũ",
            observed_duration_us=1_600_000,
        )
    ]
    store.save_checkpoint(job_id, JobStage.TTS, payload)
    store.update_status(
        job_id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        force=True,
    )

    rewriter = FakeTimingRewriter(["Một gọn"])
    resumed_synthesizer = RewriteAwareSynthesizer(
        {"Một gọn": 600_000},
        default_us=1_600_000,
    )
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED, (
        resumed.error_code,
        resumed.error_message,
        resumed.details.get("timing_failure"),
    )
    assert resumed_synthesizer.calls == ["Một gọn"]
    assert len(rewriter.rewrite_contexts) == 1
    assert rewriter.rewrite_contexts[0]["source_text"] == "One"
    assert rewriter.rewrite_contexts[0]["prior_target_text"] == "Một"
    rewrites = {
        item["ordinal"]: item
        for item in store.get_checkpoint(job_id, JobStage.TTS).payload[
            "timing_rewrites"
        ]
    }
    recovered = rewrites[0]
    assert recovered["strategy"] == "critical-group-neighbor-v1"
    assert recovered["failure_ordinal"] == 1
    assert recovered["critical_group_start_ordinal"] == 0
    assert recovered["critical_group_end_ordinal"] == 1
    assert recovered["schedule_deficit_us"] == 666_668


@pytest.mark.asyncio
async def test_phase4_invalid_owner_rewrites_try_next_group_candidate(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    rewriter = FakeTimingRewriter(
        ["quá nhiều từ", "vẫn quá dài", "chưa đủ ngắn", "Một gọn"]
    )
    synthesizer = RewriteAwareSynthesizer(
        {"Một": 1_600_000, "Hai": 1_600_000, "Một gọn": 600_000},
        default_us=1_600_000,
    )

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    assert [item["source_text"] for item in rewriter.rewrite_contexts] == [
        "Two",
        "Two",
        "Two",
        "One",
    ]
    assert synthesizer.calls == ["Một", "Hai", "Một gọn"]
    rewrites = store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ]
    assert len(rewrites) == 1
    assert rewrites[0]["ordinal"] == 0
    assert rewrites[0]["strategy"] == "critical-group-neighbor-v1"


@pytest.mark.asyncio
async def test_phase4_all_candidate_local_semantic_failures_are_bounded(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    rewriter = FakeTimingRewriter(
        [
            "quá nhiều từ",
            "vẫn quá dài",
            "chưa đủ ngắn",
            "cũng quá nhiều",
            "không thể dùng",
            "vẫn chưa đạt",
        ]
    )
    synthesizer = RewriteAwareSynthesizer(
        {"Một": 1_600_000, "Hai": 1_600_000},
        default_us=1_600_000,
    )

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_group_budget_impossible"
    assert len(rewriter.calls) == 6
    assert synthesizer.calls == ["Một", "Hai"]


@pytest.mark.asyncio
async def test_phase4_all_group_candidates_exhausted_fails_without_loop(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    initial = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Một": 1_600_000, "Hai": 1_600_000},
            default_us=1_600_000,
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert initial.error_code == "timing_rewrite_required"
    payload = dict(store.get_checkpoint(job_id, JobStage.TTS).payload)
    _replace_tts_block_text(payload, ordinal=0, text="Một cũ")
    _replace_tts_block_text(payload, ordinal=1, text="Hai cũ")
    payload["timing_rewrites"] = [
        _exhausted_v2_rewrite(
            ordinal=0,
            source_text="One",
            original_text="Một",
            rewritten_text="Một cũ",
            observed_duration_us=1_600_000,
        ),
        _exhausted_v2_rewrite(
            ordinal=1,
            source_text="Two",
            original_text="Hai",
            rewritten_text="Hai cũ",
            observed_duration_us=1_600_000,
        ),
    ]
    store.save_checkpoint(job_id, JobStage.TTS, payload)
    store.update_status(
        job_id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        force=True,
    )
    rewriter = FakeTimingRewriter(["không được gọi"])
    resumed_synthesizer = RewriteAwareSynthesizer({}, default_us=1_600_000)

    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.FAILED
    assert resumed.error_code == "timing_group_budget_impossible"
    assert resumed.retryable is False
    assert resumed_synthesizer.calls == []
    assert rewriter.calls == []


@pytest.mark.asyncio
async def test_phase4_intrinsic_owner_overflow_never_rewrites_predecessor(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    rewriter = FakeTimingRewriter(["Ý1", "Ý2", "Ý3"])
    resumed_synthesizer = RewriteAwareSynthesizer(
        {},
        default_us=3_200_000,
    )

    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.FAILED
    assert resumed.error_code == "timing_single_block_budget_impossible"
    assert resumed_synthesizer.calls == ["Ý1", "Ý2", "Ý3"]
    assert {item[0] for item in rewriter.calls} == {"Two"}
    assert all(
        item["context_before_vi"] == "Một"
        and item["context_after_vi"] is None
        for item in rewriter.rewrite_contexts
    )


@pytest.mark.asyncio
async def test_phase4_single_block_semantic_recovery_measures_tts_and_accepts_once(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    rewriter = FakeTimingRewriter(["Gọn"])
    synthesizer = RewriteAwareSynthesizer({"Gọn": 800_000}, default_us=3_200_000)
    fitter = FakeTimingFitter()
    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=FakeExporter(),
        rewriter=rewriter,
    )

    result = await stage.run(job_id)

    assert result.status is JobStatus.COMPLETED, (
        result.error_code,
        result.error_message,
        result.details.get("timing_failure"),
    )
    assert synthesizer.calls == ["Gọn"]
    context = rewriter.rewrite_contexts[0]
    assert context["semantic_recovery"] is True
    assert context["adaptive_attempt"] == 1
    assert context["max_output_words"] == 1
    assert context["context_before_vi"] == "Một"
    assert context["context_after_vi"] is None
    assert hashlib.sha256("Hai".encode()).hexdigest() in set(
        context["rejected_candidate_hashes"]
    )
    assert fitter.maximum_speeds and max(
        value for value in fitter.maximum_speeds if value is not None
    ) <= 1.20
    checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    rewrite = checkpoint["timing_rewrites"][0]
    assert rewrite["prompt_version"] == "timing-rewrite-v3"
    assert rewrite["semantic_attempt"] == 1
    assert rewrite["semantic_candidate_pending"] is False
    assert rewrite["observed_duration_us"] == 800_000
    assert rewrite["accepted"] is True
    warnings = [
        item
        for item in result.details.get("warnings", [])
        if item["code"] == "timing_translation_rewritten_semantic"
    ]
    assert len(warnings) == 1

    artifact = load_translation_artifact(
        Path(str(result.details["translated_transcript_path"])),
        expected_sha256=str(result.details["translated_transcript_sha256"]),
    )
    stage._accept_timing_rewrites(job_id, translation=artifact)
    warnings_after = [
        item
        for item in store.get_job(job_id).details.get("warnings", [])
        if item["code"] == "timing_translation_rewritten_semantic"
    ]
    assert warnings_after == warnings


@pytest.mark.asyncio
async def test_phase4_v3_candidate_checkpoint_resumes_without_llm_recall(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    failed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {}, default_us=3_200_000, fail_text="Gọn"
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=FakeTimingRewriter(["Gọn"]),
    ).run(job_id)
    assert failed.error_code == "tts_failed"
    saved = store.get_checkpoint(job_id, JobStage.TTS).payload["timing_rewrites"][0]
    assert saved["prompt_version"] == "timing-rewrite-v3"
    assert saved["semantic_candidate_pending"] is False
    assert saved.get("observed_duration_us") is None

    store.resume(job_id)
    resumed_rewriter = FakeTimingRewriter(["không được gọi"])
    resumed_synthesizer = RewriteAwareSynthesizer(
        {"Gọn": 800_000}, default_us=3_200_000
    )
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=resumed_rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED, (
        resumed.error_code,
        resumed.error_message,
        resumed.details.get("timing_failure"),
    )
    assert resumed_rewriter.calls == []
    assert resumed_synthesizer.calls == ["Gọn"]


@pytest.mark.asyncio
async def test_phase4_semantic_crash_ledger_skips_consumed_strategy_on_resume(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    failed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer({}, default_us=3_200_000),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=CrashingTimingRewriter(["Ứng viên lỗi"]),
    ).run(job_id)
    assert failed.error_code == "phase4_failed"
    ledger = store.get_checkpoint(job_id, JobStage.TTS).payload["timing_rewrites"][0]
    assert ledger["semantic_attempt"] == 1
    assert ledger["semantic_candidate_pending"] is True

    store.resume(job_id)
    resumed_rewriter = FakeTimingRewriter(["Gọn"])
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Gọn": 800_000}, default_us=3_200_000
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=resumed_rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED, (
        resumed.error_code,
        resumed.error_message,
        resumed.details.get("timing_failure"),
    )
    assert [
        item["adaptive_attempt"] for item in resumed_rewriter.rewrite_contexts
    ] == [2]


@pytest.mark.asyncio
async def test_phase4_semantic_transient_failure_does_not_consume_strategy(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    failed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer({}, default_us=3_200_000),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=FailingTimingRewriter([]),
    ).run(job_id)
    assert failed.error_code == "timing_rewrite_translation_failed"
    restored = store.get_checkpoint(job_id, JobStage.TTS).payload["timing_rewrites"][0]
    assert restored["prompt_version"] == "timing-rewrite-v2"

    store.resume(job_id)
    resumed_rewriter = FakeTimingRewriter(["Gọn"])
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Gọn": 800_000}, default_us=3_200_000
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=resumed_rewriter,
    ).run(job_id)
    assert resumed.status is JobStatus.COMPLETED, (
        resumed.error_code,
        resumed.error_message,
        resumed.details.get("timing_failure"),
    )
    assert [
        item["adaptive_attempt"] for item in resumed_rewriter.rewrite_contexts
    ] == [1]


@pytest.mark.asyncio
async def test_phase4_semantic_transient_restores_latest_rejected_attempt(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    await _seed_exhausted_single_block_rewrite(tmp_path, store, job_id)
    failed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer({}, default_us=3_200_000),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=InvalidThenFailTimingRewriter(["x", "y"]),
    ).run(job_id)
    assert failed.error_code == "timing_rewrite_translation_failed"
    ledger = store.get_checkpoint(job_id, JobStage.TTS).payload["timing_rewrites"][0]
    assert ledger["prompt_version"] == "timing-rewrite-v3"
    assert ledger["semantic_attempt"] == 1
    assert ledger["semantic_candidate_pending"] is True

    store.resume(job_id)
    resumed_rewriter = FakeTimingRewriter(["Gọn"])
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Gọn": 800_000}, default_us=3_200_000
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=resumed_rewriter,
    ).run(job_id)
    assert resumed.status is JobStatus.COMPLETED
    assert [
        item["adaptive_attempt"] for item in resumed_rewriter.rewrite_contexts
    ] == [2]


def test_single_block_semantic_recovery_accepts_elastic_postvalidation_only() -> None:
    details = {
        "failure_kind": "elastic_postvalidation",
        "critical_group_start_ordinal": 4,
        "critical_group_end_ordinal": 4,
    }
    error = TimingError(
        "timing_rewrite_required", "fixture", retryable=False, details=details
    )
    candidate = ({"ordinal": 4},)
    assert Phase4Stage._valid_single_block_semantic_recovery(
        error, owner=4, candidates=candidate
    )

    details["critical_group_start_ordinal"] = 3
    assert not Phase4Stage._valid_single_block_semantic_recovery(
        TimingError(
            "timing_rewrite_required", "fixture", retryable=False, details=details
        ),
        owner=4,
        candidates=candidate,
    )


@pytest.mark.parametrize("invalid_attempt", (True, 1.0))
def test_phase4_rejects_non_integer_semantic_attempt_checkpoint(
    tmp_path: Path,
    invalid_attempt: object,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    artifact_path = tmp_path / "jobs" / job_id / "translated-transcript.json"
    translation = load_translation_artifact(artifact_path)
    text = "Gọn"
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    original_sha = hashlib.sha256("Hai".encode()).hexdigest()
    payload = {
        "timing_rewrites": [
            {
                "ordinal": 1,
                "text": text,
                "text_sha256": text_sha,
                "source_text_sha256": hashlib.sha256("Two".encode()).hexdigest(),
                "original_translation_sha256": original_sha,
                "attempt": invalid_attempt,
                "semantic_attempt": invalid_attempt,
                "semantic_recovery": True,
                "semantic_candidate_pending": False,
                "available_duration_us": 1_000_000,
                "target_duration_us": 800_000,
                "max_words": 1,
                "previous_text_sha256": hashlib.sha256("Hai cũ".encode()).hexdigest(),
                "previous_target_duration_us": 900_000,
                "previous_observed_duration_us": 1_100_000,
                "observed_duration_us": 800_000,
                "model_id": "mt-test",
                "model_tree_sha256": MODEL_SHA,
                "prompt_version": "timing-rewrite-v3",
                "accepted": False,
                "history": [],
                "rejected_candidate_hashes": [original_sha],
                "context_before_vi": "Một",
                "context_after_vi": None,
                "strategy": "single-block-semantic-v1",
                "failure_ordinal": 1,
                "critical_group_start_ordinal": 1,
                "critical_group_end_ordinal": 1,
                "schedule_deficit_us": 100_000,
            }
        ]
    }

    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    )
    assert not stage._valid_timing_rewrites(
        payload,
        translation=translation,
    )


def test_phase4_pending_semantic_ledger_is_never_accepted_or_warned(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    translation = load_translation_artifact(
        tmp_path / "jobs" / job_id / "translated-transcript.json"
    )
    original_sha = hashlib.sha256("Hai".encode()).hexdigest()
    text = "Hai cũ"
    payload = {
        "schema_version": 4,
        "timing_rewrites": [
            {
                **_exhausted_v2_rewrite(
                    ordinal=1,
                    source_text="Two",
                    original_text="Hai",
                    rewritten_text=text,
                    observed_duration_us=3_200_000,
                ),
                "prompt_version": "timing-rewrite-v3",
                "semantic_attempt": 1,
                "semantic_recovery": True,
                "semantic_candidate_pending": True,
                "attempt": 1,
                "max_words": 1,
                "history": [],
                "rejected_candidate_hashes": [original_sha],
                "context_before_vi": "Một",
                "context_after_vi": None,
                "strategy": "single-block-semantic-v1",
                "failure_ordinal": 1,
                "critical_group_start_ordinal": 1,
                "critical_group_end_ordinal": 1,
                "schedule_deficit_us": 100_000,
            }
        ],
    }
    store.save_checkpoint(job_id, JobStage.TTS, payload)
    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    )
    assert stage._valid_timing_rewrites(payload, translation=translation)

    stage._accept_timing_rewrites(job_id, translation=translation)

    saved = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert saved["timing_rewrites"][0]["accepted"] is False
    assert not any(
        item.get("code") == "timing_translation_rewritten_semantic"
        for item in store.get_job(job_id).details.get("warnings", [])
    )


@pytest.mark.asyncio
async def test_phase4_natural_profile_rewrites_only_overflowing_block(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    artifact_path = tmp_path / "jobs" / job_id / "translated-transcript.json"
    original_artifact = artifact_path.read_bytes()
    synthesizer = RewriteAwareSynthesizer(
        {
            "Một": 3_200_000,
            "Hai": 800_000,
            "Một gọn": 1_200_000,
        },
        default_us=3_200_000,
    )
    rewriter = FakeTimingRewriter(["Một gọn"])
    fitter = FakeTimingFitter()
    exporter = FakeExporter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=exporter,
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    assert rewriter.calls == [("One", 1_867_680, "en")]
    assert rewriter.rewrite_contexts == [
        {
            "source_text": "One",
            "canonical_vi": "Một",
            "prior_target_text": "Một",
            "observed_duration_us": 3_200_000,
            "target_duration_us": 1_867_680,
            "max_output_words": 2,
            "source_language": "en",
            "target_language": "vi",
            "adaptive_attempt": 1,
        }
    ]
    assert rewriter.start_count == rewriter.close_count == 1
    assert synthesizer.calls == ["Một", "Hai", "Một gọn"]
    assert fitter.calls == ["Một gọn", "Hai"]
    assert all(value == 1.2 for value in fitter.maximum_speeds)
    assert exporter.calls == 1
    assert artifact_path.read_bytes() == original_artifact
    tts_payload = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert tts_payload["timing_rewrites"] == [
        {
            "ordinal": 0,
            "text": "Một gọn",
            "text_sha256": hashlib.sha256("Một gọn".encode()).hexdigest(),
            "source_text_sha256": hashlib.sha256(b"One").hexdigest(),
            "original_translation_sha256": hashlib.sha256("Một".encode()).hexdigest(),
            "attempt": 1,
            "adaptive_attempt": 1,
            "legacy_attempt_count": 0,
            "available_duration_us": 1_800_000,
            "target_duration_us": 1_867_680,
            "max_words": 2,
            "previous_text_sha256": hashlib.sha256("Một".encode()).hexdigest(),
            "previous_target_duration_us": 3_200_000,
            "previous_observed_duration_us": 3_200_000,
            "model_id": "mt-test",
            "model_tree_sha256": MODEL_SHA,
            "prompt_version": "timing-rewrite-v2",
            "accepted": True,
            "history": [
                {
                    "text_sha256": hashlib.sha256("Một".encode()).hexdigest(),
                    "target_duration_us": 3_200_000,
                    "observed_duration_us": 3_200_000,
                }
            ],
            "observed_duration_us": 1_200_000,
            "strategy": "failure-owner-v1",
            "failure_ordinal": 0,
            "critical_group_start_ordinal": 0,
            "critical_group_end_ordinal": 0,
            "schedule_deficit_us": 866_667,
        }
    ]
    assert Path(result.result["srt_path"]).read_text(encoding="utf-8").find(
        "Một gọn"
    ) >= 0
    assert any(
        warning["code"] == "timing_translation_rewritten_adaptive"
        for warning in result.details["warnings"]
    )


@pytest.mark.asyncio
async def test_phase4_keeps_rewritten_script_when_tts_normalizes_spoken_text(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = NormalizingRewriteSynthesizer(
        {
            "Một": 3_200_000,
            "Hai": 800_000,
            "Một gọn": 1_200_000,
        },
        default_us=3_200_000,
    )

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=FakeTimingRewriter(["Một gọn"]),
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    first_block = checkpoint["blocks"][0]
    assert first_block["text"] == "Một gọn"
    assert first_block["tts_normalized_text"] == "đã chuẩn hóa: Một gọn"
    assert first_block["text_sha256"] == hashlib.sha256(
        "Một gọn".encode()
    ).hexdigest()
    srt = Path(result.result["srt_path"]).read_text(encoding="utf-8")
    assert "Một gọn" in srt
    assert "đã chuẩn hóa:" not in srt


@pytest.mark.asyncio
async def test_phase4_timing_rewrite_stops_after_three_measured_attempts(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = RewriteAwareSynthesizer(
        {
            "Hai": 800_000,
            "Gọn một": 2_800_000,
            "Gọn hai": 2_500_000,
            "Gọn ba": 2_200_000,
        },
        default_us=3_200_000,
    )
    rewriter = FakeTimingRewriter(
        ["Gọn một", "Gọn hai", "Gọn ba", "Ý1", "Ý2", "Ý3"]
    )
    exporter = FakeExporter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=exporter,
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_single_block_budget_impossible"
    assert result.retryable is False
    assert len(rewriter.calls) == 6
    assert rewriter.start_count == rewriter.close_count == 6
    assert [item["prior_target_text"] for item in rewriter.rewrite_contexts[:3]] == [
        "Một",
        "Gọn một",
        "Gọn hai",
    ]
    assert [
        item["observed_duration_us"] for item in rewriter.rewrite_contexts[:3]
    ] == [3_200_000, 2_800_000, 2_500_000]
    assert [item["adaptive_attempt"] for item in rewriter.rewrite_contexts[:3]] == [
        1,
        2,
        3,
    ]
    assert [item["max_output_words"] for item in rewriter.rewrite_contexts[:3]] == [
        2,
        2,
        2,
    ]
    rewrite_targets = [
        item["target_duration_us"] for item in rewriter.rewrite_contexts[:3]
    ]
    assert rewrite_targets == sorted(rewrite_targets, reverse=True)
    assert synthesizer.calls == [
        "Một",
        "Hai",
        "Gọn một",
        "Gọn hai",
        "Gọn ba",
        "Ý1",
        "Ý2",
        "Ý3",
    ]
    assert exporter.calls == 0
    checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert checkpoint["timing_rewrites"][0]["attempt"] == 3
    assert checkpoint["timing_rewrites"][0]["prompt_version"] == "timing-rewrite-v3"
    assert checkpoint["timing_rewrites"][0]["observed_duration_us"] == 3_200_000
    assert checkpoint["timing_rewrites"][0]["accepted"] is False
    assert not any(
        warning["code"] == "timing_translation_rewritten_adaptive"
        for warning in result.details.get("warnings", [])
    )


@pytest.mark.asyncio
async def test_phase4_empty_rewrite_output_is_bounded_without_tts_fallback(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = RewriteAwareSynthesizer(
        {"Hai": 800_000},
        default_us=3_200_000,
    )
    rewriter = FakeTimingRewriter(["", "   ", "\n"])

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_group_budget_impossible"
    assert result.retryable is False
    assert len(rewriter.calls) == 3
    assert rewriter.start_count == rewriter.close_count == 3
    assert synthesizer.calls == ["Một", "Hai"]
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ] == []


@pytest.mark.asyncio
async def test_phase4_retries_llama_invalid_output_three_times(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = RewriteAwareSynthesizer(
        {"Hai": 800_000},
        default_us=3_200_000,
    )
    rewriter = InvalidOutputTimingRewriter([])

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_group_budget_impossible"
    assert result.retryable is False
    assert [item["adaptive_attempt"] for item in rewriter.rewrite_contexts] == [
        1,
        2,
        3,
    ]
    assert [item["observed_duration_us"] for item in rewriter.rewrite_contexts] == [
        3_200_000,
        3_200_000,
        3_200_000,
    ]
    targets = [item["target_duration_us"] for item in rewriter.rewrite_contexts]
    assert targets == sorted(targets, reverse=True)
    assert rewriter.start_count == rewriter.close_count == 3
    assert synthesizer.calls == ["Một", "Hai"]
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ] == []


@pytest.mark.asyncio
async def test_phase4_keeps_transient_rewriter_failure_retryable(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = RewriteAwareSynthesizer(
        {"Hai": 800_000},
        default_us=3_200_000,
    )
    rewriter = FailingTimingRewriter([])
    exporter = FakeExporter()

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=exporter,
        rewriter=rewriter,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "timing_rewrite_translation_failed"
    assert result.retryable is True
    assert len(rewriter.calls) == 1
    assert rewriter.start_count == rewriter.close_count == 1
    assert synthesizer.calls == ["Một", "Hai"]
    assert exporter.calls == 0
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ] == []


@pytest.mark.asyncio
async def test_phase4_resume_uses_persisted_rewrite_without_calling_model_again(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    separator = FakeSeparator()
    rewriter = FakeTimingRewriter(["Một gọn"])
    first_synthesizer = RewriteAwareSynthesizer(
        {"Một": 3_200_000, "Hai": 800_000, "Một gọn": 1_200_000},
        default_us=3_200_000,
        fail_text="Một gọn",
    )

    first = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=first_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert first.status is JobStatus.FAILED
    assert first.error_code == "tts_failed"
    assert first.retryable is True
    checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert checkpoint["timing_rewrites"][0]["text"] == "Một gọn"
    assert checkpoint["completed"] is False
    assert len(rewriter.calls) == 1

    store.resume(job_id)
    resumed_synthesizer = RewriteAwareSynthesizer(
        {"Một gọn": 1_200_000, "Hai": 800_000},
        default_us=3_200_000,
    )
    resumed = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert separator.calls == 1
    assert resumed_synthesizer.calls == ["Một gọn"]
    assert len(rewriter.calls) == 1
    warnings = [
        item
        for item in resumed.details["warnings"]
        if item["code"] == "timing_translation_rewritten_adaptive"
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_phase4_accept_warning_survives_checkpoint_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    rewriter = FakeTimingRewriter(["Một gọn"])
    original_save_checkpoint = store.save_checkpoint
    failed_once = False

    def fail_first_accepted_checkpoint(
        current_job_id: str,
        stage: JobStage,
        payload: dict[str, Any],
    ) -> None:
        nonlocal failed_once
        rewrites = payload.get("timing_rewrites", [])
        if (
            not failed_once
            and stage is JobStage.TTS
            and isinstance(rewrites, list)
            and rewrites
            and isinstance(rewrites[0], dict)
            and rewrites[0].get("accepted") is True
        ):
            failed_once = True
            raise OSError("simulated checkpoint failure after warning")
        original_save_checkpoint(current_job_id, stage, payload)

    monkeypatch.setattr(store, "save_checkpoint", fail_first_accepted_checkpoint)
    first = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Một": 3_200_000, "Hai": 800_000, "Một gọn": 1_200_000},
            default_us=3_200_000,
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert first.status is JobStatus.FAILED
    assert first.error_code == "phase4_failed"
    assert failed_once is True
    assert sum(
        item["code"] == "timing_translation_rewritten_adaptive"
        for item in first.details.get("warnings", [])
    ) == 1

    monkeypatch.setattr(store, "save_checkpoint", original_save_checkpoint)
    store.resume(job_id)
    resumed_synthesizer = RewriteAwareSynthesizer({}, default_us=3_200_000)
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert resumed_synthesizer.calls == []
    assert len(rewriter.calls) == 1
    assert sum(
        item["code"] == "timing_translation_rewritten_adaptive"
        for item in resumed.details.get("warnings", [])
    ) == 1
    rewrite = store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ][0]
    assert rewrite["accepted"] is True


@pytest.mark.asyncio
async def test_phase4_resumes_legacy_v1_attempt_three_with_measured_v2_rewrite(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    initial = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Một": 3_200_000, "Hai": 800_000},
            default_us=3_200_000,
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert initial.status is JobStatus.FAILED
    assert initial.error_code == "timing_rewrite_required"
    tts_payload = dict(store.get_checkpoint(job_id, JobStage.TTS).payload)
    blocks = [dict(item) for item in tts_payload["blocks"]]
    blocks[0]["text"] = "Một cũ"
    blocks[0]["tts_normalized_text"] = "Một cũ"
    blocks[0]["text_sha256"] = hashlib.sha256("Một cũ".encode()).hexdigest()
    tts_payload["blocks"] = blocks
    tts_payload["timing_rewrites"] = [
        {
            "ordinal": 0,
            "text": "Một cũ",
            "text_sha256": hashlib.sha256("Một cũ".encode()).hexdigest(),
            "source_text_sha256": hashlib.sha256(b"One").hexdigest(),
            "original_translation_sha256": hashlib.sha256(
                "Một".encode()
            ).hexdigest(),
            "attempt": 3,
            "available_duration_us": 1_800_000,
            "target_duration_us": 1_116_000,
            "model_id": "mt-test",
            "model_tree_sha256": MODEL_SHA,
            "prompt_version": "timing-rewrite-v1",
            # Deliberately stale: the current block checkpoint below measures
            # 3.2 seconds and must take precedence for adaptive feedback.
            "observed_duration_us": 4_600_000,
        }
    ]
    store.save_checkpoint(job_id, JobStage.TTS, tts_payload)
    store.update_status(
        job_id,
        JobStatus.READY_TTS,
        stage=JobStage.TTS,
        progress_permille=650,
        force=True,
    )

    rewriter = FakeTimingRewriter(["Một gọn"])
    resumed_synthesizer = RewriteAwareSynthesizer(
        {"Một gọn": 1_200_000, "Hai": 800_000},
        default_us=3_200_000,
    )
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=resumed_synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert resumed_synthesizer.calls == ["Một gọn"]
    assert len(rewriter.rewrite_contexts) == 1
    context = rewriter.rewrite_contexts[0]
    assert context["prior_target_text"] == "Một cũ"
    assert context["observed_duration_us"] == 3_200_000
    assert context["adaptive_attempt"] == 1
    assert context["target_duration_us"] < 1_116_000
    rewrite = store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ][0]
    assert rewrite["prompt_version"] == "timing-rewrite-v2"
    assert rewrite["legacy_attempt_count"] == 3
    assert rewrite["adaptive_attempt"] == 1
    assert rewrite["previous_observed_duration_us"] == 3_200_000
    assert rewrite["accepted"] is True
    warnings = [
        item
        for item in resumed.details["warnings"]
        if item["code"] == "timing_translation_rewritten_adaptive"
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_phase4_migrates_legacy_strict_checkpoints_without_rebuilding(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)
    for stage in (JobStage.TTS, JobStage.TIMING):
        checkpoint = store.get_checkpoint(job_id, stage)
        legacy_payload = dict(checkpoint.payload)
        legacy_payload.pop("timing_profile")
        legacy_payload.pop("silence_trim_version")
        store.save_checkpoint(job_id, stage, legacy_payload)
    store.update_status(
        job_id,
        JobStatus.VERIFYING,
        stage=JobStage.VERIFY,
        progress_permille=985,
        force=True,
    )
    synthesizer = FakeSynthesizer()
    fitter = FakeTimingFitter()
    exporter = FakeExporter()

    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=fitter,
        exporter=exporter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert synthesizer.calls == []
    assert fitter.calls == []
    assert exporter.calls == 0
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_profile"
    ] == "strict"
    assert store.get_checkpoint(job_id, JobStage.TIMING).payload[
        "timing_profile"
    ] == "strict"
    assert "silence_trim_version" not in store.get_checkpoint(
        job_id, JobStage.TTS
    ).payload


@pytest.mark.asyncio
async def test_phase4_warns_when_output_follows_shorter_video_timeline(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)

    result = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(duration_us=1_600_000),
    ).run(job_id)

    assert result.status is JobStatus.COMPLETED
    warnings = result.details["warnings"]
    assert any(
        item["code"] == "output_duration_adjusted_to_video" for item in warnings
    )
    assert result.result is not None
    assert result.result["duration_us"] == 1_600_000


@pytest.mark.asyncio
async def test_phase4_deduplicates_separation_duration_warning_on_resume(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    separator = FakeSeparator(duration_us=2_200_000)
    first = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=FakeSynthesizer(fail_text="Hai"),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert first.status is JobStatus.FAILED
    assert first.retryable is True
    assert sum(
        item["code"] == "audio_separation_duration_adjusted"
        for item in first.details["warnings"]
    ) == 1

    store.resume(job_id)
    resumed = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert separator.calls == 1
    assert sum(
        item["code"] == "audio_separation_duration_adjusted"
        for item in resumed.details["warnings"]
    ) == 1


@pytest.mark.asyncio
async def test_phase4_resume_reuses_separation_and_completed_tts_blocks(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    separator = FakeSeparator()
    first_synth = FakeSynthesizer(fail_text="Hai")
    first = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=first_synth,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert first.status is JobStatus.FAILED
    assert first.retryable is True
    assert store.get_checkpoint(job_id, JobStage.TTS).payload["blocks"][0][
        "ordinal"
    ] == 0
    store.resume(job_id)

    resumed_synth = FakeSynthesizer()
    exporter = FakeExporter()
    resumed = await _stage(
        tmp_path,
        store,
        separator=separator,
        synthesizer=resumed_synth,
        fitter=FakeTimingFitter(),
        exporter=exporter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert separator.calls == 1
    assert "Một" not in resumed_synth.calls
    assert resumed_synth.calls and set(resumed_synth.calls) == {"Hai"}
    assert exporter.calls == 1


@pytest.mark.asyncio
async def test_phase4_resume_reuses_hash_sealed_completed_export(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    first_exporter = FakeExporter()
    first = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=first_exporter,
    ).run(job_id)
    assert first.status is JobStatus.COMPLETED
    assert first_exporter.calls == 1

    # Simulate a process dying after the EXPORT checkpoint was committed but
    # before (or while) the final COMPLETED state was durably observed.
    store.update_status(
        job_id,
        JobStatus.VERIFYING,
        stage=JobStage.VERIFY,
        progress_permille=985,
        force=True,
    )
    resumed_exporter = FakeExporter()
    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=resumed_exporter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert resumed_exporter.calls == 0


@pytest.mark.asyncio
async def test_phase4_resume_reexports_when_sealed_mp4_was_modified(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)
    await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)
    output = tmp_path / "output" / f"{job_id}.mp4"
    output.write_bytes(b"tampered")
    store.update_status(
        job_id,
        JobStatus.VERIFYING,
        stage=JobStage.VERIFY,
        progress_permille=985,
        force=True,
    )
    exporter = FakeExporter()

    resumed = await _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=FakeSynthesizer(),
        fitter=FakeTimingFitter(),
        exporter=exporter,
    ).run(job_id)

    assert resumed.status is JobStatus.COMPLETED
    assert exporter.calls == 1


@pytest.mark.asyncio
async def test_phase4_cancel_aborts_blocked_timing_rewrite_without_http_timeout(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    synthesizer = RewriteAwareSynthesizer(
        {"Hai": 800_000},
        default_us=3_200_000,
    )
    # A cleanup error after the successful abort must not replace the durable
    # cancellation signal or turn the job into a retryable Phase 4 failure.
    rewriter = BlockingTimingRewriter(fail_close=True)
    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    )
    run_task = asyncio.create_task(stage.run(job_id))
    assert await asyncio.to_thread(rewriter.request_started.wait, 5.0)

    started = asyncio.get_running_loop().time()
    store.request_cancel(job_id)
    result = await asyncio.wait_for(run_task, timeout=1.0)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.5
    assert result.status is JobStatus.CANCELLING
    assert rewriter.abort_count >= 1
    assert rewriter.close_count == 1
    assert store.get_checkpoint(job_id, JobStage.TTS).payload[
        "timing_rewrites"
    ] == []


@pytest.mark.asyncio
async def test_phase4_task_cancellation_is_not_masked_by_rewriter_cleanup_error(
    tmp_path: Path,
) -> None:
    store, job_id, _source = _ready_job(tmp_path, timing_profile="natural")
    rewriter = BlockingTimingRewriter(fail_close=True)
    stage = _stage(
        tmp_path,
        store,
        separator=FakeSeparator(),
        synthesizer=RewriteAwareSynthesizer(
            {"Hai": 800_000},
            default_us=3_200_000,
        ),
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
        rewriter=rewriter,
    )
    run_task = asyncio.create_task(stage.run(job_id))
    assert await asyncio.to_thread(rewriter.request_started.wait, 5.0)

    started = asyncio.get_running_loop().time()
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=1.0)

    assert asyncio.get_running_loop().time() - started < 0.5
    assert rewriter.abort_count >= 1
    assert rewriter.close_count == 1


@pytest.mark.asyncio
async def test_phase4_cancellation_stops_before_tts(tmp_path) -> None:
    store, job_id, _source = _ready_job(tmp_path)

    class CancellingSeparator(FakeSeparator):
        async def separate(self, source: Path, output: Path, **kwargs: Any):
            store.request_cancel(job_id)
            assert kwargs["cancellation"]() is True
            raise AudioSeparationCancelled()

    synthesizer = FakeSynthesizer()
    result = await _stage(
        tmp_path,
        store,
        separator=CancellingSeparator(),
        synthesizer=synthesizer,
        fitter=FakeTimingFitter(),
        exporter=FakeExporter(),
    ).run(job_id)

    assert result.status is JobStatus.CANCELLING
    assert synthesizer.calls == []
    assert not (tmp_path / "output" / f"{job_id}.mp4").exists()
