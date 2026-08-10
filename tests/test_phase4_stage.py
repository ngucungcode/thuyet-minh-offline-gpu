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
from dub_server.timing import FittedNarrationBlock, TimingProfile, TimingQuality
from dub_server.translation_artifact import (
    TranslationResult,
    TranslationSegment,
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
        self.start_count = 0
        self.close_count = 0
        self.abort_count = 0

    def start(self) -> None:
        self.start_count += 1

    def translate_batch_for_durations(
        self,
        texts: list[str],
        target_durations_us: list[int],
        *,
        source_language: str,
        target_language: str = "vi",
        **_kwargs: Any,
    ) -> tuple[str, ...]:
        self.calls.append((texts[0], target_durations_us[0], source_language))
        return (self._outputs.pop(0),)

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

    def translate_batch_for_durations(
        self,
        texts: list[str],
        target_durations_us: list[int],
        *,
        source_language: str,
        target_language: str = "vi",
        **_kwargs: Any,
    ) -> tuple[str, ...]:
        self.calls.append((texts[0], target_durations_us[0], source_language))
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
    def translate_batch_for_durations(
        self,
        texts: list[str],
        target_durations_us: list[int],
        *,
        source_language: str,
        target_language: str = "vi",
        **_kwargs: Any,
    ) -> tuple[str, ...]:
        self.calls.append((texts[0], target_durations_us[0], source_language))
        raise LlamaTranslationError(
            "request_failed",
            "llama-server tạm thời không phản hồi",
            retryable=True,
        )


class InvalidOutputTimingRewriter(FakeTimingRewriter):
    def translate_batch_for_durations(
        self,
        texts: list[str],
        target_durations_us: list[int],
        *,
        source_language: str,
        target_language: str = "vi",
        **_kwargs: Any,
    ) -> tuple[str, ...]:
        self.calls.append((texts[0], target_durations_us[0], source_language))
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

    assert result.status is JobStatus.COMPLETED
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
    assert result.details["timing_failure"] == {
        "profile": "natural",
        "ordinal": 0,
        "required_duration_us": 2_666_667,
        "available_duration_us": 1_800_000,
        "maximum_total_speed": 1.2,
    }
    assert fitter.calls == []
    assert exporter.calls == 0


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
    assert rewriter.calls == [("One", 1_620_000, "en")]
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
            "available_duration_us": 1_800_000,
            "target_duration_us": 1_620_000,
            "model_id": "mt-test",
            "model_tree_sha256": MODEL_SHA,
            "prompt_version": "timing-rewrite-v1",
            "observed_duration_us": 1_200_000,
        }
    ]
    assert Path(result.result["srt_path"]).read_text(encoding="utf-8").find(
        "Một gọn"
    ) >= 0
    assert any(
        warning["code"] == "timing_translation_rewritten"
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
        {"Hai": 800_000},
        default_us=3_200_000,
    )
    rewriter = FakeTimingRewriter(["Một dài 1", "Một dài 2", "Một dài 3"])
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
    assert result.error_code == "timing_rewrite_exhausted"
    assert result.retryable is False
    assert len(rewriter.calls) == 3
    assert rewriter.start_count == rewriter.close_count == 3
    assert synthesizer.calls == [
        "Một",
        "Hai",
        "Một dài 1",
        "Một dài 2",
        "Một dài 3",
    ]
    assert exporter.calls == 0
    checkpoint = store.get_checkpoint(job_id, JobStage.TTS).payload
    assert checkpoint["timing_rewrites"][0]["attempt"] == 3


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
    assert result.error_code == "timing_rewrite_exhausted"
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
    assert result.error_code == "timing_rewrite_exhausted"
    assert result.retryable is False
    assert [target_us for _text, target_us, _language in rewriter.calls] == [
        1_620_000,
        1_350_000,
        1_116_000,
    ]
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
        if item["code"] == "timing_translation_rewritten"
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
    assert await asyncio.to_thread(rewriter.request_started.wait, 2.0)

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
    assert await asyncio.to_thread(rewriter.request_started.wait, 2.0)

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
