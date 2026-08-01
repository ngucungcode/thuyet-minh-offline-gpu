from __future__ import annotations

import hashlib
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
from dub_server.model_registry import VerifiedModel
from dub_server.narration import NarrationError, SynthesizedNarration
from dub_server.phase4_stage import Phase4Stage
from dub_server.state import JobStage, JobStatus, StateStore
from dub_server.timing import FittedNarrationBlock, TimingQuality
from dub_server.translation_artifact import (
    TranslationResult,
    TranslationSegment,
    write_translation_artifact,
)


MODEL_SHA = "a" * 64
SOURCE_SHA = "b" * 64


def _write_wav(
    path: Path,
    *,
    sample_rate: int,
    frame_count: int,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0" * frame_count * channels * 2)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_job(tmp_path: Path) -> tuple[StateStore, str, Path]:
    store = StateStore(tmp_path / "state" / "jobs.sqlite3")
    source = tmp_path / "incoming" / "movie.mkv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"local movie fixture")
    job = store.create_job(
        "release-1",
        {
            "rights_confirmed": True,
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
    def __init__(self) -> None:
        self.calls = 0

    async def separate(self, source: Path, output: Path, **kwargs: Any):
        self.calls += 1
        callback = kwargs.get("on_progress")
        if callback is not None:
            from dub_server.audio_separation import SeparationProgress

            callback(SeparationProgress("separating", 500, "Đang tách"))
        _write_wav(output, sample_rate=48_000, frame_count=96_000, channels=2)
        return AudioSeparationResult(
            accompaniment_path=output,
            checksum_sha256=_sha(output),
            model_id="separation-test",
            model_tree_sha256=MODEL_SHA,
            backend_name="fake-tiger",
            metrics=AudioSeparationMetrics(
                elapsed_ms=10,
                duration_us=2_000_000,
                real_time_factor=0.005,
                source_bytes=source.stat().st_size,
                output_bytes=output.stat().st_size,
                sample_rate=48_000,
                channels=2,
                sample_width_bytes=2,
                frame_count=96_000,
                backend={"offline": True},
            ),
        )


class FakeSynthesizer:
    backend = "fake-tts"

    def __init__(self, *, fail_text: str | None = None) -> None:
        self.fail_text = fail_text
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
        frame_count = round(19_200 / speed)
        _write_wav(output_path, sample_rate=24_000, frame_count=frame_count)
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


class FakeTimingFitter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fit(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_us: int,
        end_us: int,
        text: str,
        native_speed: float,
        **_kwargs: Any,
    ) -> FittedNarrationBlock:
        self.calls.append(text)
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
    def __init__(self) -> None:
        self.calls = 0

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
            duration_us=expected_duration_us,
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
) -> Phase4Stage:
    return Phase4Stage(
        models_lock_path=tmp_path / "models.lock.json",
        models_dir=tmp_path / "models",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
        default_separation_model_id="separation-test",
        default_tts_model_id="tts-test",
        tts_support_model_id=None,
        store=store,
        separator_factory=lambda _model: separator,  # type: ignore[arg-type]
        synthesizer_factory=lambda _model, _support: synthesizer,
        timing_fitter_factory=lambda: fitter,  # type: ignore[arg-type]
        exporter_factory=lambda: exporter,  # type: ignore[arg-type]
        model_resolver=_model_resolver,
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
    assert store.get_checkpoint(job_id, JobStage.TTS).payload["completed"]
    assert store.get_checkpoint(job_id, JobStage.TIMING).payload["completed"]
    assert store.get_checkpoint(job_id, JobStage.EXPORT).payload["audio_codec"] == "aac"
    assert separator.calls == 1
    assert synthesizer.closed is True
    assert fitter.calls == ["Một", "Hai"]
    assert exporter.calls == 1


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
