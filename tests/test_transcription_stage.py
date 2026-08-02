from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dub_server.asr import LanguageDetectionRequired, NoSpeechError, TranscriptionError
from dub_server.audio_decode import AudioDecodeError, DecodedAudio
from dub_server.domain import TranscriptSegment, TranscriptionResult
from dub_server.state import JobStage, JobStatus, StateStore
from dub_server.transcript import write_transcript_artifact
from dub_server.transcription_stage import TranscriptionStage


DURATION_US = 2_000_000
ASR_MODEL_ID = "asr-test-local"


def _ready_job(
    tmp_path: Path,
    *,
    transcript_source: str,
    subtitle_path: Path | None = None,
    source_language: str = "auto",
) -> tuple[StateStore, str, Path]:
    store = StateStore(tmp_path / "state" / "jobs.sqlite3")
    source_media = tmp_path / "incoming" / "movie.mkv"
    source_media.parent.mkdir(parents=True)
    source_media.write_bytes(b"local media fixture")
    created = store.create_job(
        "release-1",
        {
            "source_language": source_language,
            "subtitle_mode": "prefer",
            "models": {"asr": ASR_MODEL_ID},
        },
    )
    downloading = store.update_status(created.id, JobStatus.DOWNLOADING)
    matching = store.update_status(
        downloading.id,
        JobStatus.SUBTITLE_MATCHING,
        stage=JobStage.SUBTITLE,
    )
    selected_subtitle = (
        {
            "subtitle_id": "subtitle-1",
            "format": "srt",
            "language": "en",
        }
        if subtitle_path is not None
        else None
    )
    ready = store.update_status(
        matching.id,
        JobStatus.READY_OFFLINE,
        stage=JobStage.SUBTITLE,
        progress_permille=250,
        details={
            "transcript_source": transcript_source,
            "source_media_path": str(source_media),
            "source_subtitle_path": (
                str(subtitle_path) if subtitle_path is not None else None
            ),
            "selected_subtitle": selected_subtitle,
            "selected_media": {
                "duration_us": DURATION_US,
                "source_language": source_language,
                "audio_stream_index": 2,
            },
        },
    )
    return store, ready.id, source_media


class FakeDecoder:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    async def decode(
        self,
        media_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> DecodedAudio:
        self.calls.append(
            {
                "media_path": media_path,
                "output_path": output_path,
                **kwargs,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF fake mono 16 kHz PCM")
        return DecodedAudio(
            path=output_path,
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=2,
            frame_count=32_000,
            duration_us=DURATION_US,
        )


class FakeRecognizer:
    def __init__(
        self,
        calls: list[dict[str, Any]],
        outcome: TranscriptionResult | Exception,
    ) -> None:
        self.calls = calls
        self.outcome = outcome

    def transcribe(self, media_path: Path, **kwargs: Any) -> TranscriptionResult:
        self.calls.append({"media_path": media_path, **kwargs})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _asr_result(*, model_id: str = ASR_MODEL_ID) -> TranscriptionResult:
    return TranscriptionResult(
        source="asr",
        language="en",
        language_probability=0.96,
        duration_us=DURATION_US,
        model_id=model_id,
        segments=(TranscriptSegment(100_000, 1_900_000, "Hello offline world"),),
    )


def _two_segment_asr_result(*, model_id: str = ASR_MODEL_ID) -> TranscriptionResult:
    return TranscriptionResult(
        source="asr",
        language="en",
        language_probability=0.96,
        duration_us=DURATION_US,
        model_id=model_id,
        segments=(
            TranscriptSegment(100_000, 500_000, "Hello offline"),
            TranscriptSegment(500_000, 1_900_000, "world"),
        ),
    )


def _stage(
    tmp_path: Path,
    store: StateStore,
    *,
    decoder_factory: Any,
    recognizer_factory: Any,
    model_resolver: Any,
) -> TranscriptionStage:
    return TranscriptionStage(
        models_lock_path=tmp_path / "models.lock.json",
        models_dir=tmp_path / "models",
        jobs_dir=tmp_path / "jobs",
        default_asr_model_id="asr-default",
        compute_type="int8_float16",
        store=store,
        audio_decoder_factory=decoder_factory,
        recognizer_factory=recognizer_factory,
        model_resolver=model_resolver,
    )


@pytest.mark.asyncio
async def test_good_subtitle_never_resolves_or_constructs_asr_components(
    tmp_path: Path,
) -> None:
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,100 --> 00:00:01,900\nHello from subtitle\n",
        encoding="utf-8",
    )
    store, job_id, _ = _ready_job(
        tmp_path,
        transcript_source="subtitle",
        subtitle_path=subtitle,
        source_language="en",
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ASR dependency must not be touched for a valid subtitle")

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=forbidden,
        recognizer_factory=forbidden,
        model_resolver=forbidden,
    ).run(job_id)

    assert result.status is JobStatus.READY_TRANSLATION
    assert result.details["transcript_source"] == "subtitle"
    assert (tmp_path / "jobs" / job_id / "source-transcript.json").is_file()
    stored = store.list_transcript_segments(job_id)
    assert [(item.start_us, item.end_us, item.text) for item in stored] == [
        (100_000, 1_900_000, "Hello from subtitle")
    ]


@pytest.mark.asyncio
async def test_asr_fallback_verifies_local_model_decodes_pcm_and_commits(
    tmp_path: Path,
) -> None:
    store, job_id, source_media = _ready_job(
        tmp_path,
        transcript_source="asr",
    )
    model_path = tmp_path / "models" / ASR_MODEL_ID
    model_path.mkdir(parents=True)
    resolver_calls: list[tuple[Path, Path, str, str]] = []
    decoder_calls: list[dict[str, Any]] = []
    recognizer_calls: list[dict[str, Any]] = []

    def resolver(
        lock_path: Path,
        models_dir: Path,
        model_id: str,
        stage: str,
    ) -> Any:
        resolver_calls.append((lock_path, models_dir, model_id, stage))
        return SimpleNamespace(path=model_path)

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder(decoder_calls),
        recognizer_factory=lambda: FakeRecognizer(
            recognizer_calls, _asr_result()
        ),
        model_resolver=resolver,
    ).run(job_id)

    assert result.status is JobStatus.READY_TRANSLATION
    assert resolver_calls == [
        (
            tmp_path / "models.lock.json",
            tmp_path / "models",
            ASR_MODEL_ID,
            "asr",
        )
    ]
    assert decoder_calls[0]["media_path"] == source_media
    assert decoder_calls[0]["audio_stream_index"] == 2
    assert decoder_calls[0]["output_path"] == (
        tmp_path / "jobs" / job_id / "source-audio-16k.wav"
    )
    assert recognizer_calls[0]["model_path"] == model_path
    assert recognizer_calls[0]["model_id"] == ASR_MODEL_ID
    assert recognizer_calls[0]["compute_type"] == "int8_float16"
    assert recognizer_calls[0]["language"] is None
    assert result.details["asr_model_id"] == ASR_MODEL_ID
    assert result.details["asr_step"] == "finalizing"
    assert result.details["asr_processed_us"] == DURATION_US
    assert result.details["asr_duration_us"] == DURATION_US
    assert result.details["asr_segment_count"] == 1
    assert result.details["asr_progress_permille"] == 1000
    assert store.list_transcript_segments(job_id)[0].text == "Hello offline world"


@pytest.mark.asyncio
async def test_asr_progress_is_throttled_mapped_and_finalized_before_commit(
    tmp_path: Path,
) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()

    class ProgressRecognizer:
        def transcribe(self, _media_path: Path, **kwargs: Any) -> TranscriptionResult:
            progress = kwargs["on_progress"]
            progress(500_000, 1)
            progress(500_000, 1)
            progress(500_001, 1)
            progress(1_900_000, 2)
            return _two_segment_asr_result()

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder([]),
        recognizer_factory=ProgressRecognizer,
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.READY_TRANSLATION
    assert result.progress_permille == 450
    assert result.details["asr_step"] == "finalizing"
    assert result.details["asr_processed_us"] == DURATION_US
    assert result.details["asr_duration_us"] == DURATION_US
    assert result.details["asr_segment_count"] == 2
    assert result.details["asr_progress_permille"] == 1000

    asr_status_events = [
        event
        for event in store.list_events(job_id, limit=1000)
        if event.event_type == "job.status"
        and isinstance(event.payload.get("details"), dict)
        and event.payload["details"].get("asr_step") in {
            "preparing",
            "decoding",
            "recognizing",
            "finalizing",
        }
    ]
    recognizing = [
        event.payload
        for event in asr_status_events
        if event.payload["details"]["asr_step"] == "recognizing"
    ]
    assert [item["details"]["asr_progress_permille"] for item in recognizing] == [
        0,
        250,
        950,
    ]
    assert [item["progress_permille"] for item in recognizing] == [275, 319, 440]

    finalizing_index = next(
        index
        for index, event in enumerate(asr_status_events)
        if event.payload["details"].get("asr_step") == "finalizing"
    )
    assert asr_status_events[finalizing_index].payload["progress_permille"] == 449
    committed_index = next(
        index
        for index, event in enumerate(asr_status_events)
        if event.payload.get("status") == JobStatus.READY_TRANSLATION.value
    )
    assert finalizing_index < committed_index


@pytest.mark.asyncio
async def test_low_language_confidence_waits_for_user_selection(
    tmp_path: Path,
) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()
    detected = LanguageDetectionRequired(
        "en",
        0.42,
        (("en", 0.42), ("ja", 0.31)),
    )

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder([]),
        recognizer_factory=lambda: FakeRecognizer([], detected),
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.NEEDS_LANGUAGE
    assert result.details["detected_language"] == "en"
    assert result.details["source_language_probability"] == pytest.approx(0.42)
    assert result.details["language_candidates"] == [
        {"language": "en", "probability": 0.42},
        {"language": "ja", "probability": 0.31},
    ]
    assert store.list_transcript_segments(job_id) == []


@pytest.mark.asyncio
async def test_no_speech_is_persisted_as_non_retryable_failure(tmp_path: Path) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder([]),
        recognizer_factory=lambda: FakeRecognizer([], NoSpeechError()),
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "no_speech"
    assert result.retryable is False
    assert store.list_transcript_segments(job_id) == []


@pytest.mark.asyncio
async def test_typed_decode_failure_keeps_code_and_retryability(tmp_path: Path) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()

    class FailingDecoder:
        async def decode(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AudioDecodeError(
                "audio_decode_timeout",
                "Tách âm thanh vượt quá thời gian cho phép",
                retryable=True,
            )

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=FailingDecoder,
        recognizer_factory=lambda: FakeRecognizer([], _asr_result()),
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "audio_decode_timeout"
    assert result.retryable is True


@pytest.mark.asyncio
async def test_valid_checkpoint_artifact_commits_without_any_inference(
    tmp_path: Path,
) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    artifact = write_transcript_artifact(
        tmp_path / "jobs" / job_id / "source-transcript.json",
        _asr_result(),
    )
    store.save_checkpoint(
        job_id,
        JobStage.ASR,
        {
            "schema_version": 1,
            "artifact_ready": True,
            "source": "asr",
            "duration_us": DURATION_US,
            "model_id": ASR_MODEL_ID,
            "artifact_path": str(artifact.path),
            "artifact_sha256": artifact.sha256,
        },
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Valid artifact resume must not run inference")

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=forbidden,
        recognizer_factory=forbidden,
        model_resolver=forbidden,
    ).run(job_id)

    assert result.status is JobStatus.READY_TRANSLATION
    assert result.details["asr_step"] == "finalizing"
    assert result.details["asr_progress_permille"] == 1000
    assert store.list_transcript_segments(job_id)[0].text == "Hello offline world"


@pytest.mark.asyncio
async def test_cancellation_during_recognition_never_writes_or_commits(
    tmp_path: Path,
) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()

    class CancellingRecognizer:
        def transcribe(self, _media_path: Path, **kwargs: Any) -> TranscriptionResult:
            store.request_cancel(job_id)
            kwargs["on_progress"](1_000_000, 1)
            raise AssertionError("Cancellation callback should interrupt recognition")

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder([]),
        recognizer_factory=CancellingRecognizer,
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.CANCELLING
    assert result.cancel_requested is True
    assert result.details["asr_step"] == "recognizing"
    assert result.details["asr_processed_us"] == 0
    assert result.details["asr_segment_count"] == 0
    assert result.details["asr_progress_permille"] == 0
    assert store.list_transcript_segments(job_id) == []
    assert not (tmp_path / "jobs" / job_id / "source-transcript.json").exists()


@pytest.mark.asyncio
async def test_typed_asr_failure_is_persisted_without_leaking_exception(
    tmp_path: Path,
) -> None:
    store, job_id, _ = _ready_job(tmp_path, transcript_source="asr")
    model_path = tmp_path / "model"
    model_path.mkdir()
    failure = TranscriptionError(
        "native_asr_oom",
        "GPU không đủ bộ nhớ cho model ASR đã chọn",
        retryable=True,
    )

    result = await _stage(
        tmp_path,
        store,
        decoder_factory=lambda: FakeDecoder([]),
        recognizer_factory=lambda: FakeRecognizer([], failure),
        model_resolver=lambda *_args: SimpleNamespace(path=model_path),
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "native_asr_oom"
    assert result.error_message == "GPU không đủ bộ nhớ cho model ASR đã chọn"
    assert result.retryable is True
