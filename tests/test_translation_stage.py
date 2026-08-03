from __future__ import annotations

import asyncio
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dub_server.domain import TranscriptSegment, TranscriptionResult
from dub_server.model_registry import ModelVerificationError, VerifiedModel
from dub_server.state import JobStage, JobStatus, StateStore
from dub_server.transcript import write_transcript_artifact
from dub_server.translation import TranslationError
from dub_server.translation_artifact import load_translation_artifact
from dub_server.translation_stage import TranslationStage


DURATION_US = 6_000_000
SELECTED_MODEL_ID = "mt-gemma4-31b-q4"
MODEL_TREE_SHA256 = "b" * 64


def _ready_translation_job(
    tmp_path: Path,
    *,
    language: str = "en",
    texts: tuple[str, ...] = ("Hello offline world", "How are you today?"),
    selected_model: str | None = SELECTED_MODEL_ID,
    timing_profile: str | None = None,
) -> tuple[StateStore, str]:
    store = StateStore(tmp_path / "state" / "jobs.sqlite3")
    spec: dict[str, Any] = {
        "source_language": language,
        "models": {"translation": selected_model},
    }
    if timing_profile is not None:
        spec["timing_profile"] = timing_profile
    created = store.create_job(
        "release-1",
        spec,
    )
    downloading = store.update_status(created.id, JobStatus.DOWNLOADING)
    ready = store.update_status(
        downloading.id,
        JobStatus.READY_OFFLINE,
        stage=JobStage.SUBTITLE,
        progress_permille=250,
        details={"transcript_source": "asr"},
    )
    transcribing = store.update_status(
        ready.id,
        JobStatus.TRANSCRIBING,
        stage=JobStage.ASR,
        progress_permille=300,
    )
    segment_duration = 1_500_000
    segments = tuple(
        TranscriptSegment(
            index * 2_000_000,
            index * 2_000_000 + segment_duration,
            text,
        )
        for index, text in enumerate(texts)
    )
    source = TranscriptionResult(
        source="asr",
        language=language,
        language_probability=0.97,
        duration_us=DURATION_US,
        segments=segments,
        model_id="asr-fixture",
    )
    artifact = write_transcript_artifact(
        tmp_path / "jobs" / created.id / "source-transcript.json",
        source,
    )
    committed = store.commit_transcript(
        transcribing.id,
        source,
        artifact_path=artifact.path,
        artifact_sha256=artifact.sha256,
        expected_status=JobStatus.TRANSCRIBING,
    )
    assert committed.status is JobStatus.READY_TRANSLATION
    return store, committed.id


def _verified_model(tmp_path: Path, model_id: str) -> VerifiedModel:
    model_path = tmp_path / "models" / model_id
    model_path.mkdir(parents=True, exist_ok=True)
    return VerifiedModel(
        entry={
            "id": model_id,
            "stage": "mt",
            "revision": "fixture-revision",
            "prompt_template_id": "gemma4-translation-v1",
        },
        path=model_path,
        tree_sha256=MODEL_TREE_SHA256,
    )


class FakeTranslator:
    def __init__(self, outcomes: Sequence[Sequence[str] | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.started = False
        self.closed = False

    def start(self) -> "FakeTranslator":
        self.started = True
        return self

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def translate_batch(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str = "vi",
        on_progress=None,
    ) -> list[str]:
        self.calls.append(
            {
                "texts": list(texts),
                "source_language": source_language,
                "target_language": target_language,
                "on_progress": on_progress,
            }
        )
        if not self._outcomes:
            raise AssertionError("Unexpected translation call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)

    def close(self) -> None:
        self.closed = True


class DurationAwareFakeTranslator(FakeTranslator):
    def __init__(self, outcomes: Sequence[Sequence[str] | BaseException]) -> None:
        super().__init__(outcomes)
        self.duration_calls: list[dict[str, Any]] = []

    def translate_batch_for_durations(
        self,
        texts: Sequence[str],
        target_durations_us: Sequence[int],
        *,
        source_language: str,
        target_language: str = "vi",
        on_progress=None,
    ) -> list[str]:
        self.duration_calls.append(
            {
                "texts": list(texts),
                "target_durations_us": list(target_durations_us),
                "source_language": source_language,
                "target_language": target_language,
                "on_progress": on_progress,
            }
        )
        if not self._outcomes:
            raise AssertionError("Unexpected duration-aware translation call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)


def _stage(
    tmp_path: Path,
    store: StateStore,
    *,
    translator_factory,
    model_resolver,
    default_model_id: str = "mt-default",
) -> TranslationStage:
    return TranslationStage(
        models_lock_path=tmp_path / "models.lock.json",
        models_dir=tmp_path / "models",
        jobs_dir=tmp_path / "jobs",
        default_translation_model_id=default_model_id,
        store=store,
        translator_factory=translator_factory,
        model_resolver=model_resolver,
    )


@pytest.mark.asyncio
async def test_selected_local_model_translates_to_vietnamese_and_commits_artifact(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path)
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    resolver_calls: list[tuple[Path, Path, str, str]] = []
    factory_calls: list[VerifiedModel] = []
    translator = FakeTranslator(
        [["Xin chào thế giới ngoại tuyến"], ["Hôm nay bạn khỏe không?"]]
    )

    def resolver(lock_path: Path, models_dir: Path, model_id: str, stage: str):
        resolver_calls.append((lock_path, models_dir, model_id, stage))
        return verified

    def factory(model: VerifiedModel):
        factory_calls.append(model)
        return translator

    result = await _stage(
        tmp_path,
        store,
        translator_factory=factory,
        model_resolver=resolver,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert result.stage is JobStage.TTS
    assert result.progress_permille == 650
    assert resolver_calls == [
        (
            tmp_path / "models.lock.json",
            tmp_path / "models",
            SELECTED_MODEL_ID,
            "mt",
        )
    ]
    assert factory_calls == [verified]
    assert translator.started is True
    assert translator.closed is True
    assert [call["texts"] for call in translator.calls] == [
        ["Hello offline world"],
        ["How are you today?"],
    ]
    assert all(call["source_language"] == "en" for call in translator.calls)
    assert all(call["target_language"] == "vi" for call in translator.calls)
    rows = store.list_translation_blocks(job_id)
    assert [row.translated_text for row in rows] == [
        "Xin chào thế giới ngoại tuyến",
        "Hôm nay bạn khỏe không?",
    ]
    artifact_path = tmp_path / "jobs" / job_id / "translated-transcript.json"
    artifact = load_translation_artifact(
        artifact_path,
        expected_sha256=result.details["translated_transcript_sha256"],
        expected_source_transcript_sha256=result.details[
            "source_transcript_sha256"
        ],
    )
    assert artifact.result.model_id == SELECTED_MODEL_ID
    assert artifact.result.target_language == "vi"
    checkpoint = store.get_checkpoint(job_id, JobStage.TRANSLATION)
    assert checkpoint is not None
    assert checkpoint.payload["completed"] is True
    assert checkpoint.payload["completed_blocks"] == 2


@pytest.mark.asyncio
async def test_natural_profile_translates_each_block_for_its_spoken_duration(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path, timing_profile="natural")
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    translator = DurationAwareFakeTranslator(
        [["Bản dịch ngắn một"], ["Bản dịch ngắn hai"]]
    )

    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: translator,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert translator.calls == []
    assert [call["texts"] for call in translator.duration_calls] == [
        ["Hello offline world"],
        ["How are you today?"],
    ]
    assert [call["target_durations_us"] for call in translator.duration_calls] == [
        [1_500_000],
        [1_500_000],
    ]
    checkpoint = store.get_checkpoint(job_id, JobStage.TRANSLATION)
    assert checkpoint is not None
    assert checkpoint.payload["prompt_template_id"] == (
        "gemma4-translation-v1+natural-duration-v1"
    )


@pytest.mark.asyncio
async def test_legacy_job_without_profile_keeps_strict_translation_prompt(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path)
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    translator = DurationAwareFakeTranslator([["Một"], ["Hai"]])

    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: translator,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert translator.duration_calls == []
    assert [call["texts"] for call in translator.calls] == [
        ["Hello offline world"],
        ["How are you today?"],
    ]
    checkpoint = store.get_checkpoint(job_id, JobStage.TRANSLATION)
    assert checkpoint is not None
    assert checkpoint.payload["prompt_template_id"] == "gemma4-translation-v1"


@pytest.mark.asyncio
async def test_default_model_is_used_when_job_did_not_select_one(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path, selected_model=None)
    default_id = "mt-default-gemma"
    verified = _verified_model(tmp_path, default_id)
    resolved_ids: list[str] = []
    translator = FakeTranslator([["Một"], ["Hai"]])

    def resolver(_lock: Path, _models: Path, model_id: str, stage: str):
        resolved_ids.append(model_id)
        assert stage == "mt"
        return verified

    result = await _stage(
        tmp_path,
        store,
        default_model_id=default_id,
        translator_factory=lambda _verified: translator,
        model_resolver=resolver,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert resolved_ids == [default_id]
    assert result.details["translation_model_id"] == default_id


@pytest.mark.asyncio
async def test_vietnamese_source_bypasses_resolver_factory_and_model(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(
        tmp_path,
        language="vi-VN",
        texts=("Xin chào mọi người", "Đây là nội dung tiếng Việt"),
    )

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("Vietnamese bypass must not touch an MT model")

    result = await _stage(
        tmp_path,
        store,
        translator_factory=forbidden,
        model_resolver=forbidden,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert result.details["translation_model_id"] == "translation-bypass"
    rows = store.list_translation_blocks(job_id)
    assert [row.translated_text for row in rows] == [
        "Xin chào mọi người",
        "Đây là nội dung tiếng Việt",
    ]
    assert all(row.model_id == "translation-bypass" for row in rows)


@pytest.mark.asyncio
async def test_empty_output_retries_once_with_two_halves_and_never_copies_source(
    tmp_path: Path,
) -> None:
    source = "First sentence. Second sentence."
    store, job_id = _ready_translation_job(tmp_path, texts=(source,))
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    translator = FakeTranslator([[""], ["Câu đầu tiên.", "Câu thứ hai."]])

    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: translator,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert len(translator.calls) == 2
    assert translator.calls[0]["texts"] == [source]
    assert len(translator.calls[1]["texts"]) == 2
    stored = store.list_translation_blocks(job_id)
    assert stored[0].translated_text == "Câu đầu tiên. Câu thứ hai."
    assert stored[0].translated_text != source


@pytest.mark.asyncio
async def test_second_empty_output_is_retryable_failure_without_source_fallback(
    tmp_path: Path,
) -> None:
    source = "First sentence. Second sentence."
    store, job_id = _ready_translation_job(tmp_path, texts=(source,))
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    translator = FakeTranslator([[""], ["Câu đầu tiên.", ""]])

    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: translator,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "translation_output_empty"
    assert result.retryable is True
    assert translator.closed is True
    assert store.list_translation_blocks(job_id)[0].translated_text is None
    assert not (tmp_path / "jobs" / job_id / "translated-transcript.json").exists()


@pytest.mark.asyncio
async def test_process_restart_resumes_only_unfinished_blocks(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path)
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    interrupted = FakeTranslator(
        [["Khối đầu đã xong"], asyncio.CancelledError()]
    )
    first_stage = _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: interrupted,
        model_resolver=lambda *_args: verified,
    )

    with pytest.raises(asyncio.CancelledError):
        await first_stage.run(job_id)

    after_interrupt = store.get_job(job_id)
    assert after_interrupt.status is JobStatus.TRANSLATING
    rows = store.list_translation_blocks(job_id)
    assert [row.translated_text for row in rows] == ["Khối đầu đã xong", None]
    assert interrupted.closed is True

    resumed = FakeTranslator([["Khối thứ hai hoàn tất"]])
    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: resumed,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
    assert [call["texts"] for call in resumed.calls] == [["How are you today?"]]
    assert [row.translated_text for row in store.list_translation_blocks(job_id)] == [
        "Khối đầu đã xong",
        "Khối thứ hai hoàn tất",
    ]


@pytest.mark.asyncio
async def test_resume_rejects_a_changed_natural_translation_prompt(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path, timing_profile="natural")
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    interrupted = DurationAwareFakeTranslator(
        [["Khối đầu đã xong"], asyncio.CancelledError()]
    )

    with pytest.raises(asyncio.CancelledError):
        await _stage(
            tmp_path,
            store,
            translator_factory=lambda _verified: interrupted,
            model_resolver=lambda *_args: verified,
        ).run(job_id)

    checkpoint = store.get_checkpoint(job_id, JobStage.TRANSLATION)
    assert checkpoint is not None
    stale_payload = dict(checkpoint.payload)
    stale_payload["prompt_template_id"] = "gemma4-translation-v1"
    store.save_checkpoint(job_id, JobStage.TRANSLATION, stale_payload)
    resumed = DurationAwareFakeTranslator([["must not run"]])

    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: resumed,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "translation_checkpoint_invalid"
    assert result.retryable is False
    assert resumed.duration_calls == []
    assert resumed.closed is True


@pytest.mark.asyncio
async def test_model_verification_failure_is_typed_and_never_builds_backend(
    tmp_path: Path,
) -> None:
    store, job_id = _ready_translation_job(tmp_path)
    factory_called = False

    def factory(_verified: VerifiedModel):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("Unverified model must never reach factory")

    def resolver(*_args: Any):
        raise ModelVerificationError("fixture mismatch")

    result = await _stage(
        tmp_path,
        store,
        translator_factory=factory,
        model_resolver=resolver,
    ).run(job_id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "translation_model_verification_failed"
    assert result.retryable is True
    assert factory_called is False


@pytest.mark.asyncio
async def test_stage_with_fake_backend_does_not_use_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _ready_translation_job(tmp_path)
    verified = _verified_model(tmp_path, SELECTED_MODEL_ID)
    translator = FakeTranslator([["Ngoại tuyến một"], ["Ngoại tuyến hai"]])

    def reject_socket(*_args: object, **_kwargs: object):
        raise AssertionError("translation stage attempted network access")

    # Do not replace socket.socket itself: asyncio's thread wake-up path uses
    # an internal socketpair. Blocking DNS and outbound connection helpers is
    # sufficient here because the injected backend owns no raw socket.
    monkeypatch.setattr(socket, "create_connection", reject_socket)
    monkeypatch.setattr(socket, "getaddrinfo", reject_socket)
    result = await _stage(
        tmp_path,
        store,
        translator_factory=lambda _verified: translator,
        model_resolver=lambda *_args: verified,
    ).run(job_id)

    assert result.status is JobStatus.READY_TTS
