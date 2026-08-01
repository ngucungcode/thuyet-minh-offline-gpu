from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from dub_server.domain import TranscriptSegment, TranscriptionResult
from dub_server.translation_artifact import (
    TRANSLATION_ARTIFACT_SCHEMA_VERSION,
    TranslationArtifactError,
    TranslationArtifactIntegrityError,
    TranslationResult,
    TranslationSegment,
    build_translation_result,
    load_translation_artifact,
    read_translation_artifact,
    translation_file_sha256,
    write_translation_artifact,
)


SOURCE_DIGEST = "a" * 64


def _source_transcript(*, language: str = "en") -> TranscriptionResult:
    return TranscriptionResult(
        source="asr",
        language=language,
        language_probability=0.95,
        duration_us=5_000_000,
        model_id="asr-model",
        segments=(
            TranscriptSegment(100_000, 1_500_000, "Hello world"),
            TranscriptSegment(2_000_000, 4_500_000, "How are you?"),
        ),
    )


def _translation_result() -> TranslationResult:
    return build_translation_result(
        _source_transcript(),
        ["Xin ch\u00e0o th\u1ebf gi\u1edbi", "B\u1ea1n kh\u1ecfe kh\u00f4ng?"],
        target_language="VI_vn",
        model_id=" m2m100-418m-int8 ",
        model_revision=" 0123456789abcdef ",
        source_transcript_sha256=SOURCE_DIGEST.upper(),
    )


def test_build_result_preserves_source_text_and_integer_timeline() -> None:
    result = _translation_result()

    assert result.source_language == "en"
    assert result.target_language == "vi-vn"
    assert result.model_id == "m2m100-418m-int8"
    assert result.model_revision == "0123456789abcdef"
    assert result.source_transcript_sha256 == SOURCE_DIGEST
    assert result.segments == (
        TranslationSegment(100_000, 1_500_000, "Hello world", "Xin ch\u00e0o th\u1ebf gi\u1edbi"),
        TranslationSegment(2_000_000, 4_500_000, "How are you?", "B\u1ea1n kh\u1ecfe kh\u00f4ng?"),
    )


def test_build_rejects_missing_or_extra_translations() -> None:
    for translations in ([], ["M\u1ed9t"], ["M\u1ed9t", "Hai", "Ba"]):
        with pytest.raises(TranslationArtifactError, match="kh.ng kh.p"):
            build_translation_result(
                _source_transcript(),
                translations,
                target_language="vi",
                model_id="model",
                source_transcript_sha256=SOURCE_DIGEST,
            )
    with pytest.raises(TranslationArtifactError, match="Danh s.ch"):
        build_translation_result(
            _source_transcript(),
            "not-an-iterable-of-blocks",
            target_language="vi",
            model_id="model",
            source_transcript_sha256=SOURCE_DIGEST,
        )


def test_vietnamese_bypass_can_preserve_the_same_text() -> None:
    source = _source_transcript(language="vi")
    result = build_translation_result(
        source,
        [segment.text for segment in source.segments],
        target_language="vi",
        model_id="translation-bypass",
        source_transcript_sha256=SOURCE_DIGEST,
    )
    assert all(
        segment.source_text == segment.translated_text for segment in result.segments
    )


def test_segments_normalize_unicode_and_reject_empty_or_control_text() -> None:
    segment = TranslationSegment(
        0,
        1,
        "  Tie\u0302\u0301ng   Vie\u0323\u0302t ",
        "  Ti\u1ebfng   Vi\u1ec7t  ",
    )
    assert segment.source_text == "Ti\u1ebfng Vi\u1ec7t"
    assert segment.translated_text == "Ti\u1ebfng Vi\u1ec7t"
    with pytest.raises(TranslationArtifactError, match="must not be empty"):
        TranslationSegment(0, 1, "source", " \n\t ")
    with pytest.raises(TranslationArtifactError, match="control"):
        TranslationSegment(0, 1, "source\x00", "target")


def test_round_trip_publishes_canonical_versioned_json_with_checksums(
    tmp_path: Path,
) -> None:
    path = tmp_path / "job" / "translation.json"
    result = _translation_result()
    published = write_translation_artifact(path, result)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["artifact_type"] == "translation"
    assert document["schema_version"] == TRANSLATION_ARTIFACT_SCHEMA_VERSION
    assert document["source"] == {
        "language": "en",
        "transcript_sha256": SOURCE_DIGEST,
    }
    assert document["target"] == {"language": "vi-vn"}
    assert document["model"]["id"] == "m2m100-418m-int8"
    assert document["segments"][0] == {
        "ordinal": 0,
        "start_us": 100_000,
        "end_us": 1_500_000,
        "source_text": "Hello world",
        "translated_text": "Xin ch\u00e0o th\u1ebf gi\u1edbi",
    }
    assert published.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert published.sha256 == translation_file_sha256(path)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))

    loaded = load_translation_artifact(
        path,
        expected_sha256=published.sha256.upper(),
        expected_source_transcript_sha256=SOURCE_DIGEST.upper(),
    )
    assert loaded.result == result
    assert read_translation_artifact(
        path, expected_sha256=published.sha256
    ) == result


def test_same_result_has_deterministic_bytes_and_digest(tmp_path: Path) -> None:
    first = write_translation_artifact(tmp_path / "first.json", _translation_result())
    second = write_translation_artifact(tmp_path / "second.json", _translation_result())
    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()


def test_tampered_artifact_is_rejected_before_json_is_trusted(tmp_path: Path) -> None:
    path = tmp_path / "translation.json"
    artifact = write_translation_artifact(path, _translation_result())
    path.write_bytes(path.read_bytes().replace(b"Hello world", b"Hello earth"))

    with pytest.raises(TranslationArtifactIntegrityError, match="kh.ng kh.p"):
        load_translation_artifact(path, expected_sha256=artifact.sha256)
    with pytest.raises(TranslationArtifactIntegrityError, match="kh.ng h.p l."):
        load_translation_artifact(path, expected_sha256="invalid")


def test_source_transcript_digest_is_verified_independently(tmp_path: Path) -> None:
    path = tmp_path / "translation.json"
    artifact = write_translation_artifact(path, _translation_result())

    with pytest.raises(TranslationArtifactIntegrityError, match="kh.ng thu.c"):
        load_translation_artifact(
            path,
            expected_sha256=artifact.sha256,
            expected_source_transcript_sha256="b" * 64,
        )


def test_schema_ordinal_and_timeline_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "translation.json"
    write_translation_artifact(path, _translation_result())
    document = json.loads(path.read_text(encoding="utf-8"))

    document["schema_version"] = 2
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TranslationArtifactError, match="Phi.n b.n"):
        load_translation_artifact(path)

    document["schema_version"] = 1
    document["segments"][1]["ordinal"] = 3
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TranslationArtifactError, match="kh.ng li.n t.c"):
        load_translation_artifact(path)

    document["segments"][1]["ordinal"] = 1
    document["segments"][1]["start_us"] = 1_000_000
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TranslationArtifactError, match="overlap"):
        load_translation_artifact(path)


def test_oversized_artifact_is_rejected_before_json_decode(tmp_path: Path) -> None:
    path = tmp_path / "translation.json"
    path.write_bytes(b"{}" * 20)
    with pytest.raises(TranslationArtifactError, match="k.ch th..c"):
        load_translation_artifact(path, max_bytes=8)
    with pytest.raises(TranslationArtifactError, match="k.ch th..c"):
        translation_file_sha256(path, max_bytes=8)


def test_atomic_replace_failure_preserves_old_file_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dub_server.translation_artifact as module

    path = tmp_path / "translation.json"
    path.write_bytes(b"old-content")

    def reject_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated failure")

    monkeypatch.setattr(module.os, "replace", reject_replace)
    with pytest.raises(TranslationArtifactError, match="Kh.ng th. ghi"):
        write_translation_artifact(path, _translation_result())
    assert path.read_bytes() == b"old-content"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_artifact_helpers_are_fully_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "translation.json"
    artifact = write_translation_artifact(path, _translation_result())

    def reject_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", reject_socket)
    loaded = load_translation_artifact(
        path,
        expected_sha256=artifact.sha256,
        expected_source_transcript_sha256=SOURCE_DIGEST,
    )
    assert loaded.result == _translation_result()
