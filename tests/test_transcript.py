from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from dub_server.domain import SubtitleFormat, TranscriptSegment, TranscriptionResult
from dub_server.transcript import (
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptError,
    TranscriptIntegrityError,
    load_transcript_artifact,
    parse_subtitle_bytes,
    parse_subtitle_file,
    read_transcript_artifact,
    transcript_file_sha256,
    write_transcript_artifact,
)


def test_srt_strips_markup_and_uses_exact_integer_microseconds() -> None:
    payload = (
        "1\n00:00:01,001 --> 00:00:02,234\n<i>Hello</i> &amp; <b>world</b>\n\n"
        "2\n00:02.500 --> 00:03.000 align:start\nSecond line\n"
    ).encode("utf-8-sig")

    result = parse_subtitle_bytes(
        payload,
        subtitle_format=SubtitleFormat.SRT,
        language="EN_us",
        duration_us=4_000_000,
    )

    assert result.source == "subtitle"
    assert result.language == "en-us"
    assert result.segments == (
        TranscriptSegment(1_001_000, 2_234_000, "Hello & world"),
        TranscriptSegment(2_500_000, 3_000_000, "Second line"),
    )


def test_vtt_skips_note_style_and_cue_identifier() -> None:
    payload = b"""WEBVTT - generated

NOTE this is not dialogue
00:00.000 --> 00:09.000

STYLE
::cue { color: lime }

cue-42
00:01.250 --> 00:02.500 position:20%
<v Narrator>Hello <c.yellow>there</c></v>
"""

    result = parse_subtitle_bytes(
        payload,
        subtitle_format="vtt",
        language="en",
        duration_us=5_000_000,
    )

    assert [(item.start_us, item.end_us, item.text) for item in result.segments] == [
        (1_250_000, 2_500_000, "Hello there")
    ]


def test_ass_uses_events_format_and_removes_override_tags() -> None:
    payload = """[Script Info]
Title: Fixture

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.01,0:00:02.25,Default,,0,0,0,,{\\an8}<i>Xin</i>\\Nch\u00e0o, b\u1ea1n
Comment: 0,0:00:00.00,0:00:09.00,Default,,0,0,0,,Not speech
""".encode("utf-8")

    result = parse_subtitle_bytes(
        payload,
        subtitle_format="ssa",
        language="vi",
        duration_us=5_000_000,
    )

    assert result.segments == (
        TranscriptSegment(1_010_000, 2_250_000, "Xin ch\u00e0o, b\u1ea1n"),
    )


def test_timestamps_are_sorted_clamped_and_overlap_is_removed() -> None:
    payload = b"""1
00:00:04,000 --> 00:00:07,000
Last

2
00:00:00,000 --> 00:00:03,000
First

3
00:00:02,000 --> 00:00:04,500
Overlap

4
00:00:04,200 --> 00:00:04,300
Fully covered
"""

    result = parse_subtitle_bytes(
        payload,
        subtitle_format="srt",
        language="en",
        duration_us=5_000_000,
    )

    assert result.segments == (
        TranscriptSegment(0, 3_000_000, "First"),
        TranscriptSegment(3_000_000, 4_500_000, "Overlap"),
        TranscriptSegment(4_500_000, 5_000_000, "Last"),
    )


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_utf16_with_and_without_bom_is_supported(encoding: str) -> None:
    source = "1\n00:00:00,000 --> 00:00:01,000\nTi\u1ebfng Vi\u1ec7t\n"
    result = parse_subtitle_bytes(
        source.encode(encoding),
        subtitle_format="srt",
        language="vi",
        duration_us=2_000_000,
    )
    assert result.segments[0].text == "Ti\u1ebfng Vi\u1ec7t"


def test_cp1258_fallback_is_explicitly_controllable() -> None:
    cp1258_source = "1\n00:00:00,000 --> 00:00:01,000\nTi\u00ea\u0301ng Vi\u00ea\u0323t\n"
    payload = cp1258_source.encode("cp1258")
    parsed = parse_subtitle_bytes(
        payload,
        subtitle_format="srt",
        language="vi",
        duration_us=2_000_000,
    )
    assert parsed.segments[0].text == "Ti\u1ebfng Vi\u1ec7t"
    with pytest.raises(TranscriptError, match="Encoding"):
        parse_subtitle_bytes(
            payload,
            subtitle_format="srt",
            language="vi",
            duration_us=2_000_000,
            allow_legacy_cp1258=False,
        )


def test_empty_or_out_of_range_cues_report_no_speech() -> None:
    payload = b"1\n00:00:02,000 --> 00:00:03,000\n<i></i>\n"
    with pytest.raises(TranscriptError, match="kh.ng c. l.i tho.i"):
        parse_subtitle_bytes(
            payload,
            subtitle_format="srt",
            language="en",
            duration_us=1_000_000,
        )


def test_parse_file_detects_format_and_bounds_input(tmp_path: Path) -> None:
    path = tmp_path / "fixture.SRT"
    path.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    assert parse_subtitle_file(path, language="en", duration_us=2_000_000).segments
    with pytest.raises(TranscriptError, match="k.ch th..c"):
        parse_subtitle_file(
            path,
            language="en",
            duration_us=2_000_000,
            max_bytes=4,
        )


def _asr_result() -> TranscriptionResult:
    return TranscriptionResult(
        source="asr",
        language="ja",
        language_probability=0.875,
        duration_us=4_000_000,
        model_id="asr-faster-whisper-small",
        segments=(
            TranscriptSegment(
                100_000,
                1_500_000,
                "\u3053\u3093\u306b\u3061\u306f",
                average_log_probability=-0.125,
                no_speech_probability=0.01,
            ),
            TranscriptSegment(1_500_000, 3_750_000, "\u4e16\u754c"),
        ),
    )


def test_artifact_round_trip_is_canonical_atomic_and_hash_verified(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "source-transcript.json"
    written = write_transcript_artifact(path, _asr_result())

    assert written.path == path
    assert written.schema_version == TRANSCRIPT_SCHEMA_VERSION
    assert written.sha256 == transcript_file_sha256(path)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    loaded = load_transcript_artifact(path, expected_sha256=written.sha256.upper())
    assert loaded.result == _asr_result()
    assert read_transcript_artifact(path, expected_sha256=written.sha256) == _asr_result()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_artifact_tampering_and_invalid_hash_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    artifact = write_transcript_artifact(path, _asr_result())
    path.write_bytes(path.read_bytes().replace("\u4e16\u754c".encode(), "\u4e16\u754c!".encode()))

    with pytest.raises(TranscriptIntegrityError, match="kh.ng kh.p"):
        load_transcript_artifact(path, expected_sha256=artifact.sha256)
    with pytest.raises(TranscriptIntegrityError, match="kh.ng h.p l."):
        load_transcript_artifact(path, expected_sha256="not-a-digest")


def test_artifact_schema_and_timeline_are_strictly_validated(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    document = {
        "schema_version": 999,
        "source": "subtitle",
        "language": "en",
        "language_probability": 1.0,
        "duration_us": 10,
        "model_id": None,
        "segments": [{"start_us": 0, "end_us": 10, "text": "Hi"}],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TranscriptError, match="Phi.n b.n"):
        load_transcript_artifact(path)

    document["schema_version"] = 1
    document["segments"] = [
        {"start_us": 5, "end_us": 9, "text": "A"},
        {"start_us": 8, "end_us": 10, "text": "B"},
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TranscriptError, match="ch.ng l.n"):
        load_transcript_artifact(path)


def test_parser_and_artifact_reader_do_not_need_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = parse_subtitle_bytes(
        b"1\n00:00:00,000 --> 00:00:01,000\nOffline\n",
        subtitle_format="srt",
        language="en",
        duration_us=2_000_000,
    )
    path = tmp_path / "transcript.json"
    artifact = write_transcript_artifact(path, result)

    def reject_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", reject_socket)
    assert load_transcript_artifact(path, expected_sha256=artifact.sha256).result == result
