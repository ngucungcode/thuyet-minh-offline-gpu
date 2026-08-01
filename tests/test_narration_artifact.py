from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dub_server.narration_artifact import (
    NarrationArtifactError,
    SrtCue,
    build_srt_cues,
    build_timing_report,
    write_srt_artifact,
    write_timing_report,
)
from dub_server.timing import (
    FittedNarrationBlock,
    TimingQuality,
    microseconds_to_samples,
)


def _block(
    tmp_path: Path,
    *,
    start_us: int,
    end_us: int,
    text: str,
    quality: TimingQuality = TimingQuality.NORMAL,
    total_speed: float = 1.0,
) -> FittedNarrationBlock:
    frames = microseconds_to_samples(end_us) - microseconds_to_samples(start_us)
    return FittedNarrationBlock(
        path=tmp_path / f"{start_us}.wav",
        start_us=start_us,
        end_us=end_us,
        text=text,
        source_duration_us=end_us - start_us + 123,
        target_frame_count=frames,
        output_frame_count=frames,
        native_speed=1.1,
        atempo_speed=total_speed / 1.1,
        total_speed=total_speed,
        padded_frame_count=12,
        quality=quality,
    )


class CancelImmediately:
    def is_cancelled(self) -> bool:
        return True


def test_timing_report_is_canonical_atomic_and_contains_quality_summary(
    tmp_path: Path,
) -> None:
    blocks = (
        _block(tmp_path, start_us=100, end_us=1_000_001, text="Xin chào"),
        _block(
            tmp_path,
            start_us=1_500_000,
            end_us=2_500_000,
            text="Việt Nam",
            quality=TimingQuality.SEVERE,
            total_speed=1.8,
        ),
    )
    report = build_timing_report(
        blocks,
        duration_us=3_000_001,
        tts_model_id="tts-piper-vi-vais1000-medium",
        tts_backend="piper",
    )
    destination = tmp_path / "artifacts" / "timing-report.json"
    destination.parent.mkdir()
    destination.write_bytes(b"old")
    progress: list[tuple[int, int]] = []

    artifact = write_timing_report(
        destination,
        report,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    payload = destination.read_bytes()
    document = json.loads(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)
    assert document["artifact_type"] == "timing-report"
    assert document["schema_version"] == 1
    assert document["timeline_frame_count"] == microseconds_to_samples(3_000_001)
    assert document["tts"]["backend"] == "piper"
    assert document["quality_summary"] == {
        "normal": 1,
        "severe": 1,
        "warning": 0,
    }
    assert document["blocks"][1]["quality"] == "severe"
    assert document["blocks"][0]["padded_frame_count"] == 12
    assert progress == [(1, 2), (2, 2)]
    assert payload.endswith(b"\n")


def test_srt_uses_floor_start_ceil_end_and_utf8_atomic(tmp_path: Path) -> None:
    cues = (
        SrtCue(start_us=1_234, end_us=2_345_678, text="  Xin   chào  "),
        SrtCue(start_us=3_000_001, end_us=3_000_002, text="Tiếng Việt"),
    )
    destination = tmp_path / "narration.vi.srt"
    progress: list[tuple[int, int]] = []

    artifact = write_srt_artifact(
        destination,
        cues,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert destination.read_text(encoding="utf-8") == (
        "1\n00:00:00,001 --> 00:00:02,346\nXin chào\n\n"
        "2\n00:00:03,000 --> 00:00:03,001\nTiếng Việt\n\n"
    )
    assert artifact.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert progress == [(1, 2), (2, 2)]


def test_build_srt_cues_preserves_fitted_timeline(tmp_path: Path) -> None:
    block = _block(
        tmp_path, start_us=10_000, end_us=20_000, text="  Một   câu  "
    )
    assert build_srt_cues((block,)) == (
        SrtCue(start_us=10_000, end_us=20_000, text="Một câu"),
    )


def test_cancelled_artifact_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "narration.srt"
    destination.write_bytes(b"old")
    with pytest.raises(NarrationArtifactError) as captured:
        write_srt_artifact(
            destination,
            (SrtCue(0, 1_000_000, "Xin chào"),),
            cancellation=CancelImmediately(),
        )
    assert captured.value.code == "artifact_cancelled"
    assert captured.value.retryable is True
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".narration.srt.*.part"))


def test_srt_overlap_and_size_limit_are_typed(tmp_path: Path) -> None:
    with pytest.raises(NarrationArtifactError) as overlap:
        write_srt_artifact(
            tmp_path / "overlap.srt",
            (
                SrtCue(0, 2_000_000, "Một"),
                SrtCue(1_000_000, 3_000_000, "Hai"),
            ),
        )
    assert overlap.value.code == "srt_timeline_invalid"

    with pytest.raises(NarrationArtifactError) as too_large:
        write_srt_artifact(
            tmp_path / "large.srt",
            (SrtCue(0, 1_000_000, "Nội dung"),),
            max_bytes=4,
        )
    assert too_large.value.code == "narration_artifact_too_large"
