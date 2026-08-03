from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dub_server.audio_mix_export import (
    FfmpegAudioMixExporter,
    MediaExportError,
    MediaExportErrorCode,
)


def _probe_payload(
    *,
    duration: str = "2.000000",
    video_start: str = "0.000000",
    audio_start: str = "0.000000",
    audio_codec: str = "aac",
    extra_stream: dict[str, object] | None = None,
) -> str:
    streams: list[dict[str, object]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "start_time": video_start,
            "duration": duration,
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": audio_codec,
            "start_time": audio_start,
            "duration": duration,
        },
    ]
    if extra_stream is not None:
        streams.append(extra_stream)
    return json.dumps({"streams": streams, "format": {"duration": duration}})


class FakeRunner:
    def __init__(
        self,
        *,
        ffmpeg_returncode: int = 0,
        ffmpeg_stderr: str = "",
        probe_stdout: str | None = None,
        probe_returncode: int = 0,
    ) -> None:
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffmpeg_stderr = ffmpeg_stderr
        self.probe_stdout = probe_stdout or _probe_payload()
        self.probe_returncode = probe_returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def __call__(self, command, **kwargs):
        normalized = tuple(str(item) for item in command)
        self.calls.append((normalized, kwargs))
        if Path(normalized[0]).name.lower().startswith("ffprobe"):
            return subprocess.CompletedProcess(
                list(normalized),
                self.probe_returncode,
                self.probe_stdout,
                "probe failed" if self.probe_returncode else "",
            )
        callback = kwargs.get("on_progress")
        expected = kwargs.get("expected_duration_us")
        if callback is not None and isinstance(expected, int):
            from dub_server.audio_mix_export import ExportProgress

            callback(ExportProgress(expected // 2, expected, 0.5))
        if self.ffmpeg_returncode == 0:
            Path(normalized[-1]).write_bytes(b"valid mp4 placeholder")
        return subprocess.CompletedProcess(
            list(normalized),
            self.ffmpeg_returncode,
            "",
            self.ffmpeg_stderr,
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.mp4"
    accompaniment = tmp_path / "accompaniment.wav"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "result" / "dubbed.mp4"
    source.write_bytes(b"source with original audio")
    accompaniment.write_bytes(b"dialogue-reduced stem")
    narration.write_bytes(b"Vietnamese narration")
    return source, accompaniment, narration, output


def test_export_maps_no_source_audio_and_atomically_publishes(tmp_path: Path) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old output")
    runner = FakeRunner()
    progress = []

    artifact = asyncio.run(
        FfmpegAudioMixExporter(runner=runner).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_000_000,
            on_progress=progress.append,
        )
    )

    assert artifact.path == output.resolve()
    assert artifact.audio_codec == "aac"
    assert artifact.duration_us == 2_000_000
    assert output.read_bytes() == b"valid mp4 placeholder"
    assert not output.with_name(".dubbed.part.mp4").exists()
    assert [item.fraction for item in progress] == [0.5, 1.0]

    command = runner.calls[0][0]
    assert command[0] == "ffmpeg"
    input_positions = [index for index, item in enumerate(command) if item == "-i"]
    assert [command[index + 1] for index in input_positions] == [
        str(source.resolve()),
        str(accompaniment.resolve()),
        str(narration.resolve()),
    ]
    maps = [command[index + 1] for index, item in enumerate(command) if item == "-map"]
    assert maps == ["0:v:0", "[mixed]"]
    assert all(not item.startswith("0:a") for item in maps)
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-map_chapters") + 1] == "-1"
    assert command[command.index("-write_tmcd") + 1] == "0"
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    assert "shell" not in runner.calls[0][1]

    graph = command[command.index("-filter_complex") + 1]
    assert "loudnorm=I=-24.0" in graph
    assert "loudnorm=I=-16.0" in graph
    assert "sidechaincompress=" in graph
    assert "ratio=3.00" in graph
    assert "amix=inputs=2" in graph
    assert "atrim=duration=2.000000" in graph
    assert "alimiter=" in graph


@pytest.mark.parametrize(
    ("missing_index", "expected_code"),
    [
        (0, MediaExportErrorCode.SOURCE_VIDEO_MISSING),
        (1, MediaExportErrorCode.ACCOMPANIMENT_MISSING),
        (2, MediaExportErrorCode.NARRATION_MISSING),
    ],
)
def test_missing_input_is_typed_and_does_not_start_ffmpeg(
    tmp_path: Path,
    missing_index: int,
    expected_code: MediaExportErrorCode,
) -> None:
    paths = list(_inputs(tmp_path))
    paths[missing_index].unlink()
    runner = FakeRunner()

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                paths[0], paths[1], paths[2], paths[3], expected_duration_us=2_000_000
            )
        )

    assert captured.value.code == expected_code
    assert captured.value.message_vi
    assert captured.value.retryable is True
    assert runner.calls == []


def test_ffmpeg_failure_keeps_previous_output_and_removes_partial(tmp_path: Path) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep previous")
    runner = FakeRunner(
        ffmpeg_returncode=1,
        ffmpeg_stderr="No space left on device",
    )

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_000_000,
            )
        )

    assert captured.value.code == MediaExportErrorCode.EXPORT_FAILED
    assert "dung lượng" in captured.value.message_vi
    assert captured.value.retryable is True
    assert output.read_bytes() == b"keep previous"
    assert not output.with_name(".dubbed.part.mp4").exists()
    assert len(runner.calls) == 1


def test_verification_rejects_any_extra_or_original_audio_track(tmp_path: Path) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep previous")
    runner = FakeRunner(
        probe_stdout=_probe_payload(
            extra_stream={
                "index": 2,
                "codec_type": "audio",
                "codec_name": "aac",
                "codec_tag_string": "mp4a",
                "start_time": "0.0",
                "duration": "2.0",
            }
        )
    )

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_000_000,
            )
        )

    assert captured.value.code == MediaExportErrorCode.TRACK_LAYOUT_INVALID
    assert captured.value.retryable is False
    assert "3 luồng" in captured.value.message_vi
    assert "2:audio/aac[mp4a]" in captured.value.message_vi
    assert output.read_bytes() == b"keep previous"
    assert not output.with_name(".dubbed.part.mp4").exists()


@pytest.mark.parametrize(
    ("probe", "expected_code"),
    [
        (_probe_payload(duration="2.100001"), MediaExportErrorCode.DURATION_MISMATCH),
        (
            _probe_payload(audio_start="0.100001"),
            MediaExportErrorCode.SYNC_MISMATCH,
        ),
        (
            _probe_payload(audio_codec="opus"),
            MediaExportErrorCode.AUDIO_CODEC_INVALID,
        ),
    ],
)
def test_verification_enforces_duration_sync_and_aac(
    tmp_path: Path,
    probe: str,
    expected_code: MediaExportErrorCode,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(probe_stdout=probe)

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_000_000,
            )
        )

    assert captured.value.code == expected_code
    assert not output.exists()
    assert not output.with_name(".dubbed.part.mp4").exists()


def test_pre_cancelled_export_is_typed_and_does_not_start_runner(tmp_path: Path) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner()

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_000_000,
                cancellation=lambda: True,
            )
        )

    assert captured.value.code == MediaExportErrorCode.EXPORT_CANCELLED
    assert captured.value.retryable is True
    assert runner.calls == []
    assert not output.exists()


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="FFmpeg/ffprobe không có trên máy chạy test",
)
def test_real_ffmpeg_mix_has_one_video_and_one_aac_audio_track(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    chapters = tmp_path / "chapters.ffmetadata"
    accompaniment = tmp_path / "accompaniment.wav"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "dubbed.mp4"
    chapters.write_text(
        ";FFMETADATA1\n"
        "title=Timecoded source\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Opening\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=2",
            "-f",
            "ffmetadata",
            "-i",
            str(chapters),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "2",
            "-map_chapters",
            "2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-timecode",
            "01:00:00:00",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    source_probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "chapter=start_time,end_time:"
            "stream=codec_type,codec_name,codec_tag_string",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    source_payload = json.loads(source_probe.stdout)
    source_streams = source_payload["streams"]
    assert len(source_streams) > 2
    assert any(
        item.get("codec_tag_string") == "tmcd" for item in source_streams
    )
    assert source_payload["chapters"]
    for target, frequency in ((accompaniment, 440), (narration, 880)):
        subprocess.run(
            [
                str(FFMPEG),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=2",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    artifact = asyncio.run(
        FfmpegAudioMixExporter(
            ffmpeg_binary=str(FFMPEG),
            ffprobe_binary=str(FFPROBE),
            duration_tolerance_us=120_000,
        ).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_000_000,
        )
    )

    assert artifact.path == output.resolve()
    assert artifact.audio_codec == "aac"
    probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "chapter=start_time,end_time:stream=codec_type,codec_name",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    output_payload = json.loads(probe.stdout)
    streams = output_payload["streams"]
    assert [(item["codec_type"], item["codec_name"]) for item in streams] == [
        ("video", "mpeg4"),
        ("audio", "aac"),
    ]
    assert output_payload.get("chapters", []) == []
