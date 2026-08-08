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
    format_duration: str | None = None,
    video_duration: str | None = None,
    audio_duration: str | None = None,
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
            "duration": video_duration or duration,
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": audio_codec,
            "start_time": audio_start,
            "duration": audio_duration or duration,
        },
    ]
    if extra_stream is not None:
        streams.append(extra_stream)
    return json.dumps(
        {"streams": streams, "format": {"duration": format_duration or duration}}
    )


class FakeRunner:
    def __init__(
        self,
        *,
        ffmpeg_returncode: int = 0,
        ffmpeg_stderr: str = "",
        source_probe_stdout: str | None = None,
        source_probe_returncode: int = 0,
        source_probe_error: MediaExportError | None = None,
        probe_stdout: str | None = None,
        probe_returncode: int = 0,
    ) -> None:
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffmpeg_stderr = ffmpeg_stderr
        self.source_probe_stdout = source_probe_stdout or json.dumps(
            {"streams": [{"duration": "2.000000"}]}
        )
        self.source_probe_returncode = source_probe_returncode
        self.source_probe_error = source_probe_error
        self.probe_stdout = probe_stdout or _probe_payload()
        self.probe_returncode = probe_returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def __call__(self, command, **kwargs):
        normalized = tuple(str(item) for item in command)
        self.calls.append((normalized, kwargs))
        if Path(normalized[0]).name.lower().startswith("ffprobe"):
            source_probe = "-select_streams" in normalized
            if source_probe and self.source_probe_error is not None:
                raise self.source_probe_error
            payload = self.source_probe_stdout if source_probe else self.probe_stdout
            returncode = (
                self.source_probe_returncode if source_probe else self.probe_returncode
            )
            return subprocess.CompletedProcess(
                list(normalized),
                returncode,
                payload,
                "probe failed" if returncode else "",
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

    source_probe = runner.calls[0][0]
    assert source_probe[source_probe.index("-select_streams") + 1] == "V:0"
    command = runner.calls[1][0]
    assert command[0] == "ffmpeg"
    input_positions = [index for index, item in enumerate(command) if item == "-i"]
    assert [command[index + 1] for index in input_positions] == [
        str(source.resolve()),
        str(accompaniment.resolve()),
        str(narration.resolve()),
    ]
    maps = [command[index + 1] for index, item in enumerate(command) if item == "-map"]
    assert maps == ["0:V:0", "[mixed]"]
    assert all(not item.startswith("0:a") for item in maps)
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-disposition:v:0") + 1] == "default"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-map_chapters") + 1] == "-1"
    assert command[command.index("-write_tmcd") + 1] == "0"
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    assert "shell" not in runner.calls[1][1]

    graph = command[command.index("-filter_complex") + 1]
    assert "loudnorm=I=-24.0" in graph
    assert "loudnorm=I=-16.0" in graph
    assert "sidechaincompress=" in graph
    assert "ratio=3.00" in graph
    assert "amix=inputs=2" in graph
    assert "atrim=duration=2.000000" in graph
    assert "apad" in graph
    assert "alimiter=" in graph
    assert "-shortest" not in command


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
    assert len(runner.calls) == 2


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


def test_verification_rejects_output_video_still_marked_as_cover_art(
    tmp_path: Path,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(
        probe_stdout=json.dumps(
            {
                "streams": [
                    {
                        "index": 0,
                        "codec_type": "video",
                        "codec_name": "h264",
                        "start_time": "0",
                        "duration": "2",
                        "disposition": {"attached_pic": 1},
                    },
                    {
                        "index": 1,
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "start_time": "0",
                        "duration": "2",
                    },
                ]
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
    assert "ảnh bìa/thumbnail" in captured.value.message_vi
    assert not output.exists()


@pytest.mark.parametrize(
    ("probe", "expected_code"),
    [
        (
            _probe_payload(audio_duration="2.100001"),
            MediaExportErrorCode.DURATION_MISMATCH,
        ),
        (
            _probe_payload(audio_start="0.080000", audio_duration="2.030001"),
            MediaExportErrorCode.DURATION_MISMATCH,
        ),
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


def test_verification_uses_track_end_times_not_container_metadata(
    tmp_path: Path,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(
        source_probe_stdout=json.dumps(
            {"streams": [{"duration": "1.920000"}]}
        ),
        probe_stdout=_probe_payload(
            format_duration="2.400000",
            video_duration="1.920000",
            video_start="0.080000",
            audio_duration="2.000000",
            audio_start="0.000000",
        )
    )

    artifact = asyncio.run(
        FfmpegAudioMixExporter(runner=runner).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_400_000,
        )
    )

    assert artifact.duration_us == 1_920_000
    assert artifact.video_start_us == 80_000
    assert artifact.audio_start_us == 0
    assert output.is_file()
    command = runner.calls[1][0]
    graph = command[command.index("-filter_complex") + 1]
    assert "atrim=duration=1.920000" in graph


@pytest.mark.parametrize(
    ("source_stream", "duration_text"),
    [
        ({"duration_ts": 180_000, "time_base": "1/90000"}, "2.000000"),
        ({"tags": {"DURATION": "00:00:02.250000000"}}, "2.250000"),
    ],
)
def test_source_video_duration_supports_timestamp_and_matroska_tag_fallbacks(
    tmp_path: Path,
    source_stream: dict[str, object],
    duration_text: str,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(
        source_probe_stdout=json.dumps({"streams": [source_stream]}),
        probe_stdout=_probe_payload(duration=duration_text),
    )

    asyncio.run(
        FfmpegAudioMixExporter(runner=runner).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_600_000,
        )
    )

    command = runner.calls[1][0]
    graph = command[command.index("-filter_complex") + 1]
    assert f"atrim=duration={duration_text}" in graph


@pytest.mark.parametrize(
    "source_stream",
    [
        {"duration_ts": 1, "time_base": "1/0"},
        {"duration_ts": "NaN", "time_base": "1/90000"},
        {"tags": {"DURATION": "invalid"}},
    ],
)
def test_malformed_source_video_timing_is_rejected_safely(
    tmp_path: Path,
    source_stream: dict[str, object],
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(
        source_probe_stdout=json.dumps({"streams": [source_stream]}),
    )

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_600_000,
            )
        )

    assert captured.value.code == MediaExportErrorCode.INVALID_OUTPUT
    assert captured.value.retryable is True
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("source_probe_stdout", "source_probe_returncode", "retryable"),
    [
        (json.dumps({"streams": [{"duration": "2.0"}]}), 1, True),
        ("not-json", 0, True),
        (json.dumps({"streams": []}), 0, False),
        (json.dumps({"streams": [{}]}), 0, True),
    ],
)
def test_invalid_source_video_timeline_fails_before_ffmpeg(
    tmp_path: Path,
    source_probe_stdout: str,
    source_probe_returncode: int,
    retryable: bool,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(
        source_probe_stdout=source_probe_stdout,
        source_probe_returncode=source_probe_returncode,
    )

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_600_000,
            )
        )

    assert captured.value.code == MediaExportErrorCode.INVALID_OUTPUT
    assert captured.value.retryable is retryable
    assert len(runner.calls) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    ("probe_error", "expected_code", "retryable"),
    [
        (
            MediaExportError(
                MediaExportErrorCode.FFMPEG_UNAVAILABLE,
                "runner missing",
                retryable=False,
            ),
            MediaExportErrorCode.FFPROBE_UNAVAILABLE,
            False,
        ),
        (
            MediaExportError(
                MediaExportErrorCode.EXPORT_FAILED,
                "runner timed out",
                retryable=True,
            ),
            MediaExportErrorCode.INVALID_OUTPUT,
            True,
        ),
    ],
)
def test_source_video_probe_errors_are_mapped_to_probe_context(
    tmp_path: Path,
    probe_error: MediaExportError,
    expected_code: MediaExportErrorCode,
    retryable: bool,
) -> None:
    source, accompaniment, narration, output = _inputs(tmp_path)
    runner = FakeRunner(source_probe_error=probe_error)

    with pytest.raises(MediaExportError) as captured:
        asyncio.run(
            FfmpegAudioMixExporter(runner=runner).export(
                source,
                accompaniment,
                narration,
                output,
                expected_duration_us=2_600_000,
            )
        )

    assert captured.value.code == expected_code
    assert captured.value.retryable is retryable
    assert len(runner.calls) == 1
    assert not output.exists()


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
def test_real_ffmpeg_ignores_embedded_cover_before_content_video(
    tmp_path: Path,
) -> None:
    main = tmp_path / "main.mp4"
    cover = tmp_path / "cover.jpg"
    source = tmp_path / "source-cover-first.mp4"
    accompaniment = tmp_path / "accompaniment.wav"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "dubbed.mp4"

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
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            str(main),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
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
            "color=c=red:size=64x64",
            "-frames:v",
            "1",
            "-c:v",
            "mjpeg",
            str(cover),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    subprocess.run(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(main),
            "-i",
            str(cover),
            "-map",
            "1:v:0",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-disposition:v:0",
            "attached_pic",
            "-disposition:v:1",
            "default",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
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

    source_probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type:stream_disposition=attached_pic",
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
    source_videos = [
        item
        for item in json.loads(source_probe.stdout)["streams"]
        if item["codec_type"] == "video"
    ]
    assert len(source_videos) == 2
    assert source_videos[0]["disposition"]["attached_pic"] == 1
    assert source_videos[1]["disposition"]["attached_pic"] == 0

    artifact = asyncio.run(
        FfmpegAudioMixExporter(
            ffmpeg_binary=str(FFMPEG),
            ffprobe_binary=str(FFPROBE),
            timeout_seconds=30.0,
        ).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_000_000,
        )
    )

    assert artifact.path == output.resolve()
    output_probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name:stream_disposition=attached_pic",
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
    output_streams = json.loads(output_probe.stdout)["streams"]
    assert [(item["codec_type"], item["codec_name"]) for item in output_streams] == [
        ("video", "mpeg4"),
        ("audio", "aac"),
    ]
    assert output_streams[0]["disposition"]["attached_pic"] == 0


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
            timeout_seconds=30.0,
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


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="FFmpeg/ffprobe không có trên máy chạy test",
)
def test_real_ffmpeg_aligns_replacement_audio_to_shorter_video_track(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-with-longer-audio.mp4"
    accompaniment = tmp_path / "accompaniment.wav"
    narration = tmp_path / "narration.wav"
    output = tmp_path / "dubbed.mp4"
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
            "testsrc2=size=64x64:rate=25:duration=2.0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=2.6",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
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
                f"sine=frequency={frequency}:sample_rate=48000:duration=2.6",
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

    source_probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
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
    source_video = next(
        item
        for item in source_payload["streams"]
        if item["codec_type"] == "video"
    )
    assert abs(float(source_payload["format"]["duration"]) - 2.6) <= 0.1
    assert abs(float(source_video["duration"]) - 2.0) <= 0.1

    artifact = asyncio.run(
        FfmpegAudioMixExporter(
            ffmpeg_binary=str(FFMPEG),
            ffprobe_binary=str(FFPROBE),
            timeout_seconds=30.0,
        ).export(
            source,
            accompaniment,
            narration,
            output,
            expected_duration_us=2_600_000,
        )
    )

    assert abs(artifact.duration_us - 2_000_000) <= 100_000
    probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time,duration",
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
    streams = json.loads(probe.stdout)["streams"]
    video = next(item for item in streams if item["codec_type"] == "video")
    audio = next(item for item in streams if item["codec_type"] == "audio")
    video_end = float(video.get("start_time", 0)) + float(video["duration"])
    audio_end = float(audio.get("start_time", 0)) + float(audio["duration"])
    assert abs(video_end - audio_end) <= 0.1
