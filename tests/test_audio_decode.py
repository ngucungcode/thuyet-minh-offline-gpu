from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave
from array import array
from pathlib import Path

import pytest

from dub_server.audio_decode import AudioDecodeError, FfmpegAudioDecoder


def _write_wav(
    path: Path,
    *,
    frame_count: int = 32_000,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\0" * frame_count * channels * sample_width)


class CompletingProcess:
    def __init__(
        self,
        output_path: Path,
        *,
        frame_count: int = 32_000,
        channels: int = 1,
        sample_rate: int = 16_000,
        sample_width: int = 2,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.returncode: int | None = None
        self._configured_returncode = returncode
        self._stderr = stderr
        self._output_path = output_path
        self._frame_count = frame_count
        self._channels = channels
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes | None, bytes | None]:
        await asyncio.sleep(0)
        if self._configured_returncode == 0:
            _write_wav(
                self._output_path,
                frame_count=self._frame_count,
                sample_rate=self._sample_rate,
                channels=self._channels,
                sample_width=self._sample_width,
            )
        self.returncode = self._configured_returncode
        return None, self._stderr

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class HangingProcess:
    def __init__(self, output_path: Path, *, finish_on_terminate: bool) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._finish_on_terminate = finish_on_terminate
        self._finished = asyncio.Event()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"partial")

    async def communicate(self) -> tuple[bytes | None, bytes | None]:
        await self._finished.wait()
        return None, b""

    def terminate(self) -> None:
        self.terminated = True
        if self._finish_on_terminate:
            self.returncode = -15
            self._finished.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._finished.set()


class CancelOnSecondCheck:
    def __init__(self) -> None:
        self.checks = 0

    def is_cancelled(self) -> bool:
        self.checks += 1
        return self.checks >= 2


@pytest.mark.parametrize(
    ("audio_stream_index", "expected_selector"),
    [(None, "0:a:0"), (3, "0:3")],
)
def test_decode_uses_offline_ffmpeg_arguments_and_atomic_publish(
    tmp_path: Path,
    audio_stream_index: int | None,
    expected_selector: str,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"fixture")
    output = tmp_path / "job" / "audio.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old artifact")
    commands: list[tuple[str, ...]] = []
    process_kwargs: list[dict[str, object]] = []

    async def process_factory(*command: str, **kwargs: object):
        commands.append(command)
        process_kwargs.append(kwargs)
        return CompletingProcess(Path(command[-1]))

    artifact = asyncio.run(
        FfmpegAudioDecoder(process_factory=process_factory).decode(
            source,
            output,
            expected_duration_us=2_000_000,
            audio_stream_index=audio_stream_index,
        )
    )

    assert artifact.path == output.resolve()
    assert artifact.sample_rate == 16_000
    assert artifact.channels == 1
    assert artifact.sample_width_bytes == 2
    assert artifact.frame_count == 32_000
    assert artifact.duration_us == 2_000_000
    assert output.read_bytes().startswith(b"RIFF")
    assert not output.with_name(f".{output.name}.part").exists()
    command = commands[0]
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    assert command[command.index("-map") + 1] == expected_selector
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert "shell" not in process_kwargs[0]
    assert all("://" not in argument for argument in command)


@pytest.mark.parametrize(
    ("frame_count", "should_pass"),
    [(32_320, True), (32_321, False)],
)
def test_duration_validation_uses_inclusive_twenty_millisecond_tolerance(
    tmp_path: Path,
    frame_count: int,
    should_pass: bool,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"fixture")
    output = tmp_path / "audio.wav"

    async def process_factory(*command: str, **kwargs: object):
        return CompletingProcess(Path(command[-1]), frame_count=frame_count)

    operation = FfmpegAudioDecoder(process_factory=process_factory).decode(
        source,
        output,
        expected_duration_us=2_000_000,
    )
    if should_pass:
        artifact = asyncio.run(operation)
        assert artifact.duration_us == 2_020_000
    else:
        with pytest.raises(AudioDecodeError) as captured:
            asyncio.run(operation)
        assert captured.value.code == "decoded_audio_duration_mismatch"
        assert captured.value.retryable is False
        assert not output.exists()


def test_invalid_wav_shape_is_rejected_without_overwriting_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"fixture")
    output = tmp_path / "audio.wav"
    output.write_bytes(b"keep me")

    async def process_factory(*command: str, **kwargs: object):
        return CompletingProcess(Path(command[-1]), channels=2)

    with pytest.raises(AudioDecodeError) as captured:
        asyncio.run(
            FfmpegAudioDecoder(process_factory=process_factory).decode(
                source,
                output,
                expected_duration_us=2_000_000,
            )
        )

    assert captured.value.code == "invalid_decoded_audio"
    assert output.read_bytes() == b"keep me"
    assert not output.with_name(f".{output.name}.part").exists()


def test_missing_audio_stream_is_a_typed_non_retryable_error(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "audio.wav"

    async def process_factory(*command: str, **kwargs: object):
        return CompletingProcess(
            Path(command[-1]),
            returncode=1,
            stderr=b"Stream map '0:a:0' matches no streams.",
        )

    with pytest.raises(AudioDecodeError) as captured:
        asyncio.run(
            FfmpegAudioDecoder(process_factory=process_factory).decode(
                source,
                output,
                expected_duration_us=2_000_000,
            )
        )

    assert captured.value.code == "no_audio_stream"
    assert captured.value.message_vi
    assert captured.value.retryable is False
    assert not output.exists()


@pytest.mark.parametrize("finish_on_terminate", [True, False])
def test_cancellation_stops_ffmpeg_and_removes_partial_output(
    tmp_path: Path,
    finish_on_terminate: bool,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"fixture")
    output = tmp_path / "audio.wav"
    processes: list[HangingProcess] = []

    async def process_factory(*command: str, **kwargs: object):
        process = HangingProcess(
            Path(command[-1]),
            finish_on_terminate=finish_on_terminate,
        )
        processes.append(process)
        return process

    with pytest.raises(AudioDecodeError) as captured:
        asyncio.run(
            FfmpegAudioDecoder(
                process_factory=process_factory,
                poll_interval_seconds=0.001,
                stop_grace_seconds=0.05,
            ).decode(
                source,
                output,
                expected_duration_us=2_000_000,
                cancellation=CancelOnSecondCheck(),
            )
        )

    assert captured.value.code == "audio_decode_cancelled"
    assert captured.value.retryable is True
    assert processes[0].terminated is True
    assert processes[0].killed is (not finish_on_terminate)
    assert not output.exists()
    assert not output.with_name(f".{output.name}.part").exists()


def test_external_task_cancellation_stops_process_and_propagates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"fixture")
    output = tmp_path / "audio.wav"
    started = asyncio.Event()
    processes: list[HangingProcess] = []

    async def process_factory(*command: str, **kwargs: object):
        process = HangingProcess(Path(command[-1]), finish_on_terminate=True)
        processes.append(process)
        started.set()
        return process

    async def scenario() -> None:
        task = asyncio.create_task(
            FfmpegAudioDecoder(process_factory=process_factory).decode(
                source,
                output,
                expected_duration_us=2_000_000,
            )
        )
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert processes[0].terminated is True
    assert not output.exists()
    assert not output.with_name(f".{output.name}.part").exists()


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="FFmpeg không có trên máy chạy test")
def test_real_ffmpeg_decodes_generated_stereo_media_to_mono_16khz(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.mkv"
    output = tmp_path / "decoded.wav"
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
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=2",
            "-filter_complex",
            "[1:a][2:a]amerge=inputs=2[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    artifact = asyncio.run(
        FfmpegAudioDecoder().decode(
            source,
            output,
            expected_duration_us=2_000_000,
        )
    )

    assert artifact.frame_count == 32_000
    with wave.open(str(output), "rb") as stream:
        samples = array("h")
        samples.frombytes(stream.readframes(stream.getnframes()))
    assert samples
    assert max(abs(sample) for sample in samples) > 100
