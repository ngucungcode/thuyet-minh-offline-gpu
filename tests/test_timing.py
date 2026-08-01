from __future__ import annotations

import asyncio
import os
import re
import wave
from array import array
from pathlib import Path

import pytest

from dub_server.timing import (
    FfmpegTimingFitter,
    FittedNarrationBlock,
    TimingError,
    TimingQuality,
    build_timeline_wav,
    classify_timing_quality,
    decompose_atempo,
    microseconds_to_samples,
)


def _write_wav(
    path: Path,
    *,
    frames: int,
    sample_rate: int = 48_000,
    sample: int = 1_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array("h", [sample]) * frames
    with wave.open(os.fspath(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(values.tobytes())


class CompletingFfmpeg:
    def __init__(
        self,
        output_path: Path,
        frame_count: int,
        *,
        returncode: int = 0,
    ) -> None:
        self.returncode: int | None = None
        self._configured_returncode = returncode
        self._output_path = output_path
        self._frame_count = frame_count
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes | None, bytes | None]:
        await asyncio.sleep(0)
        if self._configured_returncode == 0:
            _write_wav(self._output_path, frames=self._frame_count)
        self.returncode = self._configured_returncode
        return None, b"failure" if self._configured_returncode else b""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class CancelImmediately:
    def is_cancelled(self) -> bool:
        return True


def _fake_fitted_block(
    path: Path,
    *,
    start_us: int,
    end_us: int,
    text: str,
) -> FittedNarrationBlock:
    frames = microseconds_to_samples(end_us) - microseconds_to_samples(start_us)
    return FittedNarrationBlock(
        path=path,
        start_us=start_us,
        end_us=end_us,
        text=text,
        source_duration_us=end_us - start_us,
        target_frame_count=frames,
        output_frame_count=frames,
        native_speed=1.0,
        atempo_speed=1.0,
        total_speed=1.0,
        padded_frame_count=0,
        quality=TimingQuality.NORMAL,
    )


def test_ffmpeg_fitter_uses_atempo_padding_fades_and_exact_samples(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tts.wav"
    _write_wav(source, frames=22_050, sample_rate=22_050)
    output = tmp_path / "fitted.wav"
    commands: list[tuple[str, ...]] = []
    kwargs_seen: list[dict[str, object]] = []
    progress: list[tuple[int, int]] = []

    async def process_factory(*command: str, **kwargs: object):
        commands.append(command)
        kwargs_seen.append(kwargs)
        filter_graph = command[command.index("-filter:a") + 1]
        target = int(re.search(r"atrim=end_sample=(\d+)", filter_graph).group(1))
        return CompletingFfmpeg(Path(command[-1]), target)

    block = asyncio.run(
        FfmpegTimingFitter(process_factory=process_factory).fit(
            source,
            output,
            start_us=100_000,
            end_us=2_100_000,
            text="Xin chào",
            native_speed=1.2,
            on_progress=lambda done, total: progress.append((done, total)),
        )
    )

    assert block.target_frame_count == 96_000
    assert block.output_frame_count == 96_000
    assert block.atempo_speed == pytest.approx(0.8)
    assert block.total_speed == pytest.approx(0.96)
    assert block.padded_frame_count == 36_000
    assert block.quality is TimingQuality.NORMAL
    assert progress == [(0, 1), (1, 1)]
    command = commands[0]
    graph = command[command.index("-filter:a") + 1]
    assert "atempo=0.8000000000" in graph
    assert "apad=whole_len=96000" in graph
    assert "atrim=end_sample=96000" in graph
    assert "afade=t=in:ss=0:ns=240" in graph
    assert "afade=t=out:ss=95760:ns=240" in graph
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    assert command[command.index("-ar") + 1] == "48000"
    assert "shell" not in kwargs_seen[0]
    assert output.read_bytes().startswith(b"RIFF")


def test_atempo_is_decomposed_and_quality_thresholds_are_locked() -> None:
    factors = decompose_atempo(9.0)
    assert factors == pytest.approx((2.0, 2.0, 2.0, 1.125))
    product = 1.0
    for factor in factors:
        assert 0.5 <= factor <= 2.0
        product *= factor
    assert product == pytest.approx(9.0)
    assert classify_timing_quality(1.35) is TimingQuality.NORMAL
    assert classify_timing_quality(1.350001) is TimingQuality.WARNING
    assert classify_timing_quality(1.70) is TimingQuality.WARNING
    assert classify_timing_quality(1.700001) is TimingQuality.SEVERE


def test_microsecond_rounding_and_timeline_placement_are_sample_exact(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    first = _fake_fitted_block(
        first_path, start_us=100_001, end_us=200_003, text="Một"
    )
    second = _fake_fitted_block(
        second_path, start_us=250_005, end_us=300_007, text="Hai"
    )
    _write_wav(first_path, frames=first.target_frame_count, sample=1_000)
    _write_wav(second_path, frames=second.target_frame_count, sample=2_000)
    output = tmp_path / "timeline.wav"
    progress: list[tuple[int, int]] = []

    timeline = build_timeline_wav(
        (first, second),
        output,
        duration_us=350_011,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert timeline.frame_count == microseconds_to_samples(350_011)
    assert progress == [(1, 2), (2, 2)]
    with wave.open(os.fspath(output), "rb") as stream:
        samples = array("h")
        samples.frombytes(stream.readframes(stream.getnframes()))
    first_start = microseconds_to_samples(first.start_us)
    first_end = microseconds_to_samples(first.end_us)
    second_start = microseconds_to_samples(second.start_us)
    second_end = microseconds_to_samples(second.end_us)
    assert all(value == 0 for value in samples[:first_start])
    assert all(value == 1_000 for value in samples[first_start:first_end])
    assert all(value == 0 for value in samples[first_end:second_start])
    assert all(value == 2_000 for value in samples[second_start:second_end])
    assert all(value == 0 for value in samples[second_end:])


def test_timeline_cancel_is_typed_atomic_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    block_path = tmp_path / "block.wav"
    block = _fake_fitted_block(
        block_path, start_us=100_000, end_us=200_000, text="Một"
    )
    _write_wav(block_path, frames=block.target_frame_count)
    output = tmp_path / "timeline.wav"
    output.write_bytes(b"old")

    with pytest.raises(TimingError) as captured:
        build_timeline_wav(
            (block,),
            output,
            duration_us=300_000,
            cancellation=CancelImmediately(),
        )
    assert captured.value.code == "timing_cancelled"
    assert captured.value.retryable is True
    assert output.read_bytes() == b"old"
    assert not output.with_name(".timeline.timeline.part.wav").exists()


def test_fitter_failure_is_typed_and_does_not_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "tts.wav"
    _write_wav(source, frames=48_000)
    output = tmp_path / "fitted.wav"
    output.write_bytes(b"old")

    async def process_factory(*command: str, **kwargs: object):
        return CompletingFfmpeg(Path(command[-1]), 48_000, returncode=1)

    with pytest.raises(TimingError) as captured:
        asyncio.run(
            FfmpegTimingFitter(process_factory=process_factory).fit(
                source, output, start_us=0, end_us=1_000_000
            )
        )
    assert captured.value.code == "timing_ffmpeg_failed"
    assert captured.value.retryable is True
    assert output.read_bytes() == b"old"
