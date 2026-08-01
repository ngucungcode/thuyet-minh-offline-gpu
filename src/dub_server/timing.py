"""Duration fitting and sample-exact 48 kHz narration timeline assembly."""

from __future__ import annotations

import asyncio
import os
import time
import wave
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .narration import Cancellation


TIMELINE_SAMPLE_RATE = 48_000
TIMELINE_CHANNELS = 1
TIMELINE_SAMPLE_WIDTH_BYTES = 2


class TimingError(RuntimeError):
    """A typed timing failure safe to persist in a job checkpoint."""

    def __init__(self, code: str, message_vi: str, *, retryable: bool) -> None:
        super().__init__(message_vi)
        self.code = code
        self.message_vi = message_vi
        self.retryable = retryable


class TimingQuality(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    SEVERE = "severe"


@dataclass(frozen=True, slots=True)
class FittedNarrationBlock:
    path: Path
    start_us: int
    end_us: int
    text: str
    source_duration_us: int
    target_frame_count: int
    output_frame_count: int
    native_speed: float
    atempo_speed: float
    total_speed: float
    padded_frame_count: int
    quality: TimingQuality

    def __post_init__(self) -> None:
        if self.start_us < 0 or self.end_us <= self.start_us:
            raise TimingError(
                "timing_slot_invalid",
                "Mốc thời gian của lời thuyết minh không hợp lệ",
                retryable=False,
            )
        if self.target_frame_count <= 0 or self.output_frame_count <= 0:
            raise TimingError(
                "timing_sample_count_invalid",
                "Số sample của lời thuyết minh không hợp lệ",
                retryable=False,
            )
        if self.output_frame_count != self.target_frame_count:
            raise TimingError(
                "timing_sample_count_mismatch",
                "Lời thuyết minh chưa khớp chính xác số sample của slot",
                retryable=False,
            )
        if self.source_duration_us <= 0 or self.native_speed <= 0 or self.atempo_speed <= 0:
            raise TimingError(
                "timing_speed_invalid",
                "Thông số tốc độ lời thuyết minh không hợp lệ",
                retryable=False,
            )
        if self.padded_frame_count < 0:
            raise TimingError(
                "timing_padding_invalid",
                "Số sample im lặng bù không hợp lệ",
                retryable=False,
            )


@dataclass(frozen=True, slots=True)
class NarrationTimeline:
    path: Path
    duration_us: int
    frame_count: int
    sample_rate: int
    channels: int
    sample_width_bytes: int
    block_count: int


TimingProgress = Callable[[int, int], None]


class ProcessHandle(Protocol):
    returncode: int | None

    async def communicate(self) -> tuple[bytes | None, bytes | None]: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[..., Awaitable[ProcessHandle]]


async def _default_process_factory(*command: str, **kwargs: object) -> ProcessHandle:
    return await asyncio.create_subprocess_exec(*command, **kwargs)


class FfmpegTimingFitter:
    """Fit a local PCM WAV to one timeline slot using FFmpeg ``atempo``.

    Speech is never truncated to solve a large under-run.  Slowdown is capped
    at 0.80x and the remaining tail is padded with silence.  FFmpeg's output is
    resampled, padded/trimmed for rounding only, and faded in/out in the same
    filter graph before an atomic publish.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        process_factory: ProcessFactory | None = None,
        minimum_slowdown: float = 0.80,
        fade_duration_us: int = 5_000,
        poll_interval_seconds: float = 0.05,
        stop_grace_seconds: float = 1.0,
        timeout_seconds: float | None = 300.0,
    ) -> None:
        if not ffmpeg_binary.strip() or "://" in ffmpeg_binary:
            raise ValueError("ffmpeg_binary must be a local executable")
        if not 0.5 <= minimum_slowdown <= 1.0:
            raise ValueError("minimum_slowdown must be between 0.5 and 1.0")
        if fade_duration_us < 0:
            raise ValueError("fade_duration_us must not be negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if stop_grace_seconds < 0:
            raise ValueError("stop_grace_seconds must not be negative")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._ffmpeg = ffmpeg_binary
        self._process_factory = process_factory or _default_process_factory
        self._minimum_slowdown = minimum_slowdown
        self._fade_duration_us = fade_duration_us
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_grace_seconds = stop_grace_seconds
        self._timeout_seconds = timeout_seconds

    async def fit(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_us: int,
        end_us: int,
        text: str = "",
        native_speed: float = 1.0,
        cancellation: Cancellation | None = None,
        on_progress: TimingProgress | None = None,
    ) -> FittedNarrationBlock:
        if start_us < 0 or end_us <= start_us:
            raise TimingError(
                "timing_slot_invalid",
                "Mốc thời gian của lời thuyết minh không hợp lệ",
                retryable=False,
            )
        if not 0.5 <= native_speed <= 2.0:
            raise TimingError(
                "timing_speed_invalid",
                "Tốc độ TTS gốc không hợp lệ",
                retryable=False,
            )
        destination = Path(output_path).resolve(strict=False)
        if destination.suffix.lower() != ".wav":
            raise TimingError(
                "timing_output_invalid",
                "File timing đầu ra phải có phần mở rộng .wav",
                retryable=False,
            )
        try:
            source = Path(input_path).resolve(strict=True)
        except OSError as exc:
            raise TimingError(
                "timing_input_missing",
                "Không tìm thấy file TTS cần khớp thời lượng",
                retryable=True,
            ) from exc
        if not source.is_file() or source == destination:
            raise TimingError(
                "timing_input_invalid",
                "File TTS đầu vào không hợp lệ",
                retryable=False,
            )

        source_metadata = _inspect_wav(source, require_timeline_format=False)
        target_frames = microseconds_to_samples(end_us) - microseconds_to_samples(start_us)
        if target_frames <= 0:
            raise TimingError(
                "timing_slot_too_short",
                "Slot thuyết minh quá ngắn để chứa một sample",
                retryable=False,
            )
        raw_atempo = (
            source_metadata.frame_count
            * TIMELINE_SAMPLE_RATE
            / (source_metadata.sample_rate * target_frames)
        )
        applied_atempo = max(self._minimum_slowdown, raw_atempo)
        estimated_content_frames = min(
            target_frames,
            round(
                source_metadata.frame_count
                * TIMELINE_SAMPLE_RATE
                / (source_metadata.sample_rate * applied_atempo)
            ),
        )
        padded_frames = max(0, target_frames - estimated_content_frames)
        total_speed = native_speed * applied_atempo
        quality = classify_timing_quality(total_speed)

        temporary = destination.with_name(f".{destination.stem}.timing.part.wav")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise TimingError(
                "timing_output_unavailable",
                "Không thể chuẩn bị nơi lưu lời thuyết minh đã khớp",
                retryable=True,
            ) from exc
        if _is_cancelled(cancellation):
            raise TimingError(
                "timing_cancelled",
                "Đã hủy khớp thời lượng lời thuyết minh",
                retryable=True,
            )

        filters = [
            *(f"atempo={factor:.10f}" for factor in decompose_atempo(applied_atempo)),
            f"aresample={TIMELINE_SAMPLE_RATE}",
            f"apad=whole_len={target_frames}",
            f"atrim=end_sample={target_frames}",
        ]
        fade_frames = min(
            microseconds_to_samples(self._fade_duration_us), target_frames // 2
        )
        if fade_frames > 0:
            filters.extend(
                [
                    f"afade=t=in:ss=0:ns={fade_frames}",
                    f"afade=t=out:ss={target_frames - fade_frames}:ns={fade_frames}",
                ]
            )
        command = (
            self._ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            os.fspath(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-filter:a",
            ",".join(filters),
            "-ac",
            str(TIMELINE_CHANNELS),
            "-ar",
            str(TIMELINE_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            os.fspath(temporary),
        )
        if any("://" in value for value in command):
            raise TimingError(
                "timing_network_forbidden",
                "Khớp thời lượng không được dùng tài nguyên mạng",
                retryable=False,
            )

        process: ProcessHandle | None = None
        communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None = None
        started = time.monotonic()
        try:
            _report_progress(on_progress, 0, 1)
            try:
                process = await self._process_factory(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                raise TimingError(
                    "timing_runtime_unavailable",
                    "Không thể khởi động FFmpeg để khớp thời lượng",
                    retryable=False,
                ) from exc
            communication = asyncio.create_task(process.communicate())
            while not communication.done():
                if _is_cancelled(cancellation):
                    await self._stop_process(process, communication)
                    raise TimingError(
                        "timing_cancelled",
                        "Đã hủy khớp thời lượng lời thuyết minh",
                        retryable=True,
                    )
                if (
                    self._timeout_seconds is not None
                    and time.monotonic() - started >= self._timeout_seconds
                ):
                    await self._stop_process(process, communication)
                    raise TimingError(
                        "timing_timeout",
                        "Khớp thời lượng vượt quá thời gian cho phép",
                        retryable=True,
                    )
                await asyncio.wait({communication}, timeout=self._poll_interval_seconds)
            _, _ = await communication
            if process.returncode != 0:
                raise TimingError(
                    "timing_ffmpeg_failed",
                    "FFmpeg không thể khớp thời lượng lời thuyết minh",
                    retryable=True,
                )
            output_metadata = _inspect_wav(temporary, require_timeline_format=True)
            if output_metadata.frame_count != target_frames:
                raise TimingError(
                    "timing_sample_count_mismatch",
                    "FFmpeg tạo sai số sample của slot thuyết minh",
                    retryable=True,
                )
            try:
                os.replace(temporary, destination)
            except OSError as exc:
                raise TimingError(
                    "timing_output_unavailable",
                    "Không thể lưu lời thuyết minh đã khớp thời lượng",
                    retryable=True,
                ) from exc
            _report_progress(on_progress, 1, 1)
            return FittedNarrationBlock(
                path=destination,
                start_us=start_us,
                end_us=end_us,
                text=" ".join(text.split()),
                source_duration_us=source_metadata.duration_us,
                target_frame_count=target_frames,
                output_frame_count=output_metadata.frame_count,
                native_speed=native_speed,
                atempo_speed=applied_atempo,
                total_speed=total_speed,
                padded_frame_count=padded_frames,
                quality=quality,
            )
        except asyncio.CancelledError:
            if process is not None and communication is not None:
                await self._stop_process(process, communication)
            raise
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    async def _stop_process(
        self,
        process: ProcessHandle,
        communication: asyncio.Task[tuple[bytes | None, bytes | None]],
    ) -> None:
        if communication.done() or process.returncode is not None:
            with suppress(Exception):
                await communication
            return
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.shield(communication), timeout=self._stop_grace_seconds
            )
            return
        except TimeoutError:
            pass
        with suppress(ProcessLookupError):
            process.kill()
        with suppress(Exception):
            await communication


def build_timeline_wav(
    blocks: Sequence[FittedNarrationBlock],
    output_path: Path,
    *,
    duration_us: int,
    cancellation: Cancellation | None = None,
    on_progress: TimingProgress | None = None,
) -> NarrationTimeline:
    """Place fitted blocks at rounded microsecond sample positions atomically."""

    if duration_us <= 0:
        raise TimingError(
            "timeline_duration_invalid",
            "Thời lượng timeline thuyết minh không hợp lệ",
            retryable=False,
        )
    if not blocks:
        raise TimingError(
            "timeline_empty",
            "Không có lời thuyết minh để tạo timeline",
            retryable=False,
        )
    destination = Path(output_path).resolve(strict=False)
    if destination.suffix.lower() != ".wav":
        raise TimingError(
            "timeline_output_invalid",
            "Timeline đầu ra phải có phần mở rộng .wav",
            retryable=False,
        )
    total_frames = microseconds_to_samples(duration_us)
    if total_frames <= 0:
        raise TimingError(
            "timeline_duration_invalid",
            "Timeline quá ngắn để chứa một sample",
            retryable=False,
        )
    temporary = destination.with_name(f".{destination.stem}.timeline.part.wav")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
    except OSError as exc:
        raise TimingError(
            "timeline_output_unavailable",
            "Không thể chuẩn bị nơi lưu timeline thuyết minh",
            retryable=True,
        ) from exc

    previous_end_us = 0
    current_frame = 0
    try:
        with wave.open(os.fspath(temporary), "wb") as destination_wav:
            destination_wav.setnchannels(TIMELINE_CHANNELS)
            destination_wav.setsampwidth(TIMELINE_SAMPLE_WIDTH_BYTES)
            destination_wav.setframerate(TIMELINE_SAMPLE_RATE)
            for ordinal, block in enumerate(blocks):
                if not isinstance(block, FittedNarrationBlock):
                    raise TimingError(
                        "timeline_block_invalid",
                        "Khối thuyết minh không hợp lệ",
                        retryable=False,
                    )
                if block.start_us < previous_end_us or block.end_us > duration_us:
                    raise TimingError(
                        "timeline_overlap",
                        "Timeline thuyết minh bị overlap hoặc vượt thời lượng video",
                        retryable=False,
                    )
                start_frame = microseconds_to_samples(block.start_us)
                end_frame = microseconds_to_samples(block.end_us)
                expected_frames = end_frame - start_frame
                if expected_frames != block.target_frame_count:
                    raise TimingError(
                        "timeline_slot_mismatch",
                        "Số sample của khối không khớp timestamp timeline",
                        retryable=False,
                    )
                metadata = _inspect_wav(block.path, require_timeline_format=True)
                if metadata.frame_count != expected_frames:
                    raise TimingError(
                        "timeline_block_length_mismatch",
                        "Khối thuyết minh không đúng số sample của slot",
                        retryable=False,
                    )
                _write_silence(
                    destination_wav,
                    start_frame - current_frame,
                    cancellation=cancellation,
                )
                _copy_wav_frames(
                    block.path,
                    destination_wav,
                    expected_frames=expected_frames,
                    cancellation=cancellation,
                )
                current_frame = end_frame
                previous_end_us = block.end_us
                _report_progress(on_progress, ordinal + 1, len(blocks))
            _write_silence(
                destination_wav,
                total_frames - current_frame,
                cancellation=cancellation,
            )
        metadata = _inspect_wav(temporary, require_timeline_format=True)
        if metadata.frame_count != total_frames:
            raise TimingError(
                "timeline_sample_count_mismatch",
                "Timeline thuyết minh không đúng tổng số sample",
                retryable=True,
            )
        if _is_cancelled(cancellation):
            raise TimingError(
                "timing_cancelled",
                "Đã hủy tạo timeline thuyết minh",
                retryable=True,
            )
        try:
            os.replace(temporary, destination)
        except OSError as exc:
            raise TimingError(
                "timeline_output_unavailable",
                "Không thể lưu timeline thuyết minh",
                retryable=True,
            ) from exc
        return NarrationTimeline(
            path=destination,
            duration_us=duration_us,
            frame_count=total_frames,
            sample_rate=TIMELINE_SAMPLE_RATE,
            channels=TIMELINE_CHANNELS,
            sample_width_bytes=TIMELINE_SAMPLE_WIDTH_BYTES,
            block_count=len(blocks),
        )
    except TimingError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise TimingError(
            "timeline_write_failed",
            "Không thể tạo timeline WAV thuyết minh",
            retryable=True,
        ) from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def microseconds_to_samples(value_us: int, *, sample_rate: int = TIMELINE_SAMPLE_RATE) -> int:
    if isinstance(value_us, bool) or not isinstance(value_us, int) or value_us < 0:
        raise TimingError(
            "timing_timestamp_invalid",
            "Timestamp thuyết minh không hợp lệ",
            retryable=False,
        )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return (value_us * sample_rate + 500_000) // 1_000_000


def decompose_atempo(speed: float) -> tuple[float, ...]:
    if not 0.5 <= speed <= 10_000:
        raise TimingError(
            "timing_speed_invalid",
            "Hệ số atempo không hợp lệ",
            retryable=False,
        )
    remaining = speed
    factors: list[float] = []
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return tuple(factors)


def classify_timing_quality(total_speed: float) -> TimingQuality:
    if total_speed <= 1.35:
        return TimingQuality.NORMAL
    if total_speed <= 1.70:
        return TimingQuality.WARNING
    return TimingQuality.SEVERE


@dataclass(frozen=True, slots=True)
class _WavMetadata:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_us: int


def _inspect_wav(path: Path, *, require_timeline_format: bool) -> _WavMetadata:
    try:
        with wave.open(os.fspath(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise TimingError(
            "timing_audio_invalid",
            "File WAV cho bước khớp thời lượng không hợp lệ",
            retryable=False,
        ) from exc
    if (
        frame_count <= 0
        or channels != 1
        or sample_width != TIMELINE_SAMPLE_WIDTH_BYTES
        or sample_rate <= 0
        or compression != "NONE"
        or (require_timeline_format and sample_rate != TIMELINE_SAMPLE_RATE)
    ):
        raise TimingError(
            "timing_audio_invalid",
            (
                "Khối thuyết minh phải là PCM mono 48 kHz 16-bit"
                if require_timeline_format
                else "Âm thanh TTS phải là PCM mono 16-bit hợp lệ"
            ),
            retryable=False,
        )
    duration_us = (frame_count * 1_000_000 + sample_rate // 2) // sample_rate
    return _WavMetadata(
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_us=duration_us,
    )


def _write_silence(
    destination: wave.Wave_write,
    frame_count: int,
    *,
    cancellation: Cancellation | None,
) -> None:
    if frame_count < 0:
        raise TimingError(
            "timeline_overlap",
            "Timeline thuyết minh bị overlap",
            retryable=False,
        )
    silence_chunk = b"\0" * (16_384 * TIMELINE_SAMPLE_WIDTH_BYTES)
    remaining = frame_count
    while remaining:
        if _is_cancelled(cancellation):
            raise TimingError(
                "timing_cancelled",
                "Đã hủy tạo timeline thuyết minh",
                retryable=True,
            )
        count = min(remaining, 16_384)
        destination.writeframesraw(silence_chunk[: count * TIMELINE_SAMPLE_WIDTH_BYTES])
        remaining -= count


def _copy_wav_frames(
    source_path: Path,
    destination: wave.Wave_write,
    *,
    expected_frames: int,
    cancellation: Cancellation | None,
) -> None:
    copied = 0
    with wave.open(os.fspath(source_path), "rb") as source:
        while copied < expected_frames:
            if _is_cancelled(cancellation):
                raise TimingError(
                    "timing_cancelled",
                    "Đã hủy tạo timeline thuyết minh",
                    retryable=True,
                )
            payload = source.readframes(min(16_384, expected_frames - copied))
            if not payload:
                break
            frames = len(payload) // TIMELINE_SAMPLE_WIDTH_BYTES
            destination.writeframesraw(payload)
            copied += frames
    if copied != expected_frames:
        raise TimingError(
            "timeline_block_length_mismatch",
            "Không đọc đủ sample của khối thuyết minh",
            retryable=False,
        )


def _is_cancelled(cancellation: Cancellation | None) -> bool:
    if cancellation is None:
        return False
    try:
        return bool(cancellation() if callable(cancellation) else cancellation.is_cancelled())
    except Exception as exc:
        raise TimingError(
            "timing_cancellation_invalid",
            "Không thể kiểm tra trạng thái hủy timing",
            retryable=False,
        ) from exc


def _report_progress(
    callback: TimingProgress | None, completed: int, total: int
) -> None:
    if callback is None:
        return
    try:
        callback(completed, total)
    except Exception as exc:
        raise TimingError(
            "timing_progress_failed",
            "Không thể cập nhật tiến độ khớp thời lượng",
            retryable=False,
        ) from exc


__all__ = [
    "FfmpegTimingFitter",
    "FittedNarrationBlock",
    "NarrationTimeline",
    "ProcessFactory",
    "TIMELINE_CHANNELS",
    "TIMELINE_SAMPLE_RATE",
    "TIMELINE_SAMPLE_WIDTH_BYTES",
    "TimingError",
    "TimingProgress",
    "TimingQuality",
    "build_timeline_wav",
    "classify_timing_quality",
    "decompose_atempo",
    "microseconds_to_samples",
]
