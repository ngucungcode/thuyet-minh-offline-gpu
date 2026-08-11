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
    NATURAL_MAX_SILENT_BORROW_US,
    NarrationTimingInput,
    TimingError,
    TimingProfile,
    TimingQuality,
    build_timeline_wav,
    classify_timing_quality,
    decompose_atempo,
    microseconds_to_samples,
    plan_narration_slots,
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


def test_natural_planner_borrows_silence_and_centres_speech() -> None:
    inputs = (
        NarrationTimingInput(1_000_000, 2_000_000, 1_600_000),
        NarrationTimingInput(3_000_000, 3_500_000, 500_000),
    )
    planned = plan_narration_slots(inputs, duration_us=6_000_000)

    assert planned[0].start_us == 700_000
    assert planned[0].end_us == 2_300_000
    assert planned[0].planned_total_speed == pytest.approx(1.0)
    assert planned[1].start_us == 3_000_000
    assert planned[1].end_us == 3_500_000
    assert planned[0].end_us <= planned[1].start_us
    assert plan_narration_slots(inputs, duration_us=6_000_000) == planned


def test_natural_planner_smooths_neighbouring_cluster_speeds() -> None:
    planned = plan_narration_slots(
        (
            NarrationTimingInput(0, 1_000_000, 1_000_000),
            NarrationTimingInput(4_000_000, 5_000_000, 3_000_000),
        ),
        duration_us=6_000_000,
    )

    assert all(slot.end_us <= 6_000_000 for slot in planned)
    assert planned[0].end_us <= planned[1].start_us
    assert max(slot.planned_total_speed for slot in planned) <= 1.20
    assert abs(
        planned[1].planned_total_speed - planned[0].planned_total_speed
    ) <= 0.080_01
    # The first block had enough room at 1.0x, but is raised gently so the
    # transition into the dense second block does not sound abrupt.
    assert planned[0].planned_total_speed > 1.0


def test_natural_planner_does_not_smooth_across_a_long_silent_scene() -> None:
    planned = plan_narration_slots(
        (
            NarrationTimingInput(0, 1_000_000, 1_000_000),
            NarrationTimingInput(600_000_000, 601_000_000, 3_000_000),
        ),
        duration_us=602_000_000,
    )

    assert planned[0].planned_total_speed == pytest.approx(1.0)
    assert planned[1].planned_total_speed > 1.1


def test_natural_planner_resolves_dense_rolling_chain_without_overlap() -> None:
    planned = plan_narration_slots(
        (
            NarrationTimingInput(0, 1_000_000, 1_200_000),
            NarrationTimingInput(1_000_000, 2_000_000, 1_200_000),
        ),
        duration_us=2_000_000,
    )

    assert [(slot.start_us, slot.end_us) for slot in planned] == [
        (0, 1_000_000),
        (1_000_000, 2_000_000),
    ]
    assert [slot.planned_total_speed for slot in planned] == pytest.approx(
        [1.2, 1.2]
    )


def test_natural_planner_does_not_split_a_gap_consumed_at_slower_speed() -> None:
    planned = plan_narration_slots(
        (
            NarrationTimingInput(0, 1_000_000, 1_800_000),
            NarrationTimingInput(2_500_000, 3_500_000, 2_600_000),
        ),
        duration_us=4_300_000,
    )

    assert [(slot.start_us, slot.end_us) for slot in planned] == [
        (0, 1_759_091),
        (1_759_091, 4_300_000),
    ]
    assert all(slot.planned_total_speed <= 1.20 for slot in planned)


def test_natural_planner_handles_a_long_connected_window_chain() -> None:
    planned = plan_narration_slots(
        tuple(
            NarrationTimingInput(
                ordinal * 1_000_000,
                (ordinal + 1) * 1_000_000,
                1_050_000,
            )
            for ordinal in range(8)
        ),
        duration_us=8_000_000,
    )

    assert [(slot.start_us, slot.end_us) for slot in planned] == [
        (ordinal * 1_000_000, (ordinal + 1) * 1_000_000)
        for ordinal in range(8)
    ]
    assert [slot.planned_total_speed for slot in planned] == pytest.approx(
        [1.05] * 8
    )


def test_natural_planner_rejects_sample_rounding_above_the_hard_cap() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            tuple(
                NarrationTimingInput(
                    ordinal * 833_351,
                    (ordinal + 1) * 833_351,
                    1_000_021,
                    source_frame_count=48_001,
                    source_sample_rate=48_000,
                )
                for ordinal in range(4)
            ),
            duration_us=3_333_404,
        )

    assert captured.value.code == "timing_rewrite_required"
    assert captured.value.details["maximum_total_speed"] == 1.2


def test_natural_planner_requires_rewrite_instead_of_speeding_above_cap() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            (NarrationTimingInput(1_000_000, 2_000_000, 3_200_000),),
            duration_us=3_000_000,
        )

    error = captured.value
    assert error.code == "timing_rewrite_required"
    assert error.retryable is False
    assert error.details == {
        "profile": "natural",
        "ordinal": 0,
        "required_duration_us": 2_666_667,
        "available_duration_us": 2_600_000,
        "maximum_total_speed": 1.2,
        "failure_kind": "single_window_capacity",
        "failure_ordinal": 0,
        "critical_group_start_ordinal": 0,
        "critical_group_end_ordinal": 0,
        "schedule_deficit_us": 66_667,
        "rewrite_candidates": [
            {
                "ordinal": 0,
                "required_duration_us": 2_666_667,
                "target_available_duration_us": 2_600_000,
                "work_duration_us": 3_200_000,
            }
        ],
    }
    assert "rút gọn" in error.message_vi
    assert "1,20×" in error.message_vi


def test_intrinsic_window_overflow_does_not_blame_a_predecessor() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            (
                NarrationTimingInput(0, 1_000_000, 1_500_000),
                NarrationTimingInput(1_000_000, 2_000_000, 3_200_000),
            ),
            duration_us=4_000_000,
        )

    details = captured.value.details
    assert details["failure_kind"] == "single_window_capacity"
    assert details["failure_ordinal"] == 1
    assert details["critical_group_start_ordinal"] == 1
    assert details["critical_group_end_ordinal"] == 1
    assert [item["ordinal"] for item in details["rewrite_candidates"]] == [1]


def test_elastic_planner_keeps_base_success_byte_identical() -> None:
    inputs = (
        NarrationTimingInput(1_000_000, 2_000_000, 1_600_000),
        NarrationTimingInput(3_000_000, 3_500_000, 500_000),
    )

    base = plan_narration_slots(inputs, duration_us=6_000_000)
    elastic_opt_in = plan_narration_slots(
        inputs,
        duration_us=6_000_000,
        maximum_silent_borrow_us=NATURAL_MAX_SILENT_BORROW_US,
    )

    assert elastic_opt_in == base


@pytest.mark.parametrize(
    "options",
    (
        {"maximum_silent_borrow_us": True},
        {"maximum_silent_borrow_us": 799_999},
        {"silence_guard_us": True},
        {"silence_guard_us": -1},
    ),
)
def test_elastic_planner_rejects_invalid_borrow_options(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        plan_narration_slots(
            (NarrationTimingInput(0, 1_000_000, 500_000),),
            duration_us=1_000_000,
            **options,
        )


def test_elastic_planner_recovers_by_borrowing_a_larger_source_gap() -> None:
    planned = plan_narration_slots(
        (NarrationTimingInput(1_000_000, 2_000_000, 3_200_000),),
        duration_us=3_000_000,
        maximum_silent_borrow_us=NATURAL_MAX_SILENT_BORROW_US,
    )

    assert (planned[0].start_us, planned[0].end_us) == (0, 3_000_000)
    assert planned[0].planned_total_speed == pytest.approx(3.2 / 3.0)
    assert planned[0].start_us < 200_000
    assert planned[0].end_us > 2_800_000


def test_elastic_postvalidation_uses_exact_frames_at_the_speed_cap() -> None:
    inputs = (
        NarrationTimingInput(
            0,
            300_000,
            1_000_188,
            native_speed=1.2,
            source_frame_count=48_009,
            source_sample_rate=48_000,
        ),
        NarrationTimingInput(
            300_000,
            600_000,
            1_000_188,
            native_speed=1.2,
            source_frame_count=48_009,
            source_sample_rate=48_000,
        ),
    )

    planned = plan_narration_slots(
        inputs,
        duration_us=2_000_376,
        maximum_silent_borrow_us=NATURAL_MAX_SILENT_BORROW_US,
    )

    assert len(planned) == 2
    assert planned[0].end_us <= planned[1].start_us
    assert all(slot.planned_total_speed == pytest.approx(1.2) for slot in planned)


def test_elastic_planner_never_extends_into_a_neighbour_source_block() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            (
                NarrationTimingInput(500_000, 1_000_000, 3_000_000),
                NarrationTimingInput(2_000_000, 2_500_000, 300_000),
            ),
            duration_us=4_000_000,
            maximum_silent_borrow_us=NATURAL_MAX_SILENT_BORROW_US,
        )

    error = captured.value
    assert error.code == "timing_rewrite_required"
    assert error.details["failure_ordinal"] == 0
    # The elastic right edge stops 120 ms before the next source block.
    assert error.details["available_duration_us"] == 1_880_000


def test_dense_failure_describes_critical_chain_and_sorted_candidates() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            (
                NarrationTimingInput(0, 1_000_000, 1_600_000),
                NarrationTimingInput(1_000_000, 2_000_000, 1_600_000),
            ),
            duration_us=2_000_000,
        )

    details = captured.value.details
    assert details["failure_kind"] == "critical_chain_capacity"
    assert details["failure_ordinal"] == 1
    assert details["critical_group_start_ordinal"] == 0
    assert details["critical_group_end_ordinal"] == 1
    assert details["schedule_deficit_us"] == 666_668
    assert details["rewrite_candidates"] == [
        {
            "ordinal": 1,
            "required_duration_us": 1_333_334,
            "target_available_duration_us": 666_666,
            "work_duration_us": 1_600_000,
        },
        {
            "ordinal": 0,
            "required_duration_us": 1_333_334,
            "target_available_duration_us": 666_666,
            "work_duration_us": 1_600_000,
        },
    ]


def test_critical_failure_details_are_deterministic() -> None:
    inputs = (
        NarrationTimingInput(0, 1_000_000, 1_500_000),
        NarrationTimingInput(1_000_000, 2_000_000, 1_700_000),
    )
    captured_details: list[dict[str, object]] = []

    for _ in range(3):
        with pytest.raises(TimingError) as captured:
            plan_narration_slots(inputs, duration_us=2_000_000)
        captured_details.append(captured.value.details)

    assert captured_details[1:] == captured_details[:-1]


def test_strict_planner_preserves_legacy_slots_even_when_speech_is_long() -> None:
    planned = plan_narration_slots(
        (NarrationTimingInput(1_000_000, 2_000_000, 3_200_000),),
        duration_us=3_000_000,
        profile=TimingProfile.STRICT,
    )

    assert [(slot.start_us, slot.end_us) for slot in planned] == [
        (1_000_000, 2_000_000)
    ]
    assert planned[0].planned_total_speed == pytest.approx(3.2)


def test_natural_planner_rejects_overlapping_source_timestamps() -> None:
    with pytest.raises(TimingError) as captured:
        plan_narration_slots(
            (
                NarrationTimingInput(0, 1_000_001, 500_000),
                NarrationTimingInput(1_000_000, 2_000_000, 500_000),
            ),
            duration_us=2_000_000,
        )
    assert captured.value.code == "timing_source_overlap"


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


def test_fitter_enforces_natural_total_speed_cap_before_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "long-tts.wav"
    _write_wav(source, frames=153_600)
    output = tmp_path / "fitted.wav"
    process_started = False

    async def process_factory(*command: str, **kwargs: object):
        nonlocal process_started
        process_started = True
        return CompletingFfmpeg(Path(command[-1]), 124_800)

    with pytest.raises(TimingError) as captured:
        asyncio.run(
            FfmpegTimingFitter(process_factory=process_factory).fit(
                source,
                output,
                start_us=200_000,
                end_us=2_800_000,
                maximum_total_speed=1.20,
            )
        )

    assert captured.value.code == "timing_rewrite_required"
    assert captured.value.details["required_duration_us"] == 2_666_667
    assert process_started is False
    assert not output.exists()


def test_fitter_rejects_even_one_sample_above_natural_hard_cap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "boundary-tts.wav"
    _write_wav(source, frames=57_601)
    output = tmp_path / "fitted.wav"
    process_started = False

    async def process_factory(*command: str, **kwargs: object):
        nonlocal process_started
        process_started = True
        return CompletingFfmpeg(Path(command[-1]), 48_000)

    with pytest.raises(TimingError) as captured:
        asyncio.run(
            FfmpegTimingFitter(process_factory=process_factory).fit(
                source,
                output,
                start_us=0,
                end_us=1_000_000,
                maximum_total_speed=1.20,
            )
        )

    assert captured.value.code == "timing_rewrite_required"
    assert process_started is False
