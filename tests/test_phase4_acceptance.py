from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dub_server.model_registry import VerifiedModel
from dub_server.narration import PiperNarrationSynthesizer


SCRIPT = Path(__file__).parents[1] / "scripts" / "phase4_acceptance.py"
SPEC = importlib.util.spec_from_file_location("phase4_acceptance_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
phase4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phase4
SPEC.loader.exec_module(phase4)


def _probe_payload(
    *,
    duration: str = "4.000000",
    video_start: str = "0.000000",
    audio_start: str = "0.000000",
    audio_codec: str = "aac",
    extra_stream: dict[str, object] | None = None,
) -> dict[str, object]:
    streams: list[dict[str, object]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "duration": duration,
            "start_time": video_start,
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": audio_codec,
            "duration": duration,
            "start_time": audio_start,
        },
    ]
    if extra_stream is not None:
        streams.append(extra_stream)
    return {"format": {"duration": duration}, "streams": streams}


def test_probe_contract_accepts_exactly_one_video_and_one_synced_aac() -> None:
    contract = phase4._validate_probe_contract(
        _probe_payload(audio_start="0.099999"),
        expected_duration_us=4_000_000,
    )

    assert contract.video_codec == "h264"
    assert contract.audio_codec == "aac"
    assert contract.duration_error_us == 0
    assert contract.sync_error_us == 99_999


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            _probe_payload(
                extra_stream={
                    "index": 2,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "4.0",
                    "start_time": "0.0",
                }
            ),
            "đúng một video track",
        ),
        (_probe_payload(audio_codec="opus"), "không phải AAC"),
        (_probe_payload(duration="4.100001"), "lệch quá 100 ms"),
        (_probe_payload(audio_start="0.100001"), "lệch quá 100 ms"),
    ],
)
def test_probe_contract_rejects_track_codec_duration_and_sync_violations(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        phase4._validate_probe_contract(
            payload,
            expected_duration_us=4_000_000,
        )


def test_quick_fixture_command_is_shell_free_local_and_bounded(tmp_path: Path) -> None:
    output = tmp_path / "fixture.mp4"
    command = phase4._fixture_command("ffmpeg", output, 4.0)

    assert command[0] == "ffmpeg"
    assert command[-1] == str(output)
    assert "shell" not in command
    assert not any("://" in item for item in command)
    maps = [command[index + 1] for index, item in enumerate(command) if item == "-map"]
    assert maps == ["0:v:0", "[mix]"]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    graph = command[command.index("-filter_complex") + 1]
    assert "music" in graph
    assert "dialogue_like" in graph
    assert "effects" in graph


@pytest.mark.parametrize("duration", [0.0, 1.999, 1_800.001])
def test_quick_fixture_rejects_unbounded_duration(
    tmp_path: Path, duration: float
) -> None:
    with pytest.raises(ValueError):
        phase4._fixture_command("ffmpeg", tmp_path / "fixture.mp4", duration)


def test_piper_builder_uses_only_verified_locked_files(tmp_path: Path) -> None:
    model_root = tmp_path / "piper"
    model_root.mkdir()
    model_file = model_root / "voice.onnx"
    config_file = model_root / "voice.onnx.json"
    model_file.write_bytes(b"model")
    config_file.write_text("{}", encoding="utf-8")
    verified = VerifiedModel(
        entry={
            "id": "tts-piper-test",
            "stage": "tts",
            "backend": "piper",
            "model_file": "voice.onnx",
            "config_file": "voice.onnx.json",
            "files": [
                {"path": "voice.onnx"},
                {"path": "voice.onnx.json"},
            ],
        },
        path=model_root,
        tree_sha256="a" * 64,
    )

    synthesizer = phase4._build_synthesizer(
        verified,
        support_model=None,
        piper_binary="piper",
        vieneu_entrypoint=tmp_path / "unused.py",
        python_binary=sys.executable,
    )

    assert isinstance(synthesizer, PiperNarrationSynthesizer)
    assert synthesizer._model_path == model_file
    assert synthesizer._config_path == config_file


def test_tts_factory_rejects_manifest_path_escape(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    verified = VerifiedModel(
        entry={
            "id": "tts-test",
            "stage": "tts",
            "backend": "piper",
            "model_file": "../outside.onnx",
        },
        path=model_root,
        tree_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="không an toàn"):
        phase4._build_synthesizer(
            verified,
            support_model=None,
            piper_binary="piper",
            vieneu_entrypoint=tmp_path / "unused.py",
            python_binary=sys.executable,
        )


def test_gpu_parser_ignores_headers_and_malformed_rows() -> None:
    assert phase4._parse_gpu_rows(
        "pid, used_memory\n123, 456\nnot-a-pid, 2\n789, 1024\n"
    ) == (
        {"pid": 123, "used_memory_mib": 456},
        {"pid": 789, "used_memory_mib": 1024},
    )


def test_atomic_report_replaces_previous_json_without_partial(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"old":true}', encoding="utf-8")

    phase4._atomic_json(report, {"passed": True, "text": "ngoại tuyến"})

    assert json.loads(report.read_text(encoding="utf-8")) == {
        "passed": True,
        "text": "ngoại tuyến",
    }
    assert list(tmp_path.glob(".report.json.*.part")) == []
