from __future__ import annotations

import asyncio
import os
import sys
import wave
from pathlib import Path

import pytest

from dub_server.narration import (
    NarrationError,
    NarrationSynthesizer,
    PiperNarrationSynthesizer,
    VieNeuNarrationSynthesizer,
    normalize_vietnamese_for_tts,
)


def _write_wav(path: Path, *, frames: int = 22_050, sample_rate: int = 22_050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(os.fspath(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x01\x00" * frames)


class CompletingProcess:
    def __init__(
        self,
        output_path: Path,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        observed_input: list[bytes | None] | None = None,
        observed_text: tuple[Path, list[str]] | None = None,
    ) -> None:
        self.returncode: int | None = None
        self._configured_returncode = returncode
        self._stderr = stderr
        self._output_path = output_path
        self._observed_input = observed_input
        self._observed_text = observed_text
        self.terminated = False
        self.killed = False

    async def communicate(
        self, input: bytes | None = None
    ) -> tuple[bytes | None, bytes | None]:
        await asyncio.sleep(0)
        if self._observed_input is not None:
            self._observed_input.append(input)
        if self._observed_text is not None:
            path, values = self._observed_text
            values.append(path.read_text(encoding="utf-8"))
        if self._configured_returncode == 0:
            _write_wav(self._output_path)
        self.returncode = self._configured_returncode
        return None, self._stderr

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class HangingProcess:
    def __init__(self, output_path: Path) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._finished = asyncio.Event()
        output_path.write_bytes(b"partial")

    async def communicate(
        self, input: bytes | None = None
    ) -> tuple[bytes | None, bytes | None]:
        await self._finished.wait()
        return None, b""

    def terminate(self) -> None:
        self.terminated = True
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


def _local_file(path: Path) -> Path:
    path.write_bytes(b"fixture")
    return path


def test_vietnamese_normalization_is_conservative_and_nfc() -> None:
    assert normalize_vietnamese_for_tts(
        "  TP. HCM  chiếm  50%  GDP & VN…\u200b  "
    ) == "Thành phố Hồ Chí Minh chiếm 50 phần trăm GDP và Việt Nam…"
    assert normalize_vietnamese_for_tts("a\u0301") == "á"
    with pytest.raises(NarrationError) as captured:
        normalize_vietnamese_for_tts("\u200b\ufeff")
    assert captured.value.code == "tts_text_empty"


def test_piper_adapter_is_local_atomic_and_injectable(tmp_path: Path) -> None:
    binary = _local_file(tmp_path / "piper")
    model = _local_file(tmp_path / "voice.onnx")
    config = _local_file(tmp_path / "voice.onnx.json")
    output = tmp_path / "blocks" / "0001.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    observed_input: list[bytes | None] = []
    progress: list[tuple[int, int]] = []

    async def process_factory(*command: str, **kwargs: object):
        calls.append((command, kwargs))
        destination = Path(command[command.index("--output_file") + 1])
        return CompletingProcess(destination, observed_input=observed_input)

    synthesizer = PiperNarrationSynthesizer(
        model,
        binary=binary,
        config_path=config,
        speaker_id=2,
        process_factory=process_factory,
    )
    assert isinstance(synthesizer, NarrationSynthesizer)
    result = asyncio.run(
        synthesizer.synthesize(
            " TP. HCM có 20% ",
            output,
            speed=1.2,
            on_progress=lambda done, total: progress.append((done, total)),
        )
    )

    assert result.path == output.resolve()
    assert result.backend == "piper"
    assert result.duration_us == 1_000_000
    assert result.text == "Thành phố Hồ Chí Minh có 20 phần trăm"
    assert observed_input == ["Thành phố Hồ Chí Minh có 20 phần trăm\n".encode()]
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == os.fspath(model.resolve())
    assert float(command[command.index("--length_scale") + 1]) == pytest.approx(1 / 1.2)
    assert command[command.index("--speaker") + 1] == "2"
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "shell" not in kwargs
    assert all("://" not in argument for argument in command)
    assert progress == [(0, 1), (1, 1)]
    assert output.read_bytes().startswith(b"RIFF")
    assert not output.with_name(".0001.tts.part.wav").exists()


def test_vieneu_adapter_uses_local_text_file_and_forces_offline_env(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    entrypoint = _local_file(tmp_path / "infer.py")
    output = tmp_path / "voice.wav"
    observed_text: list[str] = []
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def process_factory(*command: str, **kwargs: object):
        calls.append((command, kwargs))
        text_path = Path(command[command.index("--text-file") + 1])
        destination = Path(command[command.index("--output-file") + 1])
        return CompletingProcess(
            destination, observed_text=(text_path, observed_text)
        )

    result = asyncio.run(
        VieNeuNarrationSynthesizer(
            model,
            entrypoint,
            python_binary=Path(sys.executable),
            process_factory=process_factory,
            environment={"HF_HUB_OFFLINE": "0"},
        ).synthesize("Xin chào", output)
    )

    assert result.backend == "vieneu"
    assert observed_text == ["Xin chào\n"]
    command, kwargs = calls[0]
    assert "--local-files-only" in command
    assert os.fspath(model.resolve()) in command
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert not output.with_name(".voice.tts-input.part.txt").exists()


def test_vieneu_rejects_network_argument_from_custom_builder(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    entrypoint = _local_file(tmp_path / "infer.py")

    def unsafe_builder(*args: object) -> list[str]:
        return [os.fspath(entrypoint), "--model", "https://example.invalid/model"]

    synthesizer = VieNeuNarrationSynthesizer(
        model,
        entrypoint,
        python_binary=Path(sys.executable),
        argument_builder=unsafe_builder,
    )
    with pytest.raises(NarrationError) as captured:
        asyncio.run(synthesizer.synthesize("Xin chào", tmp_path / "out.wav"))
    assert captured.value.code == "tts_network_forbidden"
    assert not (tmp_path / "out.wav").exists()


def test_tts_cancellation_terminates_process_and_preserves_old_output(
    tmp_path: Path,
) -> None:
    binary = _local_file(tmp_path / "piper")
    model = _local_file(tmp_path / "voice.onnx")
    output = tmp_path / "out.wav"
    output.write_bytes(b"old")
    processes: list[HangingProcess] = []

    async def process_factory(*command: str, **kwargs: object):
        process = HangingProcess(Path(command[command.index("--output_file") + 1]))
        processes.append(process)
        return process

    with pytest.raises(NarrationError) as captured:
        asyncio.run(
            PiperNarrationSynthesizer(
                model,
                binary=binary,
                process_factory=process_factory,
                poll_interval_seconds=0.001,
            ).synthesize(
                "Xin chào",
                output,
                cancellation=CancelOnSecondCheck(),
            )
        )
    assert captured.value.code == "tts_cancelled"
    assert captured.value.retryable is True
    assert processes[0].terminated is True
    assert output.read_bytes() == b"old"
    assert not output.with_name(".out.tts.part.wav").exists()


def test_subprocess_failure_is_typed_and_does_not_publish(tmp_path: Path) -> None:
    binary = _local_file(tmp_path / "piper")
    model = _local_file(tmp_path / "voice.onnx")
    output = tmp_path / "out.wav"

    async def process_factory(*command: str, **kwargs: object):
        return CompletingProcess(
            Path(command[command.index("--output_file") + 1]),
            returncode=1,
            stderr=b"inference failed",
        )

    with pytest.raises(NarrationError) as captured:
        asyncio.run(
            PiperNarrationSynthesizer(
                model, binary=binary, process_factory=process_factory
            ).synthesize("Xin chào", output)
        )
    assert captured.value.code == "tts_failed"
    assert captured.value.retryable is True
    assert not output.exists()
