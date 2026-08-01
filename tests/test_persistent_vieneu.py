from __future__ import annotations

import asyncio
import json
import os
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from dub_server.narration import (
    NarrationError,
    PersistentVieNeuNarrationSynthesizer,
    PiperNarrationSynthesizer,
    _resolve_executable,
)
from dub_server.model_registry import VerifiedModel
from dub_server.phase4_stage import build_narration_synthesizer


def test_resolve_executable_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-system"
    target.write_bytes(b"runtime")
    link = tmp_path / "python-venv"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is not permitted")

    assert _resolve_executable(link) == os.fspath(link.absolute())


def _write_wav(path: Path, *, frames: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(os.fspath(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24_000)
        stream.writeframes(b"\x01\x00" * frames)


class _Reader:
    def __init__(self) -> None:
        self._lines: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._lines.get()

    async def read(self, size: int = -1) -> bytes:
        del size
        return await self._lines.get()

    def feed(self, value: bytes) -> None:
        self._lines.put_nowait(value)


class _Writer:
    def __init__(self, process: "_ServerProcess") -> None:
        self._process = process

    def write(self, data: bytes) -> None:
        request = json.loads(data.decode("utf-8"))
        request_type = request["type"]
        if request_type == "close":
            self._process.stdout.feed(b'{"type":"closed","protocol":1}\n')
            self._process.finish(0)
            return
        self._process.requests.append(request)
        self._process.request_received.set()
        if self._process.behavior == "hang":
            Path(request["output_file"]).write_bytes(b"partial")
            return
        if self._process.behavior == "bad_protocol":
            Path(request["output_file"]).write_bytes(b"partial")
            self._process.stdout.feed(b'{"type":"wrong","protocol":1}\n')
            return
        _write_wav(Path(request["output_file"]))
        self._process.stdout.feed(
            (
                json.dumps(
                    {
                        "type": "result",
                        "protocol": 1,
                        "id": request["id"],
                        "ok": True,
                    }
                )
                + "\n"
            ).encode("utf-8")
        )

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        return

    async def wait_closed(self) -> None:
        return


class _ServerProcess:
    def __init__(self, behavior: str = "success") -> None:
        self.behavior = behavior
        self.returncode: int | None = None
        self.stdout = _Reader()
        self.stderr = None
        self.stdin = _Writer(self)
        self.requests: list[dict[str, object]] = []
        self.request_received = asyncio.Event()
        self.terminated = False
        self.killed = False
        self._finished = asyncio.Event()
        self.stdout.feed(b'{"type":"ready","protocol":1}\n')

    async def wait(self) -> int:
        await self._finished.wait()
        assert self.returncode is not None
        return self.returncode

    def finish(self, returncode: int) -> None:
        self.returncode = returncode
        self._finished.set()

    def terminate(self) -> None:
        self.terminated = True
        self.finish(-15)
        self.stdout.feed(b"")

    def kill(self) -> None:
        self.killed = True
        self.finish(-9)
        self.stdout.feed(b"")


def _synthesizer(
    tmp_path: Path,
    factory: Callable[..., object],
    *,
    timeout_seconds: float = 1.0,
) -> PersistentVieNeuNarrationSynthesizer:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    entrypoint = tmp_path / "vieneu-offline.py"
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    return PersistentVieNeuNarrationSynthesizer(
        model,
        entrypoint,
        python_binary=Path(sys.executable),
        process_factory=factory,  # type: ignore[arg-type]
        poll_interval_seconds=0.001,
        stop_grace_seconds=0.01,
        startup_timeout_seconds=1.0,
        timeout_seconds=timeout_seconds,
        environment={"HF_HUB_OFFLINE": "0"},
    )


def test_phase4_builder_selects_persistent_vieneu_server(tmp_path: Path) -> None:
    model = tmp_path / "model"
    codec = tmp_path / "codec"
    model.mkdir()
    codec.mkdir()
    entrypoint = tmp_path / "vieneu-offline.py"
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    verified_model = VerifiedModel(
        entry={"id": "tts-vieneu", "stage": "tts", "backend": "vieneu"},
        path=model,
        tree_sha256="a" * 64,
    )
    verified_codec = VerifiedModel(
        entry={"id": "tts-codec", "stage": "tts-support", "backend": "onnx"},
        path=codec,
        tree_sha256="b" * 64,
    )

    synthesizer = build_narration_synthesizer(
        verified_model,
        verified_codec,
        vieneu_entrypoint=entrypoint,
        python_executable=sys.executable,
    )

    assert isinstance(synthesizer, PersistentVieNeuNarrationSynthesizer)


def test_phase4_builder_finds_piper_next_to_virtualenv_python(tmp_path: Path) -> None:
    model_dir = tmp_path / "voice"
    model_dir.mkdir()
    (model_dir / "voice.onnx").write_bytes(b"model")
    (model_dir / "voice.onnx.json").write_text("{}", encoding="utf-8")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    piper = venv_bin / "piper"
    python.write_bytes(b"python")
    piper.write_bytes(b"piper")
    verified = VerifiedModel(
        entry={
            "id": "tts-piper",
            "stage": "tts",
            "backend": "piper",
            "model_file": "voice.onnx",
            "config_file": "voice.onnx.json",
        },
        path=model_dir,
        tree_sha256="c" * 64,
    )

    synthesizer = build_narration_synthesizer(
        verified,
        python_executable=python,
    )

    assert isinstance(synthesizer, PiperNarrationSynthesizer)
    assert synthesizer._binary == piper


@pytest.mark.asyncio
async def test_persistent_vieneu_reuses_one_process_and_closes_it(
    tmp_path: Path,
) -> None:
    process = _ServerProcess()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def factory(*command: str, **kwargs: object) -> _ServerProcess:
        calls.append((command, kwargs))
        return process

    synthesizer = _synthesizer(tmp_path, factory)
    first = await synthesizer.synthesize("Xin chào", tmp_path / "one.wav")
    second = await synthesizer.synthesize("Việt Nam", tmp_path / "two.wav", speed=1.1)
    await synthesizer.close()
    await synthesizer.close()

    assert first.duration_us == second.duration_us == 1_000_000
    assert len(calls) == 1
    assert len(process.requests) == 2
    command, options = calls[0]
    assert "--server" in command
    assert "--local-files-only" in command
    assert options["env"]["HF_HUB_OFFLINE"] == "1"  # type: ignore[index]
    assert process.returncode == 0
    with pytest.raises(NarrationError) as captured:
        await synthesizer.synthesize("Không chạy nữa", tmp_path / "three.wav")
    assert captured.value.code == "tts_runtime_closed"


@pytest.mark.asyncio
async def test_persistent_vieneu_cancellation_preserves_output_and_restarts(
    tmp_path: Path,
) -> None:
    hanging = _ServerProcess("hang")
    replacement = _ServerProcess()
    processes = iter((hanging, replacement))

    async def factory(*command: str, **kwargs: object) -> _ServerProcess:
        del command, kwargs
        return next(processes)

    synthesizer = _synthesizer(tmp_path, factory)
    output = tmp_path / "voice.wav"
    output.write_bytes(b"old")
    cancelled = asyncio.Event()
    task = asyncio.create_task(
        synthesizer.synthesize(
            "Xin chào",
            output,
            cancellation=cancelled.is_set,
        )
    )
    await hanging.request_received.wait()
    cancelled.set()
    with pytest.raises(NarrationError) as captured:
        await task
    assert captured.value.code == "tts_cancelled"
    assert hanging.terminated is True
    assert output.read_bytes() == b"old"
    assert not output.with_name(".voice.tts.part.wav").exists()

    result = await synthesizer.synthesize("Thử lại", output)
    assert result.path == output.resolve()
    assert len(replacement.requests) == 1
    await synthesizer.close()


@pytest.mark.asyncio
async def test_persistent_vieneu_protocol_error_is_typed_and_restarts(
    tmp_path: Path,
) -> None:
    invalid = _ServerProcess("bad_protocol")
    replacement = _ServerProcess()
    processes = iter((invalid, replacement))

    async def factory(*command: str, **kwargs: object) -> _ServerProcess:
        del command, kwargs
        return next(processes)

    synthesizer = _synthesizer(tmp_path, factory)
    output = tmp_path / "voice.wav"
    with pytest.raises(NarrationError) as captured:
        await synthesizer.synthesize("Xin chào", output)
    assert captured.value.code == "tts_protocol_error"
    assert captured.value.retryable is True
    assert invalid.terminated is True
    assert not output.exists()

    await synthesizer.synthesize("Thử lại", output)
    assert output.read_bytes().startswith(b"RIFF")
    await synthesizer.close()


@pytest.mark.asyncio
async def test_persistent_vieneu_timeout_terminates_resident_process(
    tmp_path: Path,
) -> None:
    hanging = _ServerProcess("hang")

    async def factory(*command: str, **kwargs: object) -> _ServerProcess:
        del command, kwargs
        return hanging

    synthesizer = _synthesizer(tmp_path, factory, timeout_seconds=0.005)
    with pytest.raises(NarrationError) as captured:
        await synthesizer.synthesize("Xin chào", tmp_path / "voice.wav")
    assert captured.value.code == "tts_timeout"
    assert hanging.terminated is True
    await synthesizer.close()
