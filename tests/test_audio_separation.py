from __future__ import annotations

import asyncio
import hashlib
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from dub_server.audio_separation import (
    AudioSeparationCancelled,
    AudioSeparationError,
    BackendSeparationResult,
    CinematicAudioSeparator,
    SeparationBackendRequest,
    SeparationProgress,
    TigerDnrSubprocessRunner,
)


MODEL_HASH = "a" * 64


def _write_wav(
    path: Path,
    *,
    frames: int = 48_000,
    sample_rate: int = 48_000,
    channels: int = 2,
    sample_width: int = 2,
    byte_value: int = 7,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(os.fspath(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(
            bytes([byte_value]) * frames * channels * sample_width
        )


@dataclass
class WritingRunner:
    fail: bool = False

    def __post_init__(self) -> None:
        self.requests: list[SeparationBackendRequest] = []

    async def run(
        self,
        request: SeparationBackendRequest,
        *,
        cancellation=None,
        on_progress=None,
    ) -> BackendSeparationResult:
        self.requests.append(request)
        if on_progress is not None:
            on_progress(0.25)
            on_progress(0.75)
        _write_wav(request.accompaniment_path)
        if self.fail:
            raise RuntimeError("backend failed after writing a partial artifact")
        if on_progress is not None:
            on_progress(1.0)
        return BackendSeparationResult(
            accompaniment_path=request.accompaniment_path,
            backend_name="fake-tiger",
            model_id="tiger-dnr-test",
            model_tree_sha256=MODEL_HASH,
            metrics={"gpu_ms": 125, "dialogue_excluded": True},
        )


def test_separator_atomically_publishes_accompaniment_with_hash_and_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "movie.mkv"
    source.write_bytes(b"local movie")
    output = tmp_path / "artifacts" / "accompaniment.wav"
    output.parent.mkdir()
    output.write_bytes(b"old artifact")
    runner = WritingRunner()
    progress: list[SeparationProgress] = []

    result = asyncio.run(
        CinematicAudioSeparator(runner).separate(
            source,
            output,
            on_progress=progress.append,
        )
    )

    assert result.path == output.resolve()
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(result.sha256) == 64
    assert result.backend_name == "fake-tiger"
    assert result.model_id == "tiger-dnr-test"
    assert result.model_tree_sha256 == MODEL_HASH
    assert result.metrics.duration_us == 1_000_000
    assert result.metrics.sample_rate == 48_000
    assert result.metrics.channels == 2
    assert result.metrics.sample_width_bytes == 2
    assert result.metrics.frame_count == 48_000
    assert result.metrics.source_bytes == len(b"local movie")
    assert result.metrics.output_bytes == output.stat().st_size
    assert result.metrics.backend["dialogue_excluded"] is True
    assert output.read_bytes().startswith(b"RIFF")
    assert [event.completed_permille for event in progress] == sorted(
        event.completed_permille for event in progress
    )
    assert progress[0].stage == "preparing"
    assert progress[-1] == SeparationProgress(
        stage="complete",
        completed_permille=1000,
        message_vi="Đã tách xong âm thanh nền",
    )
    assert not list(output.parent.glob("*.separation-part"))


def test_backend_failure_preserves_previous_artifact_and_cleans_staging(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source)
    output = tmp_path / "accompaniment.wav"
    output.write_bytes(b"keep this")

    with pytest.raises(AudioSeparationError) as captured:
        asyncio.run(
            CinematicAudioSeparator(WritingRunner(fail=True)).separate(
                source,
                output,
            )
        )

    assert captured.value.code == "audio_separation_failed"
    assert captured.value.retryable is True
    assert output.read_bytes() == b"keep this"
    assert not list(tmp_path.glob("*.separation-part"))


def test_invalid_backend_wav_never_replaces_previous_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    output = tmp_path / "accompaniment.wav"
    output.write_bytes(b"old")

    class InvalidRunner:
        async def run(self, request, **_kwargs):
            request.accompaniment_path.write_bytes(b"not a wav")
            return BackendSeparationResult(
                accompaniment_path=request.accompaniment_path,
                backend_name="invalid",
                model_id="invalid",
                model_tree_sha256=MODEL_HASH,
                metrics={},
            )

    with pytest.raises(AudioSeparationError) as captured:
        asyncio.run(CinematicAudioSeparator(InvalidRunner()).separate(source, output))

    assert captured.value.code == "invalid_audio_separation_output"
    assert captured.value.retryable is False
    assert output.read_bytes() == b"old"


def test_backend_cannot_publish_an_artifact_outside_its_staging_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    outside = tmp_path / "outside.wav"
    _write_wav(outside)

    class EscapingRunner:
        async def run(self, request, **_kwargs):
            return BackendSeparationResult(
                accompaniment_path=outside,
                backend_name="escaping",
                model_id="invalid",
                model_tree_sha256=MODEL_HASH,
                metrics={},
            )

    with pytest.raises(AudioSeparationError) as captured:
        asyncio.run(
            CinematicAudioSeparator(EscapingRunner()).separate(
                source,
                tmp_path / "output.wav",
            )
        )

    assert captured.value.code == "unsafe_audio_separation_output"
    assert outside.exists()
    assert not (tmp_path / "output.wav").exists()


def test_pre_cancelled_operation_does_not_call_backend(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    runner = WritingRunner()

    with pytest.raises(AudioSeparationCancelled) as captured:
        asyncio.run(
            CinematicAudioSeparator(runner).separate(
                source,
                tmp_path / "output.wav",
                cancellation=lambda: True,
            )
        )

    assert captured.value.code == "audio_separation_cancelled"
    assert captured.value.retryable is True
    assert runner.requests == []


def test_cancel_after_backend_does_not_publish_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    output = tmp_path / "output.wav"
    checks = 0

    def cancellation() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(AudioSeparationCancelled):
        asyncio.run(
            CinematicAudioSeparator(WritingRunner()).separate(
                source,
                output,
                cancellation=cancellation,
            )
        )

    assert not output.exists()
    assert not list(tmp_path.glob("*.separation-part"))


class CompletingProcess:
    def __init__(
        self,
        output_path: Path,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.returncode: int | None = None
        self._configured_returncode = returncode
        self._stderr = stderr
        self._output = output_path

    async def communicate(self):
        await asyncio.sleep(0)
        if self._configured_returncode == 0:
            _write_wav(self._output, sample_rate=44_100, frames=44_100)
        self.returncode = self._configured_returncode
        return b"", self._stderr

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class ProgressProcess(CompletingProcess):
    def __init__(self, output_path: Path, progress_path: Path) -> None:
        super().__init__(output_path)
        self._progress_path = progress_path

    async def communicate(self):
        for value in ("0.250000000\n", "0.750000000\n"):
            self._progress_path.write_text(value, encoding="ascii")
            await asyncio.sleep(0.01)
        _write_wav(self._output, sample_rate=44_100, frames=44_100)
        self.returncode = 0
        return b"", b""


def test_tiger_runner_uses_only_local_paths_and_forces_offline_environment(
    tmp_path: Path,
) -> None:
    model = tmp_path / "verified-model"
    model.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    work = tmp_path / "work"
    work.mkdir()
    output = work / "accompaniment.wav"
    commands: list[tuple[str, ...]] = []
    process_kwargs: list[dict[str, object]] = []

    async def process_factory(*command: str, **kwargs: object):
        commands.append(command)
        process_kwargs.append(kwargs)
        return CompletingProcess(Path(command[command.index("--output") + 1]))

    runner = TigerDnrSubprocessRunner(
        model_path=model,
        model_id="tiger-dnr-local",
        model_tree_sha256=MODEL_HASH,
        python_executable="/venv/bin/python",
        process_factory=process_factory,
    )
    progress: list[float] = []
    result = asyncio.run(
        runner.run(
            SeparationBackendRequest(source, work, output),
            on_progress=progress.append,
        )
    )

    command = commands[0]
    assert command[:4] == (
        "/venv/bin/python",
        "-m",
        "dub_server.audio_separation",
        "_tiger-runtime",
    )
    assert command[command.index("--input") + 1] == os.fspath(source.resolve())
    assert command[command.index("--model-path") + 1] == os.fspath(model.resolve())
    assert command[command.index("--chunk-seconds") + 1] == "120.000000"
    assert command[command.index("--context-seconds") + 1] == "4.000000"
    assert command[command.index("--batch-size") + 1] == "1"
    progress_file = Path(command[command.index("--progress-file") + 1])
    assert progress_file.parent == work.resolve()
    assert all("://" not in argument for argument in command)
    environment = process_kwargs[0]["env"]
    assert isinstance(environment, dict)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["WANDB_DISABLED"] == "true"
    assert environment["TIGER_DNR_MODEL_PATH"] == os.fspath(model.resolve())
    assert process_kwargs[0]["cwd"] == os.fspath(work.resolve())
    assert "shell" not in process_kwargs[0]
    assert progress == [0.0, 1.0]
    assert result.accompaniment_path == output.resolve()
    assert result.metrics["offline"] is True


def test_tiger_runner_reports_bounded_chunk_progress(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "source.wav"
    _write_wav(source)
    work = tmp_path / "work"
    work.mkdir()

    async def process_factory(*command: str, **_kwargs: object):
        return ProgressProcess(
            Path(command[command.index("--output") + 1]),
            Path(command[command.index("--progress-file") + 1]),
        )

    progress: list[float] = []
    runner = TigerDnrSubprocessRunner(
        model_path=model,
        model_id="tiger",
        model_tree_sha256=MODEL_HASH,
        process_factory=process_factory,
        poll_interval_seconds=0.001,
    )
    asyncio.run(
        runner.run(
            SeparationBackendRequest(source, work, work / "accompaniment.wav"),
            on_progress=progress.append,
        )
    )

    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert any(0.2 <= value <= 0.3 for value in progress)
    assert any(0.7 <= value <= 0.8 for value in progress)


@pytest.mark.parametrize(
    ("diagnostic", "expected_code", "retryable"),
    [
        (b"torch.cuda.OutOfMemoryError: CUDA out of memory", "native_oom", True),
        (
            b"ModuleNotFoundError: No module named 'look2hear'",
            "audio_separator_unavailable",
            False,
        ),
        (b"unexpected model failure", "audio_separation_failed", True),
    ],
)
def test_tiger_runner_maps_backend_failures_to_safe_typed_errors(
    tmp_path: Path,
    diagnostic: bytes,
    expected_code: str,
    retryable: bool,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "source.wav"
    _write_wav(source)
    work = tmp_path / "work"
    work.mkdir()

    async def process_factory(*command: str, **_kwargs: object):
        output = Path(command[command.index("--output") + 1])
        return CompletingProcess(output, returncode=1, stderr=diagnostic)

    runner = TigerDnrSubprocessRunner(
        model_path=model,
        model_id="tiger",
        model_tree_sha256=MODEL_HASH,
        process_factory=process_factory,
    )
    with pytest.raises(AudioSeparationError) as captured:
        asyncio.run(
            runner.run(
                SeparationBackendRequest(
                    source,
                    work,
                    work / "accompaniment.wav",
                )
            )
        )

    assert captured.value.code == expected_code
    assert captured.value.retryable is retryable
    assert diagnostic.decode() not in captured.value.message_vi


class HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._finished = asyncio.Event()

    async def communicate(self):
        await self._finished.wait()
        return b"", b""

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._finished.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._finished.set()


def test_tiger_runner_cooperative_cancellation_terminates_subprocess(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    source = tmp_path / "source.wav"
    _write_wav(source)
    work = tmp_path / "work"
    work.mkdir()
    process = HangingProcess()
    checks = 0

    async def process_factory(*_command: str, **_kwargs: object):
        return process

    def cancellation() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    runner = TigerDnrSubprocessRunner(
        model_path=model,
        model_id="tiger",
        model_tree_sha256=MODEL_HASH,
        process_factory=process_factory,
        poll_interval_seconds=0.001,
        stop_grace_seconds=0.05,
    )
    with pytest.raises(AudioSeparationCancelled):
        asyncio.run(
            runner.run(
                SeparationBackendRequest(
                    source,
                    work,
                    work / "accompaniment.wav",
                ),
                cancellation=cancellation,
            )
        )

    assert process.terminated is True
    assert process.killed is False


def test_tiger_runner_requires_an_existing_model_and_verified_tree_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(AudioSeparationError) as missing:
        TigerDnrSubprocessRunner(
            model_path=tmp_path / "missing",
            model_id="tiger",
            model_tree_sha256=MODEL_HASH,
        )
    assert missing.value.code == "model_missing"

    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError, match="SHA-256"):
        TigerDnrSubprocessRunner(
            model_path=model,
            model_id="tiger",
            model_tree_sha256="not-verified",
        )


@pytest.mark.parametrize(
    "values",
    (
        {"chunk_seconds": 11.9},
        {"chunk_seconds": 12.0, "context_seconds": 6.1},
        {"batch_size": 0},
    ),
)
def test_tiger_runner_rejects_unbounded_chunk_configuration(
    tmp_path: Path,
    values: dict[str, float | int],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError):
        TigerDnrSubprocessRunner(
            model_path=model,
            model_id="tiger",
            model_tree_sha256=MODEL_HASH,
            **values,
        )
