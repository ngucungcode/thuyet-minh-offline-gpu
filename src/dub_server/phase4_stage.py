"""Checkpointed Phase 4 cinematic dubbing pipeline.

This stage consumes the hash-authenticated Phase 3 translation artifact and a
local source movie.  It then removes the original dialogue with TIGER-DnR,
synthesizes Vietnamese speech, builds an exact 48 kHz timeline, and publishes
an MP4 containing one copied video stream and one AAC mix.  Every inference
model is resolved through the immutable local model registry before use.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from .audio_mix_export import (
    ExportProgress,
    ExportedMedia,
    FfmpegAudioMixExporter,
    MediaExportError,
    MediaExportErrorCode,
    MixSettings,
)
from .audio_separation import (
    AudioSeparationCancelled,
    AudioSeparationError,
    AudioSeparationMetrics,
    AudioSeparationResult,
    CinematicAudioSeparator,
    SeparationProgress,
    TigerDnrSubprocessRunner,
)
from .model_registry import (
    ModelNotFoundError,
    ModelRegistryError,
    ModelVerificationError,
    VerifiedModel,
    resolve_verified_model,
)
from .narration import (
    NarrationError,
    NarrationSynthesizer,
    PersistentVieNeuNarrationSynthesizer,
    PiperNarrationSynthesizer,
    SynthesizedNarration,
    TTS_SILENCE_TRIM_VERSION,
    trim_synthesized_narration_silence,
)
from .narration_artifact import (
    NarrationArtifactError,
    build_srt_cues,
    build_timing_report,
    write_srt_artifact,
    write_timing_report,
)
from .state import InvalidTransition, JobRecord, JobStage, JobStatus, StateStore
from .timing import (
    FfmpegTimingFitter,
    FittedNarrationBlock,
    NATURAL_MAX_TOTAL_SPEED,
    NarrationTimingInput,
    TimingError,
    TimingProfile,
    TimingQuality,
    build_timeline_wav,
    plan_narration_slots,
)
from .translation_artifact import (
    TranslationArtifact,
    TranslationArtifactError,
    load_translation_artifact,
)


ModelResolver = Callable[[Path, Path, str, str], VerifiedModel]
SeparatorFactory = Callable[[VerifiedModel], CinematicAudioSeparator]
SynthesizerFactory = Callable[
    [VerifiedModel, VerifiedModel | None], NarrationSynthesizer
]
TimingFitterFactory = Callable[[], FfmpegTimingFitter]
ExporterFactory = Callable[[], FfmpegAudioMixExporter]

_PHASE4_STATUSES = frozenset(
    {
        JobStatus.READY_TTS,
        JobStatus.SEPARATING,
        JobStatus.SYNTHESIZING,
        JobStatus.TIMING,
        JobStatus.MIXING,
        JobStatus.MUXING,
        JobStatus.VERIFYING,
    }
)
_SHA256 = frozenset("0123456789abcdef")


class _StageCancelled(Exception):
    pass


def _next_block_progress(
    *,
    completed: int,
    total: int,
    range_start: int,
    range_size: int,
    last_persisted: int,
) -> int | None:
    """Coalesce per-block database events to visible progress changes."""

    if total <= 0:
        return None
    mapped = range_start + round(min(max(completed, 0), total) * range_size / total)
    return mapped if mapped > last_persisted else None


def build_audio_separator(
    model: VerifiedModel,
    *,
    tiger_source_dir: Path = Path("/opt/tiger"),
    python_executable: str | Path = sys.executable,
    chunk_seconds: float = 120.0,
    context_seconds: float = 4.0,
    batch_size: int = 1,
) -> CinematicAudioSeparator:
    """Build production TIGER-DnR from a verified local model directory."""

    backend = str(model.entry.get("backend", "")).strip().casefold()
    if backend not in {"tiger", "tiger-dnr"}:
        raise ModelRegistryError("Backend tách âm thanh không được hỗ trợ")
    source_dir = Path(tiger_source_dir).resolve(strict=False)
    if not source_dir.is_dir() or not (source_dir / "look2hear").is_dir():
        raise ModelRegistryError("Không tìm thấy mã nguồn TIGER-DnR đã khóa")
    return CinematicAudioSeparator(
        TigerDnrSubprocessRunner(
            model_path=model.path,
            model_id=model.model_id,
            model_tree_sha256=model.tree_sha256,
            python_executable=python_executable,
            source_dir=source_dir,
            chunk_seconds=chunk_seconds,
            context_seconds=context_seconds,
            batch_size=batch_size,
        )
    )


def build_narration_synthesizer(
    model: VerifiedModel,
    support_model: VerifiedModel | None = None,
    *,
    vieneu_entrypoint: Path = Path("/opt/vieneu/vieneu-offline.py"),
    python_executable: str | Path = sys.executable,
    piper_binary: str | Path = "piper",
) -> NarrationSynthesizer:
    """Build VieNeu or Piper using verified local artifacts only."""

    backend = str(model.entry.get("backend", "")).strip().casefold()
    if backend in {"vieneu", "vieneu-tts"}:
        if support_model is None:
            raise ModelRegistryError("VieNeu thiếu codec cục bộ đã xác minh")
        raw_entrypoint = model.entry.get("runtime_entrypoint")
        entrypoint = (
            Path(raw_entrypoint)
            if isinstance(raw_entrypoint, str) and raw_entrypoint.strip()
            else Path(vieneu_entrypoint)
        )
        if "://" in str(entrypoint) or "\x00" in str(entrypoint):
            raise ModelRegistryError("Entrypoint VieNeu không phải file cục bộ")
        return PersistentVieNeuNarrationSynthesizer(
            model.path,
            entrypoint,
            python_binary=python_executable,
            environment={"VIENEU_CODEC_PATH": str(support_model.path)},
        )
    if backend == "piper":
        model_file = _verified_entry_file(model, "model_file", suffix=".onnx")
        raw_config = model.entry.get("config_file")
        config_file = (
            _verified_entry_file(model, "config_file", suffix=".json")
            if isinstance(raw_config, str) and raw_config
            else None
        )
        selected_piper_binary: str | Path = piper_binary
        if os.fspath(piper_binary) == "piper":
            venv_piper = Path(python_executable).absolute().parent / "piper"
            if venv_piper.is_file():
                selected_piper_binary = venv_piper
        return PiperNarrationSynthesizer(
            model_file,
            binary=selected_piper_binary,
            config_path=config_file,
        )
    raise ModelRegistryError("Backend TTS không được hỗ trợ")


def _verified_entry_file(
    model: VerifiedModel,
    field: str,
    *,
    suffix: str,
) -> Path:
    raw = model.entry.get(field)
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ModelRegistryError(f"Model TTS thiếu {field} an toàn")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ModelRegistryError(f"Model TTS khai báo {field} không an toàn")
    root = model.path.resolve(strict=True)
    candidate = model.path.joinpath(*relative.parts).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ModelRegistryError(f"Model TTS khai báo {field} không an toàn") from error
    if not candidate.is_file() or candidate.suffix.casefold() != suffix:
        raise ModelRegistryError(f"File {field} của model TTS không hợp lệ")
    return candidate


class Phase4Stage:
    """Run separation, Vietnamese TTS, timing, mixing, muxing, and verify."""

    def __init__(
        self,
        *,
        models_lock_path: Path,
        models_dir: Path,
        jobs_dir: Path,
        output_dir: Path,
        default_separation_model_id: str,
        default_tts_model_id: str,
        tts_support_model_id: str | None,
        store: StateStore,
        tiger_source_dir: Path = Path("/opt/tiger"),
        vieneu_entrypoint: Path = Path("/opt/vieneu/vieneu-offline.py"),
        python_executable: str | Path = sys.executable,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        narration_target_lufs: float = -18.0,
        accompaniment_target_lufs: float = -24.0,
        separation_chunk_seconds: float = 120.0,
        separation_context_seconds: float = 4.0,
        separation_batch_size: int = 1,
        separator_factory: SeparatorFactory | None = None,
        synthesizer_factory: SynthesizerFactory | None = None,
        timing_fitter_factory: TimingFitterFactory | None = None,
        exporter_factory: ExporterFactory | None = None,
        model_resolver: ModelResolver = resolve_verified_model,
        shutdown_requested: Callable[[], bool] | None = None,
    ) -> None:
        if not default_separation_model_id.strip() or not default_tts_model_id.strip():
            raise ValueError("ID model Phase 4 mặc định không được để trống")
        self._models_lock_path = Path(models_lock_path)
        self._models_dir = Path(models_dir)
        self._jobs_dir = Path(jobs_dir)
        self._output_dir = Path(output_dir)
        self._default_separation_model_id = default_separation_model_id.strip()
        self._default_tts_model_id = default_tts_model_id.strip()
        self._tts_support_model_id = (
            tts_support_model_id.strip()
            if isinstance(tts_support_model_id, str) and tts_support_model_id.strip()
            else None
        )
        self._store = store
        self._model_resolver = model_resolver
        self._shutdown_requested = shutdown_requested or (lambda: False)
        self._tiger_source_dir = Path(tiger_source_dir)
        self._vieneu_entrypoint = Path(vieneu_entrypoint)
        self._python_executable = python_executable
        self._ffmpeg_binary = ffmpeg_binary
        self._ffprobe_binary = ffprobe_binary
        self._separation_chunk_seconds = separation_chunk_seconds
        self._separation_context_seconds = separation_context_seconds
        self._separation_batch_size = separation_batch_size

        self._separator_factory = separator_factory or self._build_separator
        self._synthesizer_factory = synthesizer_factory or self._build_synthesizer
        self._timing_fitter_factory = timing_fitter_factory or (
            lambda: FfmpegTimingFitter(ffmpeg_binary=self._ffmpeg_binary)
        )
        self._exporter_factory = exporter_factory or (
            lambda: FfmpegAudioMixExporter(
                ffmpeg_binary=self._ffmpeg_binary,
                ffprobe_binary=self._ffprobe_binary,
                settings=MixSettings(
                    narration_lufs=narration_target_lufs,
                    accompaniment_lufs=accompaniment_target_lufs,
                ),
            )
        )

    async def run(self, job_id: str) -> JobRecord:
        job = self._store.get_job(job_id)
        if job.status is JobStatus.COMPLETED or self._cancelled(job):
            return job
        if job.status not in _PHASE4_STATUSES:
            raise InvalidTransition(
                f"Job {job.id} không ở trạng thái có thể chạy Phase 4"
            )

        try:
            translation = await self._load_translation(job)
            source_media = self._source_media(job)
            source_identity = await asyncio.to_thread(self._source_identity, source_media)
            separation_model = await self._resolve_model(
                self._separation_model_id(job), "separation"
            )
            tts_model = await self._resolve_model(self._tts_model_id(job), "tts")
            tts_support = await self._resolve_tts_support(tts_model)
            timing_profile = self._timing_profile(job)
            self._raise_if_cancelled(job.id)

            separation = await self._ensure_separation(
                job.id,
                source_media=source_media,
                source_identity=source_identity,
                model=separation_model,
            )
            separation_delta_us = abs(
                separation.metrics.duration_us - translation.result.duration_us
            )
            separation_tolerance_us = max(
                2_000_000,
                round(translation.result.duration_us * 0.02),
            )
            if separation_delta_us > separation_tolerance_us:
                raise AudioSeparationError(
                    "audio_separation_duration_mismatch",
                    "Âm thanh nền đã tách không khớp thời lượng video nguồn",
                    retryable=False,
                )
            if separation_delta_us > 100_000:
                self._store.append_warning(
                    job.id,
                    "audio_separation_duration_adjusted",
                    "Âm thanh nền ngắn hoặc dài hơn container và sẽ được căn theo timeline video",
                )
            raw_blocks = await self._ensure_narration_blocks(
                job.id,
                translation=translation,
                model=tts_model,
                support_model=tts_support,
                timing_profile=timing_profile,
            )
            fitted_blocks = await self._ensure_timing(
                job.id,
                translation=translation,
                model=tts_model,
                raw_blocks=raw_blocks,
                timing_profile=timing_profile,
            )
            return await self._export(
                job.id,
                translation=translation,
                separation=separation,
                tts_model=tts_model,
                fitted_blocks=fitted_blocks,
                source_media=source_media,
            )
        except _StageCancelled:
            return self._store.get_job(job_id)
        except AudioSeparationError as error:
            if isinstance(error, AudioSeparationCancelled):
                return self._store.get_job(job_id)
            return self._fail(job_id, error.code, error.message_vi, error.retryable)
        except NarrationError as error:
            if error.code == "tts_cancelled":
                return self._store.get_job(job_id)
            return self._fail(job_id, error.code, error.message_vi, error.retryable)
        except TimingError as error:
            if error.code == "timing_cancelled":
                return self._store.get_job(job_id)
            if error.details:
                self._update_progress(
                    job_id,
                    self._store.get_job(job_id).progress_permille,
                    {"timing_failure": error.details},
                )
            return self._fail(job_id, error.code, error.message_vi, error.retryable)
        except NarrationArtifactError as error:
            if error.code == "artifact_cancelled":
                return self._store.get_job(job_id)
            return self._fail(job_id, error.code, error.message_vi, error.retryable)
        except MediaExportError as error:
            if error.code is MediaExportErrorCode.EXPORT_CANCELLED:
                return self._store.get_job(job_id)
            return self._fail(
                job_id,
                error.code.value,
                error.message_vi,
                error.retryable,
            )
        except ModelNotFoundError:
            return self._fail(
                job_id,
                "phase4_model_not_found",
                "Model tách âm thanh hoặc TTS đã chọn không có trong danh mục cục bộ",
                False,
            )
        except (ModelVerificationError, ModelRegistryError):
            return self._fail(
                job_id,
                "phase4_model_verification_failed",
                "Model Phase 4 bị thiếu hoặc không vượt qua kiểm tra toàn vẹn",
                True,
            )
        except TranslationArtifactError:
            return self._fail(
                job_id,
                "translation_artifact_invalid",
                "Artifact bản dịch đầu vào Phase 4 không hợp lệ",
                False,
            )
        except InvalidTransition:
            current = self._store.get_job(job_id)
            if self._cancelled(current):
                return current
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._fail(
                job_id,
                "phase4_failed",
                "Không thể hoàn tất tách lời, thuyết minh và xuất video",
                True,
            )

    async def process(self, job_id: str) -> JobRecord:
        return await self.run(job_id)

    async def _load_translation(self, job: JobRecord) -> TranslationArtifact:
        raw_digest = job.details.get("translated_transcript_sha256")
        source_digest = job.details.get("source_transcript_sha256")
        if not isinstance(raw_digest, str) or not isinstance(source_digest, str):
            raise TranslationArtifactError("Thiếu SHA-256 artifact bản dịch")
        artifact = await asyncio.to_thread(
            load_translation_artifact,
            self._job_dir(job.id) / "translated-transcript.json",
            expected_sha256=raw_digest,
            expected_source_transcript_sha256=source_digest,
        )
        if artifact.result.target_language.strip().lower() != "vi":
            raise TranslationArtifactError("Phase 4 chỉ chấp nhận bản dịch tiếng Việt")
        return artifact

    async def _resolve_model(self, model_id: str, stage: str) -> VerifiedModel:
        return await asyncio.to_thread(
            self._model_resolver,
            self._models_lock_path,
            self._models_dir,
            model_id,
            stage,
        )

    async def _resolve_tts_support(
        self, tts_model: VerifiedModel
    ) -> VerifiedModel | None:
        backend = str(tts_model.entry.get("backend", "")).strip().casefold()
        if backend not in {"vieneu", "vieneu-tts"}:
            return None
        raw_id = tts_model.entry.get("support_model_id") or self._tts_support_model_id
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ModelRegistryError("Model VieNeu thiếu model codec hỗ trợ")
        return await self._resolve_model(raw_id.strip(), "tts-support")

    async def _ensure_separation(
        self,
        job_id: str,
        *,
        source_media: Path,
        source_identity: Mapping[str, Any],
        model: VerifiedModel,
    ) -> AudioSeparationResult:
        checkpoint = self._store.get_checkpoint(job_id, JobStage.SEPARATION)
        resumed = await self._load_separation_checkpoint(
            job_id,
            checkpoint.payload if checkpoint is not None else None,
            source_identity=source_identity,
            model=model,
        )
        if resumed is not None:
            return resumed

        self._set_status(
            job_id,
            JobStatus.SEPARATING,
            JobStage.SEPARATION,
            650,
            force_reset=True,
            detail_updates={
                "separation_model_id": model.model_id,
                "phase4_step": "separation",
            },
        )
        separator = self._separator_factory(model)

        def progress(value: SeparationProgress) -> None:
            if self._is_cancel_requested(job_id):
                return
            mapped = 650 + round(value.completed_permille * 85 / 1000)
            self._update_progress(
                job_id,
                mapped,
                {
                    "phase4_step": "separation",
                    "phase4_message": value.message_vi,
                    "separation_progress_permille": value.completed_permille,
                },
            )

        result = await separator.separate(
            source_media,
            self._job_dir(job_id) / "accompaniment.wav",
            cancellation=lambda: self._is_cancel_requested(job_id),
            on_progress=progress,
        )
        self._raise_if_cancelled(job_id)
        self._store.save_checkpoint(
            job_id,
            JobStage.SEPARATION,
            {
                "schema_version": 1,
                "completed": True,
                "source": dict(source_identity),
                "model_id": result.model_id,
                "model_tree_sha256": result.model_tree_sha256,
                "backend": result.backend_name,
                "artifact_path": str(result.path),
                "artifact_sha256": result.sha256,
                "duration_us": result.metrics.duration_us,
                "sample_rate": result.metrics.sample_rate,
                "channels": result.metrics.channels,
                "sample_width_bytes": result.metrics.sample_width_bytes,
                "frame_count": result.metrics.frame_count,
                "metrics": {
                    "elapsed_ms": result.metrics.elapsed_ms,
                    "real_time_factor": result.metrics.real_time_factor,
                    "source_bytes": result.metrics.source_bytes,
                    "output_bytes": result.metrics.output_bytes,
                    "backend": dict(result.metrics.backend),
                },
            },
        )
        self._set_status(
            job_id,
            JobStatus.SYNTHESIZING,
            JobStage.TTS,
            735,
            detail_updates={
                "phase4_step": "tts",
                "accompaniment_path": str(result.path),
                "accompaniment_sha256": result.sha256,
            },
        )
        return result

    async def _load_separation_checkpoint(
        self,
        job_id: str,
        payload: Mapping[str, Any] | None,
        *,
        source_identity: Mapping[str, Any],
        model: VerifiedModel,
    ) -> AudioSeparationResult | None:
        if not payload or payload.get("completed") is not True:
            return None
        if (
            payload.get("source") != dict(source_identity)
            or payload.get("model_id") != model.model_id
            or payload.get("model_tree_sha256") != model.tree_sha256
        ):
            return None
        raw_path = payload.get("artifact_path")
        raw_digest = payload.get("artifact_sha256")
        if not isinstance(raw_path, str) or not self._valid_sha(raw_digest):
            return None
        path = Path(raw_path).resolve(strict=False)
        if path != (self._job_dir(job_id) / "accompaniment.wav"):
            return None
        try:
            digest = await asyncio.to_thread(self._file_sha256, path)
            values = {
                key: int(payload[key])
                for key in (
                    "duration_us",
                    "sample_rate",
                    "channels",
                    "sample_width_bytes",
                    "frame_count",
                )
            }
            stat = path.stat()
        except (OSError, KeyError, TypeError, ValueError):
            return None
        if digest != raw_digest or stat.st_size <= 0:
            return None
        raw_metrics = payload.get("metrics")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
        return AudioSeparationResult(
            accompaniment_path=path,
            checksum_sha256=digest,
            model_id=model.model_id,
            model_tree_sha256=model.tree_sha256,
            backend_name=str(payload.get("backend") or "tiger-dnr"),
            metrics=AudioSeparationMetrics(
                elapsed_ms=int(metrics.get("elapsed_ms", 0)),
                duration_us=values["duration_us"],
                real_time_factor=float(metrics.get("real_time_factor", 0.0)),
                source_bytes=int(metrics.get("source_bytes", 0)),
                output_bytes=stat.st_size,
                sample_rate=values["sample_rate"],
                channels=values["channels"],
                sample_width_bytes=values["sample_width_bytes"],
                frame_count=values["frame_count"],
                backend=(
                    dict(metrics.get("backend"))
                    if isinstance(metrics.get("backend"), dict)
                    else {}
                ),
            ),
        )

    async def _ensure_narration_blocks(
        self,
        job_id: str,
        *,
        translation: TranslationArtifact,
        model: VerifiedModel,
        support_model: VerifiedModel | None,
        timing_profile: TimingProfile,
    ) -> list[dict[str, Any]]:
        checkpoint = self._store.get_checkpoint(job_id, JobStage.TTS)
        payload = self._valid_tts_checkpoint(
            checkpoint.payload if checkpoint is not None else None,
            translation=translation,
            model=model,
            support_model=support_model,
            timing_profile=timing_profile,
        )
        if payload is None:
            payload = {
                "schema_version": 1,
                "completed": False,
                "translation_sha256": translation.sha256,
                "model_id": model.model_id,
                "model_tree_sha256": model.tree_sha256,
                "backend": str(model.entry.get("backend") or "unknown"),
                "support_model_id": (
                    support_model.model_id if support_model is not None else None
                ),
                "support_model_tree_sha256": (
                    support_model.tree_sha256 if support_model is not None else None
                ),
                "timing_profile": timing_profile.value,
                "silence_trim_version": TTS_SILENCE_TRIM_VERSION,
                "block_count": len(translation.result.segments),
                "blocks": [],
            }
            self._store.save_checkpoint(job_id, JobStage.TTS, payload)
        elif "timing_profile" not in payload:
            payload["timing_profile"] = timing_profile.value
            self._store.save_checkpoint(job_id, JobStage.TTS, payload)

        running = self._set_status(
            job_id,
            JobStatus.SYNTHESIZING,
            JobStage.TTS,
            735,
            force_reset=True,
            detail_updates={
                "phase4_step": "tts",
                "tts_model_id": model.model_id,
                "timing_profile": timing_profile.value,
                "tts_completed_blocks": len(payload["blocks"]),
                "tts_block_count": len(translation.result.segments),
            },
        )
        last_persisted_progress = running.progress_permille
        blocks_by_ordinal = {
            int(item["ordinal"]): dict(item)
            for item in payload["blocks"]
            if (
                isinstance(item, dict)
                and isinstance(item.get("ordinal"), int)
                and 0 <= int(item["ordinal"]) < len(translation.result.segments)
            )
        }
        synthesizer: NarrationSynthesizer | None = None
        try:
            if len(blocks_by_ordinal) < len(translation.result.segments):
                synthesizer = self._synthesizer_factory(model, support_model)
            for ordinal, segment in enumerate(translation.result.segments):
                self._raise_if_cancelled(job_id)
                existing = blocks_by_ordinal.get(ordinal)
                if existing is not None and await self._valid_raw_block(
                    existing,
                    job_id=job_id,
                    ordinal=ordinal,
                    text=segment.translated_text,
                    start_us=segment.start_us,
                    end_us=segment.end_us,
                ):
                    continue
                if synthesizer is None:
                    synthesizer = self._synthesizer_factory(model, support_model)
                record = await self._synthesize_block(
                    synthesizer,
                    job_id=job_id,
                    ordinal=ordinal,
                    text=segment.translated_text,
                    start_us=segment.start_us,
                    end_us=segment.end_us,
                    timing_profile=timing_profile,
                )
                blocks_by_ordinal[ordinal] = record
                payload["blocks"] = [
                    blocks_by_ordinal[index] for index in sorted(blocks_by_ordinal)
                ]
                payload["completed"] = len(blocks_by_ordinal) == len(
                    translation.result.segments
                )
                self._store.save_checkpoint(job_id, JobStage.TTS, payload)
                mapped = _next_block_progress(
                    completed=len(blocks_by_ordinal),
                    total=len(translation.result.segments),
                    range_start=735,
                    range_size=115,
                    last_persisted=last_persisted_progress,
                )
                if mapped is not None:
                    self._update_progress(
                        job_id,
                        mapped,
                        {
                            "phase4_step": "tts",
                            "tts_completed_blocks": len(blocks_by_ordinal),
                            "tts_block_count": len(translation.result.segments),
                        },
                    )
                    last_persisted_progress = mapped
        finally:
            if synthesizer is not None:
                await synthesizer.close()
        if len(blocks_by_ordinal) != len(translation.result.segments):
            raise NarrationError(
                "tts_checkpoint_incomplete",
                "Checkpoint TTS chưa hoàn tất tất cả khối thuyết minh",
                retryable=True,
            )
        payload["completed"] = True
        self._store.save_checkpoint(job_id, JobStage.TTS, payload)
        self._raise_if_cancelled(job_id)
        self._set_status(
            job_id,
            JobStatus.TIMING,
            JobStage.TIMING,
            850,
            detail_updates={
                "phase4_step": "timing",
                "tts_completed_blocks": len(blocks_by_ordinal),
            },
        )
        return [blocks_by_ordinal[index] for index in range(len(blocks_by_ordinal))]

    async def _synthesize_block(
        self,
        synthesizer: NarrationSynthesizer,
        *,
        job_id: str,
        ordinal: int,
        text: str,
        start_us: int,
        end_us: int,
        timing_profile: TimingProfile,
    ) -> dict[str, Any]:
        block_dir = self._job_dir(job_id) / "tts"
        probe_path = block_dir / f"block-{ordinal:05d}.probe.wav"
        final_path = block_dir / f"block-{ordinal:05d}.wav"
        with suppress(OSError):
            probe_path.unlink(missing_ok=True)
        probe = await synthesizer.synthesize(
            text,
            probe_path,
            speed=1.0,
            cancellation=lambda: self._is_cancel_requested(job_id),
        )
        self._raise_if_cancelled(job_id)
        probe = await asyncio.to_thread(trim_synthesized_narration_silence, probe)
        target_us = end_us - start_us
        native_speed = (
            min(1.20, max(0.90, probe.duration_us / target_us))
            if timing_profile is TimingProfile.STRICT
            else 1.0
        )
        final: SynthesizedNarration
        if abs(native_speed - 1.0) > 0.001:
            final = await synthesizer.synthesize(
                text,
                final_path,
                speed=native_speed,
                cancellation=lambda: self._is_cancel_requested(job_id),
            )
            self._raise_if_cancelled(job_id)
            final = await asyncio.to_thread(trim_synthesized_narration_silence, final)
            with suppress(OSError):
                probe_path.unlink(missing_ok=True)
        else:
            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(probe.path, final_path)
            except OSError as error:
                raise NarrationError(
                    "tts_output_unavailable",
                    "Không thể lưu khối âm thanh TTS",
                    retryable=True,
                ) from error
            final = SynthesizedNarration(
                path=final_path,
                text=probe.text,
                sample_rate=probe.sample_rate,
                channels=probe.channels,
                sample_width_bytes=probe.sample_width_bytes,
                frame_count=probe.frame_count,
                duration_us=probe.duration_us,
                native_speed=1.0,
                backend=probe.backend,
            )
        self._raise_if_cancelled(job_id)
        return {
            "ordinal": ordinal,
            "start_us": start_us,
            "end_us": end_us,
            "text_sha256": self._text_sha256(text),
            "path": str(final.path),
            "sha256": await asyncio.to_thread(self._file_sha256, final.path),
            "duration_us": final.duration_us,
            "sample_rate": final.sample_rate,
            "channels": final.channels,
            "sample_width_bytes": final.sample_width_bytes,
            "frame_count": final.frame_count,
            "native_speed": native_speed,
            "backend": final.backend,
        }

    def _valid_tts_checkpoint(
        self,
        payload: Mapping[str, Any] | None,
        *,
        translation: TranslationArtifact,
        model: VerifiedModel,
        support_model: VerifiedModel | None,
        timing_profile: TimingProfile,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, Mapping):
            return None
        expected_support_id = support_model.model_id if support_model is not None else None
        expected_support_sha = (
            support_model.tree_sha256 if support_model is not None else None
        )
        if (
            payload.get("translation_sha256") != translation.sha256
            or payload.get("model_id") != model.model_id
            or payload.get("model_tree_sha256") != model.tree_sha256
            or payload.get("support_model_id") != expected_support_id
            or payload.get("support_model_tree_sha256") != expected_support_sha
            or payload.get("block_count") != len(translation.result.segments)
            or not isinstance(payload.get("blocks"), list)
            or not self._checkpoint_profile_matches(payload, timing_profile)
            or not self._checkpoint_trim_matches(payload, timing_profile)
        ):
            return None
        return dict(payload)

    async def _valid_raw_block(
        self,
        record: Mapping[str, Any],
        *,
        job_id: str,
        ordinal: int,
        text: str,
        start_us: int,
        end_us: int,
    ) -> bool:
        if (
            record.get("ordinal") != ordinal
            or record.get("start_us") != start_us
            or record.get("end_us") != end_us
            or record.get("text_sha256") != self._text_sha256(text)
            or not self._valid_sha(record.get("sha256"))
            or not isinstance(record.get("path"), str)
        ):
            return False
        path = Path(str(record["path"])).resolve(strict=False)
        expected = self._job_dir(job_id) / "tts" / f"block-{ordinal:05d}.wav"
        if path != expected:
            return False
        try:
            return await asyncio.to_thread(self._file_sha256, path) == record["sha256"]
        except OSError:
            return False

    async def _ensure_timing(
        self,
        job_id: str,
        *,
        translation: TranslationArtifact,
        model: VerifiedModel,
        raw_blocks: Sequence[Mapping[str, Any]],
        timing_profile: TimingProfile,
    ) -> list[FittedNarrationBlock]:
        planning_inputs = tuple(
            NarrationTimingInput(
                start_us=segment.start_us,
                end_us=segment.end_us,
                source_duration_us=int(raw["duration_us"]),
                native_speed=float(raw["native_speed"]),
                source_frame_count=int(raw["frame_count"]),
                source_sample_rate=int(raw["sample_rate"]),
            )
            for segment, raw in zip(
                translation.result.segments, raw_blocks, strict=True
            )
        )
        planned_slots = await asyncio.to_thread(
            plan_narration_slots,
            planning_inputs,
            duration_us=translation.result.duration_us,
            profile=timing_profile,
        )
        checkpoint = self._store.get_checkpoint(job_id, JobStage.TIMING)
        payload = self._valid_timing_checkpoint(
            checkpoint.payload if checkpoint is not None else None,
            translation=translation,
            model=model,
            block_count=len(raw_blocks),
            timing_profile=timing_profile,
        )
        if payload is None:
            payload = {
                "schema_version": 1,
                "completed": False,
                "translation_sha256": translation.sha256,
                "tts_model_id": model.model_id,
                "tts_model_tree_sha256": model.tree_sha256,
                "timing_profile": timing_profile.value,
                "silence_trim_version": TTS_SILENCE_TRIM_VERSION,
                "block_count": len(raw_blocks),
                "blocks": [],
            }
        elif "timing_profile" not in payload:
            payload["timing_profile"] = timing_profile.value
            self._store.save_checkpoint(job_id, JobStage.TIMING, payload)
        running = self._set_status(
            job_id,
            JobStatus.TIMING,
            JobStage.TIMING,
            850,
            force_reset=True,
            detail_updates={"phase4_step": "timing"},
        )
        last_persisted_progress = running.progress_permille
        fitted_by_ordinal: dict[int, FittedNarrationBlock] = {}
        records_by_ordinal: dict[int, dict[str, Any]] = {}
        for raw in payload.get("blocks", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("ordinal"), int):
                continue
            ordinal = int(raw["ordinal"])
            if not 0 <= ordinal < len(translation.result.segments):
                continue
            block = await self._load_fitted_block(
                raw,
                job_id=job_id,
                ordinal=ordinal,
            )
            if block is not None:
                segment = translation.result.segments[ordinal]
                planned = planned_slots[ordinal]
                if (
                    block.start_us != planned.start_us
                    or block.end_us != planned.end_us
                    or block.text != segment.translated_text
                ):
                    continue
                fitted_by_ordinal[ordinal] = block
                records_by_ordinal[ordinal] = dict(raw)

        fitter = self._timing_fitter_factory()
        for ordinal, (segment, raw, planned) in enumerate(
            zip(
                translation.result.segments,
                raw_blocks,
                planned_slots,
                strict=True,
            )
        ):
            self._raise_if_cancelled(job_id)
            if ordinal in fitted_by_ordinal:
                continue
            fitted = await fitter.fit(
                Path(str(raw["path"])),
                self._job_dir(job_id) / "timing" / f"block-{ordinal:05d}.wav",
                start_us=planned.start_us,
                end_us=planned.end_us,
                text=segment.translated_text,
                native_speed=float(raw["native_speed"]),
                maximum_total_speed=(
                    NATURAL_MAX_TOTAL_SPEED
                    if timing_profile is TimingProfile.NATURAL
                    else None
                ),
                cancellation=lambda: self._is_cancel_requested(job_id),
            )
            fitted_by_ordinal[ordinal] = fitted
            records_by_ordinal[ordinal] = await self._fitted_record(ordinal, fitted)
            payload["blocks"] = [
                records_by_ordinal[index] for index in sorted(records_by_ordinal)
            ]
            # Any rebuilt block invalidates the previously sealed aggregate
            # timeline/SRT/report, even when those files still match their old
            # hashes. Rebuild all aggregate artifacts from the current blocks.
            payload["completed"] = False
            for key in (
                "timeline_path",
                "timeline_sha256",
                "timeline_frame_count",
                "timing_report_path",
                "timing_report_sha256",
                "srt_path",
                "srt_sha256",
            ):
                payload.pop(key, None)
            self._store.save_checkpoint(job_id, JobStage.TIMING, payload)
            mapped = _next_block_progress(
                completed=len(fitted_by_ordinal),
                total=len(raw_blocks),
                range_start=850,
                range_size=50,
                last_persisted=last_persisted_progress,
            )
            if mapped is not None:
                self._update_progress(
                    job_id,
                    mapped,
                    {
                        "phase4_step": "timing",
                        "timing_completed_blocks": len(fitted_by_ordinal),
                        "timing_block_count": len(raw_blocks),
                        "timing_profile": timing_profile.value,
                    },
                )
                last_persisted_progress = mapped

        fitted_blocks = [fitted_by_ordinal[index] for index in range(len(raw_blocks))]
        artifacts_valid = await self._valid_timeline_artifacts(payload, job_id=job_id)
        if not artifacts_valid:
            timeline = await asyncio.to_thread(
                build_timeline_wav,
                fitted_blocks,
                self._job_dir(job_id) / "narration-48k.wav",
                duration_us=translation.result.duration_us,
                cancellation=lambda: self._is_cancel_requested(job_id),
            )
            report = build_timing_report(
                fitted_blocks,
                duration_us=translation.result.duration_us,
                tts_model_id=model.model_id,
                tts_backend=str(model.entry.get("backend") or "unknown"),
            )
            timing_artifact = await asyncio.to_thread(
                write_timing_report,
                self._job_dir(job_id) / "timing-report.json",
                report,
                cancellation=lambda: self._is_cancel_requested(job_id),
            )
            srt_artifact = await asyncio.to_thread(
                write_srt_artifact,
                self._output_dir / f"{job_id}.vi.srt",
                build_srt_cues(fitted_blocks),
                cancellation=lambda: self._is_cancel_requested(job_id),
            )
            payload.update(
                {
                    "completed": True,
                    "timeline_path": str(timeline.path),
                    "timeline_sha256": await asyncio.to_thread(
                        self._file_sha256, timeline.path
                    ),
                    "timeline_frame_count": timeline.frame_count,
                    "timing_report_path": str(timing_artifact.path),
                    "timing_report_sha256": timing_artifact.sha256,
                    "srt_path": str(srt_artifact.path),
                    "srt_sha256": srt_artifact.sha256,
                }
            )
            self._store.save_checkpoint(job_id, JobStage.TIMING, payload)
        self._raise_if_cancelled(job_id)
        quality_counts = {
            value.value: sum(block.quality is value for block in fitted_blocks)
            for value in TimingQuality
        }
        self._set_status(
            job_id,
            JobStatus.MIXING,
            JobStage.MIX,
            900,
            detail_updates={
                "phase4_step": "mix",
                "narration_timeline_path": payload["timeline_path"],
                "timing_report_path": payload["timing_report_path"],
                "vietnamese_srt_path": payload["srt_path"],
                "timing_profile": timing_profile.value,
                "timing_quality": quality_counts,
            },
        )
        return fitted_blocks

    def _valid_timing_checkpoint(
        self,
        payload: Mapping[str, Any] | None,
        *,
        translation: TranslationArtifact,
        model: VerifiedModel,
        block_count: int,
        timing_profile: TimingProfile,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, Mapping):
            return None
        if (
            payload.get("translation_sha256") != translation.sha256
            or payload.get("tts_model_id") != model.model_id
            or payload.get("tts_model_tree_sha256") != model.tree_sha256
            or payload.get("block_count") != block_count
            or not isinstance(payload.get("blocks"), list)
            or not self._checkpoint_profile_matches(payload, timing_profile)
            or not self._checkpoint_trim_matches(payload, timing_profile)
        ):
            return None
        return dict(payload)

    async def _load_fitted_block(
        self,
        record: Mapping[str, Any],
        *,
        job_id: str,
        ordinal: int,
    ) -> FittedNarrationBlock | None:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not self._valid_sha(record.get("sha256")):
            return None
        path = Path(raw_path).resolve(strict=False)
        if path != self._job_dir(job_id) / "timing" / f"block-{ordinal:05d}.wav":
            return None
        try:
            if await asyncio.to_thread(self._file_sha256, path) != record["sha256"]:
                return None
            return FittedNarrationBlock(
                path=path,
                start_us=int(record["start_us"]),
                end_us=int(record["end_us"]),
                text=str(record["text"]),
                source_duration_us=int(record["source_duration_us"]),
                target_frame_count=int(record["target_frame_count"]),
                output_frame_count=int(record["output_frame_count"]),
                native_speed=float(record["native_speed"]),
                atempo_speed=float(record["atempo_speed"]),
                total_speed=float(record["total_speed"]),
                padded_frame_count=int(record["padded_frame_count"]),
                quality=TimingQuality(str(record["quality"])),
            )
        except (OSError, KeyError, TypeError, ValueError, TimingError):
            return None

    async def _fitted_record(
        self, ordinal: int, block: FittedNarrationBlock
    ) -> dict[str, Any]:
        return {
            "ordinal": ordinal,
            "path": str(block.path),
            "sha256": await asyncio.to_thread(self._file_sha256, block.path),
            "start_us": block.start_us,
            "end_us": block.end_us,
            "text": block.text,
            "source_duration_us": block.source_duration_us,
            "target_frame_count": block.target_frame_count,
            "output_frame_count": block.output_frame_count,
            "native_speed": block.native_speed,
            "atempo_speed": block.atempo_speed,
            "total_speed": block.total_speed,
            "padded_frame_count": block.padded_frame_count,
            "quality": block.quality.value,
        }

    async def _valid_timeline_artifacts(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
    ) -> bool:
        fields = (
            (
                "timeline_path",
                "timeline_sha256",
                self._job_dir(job_id) / "narration-48k.wav",
            ),
            (
                "timing_report_path",
                "timing_report_sha256",
                self._job_dir(job_id) / "timing-report.json",
            ),
            (
                "srt_path",
                "srt_sha256",
                self._output_dir.resolve(strict=False) / f"{job_id}.vi.srt",
            ),
        )
        if payload.get("completed") is not True:
            return False
        for path_key, digest_key, expected_path in fields:
            raw_path = payload.get(path_key)
            raw_digest = payload.get(digest_key)
            if not isinstance(raw_path, str) or not self._valid_sha(raw_digest):
                return False
            try:
                path = Path(raw_path).resolve(strict=False)
                if path != expected_path or (
                    await asyncio.to_thread(self._file_sha256, path) != raw_digest
                ):
                    return False
            except OSError:
                return False
        return True

    async def _load_export_checkpoint(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        expected_duration_us: int,
        translation_sha256: str,
        separation_sha256: str,
        timeline_sha256: str,
        tts_model_tree_sha256: str,
    ) -> ExportedMedia | None:
        """Reuse only an unchanged MP4 previously verified by the exporter."""

        if payload.get("completed") is not True:
            return None
        bindings = {
            "expected_duration_us": expected_duration_us,
            "translation_sha256": translation_sha256,
            "separation_sha256": separation_sha256,
            "timeline_sha256": timeline_sha256,
            "tts_model_tree_sha256": tts_model_tree_sha256,
            "video_track_count": 1,
            "audio_track_count": 1,
            "audio_codec": "aac",
        }
        if any(payload.get(key) != value for key, value in bindings.items()):
            return None
        raw_path = payload.get("output_path")
        raw_digest = payload.get("output_sha256")
        expected_path = (
            self._output_dir.resolve(strict=False) / f"{job_id}.mp4"
        ).absolute()
        if (
            not isinstance(raw_path, str)
            or Path(raw_path).absolute() != expected_path
            or not self._valid_sha(raw_digest)
            or expected_path.is_symlink()
        ):
            return None
        try:
            size_bytes = int(payload["size_bytes"])
            duration_us = int(payload["duration_us"])
            video_start_us = int(payload["video_start_us"])
            audio_start_us = int(payload["audio_start_us"])
            if (
                size_bytes <= 0
                or duration_us <= 0
                or video_start_us < 0
                or audio_start_us < 0
                or expected_path.stat().st_size != size_bytes
                or await asyncio.to_thread(self._file_sha256, expected_path)
                != raw_digest
                or expected_path.is_symlink()
            ):
                return None
        except (OSError, KeyError, TypeError, ValueError):
            return None
        return ExportedMedia(
            path=expected_path,
            duration_us=duration_us,
            video_start_us=video_start_us,
            audio_start_us=audio_start_us,
            audio_codec="aac",
            size_bytes=size_bytes,
        )

    async def _export(
        self,
        job_id: str,
        *,
        translation: TranslationArtifact,
        separation: AudioSeparationResult,
        tts_model: VerifiedModel,
        fitted_blocks: Sequence[FittedNarrationBlock],
        source_media: Path,
    ) -> JobRecord:
        timing = self._store.get_checkpoint(job_id, JobStage.TIMING)
        if timing is None or timing.payload.get("completed") is not True:
            raise TimingError(
                "timing_checkpoint_incomplete",
                "Checkpoint timeline chưa hoàn tất",
                retryable=True,
            )
        checkpoint = self._store.get_checkpoint(job_id, JobStage.EXPORT)
        exported = (
            None
            if checkpoint is None
            else await self._load_export_checkpoint(
                checkpoint.payload,
                job_id=job_id,
                expected_duration_us=translation.result.duration_us,
                translation_sha256=translation.sha256,
                separation_sha256=separation.checksum_sha256,
                timeline_sha256=str(timing.payload["timeline_sha256"]),
                tts_model_tree_sha256=tts_model.tree_sha256,
            )
        )

        def progress(value: ExportProgress) -> None:
            if self._is_cancel_requested(job_id):
                return
            self._update_progress(
                job_id,
                910 + round(value.fraction * 70),
                {
                    "phase4_step": "export",
                    "export_processed_us": value.processed_us,
                    "export_duration_us": value.duration_us,
                },
            )

        if exported is None:
            current = self._store.get_job(job_id)
            self._set_status(
                job_id,
                JobStatus.MUXING,
                JobStage.EXPORT,
                910,
                detail_updates={"phase4_step": "export"},
                force_reset=current.status is JobStatus.VERIFYING,
            )
            exporter = self._exporter_factory()
            output_path = self._output_dir / f"{job_id}.mp4"
            exported = await exporter.export(
                source_media,
                separation.path,
                Path(str(timing.payload["timeline_path"])),
                output_path,
                expected_duration_us=translation.result.duration_us,
                cancellation=lambda: self._is_cancel_requested(job_id),
                on_progress=progress,
            )
            self._raise_if_cancelled(job_id)
            if abs(
                exported.duration_us - translation.result.duration_us
            ) > 100_000:
                warnings = self._store.get_job(job_id).details.get("warnings", [])
                if not any(
                    isinstance(item, Mapping)
                    and item.get("code") == "output_duration_adjusted_to_video"
                    for item in (warnings if isinstance(warnings, list) else [])
                ):
                    self._store.append_warning(
                        job_id,
                        "output_duration_adjusted_to_video",
                        "Timeline đầu ra đã được căn theo luồng hình; phần tiếng thiếu "
                        "được đệm im lặng hoặc phần vượt quá hình được loại bỏ",
                    )
        self._set_status(
            job_id,
            JobStatus.VERIFYING,
            JobStage.VERIFY,
            985,
            detail_updates={"phase4_step": "verify"},
            force_reset=True,
        )
        result = await self._result_payload(
            job_id,
            exported=exported,
            translation=translation,
            separation=separation,
            tts_model=tts_model,
            fitted_blocks=fitted_blocks,
            timing_payload=timing.payload,
        )
        self._store.save_checkpoint(
            job_id,
            JobStage.EXPORT,
            {
                "schema_version": 1,
                "completed": True,
                "output_path": result["video_path"],
                "output_sha256": result["video_sha256"],
                "duration_us": result["duration_us"],
                "expected_duration_us": translation.result.duration_us,
                "size_bytes": result["size_bytes"],
                "video_start_us": result["video_start_us"],
                "audio_start_us": result["audio_start_us"],
                "video_track_count": 1,
                "audio_track_count": 1,
                "audio_codec": "aac",
                "translation_sha256": translation.sha256,
                "separation_sha256": separation.checksum_sha256,
                "timeline_sha256": str(timing.payload["timeline_sha256"]),
                "tts_model_tree_sha256": tts_model.tree_sha256,
            },
        )
        current = self._store.get_job(job_id)
        details = {
            **current.details,
            "phase4_step": "completed",
            "output_path": result["video_path"],
            "output_sha256": result["video_sha256"],
        }
        return self._store.update_status(
            job_id,
            JobStatus.COMPLETED,
            expected_status=JobStatus.VERIFYING,
            stage=JobStage.DONE,
            progress_permille=1000,
            details=details,
            result=result,
        )

    async def _result_payload(
        self,
        job_id: str,
        *,
        exported: ExportedMedia,
        translation: TranslationArtifact,
        separation: AudioSeparationResult,
        tts_model: VerifiedModel,
        fitted_blocks: Sequence[FittedNarrationBlock],
        timing_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        quality_counts = {
            value.value: sum(block.quality is value for block in fitted_blocks)
            for value in TimingQuality
        }
        return {
            "video_path": str(exported.path),
            "video_sha256": await asyncio.to_thread(
                self._file_sha256, exported.path
            ),
            "size_bytes": exported.size_bytes,
            "duration_us": exported.duration_us,
            "video_track_count": 1,
            "audio_track_count": 1,
            "audio_codec": exported.audio_codec,
            "audio_start_us": exported.audio_start_us,
            "video_start_us": exported.video_start_us,
            "srt_path": str(timing_payload["srt_path"]),
            "srt_sha256": str(timing_payload["srt_sha256"]),
            "timing_report_path": str(timing_payload["timing_report_path"]),
            "timing_report_sha256": str(timing_payload["timing_report_sha256"]),
            "translation_sha256": translation.sha256,
            "separation_model_id": separation.model_id,
            "separation_model_tree_sha256": separation.model_tree_sha256,
            "tts_model_id": tts_model.model_id,
            "tts_model_tree_sha256": tts_model.tree_sha256,
            "timing_profile": str(timing_payload["timing_profile"]),
            "timing_quality": quality_counts,
            "original_dialogue_removed": True,
            "music_and_effects_preserved": True,
            "job_id": job_id,
        }

    def _set_status(
        self,
        job_id: str,
        status: JobStatus,
        stage: JobStage,
        minimum_progress: int,
        *,
        detail_updates: Mapping[str, Any] | None = None,
        force_reset: bool = False,
    ) -> JobRecord:
        current = self._store.get_job(job_id)
        if self._cancelled(current):
            raise _StageCancelled()
        details = {**current.details, **dict(detail_updates or {})}
        if current.status is status:
            return self._store.update_progress(
                job_id,
                max(current.progress_permille, minimum_progress),
                details=details,
            )
        return self._store.update_status(
            job_id,
            status,
            expected_status=current.status,
            stage=stage,
            progress_permille=max(current.progress_permille, minimum_progress),
            details=details,
            force=force_reset,
        )

    def _update_progress(
        self,
        job_id: str,
        value: int,
        detail_updates: Mapping[str, Any],
    ) -> None:
        current = self._store.get_job(job_id)
        if self._cancelled(current):
            return
        self._store.update_progress(
            job_id,
            max(current.progress_permille, min(value, 999)),
            details={**current.details, **dict(detail_updates)},
        )

    def _fail(
        self,
        job_id: str,
        code: str,
        message_vi: str,
        retryable: bool,
    ) -> JobRecord:
        current = self._store.get_job(job_id)
        if self._cancelled(current) or current.status is JobStatus.COMPLETED:
            return current
        return self._store.update_status(
            job_id,
            JobStatus.FAILED,
            expected_status=current.status,
            stage=current.stage,
            details=current.details,
            error_code=code,
            error_message=message_vi,
            retryable=retryable,
        )

    def _build_separator(self, model: VerifiedModel) -> CinematicAudioSeparator:
        return build_audio_separator(
            model,
            tiger_source_dir=self._tiger_source_dir,
            python_executable=self._python_executable,
            chunk_seconds=self._separation_chunk_seconds,
            context_seconds=self._separation_context_seconds,
            batch_size=self._separation_batch_size,
        )

    def _build_synthesizer(
        self,
        model: VerifiedModel,
        support_model: VerifiedModel | None,
    ) -> NarrationSynthesizer:
        return build_narration_synthesizer(
            model,
            support_model,
            vieneu_entrypoint=self._vieneu_entrypoint,
            python_executable=self._python_executable,
        )

    def _source_media(self, job: JobRecord) -> Path:
        raw = job.details.get("source_media_path")
        if not isinstance(raw, str) or "://" in raw or "\x00" in raw:
            raise TranslationArtifactError("Thiếu file media nguồn cục bộ")
        try:
            path = Path(raw).resolve(strict=True)
        except OSError as error:
            raise TranslationArtifactError("Không tìm thấy file media nguồn") from error
        if not path.is_file():
            raise TranslationArtifactError("Không tìm thấy file media nguồn")
        return path

    @staticmethod
    def _source_identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _separation_model_id(self, job: JobRecord) -> str:
        return self._selected_model(job, "separation", self._default_separation_model_id)

    def _tts_model_id(self, job: JobRecord) -> str:
        return self._selected_model(job, "tts", self._default_tts_model_id)

    @staticmethod
    def _timing_profile(job: JobRecord) -> TimingProfile:
        raw = job.spec.get("timing_profile")
        # Jobs created before natural timing existed had strict per-subtitle
        # semantics.  Treating a missing field as strict preserves their sealed
        # checkpoints and makes upgrades safe.
        if raw is None:
            return TimingProfile.STRICT
        try:
            return TimingProfile(raw)
        except (TypeError, ValueError) as error:
            raise TimingError(
                "timing_profile_invalid",
                "Chế độ khớp thời lượng của job không hợp lệ",
                retryable=False,
            ) from error

    @staticmethod
    def _checkpoint_profile_matches(
        payload: Mapping[str, Any], timing_profile: TimingProfile
    ) -> bool:
        stored = payload.get("timing_profile")
        return stored == timing_profile.value or (
            stored is None and timing_profile is TimingProfile.STRICT
        )

    @staticmethod
    def _checkpoint_trim_matches(
        payload: Mapping[str, Any], timing_profile: TimingProfile
    ) -> bool:
        stored = payload.get("silence_trim_version")
        return stored == TTS_SILENCE_TRIM_VERSION or (
            stored is None and timing_profile is TimingProfile.STRICT
        )

    @staticmethod
    def _selected_model(job: JobRecord, stage: str, default: str) -> str:
        models = job.spec.get("models")
        selected = models.get(stage) if isinstance(models, dict) else None
        if isinstance(selected, str) and selected.strip():
            return selected.strip()
        return default

    def _is_cancel_requested(self, job_id: str) -> bool:
        return self._shutdown_requested() or self._cancelled(
            self._store.get_job(job_id)
        )

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self._is_cancel_requested(job_id):
            raise _StageCancelled()

    @staticmethod
    def _cancelled(job: JobRecord) -> bool:
        return job.cancel_requested or job.status in {
            JobStatus.CANCELLING,
            JobStatus.CANCELLED,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _valid_sha(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in _SHA256 for character in value)
        )

    def _job_dir(self, job_id: str) -> Path:
        return (self._jobs_dir / job_id).resolve(strict=False)

def build_phase4_stage(
    settings: Any,
    store: StateStore,
    shutdown: Any,
) -> Phase4Stage:
    """Build the production Phase 4 stage from the shared Settings object."""

    return Phase4Stage(
        models_lock_path=settings.models_lock_path,
        models_dir=settings.models_dir,
        jobs_dir=settings.jobs_dir,
        output_dir=settings.output_dir,
        default_separation_model_id=settings.default_separation_model_id,
        default_tts_model_id=settings.default_tts_model_id,
        tts_support_model_id=settings.tts_support_model_id,
        store=store,
        tiger_source_dir=settings.tiger_source_dir,
        vieneu_entrypoint=getattr(
            settings,
            "vieneu_entrypoint",
            Path("/opt/vieneu/vieneu-offline.py"),
        ),
        separation_chunk_seconds=settings.separation_chunk_seconds,
        separation_context_seconds=settings.separation_context_seconds,
        separation_batch_size=settings.separation_batch_size,
        narration_target_lufs=settings.narration_target_lufs,
        accompaniment_target_lufs=-24.0,
        shutdown_requested=shutdown.is_set,
    )


__all__ = [
    "Phase4Stage",
    "build_audio_separator",
    "build_narration_synthesizer",
    "build_phase4_stage",
]
