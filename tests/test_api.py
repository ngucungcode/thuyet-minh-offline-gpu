from __future__ import annotations

import asyncio
import errno
import gc
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
import httpx
import pytest

import dub_server.api as api_module
from dub_server.api import create_app
from dub_server.config import Settings
from dub_server.domain import (
    DownloadTask,
    MediaAsset,
    MediaKind,
    MediaQuery,
    ReleaseCandidate,
)
from dub_server.media_probe import FfprobeMediaProbe, MediaProbeError
from dub_server.state import ActiveJobExists, JobStage, JobStatus, StateStore


class FakeGpuReport:
    def model_dump(self, mode: str = "python") -> dict[str, object]:
        del mode
        return {
            "ready": False,
            "enforced": False,
            "gpus": [],
            "errors": [],
            "warnings": ["Máy test không có GPU"],
        }


@dataclass
class FakeAcquisition:
    paused_task: str | None = None
    resumed_task: str | None = None
    fail_pause: bool = False
    fail_resume: bool = False
    added_paused: bool | None = None

    def __post_init__(self) -> None:
        self.last_query: MediaQuery | None = None
        self.last_release_id: str | None = None
        self.running = False
        self.actions: list[str] = []

    async def search(self, query: MediaQuery, *args, **kwargs):
        del args, kwargs
        self.last_query = query
        return (
            ReleaseCandidate(
                release_id="release-1",
                title="Public Domain Movie 2026 1080p",
                indexer_id=7,
                protocol="torrent",
                download_uri="https://indexer.invalid/private-token",
                guid="private-guid",
                size_bytes=1234,
                seeders=5,
            ),
        )

    async def start_download(
        self,
        release: str,
        save_path: Path,
        *,
        rights_confirmed: bool,
        paused: bool = False,
    ) -> DownloadTask:
        assert rights_confirmed is True
        self.added_paused = paused
        self.last_release_id = release
        self.running = not paused
        self.actions.append("add_paused" if paused else "add_running")
        return DownloadTask(task_id="torrent-hash", name="Fixture", save_path=save_path)

    async def pause_download(self, task_id: str) -> None:
        if self.fail_pause:
            raise RuntimeError("simulated pause failure")
        self.paused_task = task_id
        self.running = False
        self.actions.append("pause")

    async def resume_download(self, task_id: str) -> None:
        self.resumed_task = task_id
        self.running = True
        self.actions.append("resume")
        if self.fail_resume:
            raise RuntimeError("simulated ambiguous resume failure")


class BlockingAcquisition(FakeAcquisition):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()
        self.resume_entered = asyncio.Event()
        self.release_resume = asyncio.Event()
        self.block_start = False
        self.block_resume = False

    async def start_download(
        self,
        release: str,
        save_path: Path,
        *,
        rights_confirmed: bool,
        paused: bool = False,
    ) -> DownloadTask:
        if self.block_start:
            self.start_entered.set()
            await self.release_start.wait()
        return await super().start_download(
            release,
            save_path,
            rights_confirmed=rights_confirmed,
            paused=paused,
        )

    async def resume_download(self, task_id: str) -> None:
        if self.block_resume:
            self.resume_entered.set()
            await self.release_resume.wait()
        await super().resume_download(task_id)


@dataclass
class FakeCoordinator:
    store: StateStore

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    async def refresh(self, job_id: str):
        self.calls.append(job_id)
        current = self.store.get_job(job_id)
        if current.status == JobStatus.DOWNLOADING:
            return self.store.update_progress(
                job_id,
                500,
                details={
                    **current.details,
                    "downloaded_bytes": 500,
                    "total_bytes": 1000,
                    "speed_bytes_per_second": 100,
                    "eta_seconds": 5,
                },
            )
        return current

    async def select_subtitle(self, job_id: str, subtitle_id: str):
        current = self.store.get_job(job_id)
        return self.store.update_status(
            job_id,
            JobStatus.READY_OFFLINE,
            details={
                **current.details,
                "selected_subtitle": {"subtitle_id": subtitle_id},
                "transcript_source": "subtitle",
            },
        )

    async def select_asr(self, job_id: str):
        current = self.store.get_job(job_id)
        return self.store.update_status(
            job_id,
            JobStatus.READY_OFFLINE,
            details={
                **current.details,
                "selected_subtitle": None,
                "transcript_source": "asr",
            },
        )


@dataclass
class FakeMediaProbe:
    duration_us: int = 4_000_000

    def __post_init__(self) -> None:
        self.paths: list[Path] = []

    async def probe(
        self,
        path: Path,
        *,
        source_language: str,
        title: str | None = None,
        media_kind: MediaKind = MediaKind.MOVIE,
        year: int | None = None,
        require_h264_passthrough: bool = False,
        allow_hevc_transcode: bool = False,
    ) -> MediaAsset:
        self.paths.append(path)
        return MediaAsset(
            path=path,
            title=title or path.stem,
            duration_us=self.duration_us,
            source_language=source_language,
            media_kind=media_kind,
            year=year,
            fps=24.0,
            audio_stream_index=1,
            video_stream_index=0,
            video_codec="h264",
        )


def _codec_media_probe(
    codec_name: str,
    *,
    cover_first: bool = False,
) -> FfprobeMediaProbe:
    async def runner(command):
        video_streams = []
        if cover_first:
            video_streams.append(
                {
                    "index": 0,
                    "codec_name": "mjpeg",
                    "codec_type": "video",
                    "avg_frame_rate": "0/0",
                    "disposition": {"attached_pic": 1},
                }
            )
        video_index = 1 if cover_first else 0
        payload = {
            "format": {"duration": "4"},
            "streams": [
                *video_streams,
                {
                    "index": video_index,
                    "codec_name": codec_name,
                    "codec_type": "video",
                    "avg_frame_rate": "24/1",
                    "disposition": {"attached_pic": 0},
                },
                {
                    "index": video_index + 1,
                    "codec_name": "aac",
                    "codec_type": "audio",
                    "disposition": {"default": 1},
                },
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    return FfprobeMediaProbe(runner=runner)


def _settings(tmp_path: Path) -> Settings:
    lock_path = tmp_path / "config" / "models.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "asr-small",
                        "stage": "asr",
                        "backend": "faster-whisper",
                        "license": "MIT",
                        "sha256": "1" * 64,
                    },
                    {
                        "id": "mt-fast",
                        "stage": "mt",
                        "backend": "ctranslate2",
                        "license": "MIT",
                        "sha256": "2" * 64,
                    },
                    {
                        "id": "tts-fast",
                        "stage": "tts",
                        "backend": "piper",
                        "license": "CC-BY-4.0",
                        "sha256": "3" * 64,
                    },
                    {
                        "id": "separation-fast",
                        "stage": "separation",
                        "backend": "tiger-dnr",
                        "license": "Apache-2.0",
                        "sha256": "4" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        database_path=tmp_path / "state" / "jobs.sqlite3",
        models_lock_path=lock_path,
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
        gpu_report_path=tmp_path / "state" / "gpu-health.json",
        gpu_report_max_age_seconds=5.0,
        prowlarr_api_key_file=None,
        qbittorrent_password_file=None,
        opensubtitles_api_key_file=None,
        default_asr_model_id="asr-small",
        default_translation_model_id="mt-fast",
        default_separation_model_id="separation-fast",
        default_tts_model_id="tts-fast",
        sse_poll_seconds=0.05,
        acquisition_monitor_seconds=30.0,
    )


def _settings_with_model_vram(
    tmp_path: Path,
    *,
    minimum_vram_mib: int,
) -> Settings:
    settings = _settings(tmp_path)
    catalog = json.loads(settings.models_lock_path.read_text(encoding="utf-8"))
    for model in catalog["models"]:
        if model["id"] == settings.default_asr_model_id:
            model["minimum_vram_mib"] = minimum_vram_mib
            break
    settings.models_lock_path.write_text(json.dumps(catalog), encoding="utf-8")
    return settings


def _write_ready_gpu_report(
    settings: Settings,
    *,
    memory_total_mib: int,
    heartbeat_at: datetime,
    name: str = "Test GPU",
    compute_capability: str = "8.6",
) -> None:
    settings.gpu_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.gpu_report_path.write_text(
        json.dumps(
            {
                "ready": True,
                "enforced": True,
                "selected_gpu_uuid": "GPU-logical-0",
                "gpus": [
                    {
                        "uuid": "GPU-logical-0",
                        "name": name,
                        "driver_version": "570.26",
                        "memory_total_mib": memory_total_mib,
                        "compute_capability": compute_capability,
                    }
                ],
                "errors": [],
                "warnings": [],
                "worker_heartbeat_at": heartbeat_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _client(
    tmp_path: Path,
    monkeypatch,
    *,
    media_probe=None,
    settings: Settings | None = None,
):
    settings = settings or _settings(tmp_path)
    store = StateStore(settings.database_path)
    acquisition = FakeAcquisition()
    acquisition.coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=acquisition.coordinator,
        media_probe=media_probe,
    )
    return TestClient(application), store, acquisition


def test_health_capabilities_and_models_are_local(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        health = client.get("/v1/health")
        capabilities = client.get("/v1/capabilities")
        models = client.get("/v1/models")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["database"]["journal_mode"] == "wal"
    assert health.json()["gpu"]["support_tier"] is None
    assert capabilities.json()["offline_inference"] is True
    assert capabilities.json()["drm_supported"] is False
    assert capabilities.json()["local_upload"] == {
        "enabled": True,
        "media_extensions": [".mp4", ".mkv"],
        "subtitle_extensions": [".srt"],
        "media_max_bytes": 100 * 1024 * 1024 * 1024,
        "subtitle_max_bytes": 16 * 1024 * 1024,
        "session_ttl_seconds": 7 * 24 * 60 * 60,
        "timing_profiles": ["natural", "strict"],
    }
    assert models.json()["models"][0]["id"] == "asr-small"
    assert models.json()["models"][0]["valid"] is False


@pytest.mark.parametrize(
    ("name", "compute_capability", "expected_tier"),
    [
        ("NVIDIA RTX 3090", "8.6", "supported"),
        ("Tesla V100-SXM2-32GB", "7.0", "maintenance-limited"),
        ("NVIDIA CMP 170HX", "8.0", "experimental"),
    ],
)
def test_health_exposes_selected_gpu_support_tier(
    tmp_path: Path,
    monkeypatch,
    name: str,
    compute_capability: str,
    expected_tier: str,
) -> None:
    settings = _settings(tmp_path)
    _write_ready_gpu_report(
        settings,
        memory_total_mib=24 * 1024,
        heartbeat_at=datetime.now(UTC),
        name=name,
        compute_capability=compute_capability,
    )
    client, _, _ = _client(tmp_path, monkeypatch, settings=settings)

    with client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["gpu"]["support_tier"] == expected_tier


def _upload_request(
    *,
    subtitle_filename: str | None = None,
    source_language: str = "auto",
) -> dict[str, object]:
    return {
        "media_filename": "fixture.mkv",
        "subtitle_filename": subtitle_filename,
        "rights_confirmed": True,
        "source_language": source_language,
        "timing_profile": "natural",
        "models": {
            "asr": None,
            "translation": None,
            "separation": None,
            "tts": None,
        },
        "voice": None,
        "voice_rights_confirmed": False,
    }


def test_local_upload_streams_media_and_atomically_creates_asr_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe = FakeMediaProbe()
    client, store, acquisition = _client(
        tmp_path,
        monkeypatch,
        media_probe=probe,
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=(chunk for chunk in (b"video-", b"bytes")),
        )
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")
        repeated = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert created.status_code == 201
    assert uploaded.status_code == 200
    assert uploaded.json()["status"] == "ready"
    assert uploaded.json()["media_size_bytes"] == len(b"video-bytes")
    assert uploaded.json()["media_sha256"] == hashlib.sha256(
        b"video-bytes"
    ).hexdigest()
    assert finalized.status_code == 202
    assert repeated.status_code == 202
    assert finalized.json()["id"] == upload_id
    assert repeated.json()["id"] == upload_id
    assert finalized.json()["status"] == "ready_offline"
    assert finalized.json()["stage"] == "subtitle"
    assert finalized.json()["spec"]["source_kind"] == "local_upload"
    assert finalized.json()["spec"]["timing_profile"] == "natural"
    assert finalized.json()["details"]["transcript_source"] == "asr"
    assert acquisition.last_release_id is None
    assert probe.paths == [tmp_path / "incoming" / upload_id / "source.mkv"]
    assert store.get_checkpoint(upload_id, JobStage.ACQUISITION) is not None
    assert store.get_checkpoint(upload_id, JobStage.SUBTITLE) is not None


def test_local_mkv_upload_accepts_h264_passthrough_before_creating_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=_codec_media_probe("h264"),
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"mkv bytes")
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert finalized.status_code == 202
    assert finalized.json()["details"]["selected_media"]["video_codec"] == "h264"
    assert store.get_job(upload_id).status is JobStatus.READY_OFFLINE


def test_local_mkv_upload_accepts_hevc_for_video_transcode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=_codec_media_probe("hevc"),
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"hevc mkv bytes")
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert finalized.status_code == 202
    assert finalized.json()["details"]["selected_media"]["video_codec"] == "hevc"
    assert store.get_job(upload_id).status is JobStatus.READY_OFFLINE


def test_retained_hevc_upload_finalizes_after_server_upgrade_without_reupload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class UpgradeAwareProbe(FakeMediaProbe):
        upgraded = False

        async def probe(self, path: Path, **kwargs) -> MediaAsset:
            self.paths.append(path)
            if not self.upgraded:
                raise MediaProbeError(
                    "unsupported_media",
                    "Luồng hình chính dùng codec HEVC; cần H.264/AVC để xuất MP4 "
                    "không mã hóa lại",
                    retryable=False,
                )
            return MediaAsset(
                path=path,
                title=path.stem,
                duration_us=self.duration_us,
                source_language=str(kwargs["source_language"]),
                media_kind=MediaKind.MOVIE,
                fps=24.0,
                audio_stream_index=1,
                video_stream_index=0,
                video_codec="hevc",
            )

    probe = UpgradeAwareProbe()
    client, store, _ = _client(tmp_path, monkeypatch, media_probe=probe)
    media_bytes = b"retained hevc bytes"
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=media_bytes,
        )
        rejected = client.post(f"/v1/uploads/{upload_id}/finalize")
        retained = client.get(f"/v1/uploads/{upload_id}")
        probe.upgraded = True
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert uploaded.status_code == 200
    assert rejected.status_code == 422
    assert retained.status_code == 200
    assert retained.json()["media_sha256"] == hashlib.sha256(media_bytes).hexdigest()
    assert finalized.status_code == 202
    assert finalized.json()["id"] == upload_id
    assert finalized.json()["details"]["selected_media"]["video_codec"] == "hevc"
    assert probe.paths == [
        tmp_path / "incoming" / upload_id / "source.mkv",
        tmp_path / "incoming" / upload_id / "source.mkv",
    ]
    assert store.get_job(upload_id).status is JobStatus.READY_OFFLINE


def test_local_upload_ignores_embedded_cover_before_h264_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=_codec_media_probe("h264", cover_first=True),
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"mkv with cover")
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert finalized.status_code == 202
    selected = finalized.json()["details"]["selected_media"]
    assert selected["video_stream_index"] == 1
    assert selected["video_codec"] == "h264"
    assert store.get_job(upload_id).status is JobStatus.READY_OFFLINE


@pytest.mark.parametrize("codec_name", ["vp8", "ffv1"])
def test_local_mkv_upload_rejects_unsupported_video_before_creating_job(
    tmp_path: Path,
    monkeypatch,
    codec_name: str,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=_codec_media_probe(codec_name),
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"mkv bytes")
        rejected = client.post(f"/v1/uploads/{upload_id}/finalize")
        retained = client.get(f"/v1/uploads/{upload_id}")

    assert rejected.status_code == 422
    assert rejected.json()["detail"] == {
        "code": "unsupported_media",
        "message": (
            f"Luồng hình chính dùng codec {codec_name.upper()}; cần H.264/AVC "
            "hoặc HEVC/H.265 để xuất MP4"
        ),
        "retryable": False,
    }
    assert retained.status_code == 200
    assert retained.json()["status"] == "ready"
    assert store.list_jobs() == []


def test_local_upload_with_srt_bypasses_provider_and_asr_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
    )
    subtitle = (
        b"1\n00:00:00,200 --> 00:00:01,500\nHello world.\n\n"
        b"2\n00:00:02,000 --> 00:00:03,500\nSecond line.\n"
    )
    with client:
        created = client.post(
            "/v1/uploads",
            json=_upload_request(
                subtitle_filename="fixture.srt",
                source_language="en",
            ),
        )
        upload_id = created.json()["id"]
        media_response = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=b"video bytes",
        )
        subtitle_response = client.put(
            f"/v1/uploads/{upload_id}/subtitle",
            content=subtitle,
        )
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")
        leftover = tmp_path / "incoming" / upload_id / "source.srt"
        leftover.write_bytes(subtitle)
        repeated = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert media_response.json()["status"] == "awaiting_subtitle"
    assert subtitle_response.json()["status"] == "ready"
    assert media_response.json()["media_sha256"] == hashlib.sha256(
        b"video bytes"
    ).hexdigest()
    assert subtitle_response.json()["subtitle_sha256"] == hashlib.sha256(
        subtitle
    ).hexdigest()
    assert finalized.status_code == 202
    assert repeated.status_code == 202
    payload = finalized.json()
    assert payload["spec"]["subtitle_mode"] == "manual"
    assert payload["details"]["transcript_source"] == "subtitle"
    subtitle_path = Path(payload["details"]["source_subtitle_path"])
    assert subtitle_path == tmp_path / "jobs" / upload_id / "source-subtitle.srt"
    assert subtitle_path.read_bytes() == subtitle
    assert not (tmp_path / "incoming" / upload_id / "source.srt").exists()
    assert acquisition.coordinator.calls == []
    checkpoint = store.get_checkpoint(upload_id, JobStage.SUBTITLE)
    assert checkpoint is not None
    assert checkpoint.payload["transcript_source"] == "subtitle"


def test_local_upload_can_replace_invalid_srt_and_finalize_same_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
    )
    valid_subtitle = (
        b"1\n00:00:00,200 --> 00:00:01,500\nCorrected subtitle.\n"
    )
    with client:
        created = client.post(
            "/v1/uploads",
            json=_upload_request(
                subtitle_filename="fixture.srt",
                source_language="en",
            ),
        )
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"video bytes")
        client.put(
            f"/v1/uploads/{upload_id}/subtitle",
            content=b"this is not a valid SRT",
        )
        rejected = client.post(f"/v1/uploads/{upload_id}/finalize")
        retained = client.get(f"/v1/uploads/{upload_id}")
        replaced = client.put(
            f"/v1/uploads/{upload_id}/subtitle",
            content=valid_subtitle,
        )
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "uploaded_subtitle_invalid"
    assert retained.status_code == 200
    assert retained.json()["status"] == "ready"
    assert replaced.status_code == 200
    assert replaced.json()["status"] == "ready"
    assert finalized.status_code == 202
    assert finalized.json()["details"]["transcript_source"] == "subtitle"
    assert store.get_job(upload_id).status is JobStatus.READY_OFFLINE
    stored_subtitle = tmp_path / "jobs" / upload_id / "source-subtitle.srt"
    assert stored_subtitle.read_bytes() == valid_subtitle


def test_local_upload_rejects_empty_undeclared_and_incomplete_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        empty = client.put(f"/v1/uploads/{upload_id}/media", content=b"")
        undeclared = client.put(
            f"/v1/uploads/{upload_id}/subtitle",
            content=b"subtitle",
        )
        incomplete = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "upload_empty"
    assert undeclared.status_code == 409
    assert undeclared.json()["detail"]["code"] == "subtitle_not_declared"
    assert incomplete.status_code == 409
    assert incomplete.json()["detail"]["code"] == "upload_incomplete"
    assert not (tmp_path / "incoming" / upload_id / "source.mkv").exists()


def test_local_upload_accepts_mp4_and_rejects_delete_after_finalize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    probe = FakeMediaProbe()
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=probe)
    request = {**_upload_request(), "media_filename": "fixture.MP4"}
    with client:
        created = client.post("/v1/uploads", json=request)
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=b"mp4 bytes",
        )
        finalized = client.post(f"/v1/uploads/{upload_id}/finalize")
        rejected_delete = client.delete(f"/v1/uploads/{upload_id}")

    assert uploaded.status_code == 200
    assert finalized.status_code == 202
    assert probe.paths == [tmp_path / "incoming" / upload_id / "source.mp4"]
    assert rejected_delete.status_code == 409
    assert rejected_delete.json()["detail"]["code"] == "upload_already_finalized"
    assert (tmp_path / "incoming" / upload_id / "source.mp4").is_file()


def test_atomic_metadata_write_fsyncs_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fsynced_modes: list[int] = []
    real_fsync = os.fsync

    def capture_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(api_module.os, "fsync", capture_fsync)
    destination = tmp_path / "checkpoint.json"
    api_module._atomic_write_bytes(destination, b"{}")

    assert destination.read_bytes() == b"{}"
    assert api_module.stat_module.S_ISREG(fsynced_modes[0])
    if os.name != "nt":
        assert api_module.stat_module.S_ISDIR(fsynced_modes[-1])


@pytest.mark.parametrize(
    "error_number",
    sorted(api_module._UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS),
)
def test_directory_fsync_ignores_unsupported_filesystems_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch,
    error_number: int,
) -> None:
    closed: list[int] = []
    with monkeypatch.context() as patch:
        patch.setattr(api_module.os, "open", lambda _path, _flags: 41)
        patch.setattr(
            api_module.os,
            "fsync",
            lambda _descriptor: (_ for _ in ()).throw(
                OSError(error_number, "directory fsync is unsupported")
            ),
        )
        patch.setattr(api_module.os, "close", closed.append)
        api_module._fsync_directory(tmp_path)

    assert closed == [41]


@pytest.mark.parametrize(
    "error_number",
    sorted(api_module._UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS),
)
def test_directory_fsync_ignores_unsupported_open_without_closing_descriptor(
    tmp_path: Path,
    monkeypatch,
    error_number: int,
) -> None:
    closed: list[int] = []

    def unsupported_open(_path: Path, _flags: int) -> int:
        raise OSError(error_number, "opening directories is unsupported")

    with monkeypatch.context() as patch:
        patch.setattr(api_module.os, "open", unsupported_open)
        patch.setattr(
            api_module.os,
            "fsync",
            lambda _descriptor: pytest.fail("fsync must not run when open failed"),
        )
        patch.setattr(api_module.os, "close", closed.append)
        api_module._fsync_directory(tmp_path)

    assert closed == []


def test_directory_fsync_propagates_real_io_failure_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    closed: list[int] = []
    with monkeypatch.context() as patch:
        patch.setattr(api_module.os, "open", lambda _path, _flags: 42)
        patch.setattr(
            api_module.os,
            "fsync",
            lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "I/O error")),
        )
        patch.setattr(api_module.os, "close", closed.append)
        with pytest.raises(OSError) as failure:
            api_module._fsync_directory(tmp_path)

    assert failure.value.errno == errno.EIO
    assert closed == [42]


def test_stream_upload_validates_malformed_and_mismatched_content_length(
    tmp_path: Path,
) -> None:
    class StreamingRequest:
        def __init__(self, length: str, chunks: tuple[bytes, ...]) -> None:
            self.headers = {"content-length": length}
            self._chunks = chunks

        async def stream(self):
            for chunk in self._chunks:
                yield chunk

    destination = tmp_path / "source.mp4"
    destination.write_bytes(b"published")

    with pytest.raises(HTTPException) as malformed:
        asyncio.run(
            api_module._stream_upload_body(
                StreamingRequest("invalid", (b"new",)),  # type: ignore[arg-type]
                destination,
                maximum=1024,
            )
        )
    assert malformed.value.status_code == 400
    assert malformed.value.detail["code"] == "content_length_invalid"

    with pytest.raises(HTTPException) as mismatch:
        asyncio.run(
            api_module._stream_upload_body(
                StreamingRequest("3", (b"ab",)),  # type: ignore[arg-type]
                destination,
                maximum=1024,
            )
        )
    assert mismatch.value.status_code == 400
    assert mismatch.value.detail["code"] == "upload_size_mismatch"
    assert destination.read_bytes() == b"published"
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({**_upload_request(), "rights_confirmed": False}, "rights_confirmation_required"),
        ({**_upload_request(), "media_filename": "../escape.mkv"}, "unsupported_upload_media"),
        ({**_upload_request(), "media_filename": "fixture.avi"}, "unsupported_upload_media"),
        (
            _upload_request(subtitle_filename="fixture.srt", source_language="auto"),
            "subtitle_language_required",
        ),
    ],
)
def test_local_upload_rejects_unsafe_or_incomplete_declarations(
    tmp_path: Path,
    monkeypatch,
    payload: dict[str, object],
    code: str,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        response = client.post("/v1/uploads", json=payload)

    assert response.status_code in {403, 422}
    assert response.json()["detail"]["code"] == code


def test_unknown_upload_uuid_does_not_leak_a_process_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    unknown = "00000000-0000-4000-8000-000000000099"

    with client:
        response = client.put(f"/v1/uploads/{unknown}/media", content=b"video")
        gc.collect()
        retained_locks = len(client.app.state.upload_session_locks)

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "upload_not_found"
    assert retained_locks == 0


def test_local_upload_limit_keeps_previous_atomic_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.upload_media_max_bytes = 1024 * 1024
    client, _, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
        settings=settings,
    )
    original = b"first-version"
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        accepted = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=original,
        )
        rejected = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=(b"x" * (1024 * 1024 + 1)),
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "upload_too_large"
    source = tmp_path / "incoming" / upload_id / "source.mkv"
    assert source.read_bytes() == original
    assert list(source.parent.glob("*.part")) == []


def test_local_upload_finalize_rejects_same_size_content_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        uploaded = client.put(
            f"/v1/uploads/{upload_id}/media",
            content=b"original",
        )
        source = tmp_path / "incoming" / upload_id / "source.mkv"
        source.write_bytes(b"modified")
        rejected = client.post(f"/v1/uploads/{upload_id}/finalize")

    assert uploaded.json()["media_size_bytes"] == len(b"original")
    assert len(b"original") == len(b"modified")
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == {
        "code": "upload_artifact_changed",
        "message": "File video đã thay đổi sau khi tải lên",
        "retryable": True,
    }
    assert store.list_jobs() == []


def test_local_upload_rejects_symlink_artifact_and_delete_cleans_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        session_directory = tmp_path / "incoming" / upload_id
        outside = tmp_path / "outside.mkv"
        outside.write_bytes(b"outside")
        source = session_directory / "source.mkv"
        try:
            source.symlink_to(outside)
        except OSError:
            pytest.skip("Môi trường test không cho phép tạo symlink")
        metadata_path = session_directory / ".upload-session.json"
        state = json.loads(metadata_path.read_text(encoding="utf-8"))
        state["status"] = "ready"
        state["media_size_bytes"] = len(b"outside")
        metadata_path.write_text(json.dumps(state), encoding="utf-8")
        rejected = client.post(f"/v1/uploads/{upload_id}/finalize")
        deleted = client.delete(f"/v1/uploads/{upload_id}")

    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "upload_artifact_invalid"
    assert deleted.status_code == 204
    assert not session_directory.exists()
    assert outside.read_bytes() == b"outside"


def test_local_upload_finalize_respects_single_heavy_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
        upload_id = created.json()["id"]
        client.put(f"/v1/uploads/{upload_id}/media", content=b"video")
        active = store.create_job("other-release", {"rights_confirmed": True})
        blocked = client.post(f"/v1/uploads/{upload_id}/finalize")
        session = client.get(f"/v1/uploads/{upload_id}")

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "active_job_exists"
    assert blocked.json()["detail"]["active_job_id"] == active.id
    assert session.json()["status"] == "ready"


def test_delete_upload_returns_retryable_error_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        upload_id = client.post("/v1/uploads", json=_upload_request()).json()["id"]

        def fail_cleanup(_path: Path) -> None:
            raise OSError("filesystem is temporarily unavailable")

        monkeypatch.setattr(api_module.shutil, "rmtree", fail_cleanup)
        response = client.delete(f"/v1/uploads/{upload_id}")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "upload_cleanup_failed",
        "message": "Không thể xóa dữ liệu phiên tải file; hãy thử lại",
        "retryable": True,
    }
    assert (tmp_path / "incoming" / upload_id).is_dir()


def test_delete_upload_rejects_unsafe_prepared_artifact_and_keeps_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch, media_probe=FakeMediaProbe())
    with client:
        upload_id = client.post("/v1/uploads", json=_upload_request()).json()["id"]
        prepared = tmp_path / "jobs" / upload_id
        prepared.write_bytes(b"unsafe artifact")
        response = client.delete(f"/v1/uploads/{upload_id}")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "upload_cleanup_failed",
        "message": "Không thể xóa dữ liệu phiên tải file; hãy thử lại",
        "retryable": True,
    }
    assert prepared.is_file()
    assert (tmp_path / "incoming" / upload_id).is_dir()


def test_expired_unfinalized_upload_is_cleaned_on_startup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.upload_session_ttl_seconds = 60
    client, _, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
        settings=settings,
    )
    with client:
        created = client.post("/v1/uploads", json=_upload_request())
    upload_id = created.json()["id"]
    session_directory = tmp_path / "incoming" / upload_id
    metadata = session_directory / ".upload-session.json"
    old = time.time() - 120
    metadata.touch()
    os.utime(metadata, (old, old))

    with client:
        missing = client.get(f"/v1/uploads/{upload_id}")

    assert missing.status_code == 404
    assert not session_directory.exists()


def test_expired_upload_cleanup_failure_does_not_block_startup(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    settings = _settings(tmp_path)
    settings.upload_session_ttl_seconds = 60
    client, _, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
        settings=settings,
    )
    with client:
        upload_id = client.post("/v1/uploads", json=_upload_request()).json()["id"]
    session_directory = tmp_path / "incoming" / upload_id
    metadata = session_directory / ".upload-session.json"
    old = time.time() - 120
    os.utime(metadata, (old, old))

    def fail_cleanup(_path: Path) -> None:
        raise OSError("filesystem is temporarily unavailable")

    monkeypatch.setattr(api_module.shutil, "rmtree", fail_cleanup)
    with caplog.at_level("ERROR", logger=api_module.__name__):
        with client:
            health = client.get("/v1/health")
            session = client.get(f"/v1/uploads/{upload_id}")

    assert health.status_code == 200
    assert session.status_code == 200
    assert session_directory.is_dir()
    assert "Could not remove expired upload session" in caplog.text


def test_expired_upload_keeps_session_when_prepared_cleanup_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.upload_session_ttl_seconds = 60
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
        settings=settings,
    )
    with client:
        upload_id = client.post("/v1/uploads", json=_upload_request()).json()["id"]
    session_directory = tmp_path / "incoming" / upload_id
    metadata = session_directory / ".upload-session.json"
    old = time.time() - 120
    os.utime(metadata, (old, old))
    prepared = tmp_path / "jobs" / upload_id
    prepared.mkdir(parents=True)
    (prepared / "artifact.part").write_bytes(b"partial")
    removed: list[Path] = []
    real_rmtree = api_module.shutil.rmtree

    def fail_prepared(path: Path) -> None:
        candidate = Path(path)
        removed.append(candidate)
        if candidate == prepared:
            raise OSError("filesystem is temporarily unavailable")
        real_rmtree(candidate)

    monkeypatch.setattr(api_module.shutil, "rmtree", fail_prepared)
    api_module._cleanup_stale_upload_sessions(
        settings,
        store,
        identifiers=(upload_id,),
    )

    assert removed == [prepared]
    assert prepared.is_dir()
    assert session_directory.is_dir()
    assert metadata.is_file()


@pytest.mark.asyncio
async def test_expired_upload_cleanup_runs_periodically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    upload_id = "00000000-0000-4000-8000-000000000001"
    calls: list[tuple[Settings, StateStore, tuple[str, ...] | None]] = []

    def fake_cleanup(
        selected_settings: Settings,
        selected_store: StateStore,
        *,
        identifiers: tuple[str, ...] | None = None,
    ) -> None:
        calls.append((selected_settings, selected_store, identifiers))

    monkeypatch.setattr(
        api_module,
        "_expired_upload_session_identifiers",
        lambda _settings: (upload_id,),
    )
    monkeypatch.setattr(api_module, "_cleanup_stale_upload_sessions", fake_cleanup)
    stop = asyncio.Event()
    session_lock = asyncio.Lock()
    await session_lock.acquire()
    task = asyncio.create_task(
        api_module._monitor_stale_upload_sessions(
            settings,
            store,
            stop,
            interval_seconds=0.01,
            session_locks={upload_id: session_lock},
        )
    )
    try:
        await asyncio.sleep(0.03)
        assert calls == []
        session_lock.release()
        for _ in range(100):
            if calls:
                break
            await asyncio.sleep(0.01)
    finally:
        if session_lock.locked():
            session_lock.release()
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    assert calls
    assert calls[0] == (settings, store, (upload_id,))


@pytest.mark.asyncio
async def test_periodic_cleanup_rechecks_ttl_after_waiting_for_upload_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.upload_session_ttl_seconds = 60
    client, store, _ = _client(
        tmp_path,
        monkeypatch,
        media_probe=FakeMediaProbe(),
        settings=settings,
    )
    with client:
        upload_id = client.post("/v1/uploads", json=_upload_request()).json()["id"]

    session_directory = tmp_path / "incoming" / upload_id
    metadata = session_directory / ".upload-session.json"
    old = time.time() - 120
    os.utime(metadata, (old, old))
    scans: list[str] = []

    def fake_expired(_settings: Settings) -> tuple[str, ...]:
        scans.append(upload_id)
        return (upload_id,)

    monkeypatch.setattr(
        api_module,
        "_expired_upload_session_identifiers",
        fake_expired,
    )
    stop = asyncio.Event()
    session_lock = asyncio.Lock()
    await session_lock.acquire()
    task = asyncio.create_task(
        api_module._monitor_stale_upload_sessions(
            settings,
            store,
            stop,
            interval_seconds=0.01,
            session_locks={upload_id: session_lock},
        )
    )
    try:
        for _ in range(100):
            if scans:
                break
            await asyncio.sleep(0.01)
        assert scans
        assert session_lock.locked()
        fresh = time.time()
        os.utime(metadata, (fresh, fresh))
        session_lock.release()
        await asyncio.sleep(0.03)
    finally:
        if session_lock.locked():
            session_lock.release()
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    assert session_directory.is_dir()
    assert metadata.is_file()


def test_embedded_dashboard_shares_the_api_origin(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        dashboard = client.get("/")
        api_health = client.get("/v1/health")

        assert dashboard.status_code == 200
        assert dashboard.headers["content-type"].startswith("text/html")
        assert "Lồng Tiếng GPU Studio" in dashboard.text
        assert client.app.state.web_dashboard_enabled is True
        asset_path = dashboard.text.split('href="/assets/', maxsplit=1)[1].split(
            '"', maxsplit=1
        )[0]
        asset = client.get(f"/assets/{asset_path}")

    assert api_health.status_code == 200
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("text/css")


def test_job_history_lists_newest_first_and_filters_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    first = store.create_job(
        "release-1",
        {"rights_confirmed": True, "private_note": "stored-but-not-a-secret"},
    )
    store.update_status(
        first.id,
        JobStatus.COMPLETED,
        stage=JobStage.DONE,
        progress_permille=1000,
        force=True,
    )
    second = store.create_job("release-2", {"rights_confirmed": True})
    store.update_status(
        second.id,
        JobStatus.FAILED,
        error_code="fixture_failed",
        error_message="Lỗi fixture",
        retryable=True,
        force=True,
    )

    with client:
        newest = client.get("/v1/jobs", params={"limit": 1})
        completed = client.get(
            "/v1/jobs",
            params=[("status", "completed"), ("limit", "10")],
        )
        oldest = client.get(
            "/v1/jobs",
            params={"newest_first": "false", "limit": 10},
        )

    assert newest.status_code == 200
    assert newest.json()["count"] == 1
    assert newest.json()["items"][0]["id"] == second.id
    assert [item["id"] for item in completed.json()["items"]] == [first.id]
    assert [item["id"] for item in oldest.json()["items"]] == [
        first.id,
        second.id,
    ]


def test_completed_job_artifacts_are_served_from_configured_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    settings = client.app.state.settings
    job = store.create_job("release-1", {"rights_confirmed": True})
    video = settings.output_dir / f"{job.id}.mp4"
    subtitle = settings.output_dir / f"{job.id}.vi.srt"
    timing = settings.jobs_dir / job.id / "timing-report.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    timing.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"mp4-result")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chào\n", encoding="utf-8")
    timing.write_text('{"schema_version":1}', encoding="utf-8")
    store.update_status(
        job.id,
        JobStatus.COMPLETED,
        stage=JobStage.DONE,
        progress_permille=1000,
        result={
            "video_path": str(video.resolve()),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "size_bytes": video.stat().st_size,
            "srt_path": str(subtitle.resolve()),
            "srt_sha256": hashlib.sha256(subtitle.read_bytes()).hexdigest(),
            "timing_report_path": str(timing.resolve()),
            "timing_report_sha256": hashlib.sha256(timing.read_bytes()).hexdigest(),
        },
        force=True,
    )

    with client:
        video_response = client.get(f"/v1/jobs/{job.id}/artifacts/video")
        subtitle_response = client.get(f"/v1/jobs/{job.id}/artifacts/subtitle")
        timing_response = client.get(f"/v1/jobs/{job.id}/artifacts/timing")

    assert video_response.status_code == 200
    assert video_response.content == b"mp4-result"
    assert video_response.headers["content-type"].startswith("video/mp4")
    assert "Xin chào" in subtitle_response.text
    assert timing_response.json() == {"schema_version": 1}


def test_artifact_endpoint_rejects_unready_and_unsealed_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    job = store.create_job("release-1", {"rights_confirmed": True})
    with client:
        unready = client.get(f"/v1/jobs/{job.id}/artifacts/video")
        assert unready.status_code == 409
        assert unready.json()["detail"]["code"] == "artifact_not_ready"

        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"must-not-leak")
        store.update_status(
            job.id,
            JobStatus.COMPLETED,
            stage=JobStage.DONE,
            result={"video_path": str(outside.resolve())},
            force=True,
        )
        rejected = client.get(f"/v1/jobs/{job.id}/artifacts/video")
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "artifact_record_invalid"


def test_artifact_endpoint_rejects_tampering_and_symlinks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    settings = client.app.state.settings
    job = store.create_job("release-1", {"rights_confirmed": True})
    video = settings.output_dir / f"{job.id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"sealed-video")
    sealed_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    store.update_status(
        job.id,
        JobStatus.COMPLETED,
        stage=JobStage.DONE,
        progress_permille=1000,
        result={
            "video_path": str(video.absolute()),
            "video_sha256": sealed_sha,
            "size_bytes": video.stat().st_size,
        },
        force=True,
    )

    video.write_bytes(b"modified-after-completion")
    with client:
        tampered = client.get(f"/v1/jobs/{job.id}/artifacts/video")
    assert tampered.status_code == 409
    assert tampered.json()["detail"]["code"] == "artifact_integrity_failed"

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"sealed-video")
    video.unlink()
    try:
        video.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is not permitted")
    with client:
        linked = client.get(f"/v1/jobs/{job.id}/artifacts/video")
    assert linked.status_code == 409
    assert linked.json()["detail"]["code"] == "artifact_record_invalid"


def test_lifespan_builds_real_coordinator_when_acquisition_is_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    acquisition = FakeAcquisition()
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
    )

    with TestClient(application) as client:
        response = client.get("/v1/health")
        coordinator_name = type(application.state.coordinator).__name__

    assert response.status_code == 200
    assert response.json()["coordinator_configured"] is True
    assert coordinator_name == "AcquisitionCoordinator"


def test_startup_marks_interrupted_created_job_as_retryable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    interrupted = store.create_job(
        "release-1",
        {"rights_confirmed": True},
    )
    acquisition = FakeAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application):
        recovered = store.get_job(interrupted.id)

    assert recovered.status is JobStatus.FAILED
    assert recovered.retryable is True
    assert recovered.error_code == "download_start_interrupted"


def test_startup_recovers_persisted_paused_download_before_monitoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "torrent-hash", "backend_started": False},
    )
    acquisition = FakeAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application):
        recovered = store.get_job(job.id)

    assert recovered.status is JobStatus.DOWNLOADING
    assert recovered.details["backend_started"] is True
    assert acquisition.resumed_task == "torrent-hash"


def test_startup_keeps_slot_if_resume_and_compensation_are_uncertain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "torrent-hash", "backend_started": False},
    )
    acquisition = FakeAcquisition(fail_resume=True, fail_pause=True)
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application):
        guarded = store.get_job(job.id)

    assert guarded.status is JobStatus.DOWNLOADING
    assert guarded.active_slot is True
    assert guarded.details["backend_state_uncertain"] is True


def test_startup_finishes_pending_cancel_and_pauses_known_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "torrent-hash"},
    )
    store.request_cancel(job.id)
    acquisition = FakeAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application):
        recovered = store.get_job(job.id)

    assert recovered.status is JobStatus.CANCELLED
    assert acquisition.paused_task == "torrent-hash"


def test_startup_finishes_pending_cancel_without_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    job = store.create_job("release-1", {"rights_confirmed": True})
    store.request_cancel(job.id)
    acquisition = FakeAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application):
        recovered = store.get_job(job.id)

    assert recovered.status is JobStatus.CANCELLED
    assert acquisition.paused_task is None


def test_cancel_without_possible_inflight_start_finishes_immediately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    job = store.create_job("release-failed", {"rights_confirmed": True})
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="download_start_failed",
        error_message="fixture",
        retryable=True,
    )

    with client:
        response = client.post(f"/v1/jobs/{job.id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_cancel_inactive_job_does_not_pause_task_owned_by_active_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    old = store.create_job("release-old", {"rights_confirmed": True})
    store.update_status(
        old.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "old-hash"},
    )
    store.update_status(
        old.id,
        JobStatus.FAILED,
        error_code="fixture",
        error_message="fixture",
        retryable=True,
    )
    active = store.create_job("release-active", {"rights_confirmed": True})
    store.update_status(
        active.id,
        JobStatus.DOWNLOADING,
        details={"task_id": "new-hash", "backend_started": True},
    )

    with client:
        response = client.post(f"/v1/jobs/{old.id}/cancel")
        store.request_cancel(active.id)
        store.finalize_cancel(active.id)
        retried = client.post(f"/v1/jobs/{old.id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert response.json()["details"]["warnings"][-1]["code"] == (
        "backend_cleanup_deferred"
    )
    assert retried.status_code == 200
    assert retried.json()["status"] == "cancelled"
    assert acquisition.paused_task == "old-hash"


def test_restart_token_never_persists_provider_url_or_api_key() -> None:
    candidate = ReleaseCandidate(
        release_id="release-1",
        title="Fixture",
        indexer_id=7,
        protocol="torrent",
        download_uri="https://prowlarr:9696/download?apikey=top-secret",
        guid="private-guid",
        info_hash="a" * 40,
    )

    snapshot = api_module._release_snapshot(candidate)
    assert snapshot is None

    safe = api_module._release_snapshot(
        ReleaseCandidate(
            release_id="release-safe",
            title="Fixture",
            indexer_id=7,
            protocol="torrent",
            download_uri=f"magnet:?xt=urn:btih:{'b' * 40}",
            guid="",
            info_hash="b" * 40,
        )
    )
    assert safe is not None
    encoded = json.dumps(safe)
    assert "top-secret" not in encoded
    assert "private-guid" not in encoded
    assert safe["download_uri"] == "magnet:?xt=urn:btih:" + "b" * 40


def test_resume_re_resolves_release_without_safe_restart_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class RestartAwareAcquisition(FakeAcquisition):
        def restore_release(self, release: ReleaseCandidate) -> None:
            self.restored_release = release

        async def search(self, query: MediaQuery, *args, **kwargs):
            del args, kwargs
            self.last_query = query
            return (
                ReleaseCandidate(
                    release_id="release-http",
                    title="Authorized Fixture",
                    indexer_id=7,
                    protocol="torrent",
                    download_uri="https://indexer.invalid/fresh-authorized-url",
                    guid="stable-guid",
                ),
            )

        async def start_download(
            self,
            release: ReleaseCandidate | str,
            save_path: Path,
            *,
            rights_confirmed: bool,
            paused: bool = False,
        ) -> DownloadTask:
            assert isinstance(release, ReleaseCandidate)
            return await super().start_download(
                release.release_id,
                save_path,
                rights_confirmed=rights_confirmed,
                paused=paused,
            )

    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    job = store.create_job(
        "release-http",
        {
            "rights_confirmed": True,
            "search_query": "Authorized Fixture",
            "year": 2024,
            "media_type": "movie",
        },
    )
    store.update_status(
        job.id,
        JobStatus.FAILED,
        error_code="download_start_interrupted",
        error_message="fixture",
        retryable=True,
    )
    acquisition = RestartAwareAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )

    with TestClient(application) as client:
        response = client.post(f"/v1/jobs/{job.id}/resume")

    assert response.status_code == 200
    assert response.json()["status"] == "downloading"
    assert acquisition.last_query == MediaQuery(
        query="Authorized Fixture",
        year=2024,
        media_kind=MediaKind.MOVIE,
    )
    assert acquisition.last_release_id == "release-http"


def test_search_uses_injected_adapter_and_hides_provider_urls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, acquisition = _client(tmp_path, monkeypatch)
    with client:
        response = client.post(
            "/v1/search",
            json={"query": "  Public   Domain ", "year": 2026, "media_type": "movie"},
        )

    assert response.status_code == 200
    assert acquisition.last_query == MediaQuery("Public Domain", 2026)
    encoded = response.text
    assert "private-token" not in encoded
    assert "private-guid" not in encoded
    assert response.json()["results"][0]["release_id"] == "release-1"


def test_job_requires_content_and_voice_rights(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        no_content_rights = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": False},
        )
        no_voice_rights = client.post(
            "/v1/jobs",
            json={
                "release_id": "release-1",
                "rights_confirmed": True,
                "voice": {"voice_id": "my-voice"},
                "voice_rights_confirmed": False,
            },
        )

    assert no_content_rights.status_code == 403
    assert no_content_rights.json()["detail"]["code"] == "rights_confirmation_required"
    assert no_voice_rights.status_code == 403
    assert no_voice_rights.json()["detail"]["code"] == "voice_rights_confirmation_required"


def test_job_rejects_unknown_or_wrong_stage_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        unknown = client.post(
            "/v1/jobs",
            json={
                "release_id": "release-1",
                "rights_confirmed": True,
                "models": {"asr": "does-not-exist"},
            },
        )
        wrong_stage = client.post(
            "/v1/jobs",
            json={
                "release_id": "release-1",
                "rights_confirmed": True,
                "models": {"asr": "mt-fast"},
            },
        )

    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "invalid_model_selection"
    assert wrong_stage.status_code == 422


def test_model_selection_rejects_insufficient_fresh_gpu_vram_for_job_and_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings_with_model_vram(tmp_path, minimum_vram_mib=8192)
    _write_ready_gpu_report(
        settings,
        memory_total_mib=6144,
        heartbeat_at=datetime.now(UTC),
    )
    client, _, _ = _client(tmp_path, monkeypatch, settings=settings)

    with client:
        upload = client.post("/v1/uploads", json=_upload_request())
        job = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )

    for response in (upload, job):
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "code": "model_vram_insufficient",
            "message": (
                "Model asr-small cần tối thiểu 8192 MiB VRAM nhưng GPU logical 0 "
                "chỉ có 6144 MiB"
            ),
            "retryable": False,
        }


@pytest.mark.parametrize("report_state", ["missing", "stale"])
def test_model_selection_without_fresh_gpu_report_remains_compatible(
    tmp_path: Path,
    monkeypatch,
    report_state: str,
) -> None:
    settings = _settings_with_model_vram(tmp_path, minimum_vram_mib=8192)
    if report_state == "stale":
        _write_ready_gpu_report(
            settings,
            memory_total_mib=6144,
            heartbeat_at=datetime.now(UTC) - timedelta(seconds=6),
        )
    client, _, _ = _client(tmp_path, monkeypatch, settings=settings)

    with client:
        upload = client.post("/v1/uploads", json=_upload_request())
        job = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )

    assert upload.status_code == 201
    assert job.status_code == 202


def test_only_one_active_gpu_job_can_be_created(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        first = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        second = client.post(
            "/v1/jobs",
            json={"release_id": "release-2", "rights_confirmed": True},
        )

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "active_job_exists"
    assert second.json()["detail"]["active_job_id"] == first.json()["id"]


def test_resume_rejects_when_another_gpu_job_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    with client:
        first = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        first_id = first.json()["id"]
        store.update_status(first_id, JobStatus.PAUSED)
        second = client.post(
            "/v1/jobs",
            json={"release_id": "release-2", "rights_confirmed": True},
        )
        resumed = client.post(f"/v1/jobs/{first_id}/resume")

    assert second.status_code == 202
    assert resumed.status_code == 409
    assert resumed.json()["detail"]["code"] == "active_job_exists"
    assert resumed.json()["detail"]["active_job_id"] == second.json()["id"]
    assert store.get_job(first_id).status is JobStatus.PAUSED


def test_manual_subtitle_can_be_selected_or_replaced_by_asr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        store.update_status(
            job_id,
            JobStatus.NEEDS_SUBTITLE_SELECTION,
            details={
                **store.get_job(job_id).details,
                "subtitle_candidates": [{"subtitle_id": "sub-1"}],
            },
            force=True,
        )
        selected = client.post(f"/v1/jobs/{job_id}/subtitles/sub-1")

    assert selected.status_code == 200
    assert selected.json()["status"] == "ready_offline"
    assert selected.json()["details"]["selected_subtitle"]["subtitle_id"] == "sub-1"

    second_root = tmp_path / "asr"
    second_client, second_store, _ = _client(second_root, monkeypatch)
    with second_client:
        created = second_client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        second_id = created.json()["id"]
        second_store.update_status(
            second_id,
            JobStatus.NEEDS_SUBTITLE_SELECTION,
            details={
                **second_store.get_job(second_id).details,
                "subtitle_candidates": [{"subtitle_id": "sub-2"}],
            },
            force=True,
        )
        use_asr = second_client.post(
            f"/v1/jobs/{second_id}/subtitles/use-asr"
        )

    assert use_asr.status_code == 200
    assert use_asr.json()["status"] == "ready_offline"
    assert use_asr.json()["details"]["transcript_source"] == "asr"


def test_create_status_cancel_and_sse_flow(tmp_path: Path, monkeypatch) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={
                "release_id": "release-1",
                "rights_confirmed": True,
                "source_language": "auto",
                "models": {
                    "asr": "asr-small",
                    "translation": "mt-fast",
                    "tts": "tts-fast",
                },
            },
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        fetched = client.get(f"/v1/jobs/{job_id}")
        events = client.get(f"/v1/jobs/{job_id}/events?once=true")
        cancelled = client.post(f"/v1/jobs/{job_id}/cancel")

    assert acquisition.last_release_id == "release-1"
    assert acquisition.added_paused is True
    assert acquisition.resumed_task == "torrent-hash"
    assert fetched.json()["details"]["task_id"] == "torrent-hash"
    assert "event: job.created" in events.text
    assert "event: job.status" in events.text
    assert cancelled.json()["status"] == "cancelled"
    assert acquisition.paused_task == "torrent-hash"
    assert store.get_job(job_id).status == JobStatus.CANCELLED
    assert store.get_job(job_id).cancel_requested is True


def test_cancel_stays_pending_when_torrent_pause_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    acquisition.fail_pause = True
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        cancelled = client.post(f"/v1/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    warnings = cancelled.json()["details"]["warnings"]
    assert warnings[-1]["code"] == "download_pause_failed"
    assert store.list_events(job_id)[-1].event_type == "job.warning"


def test_cancel_during_create_pauses_new_task_before_finalizing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        settings = _settings(tmp_path)
        store = StateStore(settings.database_path)
        acquisition = BlockingAcquisition()
        acquisition.block_start = True
        coordinator = FakeCoordinator(store)
        monkeypatch.setattr(
            api_module,
            "inspect_gpu",
            lambda **kwargs: FakeGpuReport(),
        )
        application = create_app(
            settings=settings,
            state_store=store,
            acquisition_service=acquisition,
            coordinator=coordinator,
        )

        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                creating = asyncio.create_task(
                    client.post(
                        "/v1/jobs",
                        json={
                            "release_id": "release-1",
                            "rights_confirmed": True,
                        },
                    )
                )
                await asyncio.wait_for(acquisition.start_entered.wait(), timeout=1)
                job_id = store.list_jobs(limit=1)[0].id
                cancelling = asyncio.create_task(
                    client.post(f"/v1/jobs/{job_id}/cancel")
                )
                while store.get_job(job_id).status is not JobStatus.CANCELLING:
                    await asyncio.sleep(0)
                acquisition.release_start.set()
                created_response, cancel_response = await asyncio.gather(
                    creating,
                    cancelling,
                )

        assert created_response.status_code == 202
        assert cancel_response.status_code == 200
        assert created_response.json()["status"] == "cancelled"
        assert cancel_response.json()["status"] == "cancelled"
        assert acquisition.added_paused is True
        assert acquisition.resumed_task is None
        assert acquisition.paused_task == "torrent-hash"
        assert store.get_job(job_id).status is JobStatus.CANCELLED

    asyncio.run(scenario())


def test_cancel_during_existing_task_resume_pauses_after_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        settings = _settings(tmp_path)
        store = StateStore(settings.database_path)
        job = store.create_job("release-1", {"rights_confirmed": True})
        store.update_status(
            job.id,
            JobStatus.DOWNLOADING,
            details={"task_id": "torrent-hash", "name": "Fixture"},
        )
        store.update_status(job.id, JobStatus.PAUSED)
        acquisition = BlockingAcquisition()
        acquisition.block_resume = True
        coordinator = FakeCoordinator(store)
        monkeypatch.setattr(
            api_module,
            "inspect_gpu",
            lambda **kwargs: FakeGpuReport(),
        )
        application = create_app(
            settings=settings,
            state_store=store,
            acquisition_service=acquisition,
            coordinator=coordinator,
        )

        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                resuming = asyncio.create_task(
                    client.post(f"/v1/jobs/{job.id}/resume")
                )
                await asyncio.wait_for(acquisition.resume_entered.wait(), timeout=1)
                cancelling = asyncio.create_task(
                    client.post(f"/v1/jobs/{job.id}/cancel")
                )
                while store.get_job(job.id).status is not JobStatus.CANCELLING:
                    await asyncio.sleep(0)
                second_resume = await client.post(f"/v1/jobs/{job.id}/resume")
                acquisition.release_resume.set()
                resume_response, cancel_response = await asyncio.gather(
                    resuming,
                    cancelling,
                )

        assert resume_response.status_code == 200
        assert cancel_response.status_code == 200
        assert second_resume.status_code == 200
        assert second_resume.json()["status"] == "cancelling"
        assert resume_response.json()["status"] == "cancelled"
        assert cancel_response.json()["status"] == "cancelled"
        assert acquisition.resumed_task == "torrent-hash"
        assert acquisition.paused_task == "torrent-hash"
        assert acquisition.running is False
        assert acquisition.actions[-2:] == ["resume", "pause"]
        assert store.get_job(job.id).status is JobStatus.CANCELLED

    asyncio.run(scenario())


def test_cancel_during_resume_task_creation_never_starts_torrent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        settings = _settings(tmp_path)
        store = StateStore(settings.database_path)
        job = store.create_job("release-1", {"rights_confirmed": True})
        store.update_status(
            job.id,
            JobStatus.FAILED,
            error_code="download_start_failed",
            error_message="fixture",
            retryable=True,
        )
        acquisition = BlockingAcquisition()
        acquisition.block_start = True
        coordinator = FakeCoordinator(store)
        monkeypatch.setattr(
            api_module,
            "inspect_gpu",
            lambda **kwargs: FakeGpuReport(),
        )
        application = create_app(
            settings=settings,
            state_store=store,
            acquisition_service=acquisition,
            coordinator=coordinator,
        )

        async with application.router.lifespan_context(application):
            transport = httpx.ASGITransport(app=application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                resuming = asyncio.create_task(
                    client.post(f"/v1/jobs/{job.id}/resume")
                )
                await asyncio.wait_for(acquisition.start_entered.wait(), timeout=1)
                cancelling = asyncio.create_task(
                    client.post(f"/v1/jobs/{job.id}/cancel")
                )
                while store.get_job(job.id).status is not JobStatus.CANCELLING:
                    await asyncio.sleep(0)
                acquisition.release_start.set()
                resume_response, cancel_response = await asyncio.gather(
                    resuming,
                    cancelling,
                )

        assert resume_response.status_code == 200
        assert cancel_response.status_code == 200
        assert acquisition.added_paused is True
        assert acquisition.resumed_task is None
        assert acquisition.paused_task == "torrent-hash"
        assert store.get_job(job.id).details["task_id"] == "torrent-hash"
        assert store.get_job(job.id).status is JobStatus.CANCELLED

    asyncio.run(scenario())


def test_resume_failure_is_saved_as_retryable_job_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        store.update_status(job_id, JobStatus.PAUSED)
        acquisition.fail_resume = True
        response = client.post(f"/v1/jobs/{job_id}/resume")

    assert response.status_code == 503
    failed = store.get_job(job_id)
    assert failed.status == JobStatus.FAILED
    assert failed.retryable is True
    assert failed.error_code == "download_resume_failed"
    assert acquisition.running is False
    assert acquisition.actions[-2:] == ["resume", "pause"]


def test_ambiguous_resume_and_pause_failure_keeps_active_slot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        store.update_status(job_id, JobStatus.PAUSED)
        acquisition.fail_resume = True
        acquisition.fail_pause = True
        response = client.post(f"/v1/jobs/{job_id}/resume")

    assert response.status_code == 503
    guarded = store.get_job(job_id)
    assert guarded.status is JobStatus.DOWNLOADING
    assert guarded.active_slot is True
    assert guarded.details["backend_state_uncertain"] is True
    assert guarded.error_code == "download_resume_failed"
    with pytest.raises(ActiveJobExists):
        store.create_job("release-2", {"rights_confirmed": True})


def test_get_is_side_effect_free_and_refresh_uses_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        acquisition.coordinator.calls.clear()

        read_only = client.get(f"/v1/jobs/{job_id}")
        assert acquisition.coordinator.calls == []
        refreshed = client.post(f"/v1/jobs/{job_id}/refresh")

    assert read_only.json()["progress_permille"] == 0
    assert acquisition.coordinator.calls == [job_id]
    assert refreshed.json()["progress_permille"] == 500
    assert refreshed.json()["details"]["eta_seconds"] == 5


def test_background_monitor_refreshes_only_active_acquisition_jobs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    settings.acquisition_monitor_seconds = 0.1
    store = StateStore(settings.database_path)
    acquisition = FakeAcquisition()
    coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=coordinator,
    )
    terminal = store.create_job(
        "release-terminal",
        {"rights_confirmed": True},
        job_id="terminal",
    )
    store.request_cancel(terminal.id)
    store.finalize_cancel(terminal.id)

    with TestClient(application) as client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        deadline = time.monotonic() + 1.0
        while job_id not in coordinator.calls and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.15)

    assert job_id in coordinator.calls
    assert "terminal" not in coordinator.calls
    assert store.get_job(job_id).progress_permille == 500


def test_missing_job_has_vietnamese_safe_error(tmp_path: Path, monkeypatch) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        response = client.get("/v1/jobs/not-found")

    assert response.status_code == 404
    assert response.json()["detail"] == {
        "code": "job_not_found",
        "message": "Không tìm thấy job",
        "retryable": False,
    }


def test_create_freezes_default_asr_model_into_job_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )

    assert created.status_code == 202
    assert created.json()["spec"]["models"]["asr"] == "asr-small"


@pytest.mark.parametrize(
    ("timing_payload", "expected"),
    [({}, "natural"), ({"timing_profile": "strict"}, "strict")],
)
def test_create_freezes_timing_profile_into_job_spec(
    tmp_path: Path,
    monkeypatch,
    timing_payload: dict[str, str],
    expected: str,
) -> None:
    client, _, _ = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={
                "release_id": "release-1",
                "rights_confirmed": True,
                **timing_payload,
            },
        )

    assert created.status_code == 202
    assert created.json()["spec"]["timing_profile"] == expected


def test_user_can_select_language_after_uncertain_asr_detection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, _ = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        current = store.get_job(job_id)
        store.update_status(
            job_id,
            JobStatus.NEEDS_LANGUAGE,
            force=True,
            stage=JobStage.ASR,
            details={
                **current.details,
                "language_detection_candidates": [
                    {"language": "en", "probability": 0.49}
                ],
            },
        )
        selected = client.post(
            f"/v1/jobs/{job_id}/language",
            json={"language": "eng-US"},
        )

    assert selected.status_code == 200
    assert selected.json()["status"] == "transcribing"
    assert selected.json()["stage"] == "asr"
    assert selected.json()["details"]["source_language_override"] == "en"
    assert "language_detection_candidates" not in selected.json()["details"]


def test_offline_resume_never_resumes_completed_torrent_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        current = store.get_job(job_id)
        store.update_status(
            job_id,
            JobStatus.TRANSCRIBING,
            force=True,
            stage=JobStage.ASR,
            details={**current.details, "transcript_source": "asr"},
        )
        store.update_status(
            job_id,
            JobStatus.FAILED,
            error_code="asr_failed",
            error_message="ASR failed",
            retryable=True,
        )
        acquisition.resumed_task = None
        acquisition.actions.clear()
        resumed = client.post(f"/v1/jobs/{job_id}/resume")

    assert resumed.status_code == 200
    assert resumed.json()["status"] == "transcribing"
    assert acquisition.resumed_task is None
    assert acquisition.actions == []


def test_cancel_running_asr_waits_for_worker_without_pausing_torrent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        current = store.get_job(job_id)
        store.update_status(
            job_id,
            JobStatus.TRANSCRIBING,
            force=True,
            stage=JobStage.ASR,
            details={**current.details, "transcript_source": "asr"},
        )
        acquisition.paused_task = None
        cancelled = client.post(f"/v1/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    assert cancelled.json()["details"]["offline_cancel_pending"] is True
    assert acquisition.paused_task is None
    assert store.get_job(job_id).cancel_requested is True


def test_cancel_idle_offline_job_finishes_without_pausing_torrent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, store, acquisition = _client(tmp_path, monkeypatch)
    with client:
        created = client.post(
            "/v1/jobs",
            json={"release_id": "release-1", "rights_confirmed": True},
        )
        job_id = created.json()["id"]
        current = store.get_job(job_id)
        store.update_status(
            job_id,
            JobStatus.READY_OFFLINE,
            force=True,
            details={**current.details, "transcript_source": "asr"},
        )
        acquisition.paused_task = None
        cancelled = client.post(f"/v1/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert acquisition.paused_task is None
