from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

import dub_server.api as api_module
from dub_server.api import create_app
from dub_server.config import Settings
from dub_server.domain import DownloadTask, MediaKind, MediaQuery, ReleaseCandidate
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


def _client(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    store = StateStore(settings.database_path)
    acquisition = FakeAcquisition()
    acquisition.coordinator = FakeCoordinator(store)
    monkeypatch.setattr(api_module, "inspect_gpu", lambda **kwargs: FakeGpuReport())
    application = create_app(
        settings=settings,
        state_store=store,
        acquisition_service=acquisition,
        coordinator=acquisition.coordinator,
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
    assert capabilities.json()["offline_inference"] is True
    assert capabilities.json()["drm_supported"] is False
    assert models.json()["models"][0]["id"] == "asr-small"
    assert models.json()["models"][0]["valid"] is False


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
