from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from dub_server.acquisition.coordinator import AcquisitionCoordinator
from dub_server.domain import (
    AcquisitionError,
    AcquisitionErrorCode,
    DownloadedFile,
    DownloadState,
    DownloadStatus,
    MediaAsset,
    SubtitleCandidate,
    SubtitleFormat,
    SubtitleSource,
)
from dub_server.media_probe import FfprobeMediaProbe, MediaProbeError
from dub_server.state import JobStage, JobStatus, StateStore


class FakeMediaProbe:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    async def probe(self, path: Path, *, source_language: str, title=None, media_kind=None, year=None):
        self.paths.append(path)
        return MediaAsset(
            path=path,
            title=title or path.stem,
            duration_us=120_000_000,
            source_language="en" if source_language == "auto" else source_language,
            media_kind=media_kind,
            year=year,
            fps=24.0,
        )


class FakeService:
    def __init__(
        self,
        *,
        status: DownloadStatus,
        files: tuple[DownloadedFile, ...] = (),
        candidates: tuple[SubtitleCandidate, ...] = (),
        find_error: AcquisitionError | None = None,
        materialize_failures: set[str] | None = None,
    ) -> None:
        self.status = status
        self.files = files
        self.candidates = candidates
        self.find_error = find_error
        self.materialize_failures = materialize_failures or set()
        self.calls: list[tuple[str, object]] = []

    async def download_status(self, task_id: str) -> DownloadStatus:
        self.calls.append(("status", task_id))
        return self.status

    async def download_files(self, task_id: str) -> tuple[DownloadedFile, ...]:
        self.calls.append(("files", task_id))
        return self.files

    async def find_subtitles(self, media: MediaAsset) -> tuple[SubtitleCandidate, ...]:
        self.calls.append(("subtitles", media.path))
        if self.find_error is not None:
            raise self.find_error
        return self.candidates

    async def materialize_subtitle(
        self,
        media: MediaAsset,
        candidate: SubtitleCandidate,
        destination: Path,
    ) -> Path:
        self.calls.append(("materialize", candidate.subtitle_id))
        if candidate.subtitle_id in self.materialize_failures:
            raise AcquisitionError(
                AcquisitionErrorCode.SUBTITLE_INVALID,
                "Phụ đề thử nghiệm bị hỏng",
                retryable=False,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        return destination


def _status(
    state: DownloadState,
    *,
    progress: float = 1.0,
    downloaded: int = 1000,
    total: int = 1000,
) -> DownloadStatus:
    return DownloadStatus(
        task_id="task-1",
        state=state,
        progress=progress,
        downloaded_bytes=downloaded,
        total_bytes=total,
        speed_bytes_per_second=125,
        eta_seconds=7,
    )


def _job(store: StateStore, *, mode: str = "prefer", language: str = "en"):
    job = store.create_job(
        "release-1",
        {"source_language": language, "subtitle_mode": mode},
        job_id=f"job-{mode}-{language}",
    )
    return store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        stage=JobStage.ACQUISITION,
        details={"task_id": "task-1", "name": "Legal Fixture"},
    )


def _video(incoming: Path, job_id: str, relative: Path, size: int) -> DownloadedFile:
    path = incoming / job_id / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return DownloadedFile(relative, size, 1.0)


def _candidate(identifier: str, score: int, *, high: bool) -> SubtitleCandidate:
    return SubtitleCandidate(
        subtitle_id=identifier,
        source=SubtitleSource.SIDECAR,
        language="en",
        format=SubtitleFormat.SRT,
        score=score,
        high_confidence=high,
        matched_by="test",
    )


def test_refresh_updates_download_metrics_and_checkpoint(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    service = FakeService(status=_status(DownloadState.DOWNLOADING, progress=0.25, downloaded=250))
    coordinator = AcquisitionCoordinator(
        service, store, tmp_path / "incoming", tmp_path / "jobs", FakeMediaProbe()
    )

    result = asyncio.run(coordinator.refresh(job.id))

    assert result.status is JobStatus.DOWNLOADING
    # Download owns the first 200 permille of overall pipeline progress.
    assert result.progress_permille == 50
    assert result.details["stage_progress_permille"] == 250
    assert result.details["downloaded_bytes"] == 250
    assert result.details["speed_bytes_per_second"] == 125
    assert result.details["eta_seconds"] == 7
    checkpoint = store.get_checkpoint(job.id, JobStage.ACQUISITION)
    assert checkpoint is not None and checkpoint.payload["downloaded_bytes"] == 250
    assert service.calls == [("status", "task-1")]


def test_refresh_does_not_poll_before_backend_start_ack(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    job = store.update_status(
        job.id,
        JobStatus.DOWNLOADING,
        details={**job.details, "backend_started": False},
    )
    service = FakeService(
        status=_status(DownloadState.PAUSED, progress=0.0, downloaded=0),
    )
    coordinator = AcquisitionCoordinator(
        service,
        store,
        tmp_path / "incoming",
        tmp_path / "jobs",
        FakeMediaProbe(),
    )

    result = asyncio.run(coordinator.refresh(job.id))

    assert result.status is JobStatus.DOWNLOADING
    assert result.details["backend_started"] is False
    assert service.calls == []


def test_completed_download_chooses_largest_video_and_best_confident_subtitle(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    incoming = tmp_path / "incoming"
    files = (
        _video(incoming, job.id, Path("sample.mp4"), 50),
        _video(incoming, job.id, Path("Feature", "movie.mkv"), 500),
        _video(incoming, job.id, Path("Feature", "bonus.mp4"), 100),
    )
    candidates = (
        _candidate("low-confidence", 999, high=False),
        _candidate("high-80", 80, high=True),
        _candidate("high-95", 95, high=True),
    )
    service = FakeService(status=_status(DownloadState.COMPLETED), files=files, candidates=candidates)
    probe = FakeMediaProbe()
    coordinator = AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", probe)

    result = asyncio.run(coordinator.refresh(job.id))

    assert result.status is JobStatus.READY_OFFLINE
    assert result.stage is JobStage.SUBTITLE
    assert result.details["transcript_source"] == "subtitle"
    assert result.details["selected_subtitle"]["subtitle_id"] == "high-95"
    assert Path(result.details["source_subtitle_path"]).is_file()
    assert probe.paths == [(incoming / job.id / "Feature" / "movie.mkv").resolve()]
    assert ("materialize", "high-95") in service.calls
    acquisition = store.get_checkpoint(job.id, JobStage.ACQUISITION)
    assert acquisition is not None
    assert acquisition.payload["selected_media"]["relative_path"] == "Feature/movie.mkv"


def test_asr_mode_skips_subtitle_provider(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store, mode="asr")
    incoming = tmp_path / "incoming"
    files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
    service = FakeService(status=_status(DownloadState.COMPLETED), files=files)
    coordinator = AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", FakeMediaProbe())

    result = asyncio.run(coordinator.refresh(job.id))

    assert result.status is JobStatus.READY_OFFLINE
    assert result.details["transcript_source"] == "asr"
    assert result.details["subtitle_fallback_reason"] == "asr_requested"
    assert all(call[0] != "subtitles" for call in service.calls)


def test_unknown_source_language_skips_remote_subtitle_lookup(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store, language="und")
    incoming = tmp_path / "incoming"
    files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
    service = FakeService(status=_status(DownloadState.COMPLETED), files=files)

    result = asyncio.run(
        AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", FakeMediaProbe()).refresh(job.id)
    )

    assert result.status is JobStatus.READY_OFFLINE
    assert result.progress_permille == 250
    assert result.details["transcript_source"] == "asr"
    assert result.details["subtitle_fallback_reason"] == "source_language_unknown"
    assert all(call[0] != "subtitles" for call in service.calls)


def test_refresh_calls_are_serialized_per_job(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)

    class SlowService(FakeService):
        active = 0
        maximum_active = 0

        async def download_status(self, task_id: str) -> DownloadStatus:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return await super().download_status(task_id)

    service = SlowService(status=_status(DownloadState.DOWNLOADING, progress=0.1))
    coordinator = AcquisitionCoordinator(
        service, store, tmp_path / "incoming", tmp_path / "jobs", FakeMediaProbe()
    )

    async def scenario():
        return await asyncio.gather(coordinator.refresh(job.id), coordinator.refresh(job.id))

    results = asyncio.run(scenario())

    assert service.maximum_active == 1
    assert all(result.status is JobStatus.DOWNLOADING for result in results)


def test_manual_mode_waits_only_when_candidates_exist(tmp_path: Path) -> None:
    async def scenario(has_candidates: bool):
        root = tmp_path / ("with" if has_candidates else "without")
        store = StateStore(root / "state.db")
        job = _job(store, mode="manual")
        incoming = root / "incoming"
        files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
        candidates = (_candidate("manual", 80, high=True),) if has_candidates else ()
        service = FakeService(status=_status(DownloadState.COMPLETED), files=files, candidates=candidates)
        result = await AcquisitionCoordinator(
            service, store, incoming, root / "jobs", FakeMediaProbe()
        ).refresh(job.id)
        return result

    with_candidates = asyncio.run(scenario(True))
    without_candidates = asyncio.run(scenario(False))

    assert with_candidates.status is JobStatus.NEEDS_SUBTITLE_SELECTION
    assert with_candidates.details["transcript_source"] == "pending_manual_subtitle"
    assert without_candidates.status is JobStatus.READY_OFFLINE
    assert without_candidates.details["transcript_source"] == "asr"


def test_manual_selection_materializes_candidate_and_can_choose_asr(
    tmp_path: Path,
) -> None:
    async def scenario(root: Path, *, use_asr: bool):
        store = StateStore(root / "state.db")
        job = _job(store, mode="manual")
        incoming = root / "incoming"
        candidate = _candidate("manual", 80, high=True)
        service = FakeService(
            status=_status(DownloadState.COMPLETED),
            files=(_video(incoming, job.id, Path("movie.mp4"), 100),),
            candidates=(candidate,),
        )
        coordinator = AcquisitionCoordinator(
            service, store, incoming, root / "jobs", FakeMediaProbe()
        )
        waiting = await coordinator.refresh(job.id)
        if use_asr:
            selected = await coordinator.select_asr(job.id)
        else:
            selected = await coordinator.select_subtitle(job.id, candidate.subtitle_id)
        return waiting, selected, service, store

    waiting, selected, service, store = asyncio.run(
        scenario(tmp_path / "subtitle", use_asr=False)
    )
    assert waiting.status is JobStatus.NEEDS_SUBTITLE_SELECTION
    assert selected.status is JobStatus.READY_OFFLINE
    assert selected.details["transcript_source"] == "subtitle"
    assert Path(selected.details["source_subtitle_path"]).is_file()
    assert ("materialize", "manual") in service.calls
    assert store.get_checkpoint(selected.id, JobStage.SUBTITLE) is not None

    _, asr, asr_service, _ = asyncio.run(
        scenario(tmp_path / "asr", use_asr=True)
    )
    assert asr.status is JobStatus.READY_OFFLINE
    assert asr.details["transcript_source"] == "asr"
    assert asr.details["subtitle_fallback_reason"] == "user_selected_asr"
    assert all(call[0] != "materialize" for call in asr_service.calls)


def test_prefer_without_high_confidence_falls_back_to_asr(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    incoming = tmp_path / "incoming"
    files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
    service = FakeService(
        status=_status(DownloadState.COMPLETED),
        files=files,
        candidates=(_candidate("weak", 70, high=False),),
    )
    result = asyncio.run(
        AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", FakeMediaProbe()).refresh(job.id)
    )

    assert result.status is JobStatus.READY_OFFLINE
    assert result.details["transcript_source"] == "asr"
    assert result.details["subtitle_candidates"][0]["subtitle_id"] == "weak"
    assert all(call[0] != "materialize" for call in service.calls)


def test_prefer_falls_back_when_subtitle_search_is_unavailable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    incoming = tmp_path / "incoming"
    files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
    service = FakeService(
        status=_status(DownloadState.COMPLETED),
        files=files,
        find_error=AcquisitionError(
            AcquisitionErrorCode.SUBTITLE_UNAVAILABLE,
            "Dịch vụ phụ đề tạm thời không khả dụng",
            retryable=True,
        ),
    )

    result = asyncio.run(
        AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", FakeMediaProbe()).refresh(job.id)
    )

    assert result.status is JobStatus.READY_OFFLINE
    assert result.details["transcript_source"] == "asr"
    assert result.details["subtitle_fallback_reason"] == "subtitle_search_failed"
    assert result.details["subtitle_warnings"][0]["retryable"] is True


def test_prefer_tries_next_confident_subtitle_when_best_is_corrupt(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    incoming = tmp_path / "incoming"
    files = (_video(incoming, job.id, Path("movie.mp4"), 100),)
    candidates = (
        _candidate("best-corrupt", 100, high=True),
        _candidate("second-valid", 90, high=True),
    )
    service = FakeService(
        status=_status(DownloadState.COMPLETED),
        files=files,
        candidates=candidates,
        materialize_failures={"best-corrupt"},
    )

    result = asyncio.run(
        AcquisitionCoordinator(service, store, incoming, tmp_path / "jobs", FakeMediaProbe()).refresh(job.id)
    )

    assert result.status is JobStatus.READY_OFFLINE
    assert result.details["selected_subtitle"]["subtitle_id"] == "second-valid"
    assert result.details["subtitle_warnings"][0]["subtitle_id"] == "best-corrupt"


@pytest.mark.parametrize("relative", [Path("../escape.mp4"), Path("C:/escape.mp4")])
def test_unsafe_or_missing_video_becomes_typed_failure(tmp_path: Path, relative: Path) -> None:
    store = StateStore(tmp_path / "state.db")
    job = _job(store)
    service = FakeService(
        status=_status(DownloadState.COMPLETED),
        files=(DownloadedFile(relative, 10_000, 1.0),),
    )

    result = asyncio.run(
        AcquisitionCoordinator(
            service, store, tmp_path / "incoming", tmp_path / "jobs", FakeMediaProbe()
        ).refresh(job.id)
    )

    assert result.status is JobStatus.FAILED
    assert result.error_code == "invalid_download_path"
    assert result.retryable is False
    assert store.get_checkpoint(job.id, JobStage.ACQUISITION) is not None


def test_ffprobe_media_probe_parses_duration_language_and_fps(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mkv"
    media.write_bytes(b"fixture")
    commands: list[tuple[str, ...]] = []

    async def runner(command):
        commands.append(tuple(command))
        payload = {
            "format": {"duration": "12.345678", "tags": {"title": "Embedded title"}},
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_type": "video",
                    "avg_frame_rate": "24000/1001",
                },
                {
                    "index": 2,
                    "codec_type": "audio",
                    "start_time": "0.125",
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1},
                },
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    asset = asyncio.run(
        FfprobeMediaProbe(runner=runner).probe(media, source_language="auto")
    )

    assert asset.duration_us == 12_345_678
    assert asset.source_language == "eng"
    assert asset.title == "Embedded title"
    assert asset.fps == pytest.approx(23.976, rel=1e-4)
    assert asset.audio_stream_index == 2
    assert asset.audio_start_us == 125_000
    assert asset.video_stream_index == 0
    assert asset.video_codec == "h264"
    assert commands[0][0] == "ffprobe"
    assert commands[0][1:4] == ("-v", "error", "-protocol_whitelist")
    assert commands[0][4] == "file"
    assert commands[0][-1] == str(media.resolve())


def test_ffprobe_media_probe_selects_matching_audio_language_before_default(
    tmp_path: Path,
) -> None:
    media = tmp_path / "multi-audio.mkv"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {"index": 0, "codec_type": "video", "avg_frame_rate": "25/1"},
                {
                    "index": 1,
                    "codec_type": "audio",
                    "tags": {"language": "jpn"},
                    "disposition": {"default": 1},
                },
                {
                    "index": 3,
                    "codec_type": "audio",
                    "tags": {"language": "eng"},
                    "disposition": {"default": 0},
                },
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    asset = asyncio.run(
        FfprobeMediaProbe(runner=runner).probe(media, source_language="en")
    )

    assert asset.audio_stream_index == 3
    assert asset.source_language == "en"


def test_ffprobe_media_probe_rejects_audio_only_input(tmp_path: Path) -> None:
    media = tmp_path / "audio.m4a"
    media.write_bytes(b"fixture")

    async def runner(command):
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"format": {"duration": "1"}, "streams": [{"codec_type": "audio"}]}),
            "",
        )

    with pytest.raises(MediaProbeError, match="luồng video"):
        asyncio.run(FfprobeMediaProbe(runner=runner).probe(media, source_language="en"))


def test_ffprobe_media_probe_accepts_h264_mkv_for_mp4_passthrough(
    tmp_path: Path,
) -> None:
    media = tmp_path / "feature.mkv"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_type": "video",
                    "avg_frame_rate": "24/1",
                    "disposition": {"attached_pic": 0},
                },
                {"index": 1, "codec_name": "aac", "codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    asset = asyncio.run(
        FfprobeMediaProbe(runner=runner).probe(
            media,
            source_language="en",
            require_h264_passthrough=True,
        )
    )

    assert asset.video_codec == "h264"
    assert asset.video_stream_index == 0


@pytest.mark.parametrize("codec_name", ["vp8", "hevc", "ffv1"])
def test_ffprobe_media_probe_rejects_non_h264_mp4_passthrough(
    tmp_path: Path,
    codec_name: str,
) -> None:
    media = tmp_path / f"feature-{codec_name}.mkv"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {"index": 0, "codec_name": codec_name, "codec_type": "video"},
                {"index": 1, "codec_name": "aac", "codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(MediaProbeError) as raised:
        asyncio.run(
            FfprobeMediaProbe(runner=runner).probe(
                media,
                source_language="en",
                require_h264_passthrough=True,
            )
        )

    assert raised.value.code == "unsupported_media"
    assert raised.value.retryable is False


def test_ffprobe_media_probe_keeps_legacy_non_h264_acquisition_compatible(
    tmp_path: Path,
) -> None:
    media = tmp_path / "legacy-vp8.mkv"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {"index": 0, "codec_name": "vp8", "codec_type": "video"},
                {"index": 1, "codec_name": "opus", "codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    asset = asyncio.run(
        FfprobeMediaProbe(runner=runner).probe(media, source_language="en")
    )

    assert asset.video_codec == "vp8"


def test_ffprobe_media_probe_ignores_attached_picture_before_content_video(
    tmp_path: Path,
) -> None:
    media = tmp_path / "cover-first.mkv"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_type": "video",
                    "disposition": {"attached_pic": 1},
                },
                {"index": 1, "codec_name": "h264", "codec_type": "video"},
                {"index": 2, "codec_name": "aac", "codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    asset = asyncio.run(
        FfprobeMediaProbe(runner=runner).probe(
            media,
            source_language="en",
            require_h264_passthrough=True,
        )
    )

    assert asset.video_stream_index == 1
    assert asset.video_codec == "h264"


def test_ffprobe_media_probe_rejects_cover_only_input(tmp_path: Path) -> None:
    media = tmp_path / "cover-only.mp4"
    media.write_bytes(b"fixture")

    async def runner(command):
        payload = {
            "format": {"duration": "5"},
            "streams": [
                {
                    "index": 0,
                    "codec_name": "mjpeg",
                    "codec_type": "video",
                    "disposition": {"attached_pic": 1},
                },
                {"index": 1, "codec_name": "aac", "codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(MediaProbeError) as raised:
        asyncio.run(
            FfprobeMediaProbe(runner=runner).probe(
                media,
                source_language="en",
                require_h264_passthrough=True,
            )
        )

    assert raised.value.code == "unsupported_media"
    assert "chỉ chứa ảnh bìa" in raised.value.message_vi
    assert raised.value.retryable is False
