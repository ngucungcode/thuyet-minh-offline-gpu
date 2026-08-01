"""FastAPI control plane for acquisition and durable job orchestration."""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import stat as stat_module
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Sequence

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import (
    ModelCatalog,
    Settings,
    get_settings,
    load_model_catalog,
    read_secret,
)
from .asr import TranscriptionError, normalize_whisper_language
from .domain import (
    AcquisitionError,
    AcquisitionErrorCode,
    MediaKind,
    MediaQuery,
    ReleaseCandidate,
)
from .gpu import inspect_gpu, read_gpu_report
from .state import (
    ActiveJobExists,
    InvalidTransition,
    JobNotFound,
    JobRecord,
    JobStage,
    JobStatus,
    StateStore,
)


_SHA256_HEX = frozenset("0123456789abcdef")


def _sealed_file_sha256(path: Path) -> tuple[str, int]:
    """Hash one regular file without following a final-component symlink."""

    if path.is_symlink():
        raise OSError("artifact path is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode):
            raise OSError("artifact is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.is_symlink():
            raise OSError("artifact path changed to a symlink")
        return digest.hexdigest(), metadata.st_size
    finally:
        os.close(descriptor)


class AcquisitionPort(Protocol):
    async def search(
        self,
        query: str | MediaQuery,
        year: int | None = None,
        media_type: MediaKind | str = MediaKind.MOVIE,
    ) -> Sequence[ReleaseCandidate]: ...

    async def start_download(
        self,
        release: ReleaseCandidate | str,
        save_path: Path,
        *,
        rights_confirmed: bool,
        paused: bool = False,
    ) -> Any: ...


class CoordinatorPort(Protocol):
    async def refresh(self, job_id: str) -> JobRecord | None: ...

    async def select_subtitle(
        self, job_id: str, subtitle_id: str
    ) -> JobRecord: ...

    async def select_asr(self, job_id: str) -> JobRecord: ...


_LOGGER = logging.getLogger(__name__)
_WEB_STATIC_ROOT = Path(__file__).resolve().with_name("web_static")


async def _monitor_acquisition_jobs(
    coordinator: CoordinatorPort,
    store: StateStore,
    stop: asyncio.Event,
    interval_seconds: float,
) -> None:
    """Refresh only network acquisition stages; inference stays offline."""

    monitored = (JobStatus.DOWNLOADING, JobStatus.SUBTITLE_MATCHING)
    while not stop.is_set():
        for record in store.list_jobs(monitored, limit=100):
            if stop.is_set():
                return
            try:
                await coordinator.refresh(record.id)
            except JobNotFound:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # Adapter/coordinator errors are expected to be persisted as
                # typed job failures.  One broken job must not stop monitoring.
                _LOGGER.exception("Acquisition refresh failed for job %s", record.id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except TimeoutError:
            pass


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    year: int | None = Field(default=None, ge=1888, le=2200)
    media_type: Literal["movie"] = "movie"


class ReleaseResponse(BaseModel):
    release_id: str
    title: str
    indexer_id: int
    protocol: str
    info_hash: str | None = None
    size_bytes: int | None = None
    seeders: int | None = None
    leechers: int | None = None
    published_at: str | None = None
    categories: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[ReleaseResponse]


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asr: str | None = None
    translation: str | None = None
    separation: str | None = None
    tts: str | None = None


class VoiceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_id: str | None = Field(default=None, max_length=200)
    reference_path: str | None = Field(default=None, max_length=1000)


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_id: str = Field(min_length=1, max_length=500)
    rights_confirmed: bool = False
    source_language: str = Field(default="auto", min_length=2, max_length=35)
    subtitle_mode: Literal["prefer", "manual", "asr"] = "prefer"
    models: ModelSelection = Field(default_factory=ModelSelection)
    voice: VoiceSelection | None = None
    voice_rights_confirmed: bool = False


class LanguageSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(min_length=2, max_length=35)


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    release_id: str
    status: str
    stage: str
    progress_permille: int
    spec: dict[str, Any]
    details: dict[str, Any]
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    cancel_requested: bool
    revision: int
    created_at: str
    updated_at: str


class JobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[JobResponse]
    count: int


def _public_release(candidate: ReleaseCandidate) -> ReleaseResponse:
    """Never expose provider URLs or opaque GUIDs through the public API."""

    return ReleaseResponse(
        release_id=candidate.release_id,
        title=candidate.title,
        indexer_id=candidate.indexer_id,
        protocol=candidate.protocol,
        info_hash=candidate.info_hash,
        size_bytes=candidate.size_bytes,
        seeders=candidate.seeders,
        leechers=candidate.leechers,
        published_at=candidate.published_at,
        categories=list(candidate.categories),
    )


def _job_response(record: JobRecord) -> JobResponse:
    return JobResponse.model_validate(record.to_dict())


def _adapter_value(value: Any, name: str, default: Any = None) -> Any:
    if is_dataclass(value):
        return asdict(value).get(name, default)
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _release_snapshot(candidate: ReleaseCandidate) -> dict[str, Any] | None:
    """Build a restart token without persisting provider URLs or credentials."""

    info_hash = (candidate.info_hash or "").strip().lower()
    if len(info_hash) != 40 or any(
        character not in "0123456789abcdef" for character in info_hash
    ):
        return None
    safe_magnet = f"magnet:?xt=urn:btih:{info_hash}"
    if candidate.download_uri.strip().lower() != safe_magnet:
        # HTTP URLs and magnets with tracker parameters may contain private
        # tokens. Re-resolve them from the saved search query after restart.
        return None
    return {
        "release_id": candidate.release_id,
        "title": candidate.title,
        "indexer_id": candidate.indexer_id,
        "protocol": "torrent",
        "download_uri": safe_magnet,
        "guid": "",
        "info_hash": info_hash,
        "size_bytes": candidate.size_bytes,
        "seeders": candidate.seeders,
        "leechers": candidate.leechers,
        "published_at": candidate.published_at,
        "categories": list(candidate.categories),
    }


def _release_from_snapshot(
    payload: object,
    *,
    expected_release_id: str,
) -> ReleaseCandidate | None:
    if not isinstance(payload, dict):
        return None
    try:
        values = dict(payload)
        values["categories"] = tuple(values.get("categories") or ())
        candidate = ReleaseCandidate(**values)
    except (TypeError, ValueError):
        return None
    expected_uri = f"magnet:?xt=urn:btih:{(candidate.info_hash or '').lower()}"
    if (
        candidate.release_id != expected_release_id
        or candidate.guid
        or candidate.download_uri.lower() != expected_uri
    ):
        return None
    return candidate


def _acquisition_error_status(error: AcquisitionError) -> int:
    code = error.code.value
    if code == "rights_confirmation_required":
        return status.HTTP_403_FORBIDDEN
    if code in {"download_unavailable", "download_not_found"}:
        return status.HTTP_409_CONFLICT
    if code == "invalid_response":
        return status.HTTP_502_BAD_GATEWAY
    return status.HTTP_503_SERVICE_UNAVAILABLE if error.retryable else 400


def _validate_model_selection(
    selection: ModelSelection,
    catalog: ModelCatalog,
) -> None:
    by_id = {entry.id: entry for entry in catalog.models}
    requested = {
        "asr": selection.asr,
        "mt": selection.translation,
        "separation": selection.separation,
        "tts": selection.tts,
    }
    for expected_stage, model_id in requested.items():
        if model_id is None:
            continue
        entry = by_id.get(model_id)
        if entry is None or entry.stage != expected_stage:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_model_selection",
                    "message": (
                        f"Model {model_id} không thuộc lựa chọn {expected_stage}"
                    ),
                    "retryable": False,
                },
            )


def create_app(
    *,
    settings: Settings | None = None,
    state_store: StateStore | None = None,
    acquisition_service: AcquisitionPort | None = None,
    coordinator: CoordinatorPort | None = None,
) -> FastAPI:
    """Create an app with explicit adapter injection for deterministic tests."""

    configured_settings = settings or get_settings()
    active_job_operations: dict[str, asyncio.Task[Any]] = {}
    cancellation_reconciliations: dict[str, asyncio.Task[None]] = {}
    backend_mutation_lock = asyncio.Lock()

    async def mutate_backend(
        operation: Any,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        async def invoke() -> Any:
            async with backend_mutation_lock:
                return await operation(*args, **kwargs)

        if timeout is None:
            return await invoke()
        return await asyncio.wait_for(invoke(), timeout=max(0.05, timeout))

    async def pause_backend_for_cancel(
        store: StateStore,
        service: Any,
        job_id: str,
        task_id: str,
        *,
        timeout: float,
    ) -> str:
        """Pause one task without racing a new owner's backend mutation."""

        async def pause_if_unowned() -> str:
            async with backend_mutation_lock:
                current = store.get_job(job_id)
                if (
                    not current.active_slot
                    and any(
                        active.id != job_id
                        for active in store.list_active_jobs(limit=1000)
                    )
                ):
                    return "deferred"
                pause_download = getattr(service, "pause_download", None)
                if not callable(pause_download):
                    return "unsupported"
                await pause_download(task_id)
                return "paused"

        try:
            return await asyncio.wait_for(
                pause_if_unowned(),
                timeout=max(0.05, timeout),
            )
        except Exception:
            return "failed"

    def schedule_cancel_reconciliation(
        owner: asyncio.Task[Any],
        store: StateStore,
        service: Any,
        job_id: str,
    ) -> None:
        existing = cancellation_reconciliations.get(job_id)
        if existing is not None and not existing.done():
            return

        async def reconcile() -> None:
            try:
                await asyncio.shield(owner)
            except asyncio.CancelledError:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise
            except Exception:
                pass
            current = store.get_job(job_id)
            if current.status is not JobStatus.CANCELLING:
                return
            task_id = current.details.get("task_id")
            if not task_id:
                store.finalize_cancel(job_id)
                return
            result = await pause_backend_for_cancel(
                store,
                service,
                job_id,
                str(task_id),
                timeout=1.5,
            )
            if result == "paused":
                store.finalize_cancel(job_id)
            elif result == "deferred":
                store.append_warning(
                    job_id,
                    "backend_cleanup_deferred",
                    "Chưa dừng torrent cũ vì một job khác đang hoạt động",
                )
            else:
                store.append_warning(
                    job_id,
                    "download_pause_failed",
                    "Chưa thể hoàn tất hủy torrent sau thao tác tải đang chạy",
                )

        task = asyncio.create_task(
            reconcile(),
            name=f"cancel-reconcile-{job_id}",
        )
        cancellation_reconciliations[job_id] = task

        def discard(completed: asyncio.Task[None]) -> None:
            if cancellation_reconciliations.get(job_id) is completed:
                cancellation_reconciliations.pop(job_id, None)

        task.add_done_callback(discard)

    def track_job_operation(job_id: str) -> None:
        """Expose in-flight torrent mutations to the cancellation endpoint.

        Cancellation first records the durable CANCELLING state.  The request
        that currently owns a qBittorrent mutation then performs the final
        pause, which prevents a late resume/add call from running after a
        cancellation has already been finalized.
        """

        task = asyncio.current_task()
        if task is None:
            return
        existing = active_job_operations.get(job_id)
        if existing is not None and existing is not task and not existing.done():
            raise InvalidTransition("Job đang có một thao tác tải nguồn khác")
        active_job_operations[job_id] = task

        def discard(completed: asyncio.Task[Any]) -> None:
            if active_job_operations.get(job_id) is completed:
                active_job_operations.pop(job_id, None)

        task.add_done_callback(discard)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if state_store is None:
            configured_settings.ensure_local_directories()
            application.state.job_store = StateStore(
                configured_settings.database_path
            )
        else:
            application.state.job_store = state_store
        for interrupted in application.state.job_store.list_jobs(
            (JobStatus.CREATED,),
            limit=1000,
        ):
            application.state.job_store.update_status(
                interrupted.id,
                JobStatus.FAILED,
                error_code="download_start_interrupted",
                error_message=(
                    "API đã dừng khi đang khởi tạo tải nguồn; hãy chọn Tiếp tục"
                ),
                retryable=True,
            )
        client: httpx.AsyncClient | None = None
        configured_acquisition = acquisition_service
        if configured_acquisition is None:
            prowlarr_key = read_secret(configured_settings.prowlarr_api_key_file)
            qbittorrent_password = read_secret(
                configured_settings.qbittorrent_password_file
            )
            if prowlarr_key and qbittorrent_password:
                from .acquisition import build_acquisition_service

                client = httpx.AsyncClient(follow_redirects=False)
                configured_acquisition = build_acquisition_service(
                    client=client,
                    prowlarr_url=configured_settings.prowlarr_url,
                    prowlarr_api_key=prowlarr_key,
                    qbittorrent_url=configured_settings.qbittorrent_url,
                    qbittorrent_username=configured_settings.qbittorrent_username,
                    qbittorrent_password=qbittorrent_password,
                    opensubtitles_api_key=read_secret(
                        configured_settings.opensubtitles_api_key_file
                    ),
                    opensubtitles_token=read_secret(
                        configured_settings.opensubtitles_token_file
                    ),
                    opensubtitles_base_url=(
                        configured_settings.opensubtitles_url.rstrip("/")
                        .removesuffix("/api/v1")
                    ),
                    opensubtitles_user_agent=(
                        configured_settings.opensubtitles_user_agent
                    ),
                )
        configured_coordinator = coordinator
        if configured_coordinator is None and configured_acquisition is not None:
            from .acquisition.coordinator import AcquisitionCoordinator

            configured_coordinator = AcquisitionCoordinator(
                configured_acquisition,
                application.state.job_store,
                configured_settings.incoming_dir,
                configured_settings.jobs_dir,
            )
        application.state.acquisition = configured_acquisition
        application.state.coordinator = configured_coordinator
        for pending_start in application.state.job_store.list_jobs(
            (JobStatus.DOWNLOADING,),
            limit=1000,
        ):
            if pending_start.details.get("backend_started") is not False:
                continue
            task_id = pending_start.details.get("task_id")
            resume_download = getattr(configured_acquisition, "resume_download", None)
            if not task_id or not callable(resume_download):
                application.state.job_store.update_status(
                    pending_start.id,
                    JobStatus.FAILED,
                    error_code="download_start_interrupted",
                    error_message=(
                        "API đã dừng trước khi xác nhận trình tải nguồn bắt đầu"
                    ),
                    retryable=True,
                )
                continue
            try:
                await mutate_backend(
                    resume_download,
                    str(task_id),
                    timeout=15.0,
                )
            except Exception:
                pause_result = await pause_backend_for_cancel(
                    application.state.job_store,
                    configured_acquisition,
                    pending_start.id,
                    str(task_id),
                    timeout=1.5,
                )
                message = (
                    "Không thể khôi phục trình tải nguồn sau khi API khởi động lại"
                )
                if pause_result == "paused":
                    application.state.job_store.update_status(
                        pending_start.id,
                        JobStatus.FAILED,
                        error_code="download_start_interrupted",
                        error_message=message,
                        retryable=True,
                    )
                else:
                    application.state.job_store.update_status(
                        pending_start.id,
                        JobStatus.DOWNLOADING,
                        details={
                            **pending_start.details,
                            "backend_started": True,
                            "backend_state_uncertain": True,
                        },
                        error_code="download_start_interrupted",
                        error_message=message,
                        retryable=True,
                        expected_status=JobStatus.DOWNLOADING,
                    )
            else:
                application.state.job_store.update_status(
                    pending_start.id,
                    JobStatus.DOWNLOADING,
                    details={
                        **pending_start.details,
                        "backend_started": True,
                    },
                    expected_status=JobStatus.DOWNLOADING,
                )
        for pending_cancel in application.state.job_store.list_jobs(
            (JobStatus.CANCELLING,),
            limit=1000,
        ):
            task_id = pending_cancel.details.get("task_id")
            if not task_id:
                application.state.job_store.finalize_cancel(pending_cancel.id)
            else:
                pause_result = await pause_backend_for_cancel(
                    application.state.job_store,
                    configured_acquisition,
                    pending_cancel.id,
                    str(task_id),
                    timeout=1.5,
                )
                if pause_result == "paused":
                    application.state.job_store.finalize_cancel(pending_cancel.id)
                elif pause_result == "deferred":
                    application.state.job_store.append_warning(
                        pending_cancel.id,
                        "backend_cleanup_deferred",
                        "Chưa dừng torrent cũ vì một job khác đang hoạt động",
                    )
                else:
                    application.state.job_store.append_warning(
                        pending_cancel.id,
                        "download_pause_failed",
                        "Chưa thể hoàn tất hủy torrent sau khi API khởi động lại",
                    )
        monitor_stop = asyncio.Event()
        monitor_task: asyncio.Task[None] | None = None
        if configured_coordinator is not None:
            monitor_task = asyncio.create_task(
                _monitor_acquisition_jobs(
                    configured_coordinator,
                    application.state.job_store,
                    monitor_stop,
                    configured_settings.acquisition_monitor_seconds,
                ),
                name="acquisition-job-monitor",
            )
        application.state.acquisition_monitor_task = monitor_task
        try:
            yield
        finally:
            monitor_stop.set()
            if monitor_task is not None:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            pending_reconciliations = tuple(cancellation_reconciliations.values())
            for reconciliation in pending_reconciliations:
                reconciliation.cancel()
            if pending_reconciliations:
                await asyncio.gather(
                    *pending_reconciliations,
                    return_exceptions=True,
                )
            if client is not None:
                await client.aclose()

    application = FastAPI(
        title="Thuyết Minh Offline GPU",
        version="0.1.0",
        description="Điều khiển tải nguồn hợp pháp và pipeline thuyết minh cục bộ.",
        lifespan=lifespan,
    )
    application.state.settings = configured_settings

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        fields = [
            ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            for item in exception.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Dữ liệu yêu cầu không hợp lệ",
                    "retryable": False,
                    "fields": fields,
                }
            },
        )

    def job_store(request: Request) -> StateStore:
        return request.app.state.job_store

    def acquisition(request: Request) -> AcquisitionPort:
        service = request.app.state.acquisition
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "acquisition_not_configured",
                    "message": "Dịch vụ tìm kiếm và tải nguồn chưa được cấu hình",
                    "retryable": True,
                },
            )
        return service

    def job_coordinator(request: Request) -> CoordinatorPort:
        configured = request.app.state.coordinator
        if configured is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "coordinator_not_configured",
                    "message": "Bộ điều phối tải nguồn chưa được cấu hình",
                    "retryable": True,
                },
            )
        return configured

    StoreDependency = Annotated[StateStore, Depends(job_store)]
    AcquisitionDependency = Annotated[AcquisitionPort, Depends(acquisition)]
    CoordinatorDependency = Annotated[CoordinatorPort, Depends(job_coordinator)]

    @application.get("/v1/health")
    def health(
        request: Request,
        store: StoreDependency,
    ) -> Any:
        try:
            journal_mode = store.journal_mode()
            database_status = "ok"
        except sqlite3.Error:
            journal_mode = "unknown"
            database_status = "error"

        try:
            catalog = load_model_catalog(
                configured_settings.models_lock_path,
                configured_settings.models_dir,
            )
            catalog_status = "ok"
            model_count = len(catalog.models)
        except (OSError, ValueError, json.JSONDecodeError):
            catalog_status = "missing_or_invalid"
            model_count = 0

        gpu_report = read_gpu_report(
            configured_settings.gpu_report_path,
            max_age_seconds=configured_settings.gpu_report_max_age_seconds,
        )

        healthy = database_status == "ok"
        degraded = (
            not gpu_report.get("ready", False)
            or catalog_status != "ok"
            or request.app.state.acquisition is None
        )
        payload = {
            "status": "error" if not healthy else ("degraded" if degraded else "ok"),
            "api_version": "v1",
            "database": {
                "status": database_status,
                "journal_mode": journal_mode,
            },
            "model_catalog": {
                "status": catalog_status,
                "count": model_count,
            },
            "acquisition_configured": request.app.state.acquisition is not None,
            "coordinator_configured": request.app.state.coordinator is not None,
            "gpu": gpu_report,
        }
        if not healthy:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload,
            )
        return payload

    @application.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "api_version": "v1",
            "media_types": ["movie"],
            "subtitle_modes": ["prefer", "manual", "asr"],
            "source_language": {
                "automatic_detection": True,
                "vietnamese_passthrough": True,
                "availability_depends_on_selected_model": True,
            },
            "outputs": ["mp4", "srt", "timing-report.json"],
            "offline_inference": True,
            "cinematic_audio": {
                "dialogue_removed": True,
                "music_and_effects_preserved": True,
                "narration_ducking": True,
            },
            "models": {
                "runtime_selection": True,
                "explicit_install_only": True,
                "runtime_downloads": False,
            },
            "artifact_download_endpoint": True,
            "one_active_job_per_gpu": True,
            "drm_supported": False,
        }

    @application.get("/v1/models", response_model=ModelCatalog)
    def models() -> ModelCatalog:
        try:
            return load_model_catalog(
                configured_settings.models_lock_path,
                configured_settings.models_dir,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "model_catalog_missing",
                    "message": "Không tìm thấy danh mục model cục bộ",
                    "retryable": False,
                },
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "model_catalog_invalid",
                    "message": "Danh mục model cục bộ không hợp lệ",
                    "retryable": False,
                },
            ) from exc

    @application.post("/v1/search", response_model=SearchResponse)
    async def search(
        payload: SearchRequest,
        service: AcquisitionDependency,
    ) -> SearchResponse:
        query = " ".join(payload.query.split())
        if not query:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_query",
                    "message": "Từ khóa tìm kiếm không được để trống",
                    "retryable": False,
                },
            )
        try:
            results = await service.search(
                MediaQuery(
                    query=query,
                    year=payload.year,
                    media_kind=MediaKind(payload.media_type),
                )
            )
        except AcquisitionError as exc:
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                },
            ) from exc
        return SearchResponse(results=[_public_release(item) for item in results])

    @application.post(
        "/v1/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(
        payload: JobCreateRequest,
        store: StoreDependency,
        service: AcquisitionDependency,
    ) -> JobResponse:
        if payload.rights_confirmed is not True:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "rights_confirmation_required",
                    "message": "Bạn phải xác nhận có quyền tải và xử lý nội dung này",
                    "retryable": False,
                },
            )
        if payload.voice is not None and payload.voice_rights_confirmed is not True:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "voice_rights_confirmation_required",
                    "message": "Bạn phải xác nhận có quyền sử dụng giọng tham chiếu",
                    "retryable": False,
                },
            )

        try:
            catalog = load_model_catalog(
                configured_settings.models_lock_path,
                configured_settings.models_dir,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "model_catalog_unavailable",
                    "message": "Danh mục model cục bộ chưa sẵn sàng",
                    "retryable": False,
                },
            ) from exc
        effective_models = ModelSelection(
            asr=(payload.models.asr or configured_settings.default_asr_model_id),
            translation=(
                payload.models.translation
                or configured_settings.default_translation_model_id
            ),
            separation=(
                payload.models.separation
                or configured_settings.default_separation_model_id
            ),
            tts=(payload.models.tts or configured_settings.default_tts_model_id),
        )
        _validate_model_selection(effective_models, catalog)

        active_jobs = store.list_active_jobs(limit=1)
        if active_jobs:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": "Đã có một job nặng đang hoạt động trên GPU",
                    "retryable": True,
                    "active_job_id": active_jobs[0].id,
                },
            )

        spec = payload.model_dump(mode="json")
        # Freeze defaults into the durable job so changing server config can
        # never silently switch a resumed inference job to a different model.
        spec["models"] = effective_models.model_dump(mode="json")
        release_lookup = getattr(service, "release", None)
        selected_release = (
            release_lookup(payload.release_id) if callable(release_lookup) else None
        )
        query_lookup = getattr(service, "release_query", None)
        release_query = (
            query_lookup(payload.release_id) if callable(query_lookup) else None
        )
        if isinstance(release_query, MediaQuery):
            spec["year"] = release_query.year
            spec["media_type"] = release_query.media_kind.value
            spec["search_query"] = release_query.query

        try:
            record = store.create_job(
                payload.release_id,
                spec,
            )
        except ActiveJobExists as exc:
            active = store.list_active_jobs(limit=1)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": str(exc),
                    "retryable": True,
                    "active_job_id": active[0].id if active else None,
                },
            ) from exc
        if isinstance(selected_release, ReleaseCandidate):
            restart_token = _release_snapshot(selected_release)
            if restart_token is not None:
                store.save_checkpoint(
                    record.id,
                    JobStage.ACQUISITION,
                    {"release": restart_token},
                )
        track_job_operation(record.id)
        destination = configured_settings.incoming_dir / record.id
        created_task_id: str | None = None
        created_task_name = ""
        created_resume_attempted = False

        async def settle_created_task_cancel() -> JobRecord:
            current = store.get_job(record.id)
            if created_task_id is None or current.status is JobStatus.CANCELLED:
                return current
            current = store.update_status(
                record.id,
                current.status,
                details={
                    **current.details,
                    "task_id": created_task_id,
                    "name": created_task_name,
                },
            )
            pause_result = await pause_backend_for_cancel(
                store,
                service,
                record.id,
                created_task_id,
                timeout=1.5,
            )
            if pause_result != "paused":
                warning_code = (
                    "backend_cleanup_deferred"
                    if pause_result == "deferred"
                    else "download_pause_failed"
                )
                return store.append_warning(
                    record.id,
                    warning_code,
                    "Job đã hủy nhưng chưa thể tạm dừng torrent; dữ liệu nguồn vẫn được giữ lại",
                )
            return store.finalize_cancel(record.id)

        async def fail_created_download(
            code: str,
            message: str,
            *,
            retryable: bool,
        ) -> JobRecord:
            current = store.get_job(record.id)
            if created_resume_attempted and created_task_id is not None:
                pause_result = await pause_backend_for_cancel(
                    store,
                    service,
                    record.id,
                    created_task_id,
                    timeout=1.5,
                )
                if pause_result != "paused":
                    return store.update_status(
                        current.id,
                        current.status,
                        details={
                            **current.details,
                            "task_id": created_task_id,
                            "name": created_task_name,
                            "backend_started": True,
                            "backend_state_uncertain": True,
                        },
                        error_code=code,
                        error_message=message,
                        retryable=True,
                    )
            return store.update_status(
                current.id,
                JobStatus.FAILED,
                error_code=code,
                error_message=message,
                retryable=retryable,
            )

        try:
            task = await mutate_backend(
                service.start_download,
                payload.release_id,
                destination,
                rights_confirmed=True,
                paused=True,
            )
            created_task_id = str(_adapter_value(task, "task_id", ""))
            created_task_name = str(_adapter_value(task, "name", ""))
            if not created_task_id:
                raise RuntimeError("Download adapter returned no task id")
            current = store.get_job(record.id)
            if current.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                return _job_response(await settle_created_task_cancel())
            record = store.update_status(
                record.id,
                JobStatus.DOWNLOADING,
                stage=JobStage.ACQUISITION,
                details={
                    "task_id": created_task_id,
                    "name": created_task_name,
                    "downloaded_bytes": 0,
                    "total_bytes": None,
                    "speed_bytes_per_second": 0,
                    "eta_seconds": None,
                    "backend_started": False,
                },
            )
            store.save_checkpoint(
                record.id,
                JobStage.ACQUISITION,
                {"task_id": created_task_id},
            )
            current = store.get_job(record.id)
            if current.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                return _job_response(await settle_created_task_cancel())
            resume_download = getattr(service, "resume_download", None)
            if not callable(resume_download):
                raise RuntimeError("Acquisition service has no resume method")
            created_resume_attempted = True
            await mutate_backend(resume_download, created_task_id)
            current = store.get_job(record.id)
            if current.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                return _job_response(await settle_created_task_cancel())
            record = store.update_status(
                record.id,
                JobStatus.DOWNLOADING,
                details={**current.details, "backend_started": True},
                expected_status=JobStatus.DOWNLOADING,
            )
        except AcquisitionError as exc:
            current = store.get_job(record.id)
            if current.status is JobStatus.CANCELLING:
                if created_task_id is not None:
                    return _job_response(await settle_created_task_cancel())
                return _job_response(store.finalize_cancel(record.id))
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            await fail_created_download(
                exc.code.value,
                exc.message_vi,
                retryable=exc.retryable,
            )
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                    "job_id": record.id,
                },
            ) from exc
        except ActiveJobExists as exc:
            active = store.list_active_jobs(limit=1)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": str(exc),
                    "retryable": True,
                    "active_job_id": active[0].id if active else None,
                },
            ) from exc
        except InvalidTransition as exc:
            current = store.get_job(record.id)
            if current.status is JobStatus.CANCELLING:
                if created_task_id is not None:
                    return _job_response(await settle_created_task_cancel())
                return _job_response(store.finalize_cancel(record.id))
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "invalid_job_transition",
                    "message": str(exc),
                    "retryable": False,
                },
            ) from exc
        except Exception as exc:
            current = store.get_job(record.id)
            if current.status is JobStatus.CANCELLING:
                if created_task_id is not None:
                    return _job_response(await settle_created_task_cancel())
                return _job_response(store.finalize_cancel(record.id))
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            message = "Không thể khởi tạo tác vụ tải nguồn"
            await fail_created_download(
                "download_start_failed",
                message,
                retryable=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "download_start_failed",
                    "message": message,
                    "retryable": True,
                    "job_id": record.id,
                },
            ) from exc
        return _job_response(record)

    @application.get("/v1/jobs", response_model=JobListResponse)
    def list_jobs(
        store: StoreDependency,
        statuses: Annotated[
            list[JobStatus] | None,
            Query(alias="status"),
        ] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        newest_first: bool = True,
    ) -> JobListResponse:
        records = store.list_jobs(
            statuses=statuses,
            limit=limit,
            newest_first=newest_first,
        )
        items = [_job_response(record) for record in records]
        return JobListResponse(items=items, count=len(items))

    @application.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, store: StoreDependency) -> JobResponse:
        try:
            return _job_response(store.get_job(job_id))
        except JobNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc

    @application.get("/v1/jobs/{job_id}/artifacts/{kind}")
    def get_job_artifact(
        job_id: str,
        kind: Literal["video", "subtitle", "timing"],
        store: StoreDependency,
    ) -> FileResponse:
        """Serve only deterministic, completed Phase 4 artifacts.

        Paths persisted in SQLite are treated as untrusted metadata.  The
        server derives each allowed path from its configured output/job roots
        and compares it with the sealed result before opening the file.
        """

        try:
            record = store.get_job(job_id)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        if record.status is not JobStatus.COMPLETED or record.result is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "artifact_not_ready",
                    "message": "Kết quả thuyết minh chưa sẵn sàng",
                    "retryable": True,
                },
            )

        if kind == "video":
            artifact_root = configured_settings.output_dir.resolve(strict=False)
            expected = artifact_root / f"{job_id}.mp4"
            result_field = "video_path"
            digest_field = "video_sha256"
            filename = f"{job_id}.mp4"
            media_type = "video/mp4"
        elif kind == "subtitle":
            artifact_root = configured_settings.output_dir.resolve(strict=False)
            expected = artifact_root / f"{job_id}.vi.srt"
            result_field = "srt_path"
            digest_field = "srt_sha256"
            filename = f"{job_id}.vi.srt"
            media_type = "application/x-subrip"
        else:
            artifact_root = configured_settings.jobs_dir.resolve(strict=False)
            expected = artifact_root / job_id / "timing-report.json"
            result_field = "timing_report_path"
            digest_field = "timing_report_sha256"
            filename = f"{job_id}.timing-report.json"
            media_type = "application/json"

        expected = expected.absolute()
        raw_path = record.result.get(result_field)
        raw_digest = record.result.get(digest_field)
        candidate = Path(raw_path).absolute() if isinstance(raw_path, str) else None
        if (
            candidate != expected
            or not expected.is_relative_to(artifact_root)
            or not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or any(character not in _SHA256_HEX for character in raw_digest)
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "artifact_record_invalid",
                    "message": "Thông tin file kết quả không hợp lệ",
                    "retryable": False,
                },
            )
        try:
            if not expected.parent.resolve(strict=True).is_relative_to(artifact_root):
                raise OSError("artifact parent escapes configured root")
            actual_digest, actual_size = _sealed_file_sha256(expected)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={
                    "code": "artifact_missing",
                    "message": "File kết quả không còn trên máy chủ",
                    "retryable": False,
                },
            )
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "artifact_record_invalid",
                    "message": "File kết quả không phải artifact cục bộ an toàn",
                    "retryable": False,
                },
            ) from exc
        recorded_size = record.result.get("size_bytes") if kind == "video" else None
        if actual_digest != raw_digest or (
            recorded_size is not None
            and (
                isinstance(recorded_size, bool)
                or not isinstance(recorded_size, int)
                or recorded_size != actual_size
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "artifact_integrity_failed",
                    "message": "File kết quả đã thay đổi sau khi được niêm phong",
                    "retryable": False,
                },
            )
        return FileResponse(
            expected,
            media_type=media_type,
            filename=filename,
        )

    @application.post(
        "/v1/jobs/{job_id}/language",
        response_model=JobResponse,
    )
    def select_source_language(
        job_id: str,
        payload: LanguageSelectionRequest,
        store: StoreDependency,
    ) -> JobResponse:
        """Resume local ASR after an uncertain automatic detection result."""

        try:
            selected = normalize_whisper_language(payload.language)
            if selected is None:
                raise TranscriptionError(
                    "unsupported_language",
                    "Bạn phải chọn một ngôn ngữ nguồn cụ thể",
                    retryable=False,
                )
            current = store.get_job(job_id)
            if current.status is not JobStatus.NEEDS_LANGUAGE:
                raise InvalidTransition(
                    "Job không chờ người dùng chọn ngôn ngữ nguồn"
                )
            details = dict(current.details)
            details["source_language_override"] = selected
            details["source_language_selected_by_user"] = True
            details.pop("language_detection_candidates", None)
            details.pop("language_candidates", None)
            resumed = store.update_status(
                job_id,
                JobStatus.TRANSCRIBING,
                expected_status=JobStatus.NEEDS_LANGUAGE,
                stage=JobStage.ASR,
                details=details,
            )
            return _job_response(resumed)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except (InvalidTransition, TranscriptionError) as exc:
            code = (
                exc.code
                if isinstance(exc, TranscriptionError)
                else "invalid_job_transition"
            )
            message = (
                exc.message_vi if isinstance(exc, TranscriptionError) else str(exc)
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": code,
                    "message": message,
                    "retryable": False,
                },
            ) from exc

    @application.post("/v1/jobs/{job_id}/refresh", response_model=JobResponse)
    async def refresh_job(
        job_id: str,
        store: StoreDependency,
        configured: CoordinatorDependency,
    ) -> JobResponse:
        try:
            store.get_job(job_id)
            refreshed = await configured.refresh(job_id)
            record = refreshed if isinstance(refreshed, JobRecord) else store.get_job(job_id)
            return _job_response(record)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except InvalidTransition as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_job_transition",
                    "message": str(exc),
                    "retryable": False,
                },
            ) from exc
        except AcquisitionError as exc:
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "refresh_failed",
                    "message": "Không thể cập nhật tiến độ tải nguồn",
                    "retryable": True,
                },
            ) from exc

    @application.post(
        "/v1/jobs/{job_id}/subtitles/use-asr",
        response_model=JobResponse,
    )
    async def select_asr_for_job(
        job_id: str,
        store: StoreDependency,
        configured: CoordinatorDependency,
    ) -> JobResponse:
        try:
            store.get_job(job_id)
            return _job_response(await configured.select_asr(job_id))
        except JobNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except ActiveJobExists as exc:
            active = store.list_active_jobs(limit=1)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": str(exc),
                    "retryable": True,
                    "active_job_id": active[0].id if active else None,
                },
            ) from exc
        except AcquisitionError as exc:
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "subtitle_selection_failed",
                    "message": "Không thể xác nhận dùng nhận dạng giọng nói",
                    "retryable": True,
                },
            ) from exc

    @application.post(
        "/v1/jobs/{job_id}/subtitles/{subtitle_id}",
        response_model=JobResponse,
    )
    async def select_subtitle_for_job(
        job_id: str,
        subtitle_id: str,
        store: StoreDependency,
        configured: CoordinatorDependency,
    ) -> JobResponse:
        try:
            store.get_job(job_id)
            return _job_response(
                await configured.select_subtitle(job_id, subtitle_id)
            )
        except JobNotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except ActiveJobExists as exc:
            active = store.list_active_jobs(limit=1)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": str(exc),
                    "retryable": True,
                    "active_job_id": active[0].id if active else None,
                },
            ) from exc
        except AcquisitionError as exc:
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "subtitle_selection_failed",
                    "message": "Không thể sử dụng phụ đề đã chọn",
                    "retryable": True,
                },
            ) from exc

    @application.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    async def cancel_job(
        job_id: str,
        request: Request,
        store: StoreDependency,
    ) -> JobResponse:
        try:
            current = store.get_job(job_id)
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            offline_running = current.status in {
                JobStatus.TRANSCRIBING,
                JobStatus.SUBTITLE_SELECTED,
                JobStatus.TRANSLATING,
                JobStatus.SEPARATING,
                JobStatus.SYNTHESIZING,
                JobStatus.TIMING,
                JobStatus.MIXING,
                JobStatus.MUXING,
                JobStatus.VERIFYING,
            }
            offline_idle = current.status in {
                JobStatus.READY_OFFLINE,
                JobStatus.NEEDS_LANGUAGE,
                JobStatus.READY_TRANSLATION,
                JobStatus.READY_TTS,
            } or (
                current.status in {JobStatus.PAUSED, JobStatus.FAILED}
                and current.previous_status
                in {
                    JobStatus.READY_OFFLINE,
                    JobStatus.TRANSCRIBING,
                    JobStatus.SUBTITLE_SELECTED,
                    JobStatus.NEEDS_LANGUAGE,
                    JobStatus.READY_TRANSLATION,
                    JobStatus.TRANSLATING,
                    JobStatus.READY_TTS,
                    JobStatus.SEPARATING,
                    JobStatus.SYNTHESIZING,
                    JobStatus.TIMING,
                    JobStatus.MIXING,
                    JobStatus.MUXING,
                    JobStatus.VERIFYING,
                }
            )
            if offline_running or offline_idle or bool(
                current.details.get("offline_cancel_pending")
            ):
                cancelled = store.request_cancel(job_id)
                if offline_running:
                    cancelled = store.update_status(
                        job_id,
                        JobStatus.CANCELLING,
                        expected_status=JobStatus.CANCELLING,
                        details={
                            **cancelled.details,
                            "offline_cancel_pending": True,
                        },
                        cancel_requested=True,
                    )
                    # The worker owns native/GPU cleanup and finalizes only
                    # after inference has returned and VRAM is released.
                    return _job_response(cancelled)
                if bool(cancelled.details.get("offline_cancel_pending")):
                    return _job_response(cancelled)
                return _job_response(store.finalize_cancel(job_id))
            cancelled = store.request_cancel(job_id)
            if cancelled.status is JobStatus.CANCELLED:
                return _job_response(cancelled)
            in_flight = active_job_operations.get(job_id)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 1.8
            if (
                in_flight is not None
                and in_flight is not asyncio.current_task()
                and not in_flight.done()
            ):
                # Give a nearly-complete owner a chance to perform its ordered
                # compensation. Long adapter calls keep the durable
                # CANCELLING state and finish the pause themselves later.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(in_flight),
                        timeout=max(0.0, deadline - loop.time()),
                    )
                except asyncio.TimeoutError:
                    schedule_cancel_reconciliation(
                        in_flight,
                        store,
                        getattr(request.app.state, "acquisition", None),
                        job_id,
                    )
                    return _job_response(store.get_job(job_id))
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                except Exception:
                    pass
                cancelled = store.get_job(job_id)
                if cancelled.status is JobStatus.CANCELLED:
                    return _job_response(cancelled)
            task_id = cancelled.details.get("task_id")
            service = getattr(request.app.state, "acquisition", None)
            if task_id:
                pause_result = await pause_backend_for_cancel(
                    store,
                    service,
                    job_id,
                    str(task_id),
                    timeout=max(0.05, deadline - loop.time()),
                )
                if pause_result == "paused":
                    cancelled = store.finalize_cancel(job_id)
                elif pause_result == "deferred":
                    cancelled = store.append_warning(
                        job_id,
                        "backend_cleanup_deferred",
                        "Chưa dừng torrent cũ vì một job khác đang hoạt động",
                    )
                else:
                    cancelled = store.append_warning(
                        job_id,
                        "download_pause_failed",
                        "Job đã hủy nhưng chưa thể tạm dừng torrent; dữ liệu nguồn vẫn được giữ lại",
                    )
            else:
                cancelled = store.finalize_cancel(job_id)
            return _job_response(cancelled)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except InvalidTransition as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_job_transition",
                    "message": str(exc),
                    "retryable": False,
                },
            ) from exc
        except AcquisitionError as exc:
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                },
            ) from exc

    @application.post("/v1/jobs/{job_id}/resume", response_model=JobResponse)
    async def resume_job(
        job_id: str,
        store: StoreDependency,
        service: AcquisitionDependency,
    ) -> JobResponse:
        operation_task_id: str | None = None
        operation_task_name = ""
        owns_operation = False
        resume_attempted = False

        async def settle_operation_cancel(task_id: str) -> JobRecord | None:
            current = store.get_job(job_id)
            if current.status not in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                return None
            if current.status is JobStatus.CANCELLED:
                return current
            current = store.update_status(
                job_id,
                current.status,
                details={
                    **current.details,
                    "task_id": task_id,
                    "name": operation_task_name or current.details.get("name", ""),
                },
            )
            pause_result = await pause_backend_for_cancel(
                store,
                service,
                job_id,
                task_id,
                timeout=1.5,
            )
            if pause_result == "paused":
                if current.status is JobStatus.CANCELLING:
                    current = store.finalize_cancel(job_id)
            elif pause_result == "deferred":
                current = store.append_warning(
                    job_id,
                    "backend_cleanup_deferred",
                    "Chưa dừng torrent cũ vì một job khác đang hoạt động",
                )
            else:
                current = store.append_warning(
                    job_id,
                    "download_pause_failed",
                    "Job đã hủy nhưng chưa thể tạm dừng torrent; dữ liệu nguồn vẫn được giữ lại",
                )
            return current

        async def fail_resume_download(
            code: str,
            message: str,
            *,
            retryable: bool,
        ) -> JobRecord:
            current = store.get_job(job_id)
            if resume_attempted and operation_task_id is not None:
                pause_result = await pause_backend_for_cancel(
                    store,
                    service,
                    job_id,
                    operation_task_id,
                    timeout=1.5,
                )
                if pause_result not in {"paused", "deferred"} and current.active_slot:
                    return store.update_status(
                        current.id,
                        current.status,
                        details={
                            **current.details,
                            "task_id": operation_task_id,
                            "name": operation_task_name,
                            "backend_started": True,
                            "backend_state_uncertain": True,
                        },
                        error_code=code,
                        error_message=message,
                        retryable=True,
                    )
            return store.update_status(
                current.id,
                JobStatus.FAILED,
                error_code=code,
                error_message=message,
                retryable=retryable,
                force=(current.status is JobStatus.PAUSED),
            )

        try:
            initial = store.get_job(job_id)
            initial_task_id = initial.details.get("task_id")
            if initial_task_id:
                operation_task_id = str(initial_task_id)
                operation_task_name = str(initial.details.get("name", ""))
            track_job_operation(job_id)
            owns_operation = True
            active_jobs = [
                active
                for active in store.list_active_jobs(limit=2)
                if active.id != job_id
            ]
            if active_jobs:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "active_job_exists",
                        "message": "Đã có một job nặng đang hoạt động trên GPU",
                        "retryable": True,
                        "active_job_id": active_jobs[0].id,
                    },
                )
            record = store.resume(job_id)
            if record.status in {
                JobStatus.READY_OFFLINE,
                JobStatus.TRANSCRIBING,
                JobStatus.SUBTITLE_SELECTED,
                JobStatus.NEEDS_LANGUAGE,
                JobStatus.READY_TRANSLATION,
                JobStatus.TRANSLATING,
                JobStatus.READY_TTS,
                JobStatus.SEPARATING,
                JobStatus.SYNTHESIZING,
                JobStatus.TIMING,
                JobStatus.MIXING,
                JobStatus.MUXING,
                JobStatus.VERIFYING,
            }:
                # Offline stages are resumed by the GPU worker from their
                # durable artifact/checkpoint. A completed torrent task ID in
                # details must never route this request back to qBittorrent.
                return _job_response(record)
            resume_download = getattr(service, "resume_download", None)
            task_id = record.details.get("task_id")
            if task_id:
                operation_task_id = str(task_id)
                operation_task_name = str(record.details.get("name", ""))
                if record.status is JobStatus.DOWNLOADING:
                    record = store.update_status(
                        record.id,
                        record.status,
                        details={**record.details, "backend_started": False},
                        expected_status=record.status,
                    )
                if resume_download is None:
                    raise RuntimeError("Acquisition service has no resume method")
                relocate_download = getattr(service, "relocate_download", None)
                if callable(relocate_download):
                    await mutate_backend(
                        relocate_download,
                        operation_task_id,
                        configured_settings.incoming_dir / record.id,
                    )
                    current = store.get_job(job_id)
                    if current.status in {
                        JobStatus.CANCELLING,
                        JobStatus.CANCELLED,
                    }:
                        cancelled = await settle_operation_cancel(operation_task_id)
                        return _job_response(cancelled or current)
                resume_attempted = True
                await mutate_backend(resume_download, operation_task_id)
                current = store.get_job(job_id)
                if current.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                if current.status is JobStatus.DOWNLOADING:
                    current = store.update_status(
                        current.id,
                        current.status,
                        details={**current.details, "backend_started": True},
                        expected_status=JobStatus.DOWNLOADING,
                    )
                record = current
            else:
                checkpoint = store.get_checkpoint(
                    record.id,
                    JobStage.ACQUISITION,
                )
                restored_release = _release_from_snapshot(
                    checkpoint.payload.get("release") if checkpoint else None,
                    expected_release_id=record.release_id,
                )
                restore_release = getattr(service, "restore_release", None)
                if restored_release is not None and callable(restore_release):
                    restore_release(restored_release)
                elif callable(restore_release):
                    search_query = record.spec.get("search_query")
                    if isinstance(search_query, str) and search_query.strip():
                        try:
                            media_kind = MediaKind(
                                str(record.spec.get("media_type", MediaKind.MOVIE.value))
                            )
                        except ValueError:
                            media_kind = MediaKind.MOVIE
                        search_results = await service.search(
                            MediaQuery(
                                query=search_query,
                                year=(
                                    int(record.spec["year"])
                                    if record.spec.get("year") is not None
                                    else None
                                ),
                                media_kind=media_kind,
                            )
                        )
                        restored_release = next(
                            (
                                candidate
                                for candidate in search_results
                                if candidate.release_id == record.release_id
                            ),
                            None,
                        )
                    if restored_release is None:
                        raise AcquisitionError(
                            AcquisitionErrorCode.DOWNLOAD_UNAVAILABLE,
                            "Không thể khôi phục bản phát hành; hãy tìm kiếm và chọn lại",
                            retryable=True,
                        )
                task = await mutate_backend(
                    service.start_download,
                    restored_release or record.release_id,
                    configured_settings.incoming_dir / record.id,
                    rights_confirmed=True,
                    paused=True,
                )
                operation_task_id = str(_adapter_value(task, "task_id", ""))
                operation_task_name = str(_adapter_value(task, "name", ""))
                task_id = operation_task_id
                if not operation_task_id:
                    raise RuntimeError("Download adapter returned no task id")
                current = store.get_job(job_id)
                if current.status in {JobStatus.CANCELLING, JobStatus.CANCELLED}:
                    current = store.update_status(
                        job_id,
                        current.status,
                        details={
                            **current.details,
                            "task_id": operation_task_id,
                            "name": operation_task_name,
                        },
                    )
                    if cancelled := await settle_operation_cancel(operation_task_id):
                        return _job_response(cancelled)
                record = store.update_status(
                    record.id,
                    JobStatus.DOWNLOADING,
                    stage=JobStage.ACQUISITION,
                    details={
                        **record.details,
                        "task_id": operation_task_id,
                        "name": operation_task_name,
                        "downloaded_bytes": 0,
                        "total_bytes": None,
                        "speed_bytes_per_second": 0,
                        "eta_seconds": None,
                        "backend_started": False,
                    },
                )
                store.save_checkpoint(
                    record.id,
                    JobStage.ACQUISITION,
                    {"task_id": operation_task_id},
                )
                current = store.get_job(job_id)
                if current.status in {
                    JobStatus.CANCELLING,
                    JobStatus.CANCELLED,
                }:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                if resume_download is None:
                    raise RuntimeError("Acquisition service has no resume method")
                resume_attempted = True
                await mutate_backend(resume_download, operation_task_id)
                current = store.get_job(job_id)
                if current.status in {
                    JobStatus.CANCELLING,
                    JobStatus.CANCELLED,
                }:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                record = store.update_status(
                    current.id,
                    JobStatus.DOWNLOADING,
                    details={**current.details, "backend_started": True},
                    expected_status=JobStatus.DOWNLOADING,
                )
            return _job_response(record)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc
        except InvalidTransition as exc:
            current = store.get_job(job_id)
            if current.status is JobStatus.CANCELLING:
                if owns_operation and operation_task_id is not None:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                if owns_operation:
                    return _job_response(store.finalize_cancel(job_id))
                return _job_response(current)
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "invalid_job_transition",
                    "message": str(exc),
                    "retryable": False,
                },
            ) from exc
        except ActiveJobExists as exc:
            active = store.list_active_jobs(limit=1)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "active_job_exists",
                    "message": str(exc),
                    "retryable": True,
                    "active_job_id": active[0].id if active else None,
                },
            ) from exc
        except AcquisitionError as exc:
            current = store.get_job(job_id)
            if current.status is JobStatus.CANCELLING:
                if owns_operation and operation_task_id is not None:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                if owns_operation:
                    return _job_response(store.finalize_cancel(job_id))
                return _job_response(current)
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            failed = await fail_resume_download(
                exc.code.value,
                exc.message_vi,
                retryable=exc.retryable,
            )
            raise HTTPException(
                status_code=_acquisition_error_status(exc),
                detail={
                    "code": exc.code.value,
                    "message": exc.message_vi,
                    "retryable": exc.retryable,
                    "job_id": failed.id,
                },
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            current = store.get_job(job_id)
            if current.status is JobStatus.CANCELLING:
                if owns_operation and operation_task_id is not None:
                    cancelled = await settle_operation_cancel(operation_task_id)
                    return _job_response(cancelled or current)
                if owns_operation:
                    return _job_response(store.finalize_cancel(job_id))
                return _job_response(current)
            if current.status is JobStatus.CANCELLED:
                return _job_response(current)
            message = "Không thể tiếp tục tác vụ tải nguồn"
            failed = await fail_resume_download(
                "download_resume_failed",
                message,
                retryable=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "download_resume_failed",
                    "message": message,
                    "retryable": True,
                    "job_id": failed.id,
                },
            ) from exc

    @application.get("/v1/jobs/{job_id}/events")
    async def job_events(
        request: Request,
        job_id: str,
        store: StoreDependency,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
        once: bool = False,
    ) -> StreamingResponse:
        try:
            store.get_job(job_id)
        except JobNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "job_not_found",
                    "message": "Không tìm thấy job",
                    "retryable": False,
                },
            ) from exc

        cursor = after
        if last_event_id is not None:
            try:
                cursor = max(cursor, int(last_event_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "invalid_event_cursor",
                        "message": "Last-Event-ID không hợp lệ",
                        "retryable": False,
                    },
                ) from exc

        async def stream():
            nonlocal cursor
            while True:
                events = store.list_events(job_id, after_id=cursor)
                for event in events:
                    cursor = event.id
                    data = json.dumps(
                        {
                            "type": event.event_type,
                            "job_id": event.job_id,
                            "created_at": event.created_at,
                            "payload": event.payload,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield f"id: {event.id}\nevent: {event.event_type}\ndata: {data}\n\n"
                if once:
                    return
                if await request.is_disconnected():
                    return
                current = store.get_job(job_id)
                if current.status in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
                    return
                if not events:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(configured_settings.sse_poll_seconds)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    web_index = _WEB_STATIC_ROOT / "index.html"
    if web_index.is_file():
        # API and documentation routes are registered first, so this final
        # catch-all mount cannot shadow /v1, /docs, or /openapi.json.
        application.mount(
            "/",
            StaticFiles(directory=_WEB_STATIC_ROOT, html=True),
            name="web-dashboard",
        )
        application.state.web_dashboard_enabled = True
    else:
        application.state.web_dashboard_enabled = False

    return application


app = create_app()
