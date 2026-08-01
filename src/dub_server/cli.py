"""Command-line client for the local FastAPI control plane."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import httpx
import typer

from .config import Settings


if sys.platform == "win32":
    # Windows shells may still expose a legacy code page.  Reconfiguring only
    # the CLI streams keeps Vietnamese help and errors printable.
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            _reconfigure(encoding="utf-8", errors="replace")


app = typer.Typer(
    name="dub",
    help="Điều khiển hệ thống thuyết minh video chạy cục bộ.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Xem danh mục model cục bộ.")
jobs_app = typer.Typer(help="Quản lý và theo dõi các job thuyết minh.")
stack_app = typer.Typer(help="Quản lý stack native trên máy chủ GPU.")
maintenance_app = typer.Typer(help="Dọn artifact và tạo SBOM phát hành.")
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")
app.add_typer(stack_app, name="stack")
app.add_typer(maintenance_app, name="maintenance")


_ACTION_REQUIRED_STATUSES = frozenset(
    {"needs_language", "needs_subtitle_selection", "paused"}
)
_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", *_ACTION_REQUIRED_STATUSES}
)
_STAGE_LABELS = {
    "acquisition": "Tải nguồn",
    "subtitle": "Chọn phụ đề",
    "asr": "Nhận dạng lời nói",
    "translation": "Dịch sang tiếng Việt",
    "separation": "Tách lời diễn viên",
    "tts": "Tổng hợp lời thuyết minh",
    "timing": "Khớp thời lượng",
    "mix": "Trộn nhạc nền và lời mới",
    "export": "Xuất MP4",
    "verify": "Kiểm tra kết quả",
    "done": "Hoàn tất",
}
_MODEL_PROFILES: dict[str, tuple[str, ...]] = {
    "minimal": (
        "asr-faster-whisper-small",
        "mt-gemma4-e2b-q4",
        "separation-tiger-dnr",
        "tts-piper-vi-vais1000-medium",
    ),
    "balanced": (
        "asr-faster-whisper-small",
        "mt-gemma4-e2b-q4",
        "separation-tiger-dnr",
        "tts-neucodec-onnx-int8",
        "tts-vieneu-v2",
    ),
    "maximum": (
        "asr-faster-whisper-large-v3-turbo",
        "mt-gemma4-31b-q4",
        "separation-tiger-dnr",
        "tts-neucodec-onnx-int8",
        "tts-vieneu-v2",
    ),
}


def _base_url(value: str | None) -> str:
    return (value or Settings().api_url).rstrip("/")


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2))


def _project_root() -> Path:
    configured = os.environ.get("DUB_PROJECT_ROOT") or os.environ.get("PROJECT_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2]
    return root.expanduser().resolve(strict=False)


def _human_bytes(value: int | float | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{int(value)} B"


def _run_project_script(
    relative_path: str,
    arguments: list[str] | None = None,
) -> None:
    script = (_project_root() / relative_path).resolve(strict=False)
    if not script.is_file() or not script.is_relative_to(_project_root()):
        typer.echo(f"Không tìm thấy script quản trị: {script}", err=True)
        raise typer.Exit(code=2)
    if script.suffix == ".sh":
        if platform.system() != "Linux":
            typer.echo("Lệnh này chỉ chạy trên máy chủ Linux native", err=True)
            raise typer.Exit(code=2)
        command = ["bash", str(script), *(arguments or [])]
    else:
        command = [sys.executable, str(script), *(arguments or [])]
    try:
        result = subprocess.run(command, cwd=_project_root(), check=False)
    except OSError as exc:
        typer.echo(f"Không thể chạy công cụ quản trị: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"API trả về HTTP {response.status_code}"
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            location = ".".join(str(part) for part in item.get("loc", ()))
            message = str(item.get("msg", "Dữ liệu không hợp lệ"))
            messages.append(f"{location}: {message}" if location else message)
        if messages:
            return "; ".join(messages)
    if isinstance(detail, dict) and detail.get("message"):
        code = detail.get("code")
        message = str(detail["message"])
        return f"{message} [{code}]" if code else message
    return f"API trả về HTTP {response.status_code}"


def _request(
    method: str,
    path: str,
    *,
    api_url: str | None,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | list[tuple[str, Any]] | None = None,
    timeout_seconds: float = 30.0,
) -> Any:
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.request(
                method,
                f"{_base_url(api_url)}{path}",
                json=payload,
                params=params,
            )
    except httpx.HTTPError as exc:
        typer.echo(f"Không thể kết nối tới API cục bộ: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if response.is_error:
        typer.echo(_error_message(response), err=True)
        raise typer.Exit(code=1)
    try:
        return response.json()
    except ValueError as exc:
        typer.echo("API trả về dữ liệu không hợp lệ", err=True)
        raise typer.Exit(code=1) from exc


def _download_artifact(
    path: str,
    destination: Path,
    *,
    api_url: str | None,
    overwrite: bool,
    resume_partial: bool = True,
) -> None:
    resolved = destination.expanduser().resolve(strict=False)
    if resolved.exists() and not overwrite:
        typer.echo(
            f"File đã tồn tại: {resolved}. Dùng --overwrite để ghi đè.",
            err=True,
        )
        raise typer.Exit(code=1)
    temporary = resolved.with_name(f".{resolved.name}.part")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if not resume_partial:
            temporary.unlink(missing_ok=True)
        timeout = httpx.Timeout(None, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            def transfer(range_start: int) -> bool:
                headers = (
                    {"Range": f"bytes={range_start}-"} if range_start > 0 else None
                )
                with client.stream(
                    "GET",
                    f"{_base_url(api_url)}{path}",
                    headers=headers,
                ) as response:
                    if response.status_code == 416 and range_start > 0:
                        return False
                    if response.is_error:
                        typer.echo(_error_message(response), err=True)
                        raise typer.Exit(code=1)
                    append = range_start > 0 and response.status_code == 206
                    if range_start > 0 and not append:
                        temporary.unlink(missing_ok=True)
                        range_start = 0
                    total: int | None = None
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
                    if match:
                        total = int(match.group(1))
                    elif response.headers.get("Content-Length", "").isdigit():
                        total = range_start + int(response.headers["Content-Length"])
                    mode = "ab" if append else "xb"
                    written = range_start
                    with temporary.open(mode) as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            stream.write(chunk)
                            written += len(chunk)
                            if sys.stderr.isatty():
                                if total:
                                    fraction = min(1.0, written / total)
                                    width = 28
                                    filled = round(width * fraction)
                                    bar = "#" * filled + "-" * (width - filled)
                                    message = (
                                        f"\r[{bar}] {fraction * 100:5.1f}% "
                                        f"({_human_bytes(written)}/{_human_bytes(total)})"
                                    )
                                else:
                                    message = f"\rĐã tải {_human_bytes(written)}"
                                typer.echo(message, err=True, nl=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if sys.stderr.isatty():
                        typer.echo(err=True)
                    return True

            start = temporary.stat().st_size if temporary.is_file() else 0
            if not transfer(start):
                temporary.unlink(missing_ok=True)
                transfer(0)
        os.replace(temporary, resolved)
    except typer.Exit:
        if not resume_partial:
            temporary.unlink(missing_ok=True)
        raise
    except (OSError, httpx.HTTPError) as exc:
        if not resume_partial:
            temporary.unlink(missing_ok=True)
        typer.echo(f"Không thể tải file kết quả: {exc}", err=True)
        if resume_partial and temporary.is_file():
            typer.echo(f"Đã giữ phần tải dở để tiếp tục: {temporary}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(resolved))


def _iter_sse(response: httpx.Response) -> Iterator[dict[str, str]]:
    event: dict[str, str] = {}
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if event or data_lines:
                if data_lines:
                    event["data"] = "\n".join(data_lines)
                yield event
            event = {}
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field in {"id", "event", "retry"}:
            event[field] = value
    if event or data_lines:
        if data_lines:
            event["data"] = "\n".join(data_lines)
        yield event


def _progress_snapshot(job: dict[str, Any]) -> str:
    progress = max(0, min(1000, int(job.get("progress_permille", 0))))
    fraction = progress / 1000.0
    width = 30
    filled = round(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    stage = str(job.get("stage", ""))
    status = str(job.get("status", "unknown"))
    details = job.get("details") if isinstance(job.get("details"), dict) else {}
    extras: list[str] = []
    downloaded = details.get("downloaded_bytes")
    total = details.get("total_bytes")
    if isinstance(downloaded, int):
        extras.append(f"{_human_bytes(downloaded)}/{_human_bytes(total)}")
    speed = details.get("speed_bytes_per_second")
    if isinstance(speed, (int, float)) and speed > 0:
        extras.append(f"{_human_bytes(speed)}/s")
    eta = details.get("eta_seconds")
    if isinstance(eta, (int, float)) and eta >= 0:
        extras.append(f"ETA {int(eta)}s")
    completed_blocks = details.get("completed_blocks")
    total_blocks = details.get("total_blocks")
    if isinstance(completed_blocks, int) and isinstance(total_blocks, int):
        extras.append(f"block {completed_blocks}/{total_blocks}")
    suffix = f" | {' | '.join(extras)}" if extras else ""
    label = _STAGE_LABELS.get(stage, stage or "Khởi tạo")
    return f"[{bar}] {fraction * 100:5.1f}% | {label} | {status}{suffix}"


def _show_progress(job: dict[str, Any], *, final: bool = False) -> None:
    message = _progress_snapshot(job)
    if sys.stdout.isatty():
        typer.echo(f"\r\x1b[2K{message}", nl=final)
    else:
        typer.echo(message)


def _watch_job(
    job_id: str,
    *,
    api_url: str | None,
    after: int = 0,
) -> dict[str, Any]:
    job = _request("GET", f"/v1/jobs/{job_id}", api_url=api_url)
    _show_progress(job, final=str(job.get("status")) in _TERMINAL_STATUSES)
    if str(job.get("status")) in _TERMINAL_STATUSES:
        return job

    cursor = max(0, after)
    reconnects = 0
    try:
        while True:
            headers = {"Last-Event-ID": str(cursor)} if cursor else None
            timeout = httpx.Timeout(None, connect=30.0)
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream(
                        "GET",
                        f"{_base_url(api_url)}/v1/jobs/{job_id}/events",
                        params={"after": cursor},
                        headers=headers,
                    ) as response:
                        if response.is_error:
                            typer.echo(_error_message(response), err=True)
                            raise typer.Exit(code=1)
                        reconnects = 0
                        for event in _iter_sse(response):
                            raw_id = event.get("id", "")
                            if raw_id.isdigit():
                                cursor = max(cursor, int(raw_id))
                            raw_data = event.get("data")
                            if not raw_data:
                                continue
                            try:
                                envelope = json.loads(raw_data)
                            except json.JSONDecodeError:
                                continue
                            payload = envelope.get("payload", {})
                            if isinstance(payload, dict) and (
                                "status" in payload or "progress_permille" in payload
                            ):
                                job.update(payload)
                            else:
                                job = _request(
                                    "GET",
                                    f"/v1/jobs/{job_id}",
                                    api_url=api_url,
                                )
                            status_value = str(job.get("status", ""))
                            terminal = status_value in _TERMINAL_STATUSES
                            _show_progress(job, final=terminal)
                            if terminal:
                                return job
                job = _request("GET", f"/v1/jobs/{job_id}", api_url=api_url)
                if str(job.get("status")) in _TERMINAL_STATUSES:
                    _show_progress(job, final=True)
                    return job
            except httpx.HTTPError as exc:
                reconnects += 1
                if reconnects > 5:
                    typer.echo(f"Mất kết nối luồng tiến độ: {exc}", err=True)
                    raise typer.Exit(code=1) from exc
                time.sleep(min(5.0, float(reconnects)))
    except KeyboardInterrupt as exc:
        if sys.stdout.isatty():
            typer.echo()
        typer.echo("Đã dừng theo dõi; job vẫn tiếp tục chạy trên máy chủ", err=True)
        raise typer.Exit(code=130) from exc


@app.command()
def version() -> None:
    """Hiển thị phiên bản CLI và Python đang dùng."""

    try:
        package_version = metadata.version("thuyet-minh-offline-gpu")
    except metadata.PackageNotFoundError:
        package_version = "0.1.0+source"
    _print_json(
        {
            "cli": package_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )


@app.command()
def doctor(
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Coi cảnh báo môi trường không phải Linux là lỗi.",
    ),
) -> None:
    """Kiểm tra source, dung lượng, runtime GPU và API mà không sửa hệ thống."""

    checks: list[dict[str, Any]] = []

    def add(name: str, state: str, message: str) -> None:
        checks.append({"name": name, "status": state, "message": message})

    root = _project_root()
    required_files = (
        root / "pyproject.toml",
        root / "config/models.lock.json",
        root / "native/components.lock.json",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    add(
        "source",
        "error" if missing else "ok",
        f"Thiếu: {', '.join(missing)}" if missing else str(root),
    )
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    add(
        "python",
        "ok" if supported_python else "error",
        platform.python_version(),
    )
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError as exc:
        add("disk", "error", str(exc))
    else:
        add(
            "disk",
            "ok" if free_bytes >= 25 * 1024**3 else "warning",
            f"Còn trống {_human_bytes(free_bytes)}",
        )

    if platform.system() == "Linux":
        for command in ("ffmpeg", "ffprobe", "nvidia-smi", "sqlite3"):
            found = shutil.which(command)
            add(command, "ok" if found else "error", found or "Không tìm thấy")
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            add(
                "gpu",
                "ok" if result.returncode == 0 else "error",
                (result.stdout or result.stderr).strip(),
            )
    else:
        add(
            "native-runtime",
            "error" if strict else "warning",
            "Stack GPU native chỉ được hỗ trợ trên Ubuntu Linux",
        )

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{_base_url(api_url)}/v1/health")
        if response.is_error:
            add("api", "error", _error_message(response))
        else:
            payload = response.json()
            api_state = str(payload.get("status", "unknown"))
            add(
                "api",
                "ok" if api_state == "ok" else "warning",
                f"{_base_url(api_url)} ({api_state})",
            )
    except (httpx.HTTPError, ValueError) as exc:
        add("api", "error", f"Không thể kiểm tra API: {exc}")

    overall = "error" if any(item["status"] == "error" for item in checks) else (
        "warning" if any(item["status"] == "warning" for item in checks) else "ok"
    )
    _print_json({"status": overall, "checks": checks})
    if overall == "error":
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Địa chỉ bind; mặc định lấy từ DUB_API_HOST."),
    port: int | None = typer.Option(None, min=1, max=65535),
) -> None:
    """Khởi động API cục bộ."""

    import uvicorn

    settings = Settings()
    uvicorn.run(
        "dub_server.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


@app.command()
def health(
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Hiển thị trạng thái API, lưu trữ, model và GPU."""

    _print_json(_request("GET", "/v1/health", api_url=api_url))


@app.command()
def capabilities(
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Hiển thị các khả năng của pipeline cục bộ."""

    _print_json(_request("GET", "/v1/capabilities", api_url=api_url))


@models_app.command("list")
def list_models(
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Liệt kê model cục bộ mà không tải dữ liệu."""

    _print_json(_request("GET", "/v1/models", api_url=api_url))


@models_app.command("install")
def install_local_model(
    model_id: str = typer.Argument(..., help="ID model trong models.lock.json."),
    lock_path: Path | None = typer.Option(None, "--lock"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
) -> None:
    """Tải model đã khóa bằng trình quản lý mạng tường minh rồi xác minh SHA-256."""

    from .model_manager import install_model
    from .model_registry import ModelRegistryError

    settings = Settings()
    try:
        verified = install_model(
            lock_path or settings.models_lock_path,
            models_dir or settings.models_dir,
            model_id,
        )
    except (OSError, ValueError, ModelRegistryError) as exc:
        typer.echo(f"Không thể cài model {model_id}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_json(
        {
            "id": verified.model_id,
            "stage": verified.stage,
            "path": str(verified.path),
            "tree_sha256": verified.tree_sha256,
            "valid": True,
        }
    )


@models_app.command("verify")
def verify_local_model(
    model_id: str | None = typer.Argument(
        None,
        help="ID model trong models.lock.json; bỏ trống khi dùng --all.",
    ),
    all_models: bool = typer.Option(False, "--all", help="Kiểm tra mọi model đã cài."),
    lock_path: Path | None = typer.Option(None, "--lock"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
) -> None:
    """Băm lại toàn bộ file của một model đã cài mà không mở mạng."""

    from .model_manager import catalog_status, verify_model
    from .model_registry import ModelRegistryError

    settings = Settings()
    if (model_id is None) == (not all_models):
        raise typer.BadParameter("Hãy truyền MODEL_ID hoặc dùng --all, nhưng không dùng cả hai")
    try:
        if all_models:
            result = catalog_status(
                lock_path or settings.models_lock_path,
                models_dir or settings.models_dir,
            )
            _print_json(result)
            invalid = [
                item
                for item in result["models"]
                if item.get("installed") and not item.get("valid")
            ]
            if invalid:
                raise typer.Exit(code=1)
            return
        assert model_id is not None
        verified = verify_model(
            lock_path or settings.models_lock_path,
            models_dir or settings.models_dir,
            model_id,
        )
    except (OSError, ValueError, ModelRegistryError) as exc:
        typer.echo(f"Model {model_id} không hợp lệ: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_json(
        {
            "id": verified.model_id,
            "stage": verified.stage,
            "path": str(verified.path),
            "tree_sha256": verified.tree_sha256,
            "valid": True,
        }
    )


@models_app.command("show")
def show_local_model(
    model_id: str = typer.Argument(...),
    lock_path: Path | None = typer.Option(None, "--lock"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
) -> None:
    """Hiển thị manifest, dung lượng và trạng thái cài của một model."""

    from .config import load_model_catalog

    settings = Settings()
    try:
        catalog = load_model_catalog(
            lock_path or settings.models_lock_path,
            models_dir or settings.models_dir,
        )
        model = next(item for item in catalog.models if item.id == model_id)
    except StopIteration as exc:
        typer.echo(f"Không có model {model_id} trong catalog", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError) as exc:
        typer.echo(f"Không thể đọc catalog model: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = model.model_dump(mode="json")
    files = payload.get("files", [])
    payload["size_bytes"] = sum(
        int(item.get("size", 0)) for item in files if isinstance(item, dict)
    )
    _print_json(payload)


@models_app.command("profiles")
def list_model_profiles(
    lock_path: Path | None = typer.Option(None, "--lock"),
) -> None:
    """Liệt kê các profile model cài sẵn và dung lượng tải dự kiến."""

    from .model_registry import read_model_catalog

    settings = Settings()
    try:
        catalog = read_model_catalog(lock_path or settings.models_lock_path)
    except (OSError, ValueError) as exc:
        typer.echo(f"Không thể đọc catalog model: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    entries = {str(item["id"]): item for item in catalog["models"]}
    profiles = []
    for name, model_ids in _MODEL_PROFILES.items():
        size = sum(
            sum(int(file["size"]) for file in entries[model_id].get("files", []))
            for model_id in model_ids
        )
        minimum_vram = max(
            int(entries[model_id].get("minimum_vram_mib", 0) or 0)
            for model_id in model_ids
        )
        profiles.append(
            {
                "name": name,
                "models": list(model_ids),
                "download_bytes": size,
                "download_size": _human_bytes(size),
                "minimum_vram_mib": minimum_vram,
            }
        )
    _print_json({"profiles": profiles})


@models_app.command("recommend")
def recommend_model_profile(
    vram_mib: int | None = typer.Option(None, "--vram-mib", min=0),
) -> None:
    """Đề xuất profile model theo VRAM hiện có."""

    detected = vram_mib
    if detected is None and shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        values = [
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        ]
        if values:
            detected = max(values)
    if detected is None:
        profile = "minimal"
        reason = "Không đọc được VRAM; chọn cấu hình an toàn nhất"
    elif detected >= 22_528:
        profile = "maximum"
        reason = "Đủ VRAM cho Gemma 4 31B Q4"
    elif detected >= 8_192:
        profile = "balanced"
        reason = "Đủ VRAM cho profile cân bằng"
    else:
        profile = "minimal"
        reason = "VRAM thấp; ưu tiên model nhỏ"
    _print_json(
        {
            "profile": profile,
            "vram_mib": detected,
            "reason": reason,
            "models": list(_MODEL_PROFILES[profile]),
        }
    )


@models_app.command("install-profile")
def install_model_profile(
    profile: str = typer.Argument(..., help="minimal, balanced hoặc maximum"),
    yes: bool = typer.Option(False, "--yes", help="Xác nhận tải toàn bộ model."),
    lock_path: Path | None = typer.Option(None, "--lock"),
    models_dir: Path | None = typer.Option(None, "--models-dir"),
) -> None:
    """Cài tuần tự và xác minh trọn bộ model của một profile."""

    from .model_manager import install_model
    from .model_registry import ModelRegistryError

    if profile not in _MODEL_PROFILES:
        raise typer.BadParameter("Profile phải là minimal, balanced hoặc maximum")
    if not yes:
        raise typer.BadParameter(
            "Profile có thể tải hàng chục GiB; hãy dùng --yes để xác nhận",
            param_hint="--yes",
        )
    settings = Settings()
    selected_lock = lock_path or settings.models_lock_path
    selected_dir = models_dir or settings.models_dir
    installed: list[dict[str, Any]] = []
    for model_id in _MODEL_PROFILES[profile]:
        typer.echo(f"Đang cài {model_id}...", err=True)
        try:
            verified = install_model(selected_lock, selected_dir, model_id)
        except (OSError, ValueError, ModelRegistryError) as exc:
            typer.echo(f"Không thể cài model {model_id}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        installed.append(
            {
                "id": verified.model_id,
                "stage": verified.stage,
                "tree_sha256": verified.tree_sha256,
                "valid": True,
            }
        )
    _print_json({"profile": profile, "models": installed, "valid": True})


@app.command()
def search(
    query: str = typer.Argument(..., help="Tên nội dung cần tìm."),
    year: int | None = typer.Option(None, min=1888, max=2200),
    media_type: str = typer.Option("movie", "--type", help="movie hoặc series"),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Tìm trên các indexer do quản trị viên cấu hình."""

    if media_type != "movie":
        raise typer.BadParameter("Phase 1 chỉ hỗ trợ loại nội dung movie")
    _print_json(
        _request(
            "POST",
            "/v1/search",
            api_url=api_url,
            payload={"query": query, "year": year, "media_type": media_type},
        )
    )


@jobs_app.command("create")
@app.command("submit")
@app.command("run")
def run_job(
    release_id: str = typer.Option(..., "--release-id"),
    rights_confirmed: bool = typer.Option(
        False,
        "--i-have-rights",
        help="Xác nhận bạn có quyền tải và xử lý nội dung.",
    ),
    source_language: str = typer.Option("auto", "--source-language"),
    subtitle_mode: str = typer.Option("prefer", "--subtitle-mode"),
    asr_model: str | None = typer.Option(None, "--asr-model"),
    translation_model: str | None = typer.Option(None, "--translation-model"),
    separation_model: str | None = typer.Option(None, "--separation-model"),
    tts_model: str | None = typer.Option(None, "--tts-model"),
    voice_id: str | None = typer.Option(None, "--voice-id"),
    voice_reference: str | None = typer.Option(None, "--voice-reference"),
    voice_rights_confirmed: bool = typer.Option(
        False,
        "--i-have-voice-rights",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Theo dõi đến khi hoàn tất hoặc cần người dùng xử lý.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Tự tải MP4 khi dùng --wait và job hoàn tất.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Tạo job tải và thuyết minh có xác nhận quyền."""

    if not rights_confirmed:
        raise typer.BadParameter(
            "Phải dùng --i-have-rights để xác nhận quyền tải và xử lý nội dung",
            param_hint="--i-have-rights",
        )
    if subtitle_mode not in {"prefer", "manual", "asr"}:
        raise typer.BadParameter("Chế độ phụ đề phải là prefer, manual hoặc asr")
    voice = None
    if voice_id or voice_reference:
        if not voice_rights_confirmed:
            raise typer.BadParameter(
                "Phải dùng --i-have-voice-rights khi chọn giọng tham chiếu",
                param_hint="--i-have-voice-rights",
            )
        voice = {"voice_id": voice_id, "reference_path": voice_reference}
    payload = {
        "release_id": release_id,
        "rights_confirmed": True,
        "source_language": source_language,
        "subtitle_mode": subtitle_mode,
        "models": {
            "asr": asr_model,
            "translation": translation_model,
            "separation": separation_model,
            "tts": tts_model,
        },
        "voice": voice,
        "voice_rights_confirmed": voice_rights_confirmed,
    }
    if output is not None and not wait:
        raise typer.BadParameter("--output chỉ dùng cùng --wait", param_hint="--output")
    created = _request("POST", "/v1/jobs", api_url=api_url, payload=payload)
    _print_json(created)
    if not wait:
        return
    final = _watch_job(str(created["id"]), api_url=api_url)
    final_status = str(final.get("status"))
    if final_status == "completed":
        if output is not None:
            _download_artifact(
                f"/v1/jobs/{created['id']}/artifacts/video",
                output,
                api_url=api_url,
                overwrite=overwrite,
            )
        return
    if final_status in _ACTION_REQUIRED_STATUSES:
        typer.echo(
            "Job đang chờ thao tác. Dùng dub status, subtitle-select hoặc language-select.",
            err=True,
        )
        raise typer.Exit(code=2)
    error = final.get("error")
    if isinstance(error, dict) and error.get("message"):
        typer.echo(str(error["message"]), err=True)
    raise typer.Exit(code=1)


@jobs_app.command("list")
def list_jobs_command(
    statuses: list[str] | None = typer.Option(
        None,
        "--status",
        help="Có thể lặp lại để lọc nhiều trạng thái.",
    ),
    limit: int = typer.Option(50, min=1, max=1000),
    oldest_first: bool = typer.Option(False, "--oldest-first"),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Liệt kê lịch sử job, mặc định mới nhất trước."""

    params: list[tuple[str, Any]] = [
        ("limit", limit),
        ("newest_first", not oldest_first),
    ]
    params.extend(("status", value) for value in statuses or [])
    _print_json(_request("GET", "/v1/jobs", api_url=api_url, params=params))


@jobs_app.command("watch")
@app.command("watch")
def watch_job_command(
    job_id: str = typer.Argument(...),
    after: int = typer.Option(0, min=0, help="ID event cuối đã nhận."),
    fetch_dir: Path | None = typer.Option(
        None,
        "--fetch-dir",
        help="Tự tải đủ MP4, SRT và timing report khi hoàn tất.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Theo dõi SSE và hiển thị thanh tiến trình chi tiết theo stage."""

    final = _watch_job(job_id, api_url=api_url, after=after)
    final_status = str(final.get("status"))
    if final_status == "completed" and fetch_dir is not None:
        target_dir = fetch_dir.expanduser().resolve(strict=False)
        for kind, suffix in {
            "video": ".mp4",
            "subtitle": ".vi.srt",
            "timing": ".timing-report.json",
        }.items():
            _download_artifact(
                f"/v1/jobs/{job_id}/artifacts/{kind}",
                target_dir / f"{job_id}{suffix}",
                api_url=api_url,
                overwrite=overwrite,
            )
    elif final_status in _ACTION_REQUIRED_STATUSES:
        raise typer.Exit(code=2)
    elif final_status != "completed":
        raise typer.Exit(code=1)


@jobs_app.command("events")
@app.command("events")
def job_events_command(
    job_id: str = typer.Argument(...),
    after: int = typer.Option(0, min=0),
    once: bool = typer.Option(False, "--once", help="Đọc event hiện có rồi thoát."),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Xuất event SSE dạng JSON Lines cho script và automation."""

    try:
        timeout = httpx.Timeout(None, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "GET",
                f"{_base_url(api_url)}/v1/jobs/{job_id}/events",
                params={"after": after, "once": once},
            ) as response:
                if response.is_error:
                    typer.echo(_error_message(response), err=True)
                    raise typer.Exit(code=1)
                for event in _iter_sse(response):
                    if not event.get("data"):
                        continue
                    try:
                        payload = json.loads(event["data"])
                    except json.JSONDecodeError:
                        payload = {"data": event["data"]}
                    payload["event_id"] = event.get("id")
                    payload["event"] = event.get("event")
                    typer.echo(json.dumps(payload, ensure_ascii=False))
    except httpx.HTTPError as exc:
        typer.echo(f"Không thể đọc luồng event: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@jobs_app.command("show")
@app.command()
def status(
    job_id: str = typer.Argument(...),
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Cập nhật tiến độ tải trước khi đọc trạng thái.",
    ),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Hiển thị trạng thái và checkpoint của một job."""

    if refresh:
        _request("POST", f"/v1/jobs/{job_id}/refresh", api_url=api_url)
    _print_json(_request("GET", f"/v1/jobs/{job_id}", api_url=api_url))


@jobs_app.command("subtitle-select")
@app.command("subtitle-select")
def subtitle_select(
    job_id: str = typer.Argument(...),
    subtitle_id: str = typer.Argument(...),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Chọn một phụ đề đã được đề xuất cho job thủ công."""

    _print_json(
        _request(
            "POST",
            f"/v1/jobs/{job_id}/subtitles/{subtitle_id}",
            api_url=api_url,
        )
    )


@jobs_app.command("subtitle-use-asr")
@app.command("subtitle-use-asr")
def subtitle_use_asr(
    job_id: str = typer.Argument(...),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Bỏ qua các phụ đề đề xuất và dùng ASR cục bộ."""

    _print_json(
        _request(
            "POST",
            f"/v1/jobs/{job_id}/subtitles/use-asr",
            api_url=api_url,
        )
    )


@jobs_app.command("language-select")
@app.command("language-select")
def language_select(
    job_id: str = typer.Argument(...),
    language: str = typer.Argument(..., help="Mã ngôn ngữ nguồn, ví dụ en, ja hoặc th."),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Chọn lại ngôn ngữ khi nhận dạng tự động chưa đủ tin cậy."""

    _print_json(
        _request(
            "POST",
            f"/v1/jobs/{job_id}/language",
            api_url=api_url,
            payload={"language": language},
        )
    )


@jobs_app.command("cancel")
@app.command()
def cancel(
    job_id: str = typer.Argument(...),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Hủy job nhưng giữ lại dữ liệu nguồn của torrent client."""

    _print_json(
        _request("POST", f"/v1/jobs/{job_id}/cancel", api_url=api_url)
    )


@jobs_app.command("resume")
@app.command()
def resume(
    job_id: str = typer.Argument(...),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Tiếp tục job tạm dừng hoặc lỗi có thể thử lại."""

    _print_json(
        _request("POST", f"/v1/jobs/{job_id}/resume", api_url=api_url)
    )


@jobs_app.command("fetch")
@app.command("fetch")
def fetch_artifact(
    job_id: str = typer.Argument(...),
    kind: str = typer.Option(
        "video",
        "--kind",
        help="video, subtitle, timing hoặc all",
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    resume_partial: bool = typer.Option(
        True,
        "--resume-partial/--no-resume-partial",
        help="Tiếp tục file .part bằng HTTP Range khi có thể.",
    ),
    api_url: str | None = typer.Option(None, envvar="DUB_API_URL"),
) -> None:
    """Tải MP4, SRT tiếng Việt hoặc timing report của job đã hoàn tất."""

    suffixes = {
        "video": ".mp4",
        "subtitle": ".vi.srt",
        "timing": ".timing-report.json",
    }
    if kind not in {*suffixes, "all"}:
        raise typer.BadParameter("Loại file phải là video, subtitle, timing hoặc all")
    if kind == "all":
        destination_dir = (output or Path.cwd()).expanduser().resolve(strict=False)
        if destination_dir.exists() and not destination_dir.is_dir():
            raise typer.BadParameter("Với --kind all, --output phải là thư mục")
        for artifact_kind, suffix in suffixes.items():
            _download_artifact(
                f"/v1/jobs/{job_id}/artifacts/{artifact_kind}",
                destination_dir / f"{job_id}{suffix}",
                api_url=api_url,
                overwrite=overwrite,
                resume_partial=resume_partial,
            )
        return
    destination = output or Path(f"{job_id}{suffixes[kind]}")
    _download_artifact(
        f"/v1/jobs/{job_id}/artifacts/{kind}",
        destination,
        api_url=api_url,
        overwrite=overwrite,
        resume_partial=resume_partial,
    )


@stack_app.command("start")
def stack_start() -> None:
    """Khởi động API, worker, Prowlarr và qBittorrent."""

    _run_project_script("scripts/native-stack.sh", ["start"])


@stack_app.command("stop")
def stack_stop() -> None:
    """Dừng sạch stack native mà không dùng SIGKILL."""

    _run_project_script("scripts/native-stack.sh", ["stop"])


@stack_app.command("restart")
def stack_restart() -> None:
    """Khởi động lại sạch toàn bộ stack native."""

    _run_project_script("scripts/native-stack.sh", ["restart"])


@stack_app.command("status")
def stack_status() -> None:
    """Hiển thị trạng thái bốn tiến trình native."""

    _run_project_script("scripts/native-stack.sh", ["status"])


@stack_app.command("logs")
def stack_logs(
    lines: int = typer.Option(100, "--lines", "-n", min=1, max=100_000),
) -> None:
    """Hiển thị phần cuối log của toàn bộ dịch vụ."""

    _run_project_script("scripts/native-stack.sh", ["logs", str(lines)])


@stack_app.command("foreground")
def stack_foreground() -> None:
    """Chạy Supervisor foreground cho startup command của nhà cung cấp VM."""

    _run_project_script("scripts/native-stack.sh", ["foreground"])


@stack_app.command("preflight")
def stack_preflight() -> None:
    """Chạy kiểm tra native đầy đủ bằng user dịch vụ."""

    _run_project_script("scripts/native-preflight.sh")


@stack_app.command("init-services")
def stack_init_services(
    rotate_secrets: bool = typer.Option(False, "--rotate-secrets"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Khóa WebUI về loopback và tạo/cập nhật secret dịch vụ."""

    if rotate_secrets and not yes:
        raise typer.BadParameter(
            "Dùng --yes để xác nhận xoay secret Prowlarr/qBittorrent",
            param_hint="--yes",
        )
    arguments = ["--rotate-secrets"] if rotate_secrets else []
    _run_project_script("scripts/native-init-services.sh", arguments)


@stack_app.command("acceptance")
def stack_acceptance(
    phase: str = typer.Option(
        "basic",
        "--phase",
        help="basic, phase2, phase3, phase4 hoặc all",
    ),
) -> None:
    """Chạy cổng nghiệm thu đã khóa cho runtime GPU."""

    scripts = {
        "basic": ["scripts/native-acceptance.sh"],
        "phase2": ["scripts/native-phase2-acceptance.sh"],
        "phase3": ["scripts/native-phase3-acceptance.sh"],
        "phase4": ["scripts/native-phase4-acceptance.sh"],
        "all": [
            "scripts/native-acceptance.sh",
            "scripts/native-phase2-acceptance.sh",
            "scripts/native-phase3-acceptance.sh",
            "scripts/native-phase4-acceptance.sh",
        ],
    }
    if phase not in scripts:
        raise typer.BadParameter("Phase phải là basic, phase2, phase3, phase4 hoặc all")
    for script in scripts[phase]:
        _run_project_script(script)


@maintenance_app.command("cleanup")
def maintenance_cleanup(
    apply: bool = typer.Option(False, "--apply", help="Thực sự xóa artifact trong plan."),
    yes: bool = typer.Option(False, "--yes", help="Xác nhận thao tác xóa."),
    retry_retention_days: int = typer.Option(7, min=0, max=3650),
) -> None:
    """Lập kế hoạch dọn job hủy/lỗi; mặc định không xóa gì."""

    if apply and not yes:
        raise typer.BadParameter(
            "Dùng --yes cùng --apply để xác nhận xóa artifact",
            param_hint="--yes",
        )
    arguments = ["--retry-retention-days", str(retry_retention_days)]
    if apply:
        arguments.append("--apply")
    _run_project_script("scripts/cleanup-job-artifacts.py", arguments)


@maintenance_app.command("sbom")
def maintenance_sbom(
    output: Path = typer.Option(
        Path("var/reports/sbom.cdx.json"),
        "--output",
        "-o",
    ),
) -> None:
    """Tạo CycloneDX SBOM từ các lockfile và môi trường đã cài."""

    _run_project_script(
        "scripts/generate-sbom.py",
        ["--output", str(output.expanduser().resolve(strict=False))],
    )


if __name__ == "__main__":
    app()
