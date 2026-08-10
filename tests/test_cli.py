from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from click import unstyle
from typer.testing import CliRunner

import dub_server.cli as cli
from dub_server.gpu import NvidiaGpu


runner = CliRunner()


def test_help_exposes_complete_command_tree() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "submit",
        "upload",
        "watch",
        "events",
        "models",
        "jobs",
        "stack",
        "maintenance",
        "doctor",
    ):
        assert command in result.stdout


def test_run_builds_frozen_model_payload(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(method, path, *, api_url, payload=None, **kwargs):
        del api_url, kwargs
        calls.append((method, path, payload))
        return {"id": "job-1", "status": "downloading"}

    monkeypatch.setattr(cli, "_request", request)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--release-id",
            "release-1",
            "--i-have-rights",
            "--source-language",
            "ja",
            "--subtitle-mode",
            "asr",
            "--asr-model",
            "asr-small",
            "--translation-model",
            "mt-fast",
            "--separation-model",
            "separation",
            "--tts-model",
            "tts-fast",
        ],
    )

    assert result.exit_code == 0
    payload = calls[0][2]
    assert calls[0][:2] == ("POST", "/v1/jobs")
    assert payload is not None
    assert payload["rights_confirmed"] is True
    assert payload["source_language"] == "ja"
    assert payload["timing_profile"] == "natural"
    assert payload["models"] == {
        "asr": "asr-small",
        "translation": "mt-fast",
        "separation": "separation",
        "tts": "tts-fast",
    }


def test_run_forwards_strict_timing_profile(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def request(method, path, *, api_url, payload=None, **kwargs):
        del method, path, api_url, kwargs
        captured.update(payload or {})
        return {"id": "job-1", "status": "downloading"}

    monkeypatch.setattr(cli, "_request", request)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--release-id",
            "release-1",
            "--i-have-rights",
            "--timing-profile",
            "strict",
        ],
    )

    assert result.exit_code == 0
    assert captured["timing_profile"] == "strict"


def test_run_requires_rights_before_calling_api(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )

    result = runner.invoke(cli.app, ["run", "--release-id", "release-1"])

    assert result.exit_code == 2
    assert not isinstance(result.exception, AssertionError)


def test_upload_command_streams_media_srt_and_finalizes_job(
    monkeypatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mkv"
    subtitle = tmp_path / "movie.srt"
    media.write_bytes(b"video")
    subtitle.write_text("fixture", encoding="utf-8")
    requests: list[tuple[str, str, dict[str, Any] | None]] = []
    uploads: list[tuple[str, Path]] = []

    def request(method, path, *, api_url, payload=None, **kwargs):
        del api_url, kwargs
        requests.append((method, path, payload))
        if path == "/v1/uploads":
            return {
                "id": "upload-1",
                "status": "awaiting_media",
                "subtitle_filename": "movie.srt",
            }
        return {"id": "upload-1", "status": "ready_offline"}

    def upload(path, source, *, api_url):
        del api_url
        uploads.append((path, source))
        return {
            "id": "upload-1",
            "status": "ready",
            "subtitle_filename": "movie.srt",
        }

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(cli, "_upload_artifact", upload)
    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(media),
            "--subtitle",
            str(subtitle),
            "--source-language",
            "en",
            "--timing-profile",
            "natural",
            "--i-have-rights",
        ],
    )

    assert result.exit_code == 0
    assert requests[0][:2] == ("POST", "/v1/uploads")
    payload = requests[0][2]
    assert payload is not None
    assert payload["media_filename"] == "movie.mkv"
    assert payload["subtitle_filename"] == "movie.srt"
    assert payload["source_language"] == "en"
    assert payload["timing_profile"] == "natural"
    assert payload["models"] == {
        "asr": None,
        "translation": None,
        "separation": None,
        "tts": None,
    }
    assert uploads == [
        ("/v1/uploads/upload-1/media", media),
        ("/v1/uploads/upload-1/subtitle", subtitle),
    ]
    assert requests[-1][:2] == ("POST", "/v1/uploads/upload-1/finalize")


def test_upload_resume_only_streams_missing_subtitle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mkv"
    subtitle = tmp_path / "movie.srt"
    media.write_bytes(b"video")
    subtitle.write_bytes(b"subtitle")
    uploads: list[tuple[str, Path]] = []
    requests: list[tuple[str, str]] = []

    def request(method, path, *, api_url, payload=None, **kwargs):
        del api_url, payload, kwargs
        requests.append((method, path))
        if method == "GET":
            return {
                "id": "upload-1",
                "status": "awaiting_subtitle",
                "media_filename": "movie.mkv",
                "subtitle_filename": "movie.srt",
                "media_size_bytes": media.stat().st_size,
                "subtitle_size_bytes": None,
                "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                "subtitle_sha256": None,
                "job_id": None,
            }
        return {"id": "upload-1", "status": "ready_offline"}

    def upload(path, source, *, api_url):
        del api_url
        uploads.append((path, source))
        return {
            "id": "upload-1",
            "status": "ready",
            "media_filename": "movie.mkv",
            "subtitle_filename": "movie.srt",
            "media_size_bytes": media.stat().st_size,
            "subtitle_size_bytes": subtitle.stat().st_size,
            "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "subtitle_sha256": hashlib.sha256(subtitle.read_bytes()).hexdigest(),
            "job_id": None,
        }

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(cli, "_upload_artifact", upload)
    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(media),
            "--subtitle",
            str(subtitle),
            "--upload-id",
            "upload-1",
            "--i-have-rights",
        ],
    )

    assert result.exit_code == 0
    assert uploads == [("/v1/uploads/upload-1/subtitle", subtitle)]
    assert requests == [
        ("GET", "/v1/uploads/upload-1"),
        ("POST", "/v1/uploads/upload-1/finalize"),
    ]


def test_upload_resume_reuploads_same_size_media_when_content_changed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mkv"
    media.write_bytes(b"new!!")
    uploads: list[tuple[str, Path]] = []

    def request(method, path, *, api_url, payload=None, **kwargs):
        del api_url, payload, kwargs
        if method == "GET":
            return {
                "id": "upload-1",
                "status": "ready",
                "media_filename": "movie.mkv",
                "subtitle_filename": None,
                "media_size_bytes": media.stat().st_size,
                "subtitle_size_bytes": None,
                "media_sha256": hashlib.sha256(b"old!!").hexdigest(),
                "subtitle_sha256": None,
                "job_id": None,
            }
        return {"id": "upload-1", "status": "ready_offline"}

    def upload(path, source, *, api_url):
        del api_url
        uploads.append((path, source))
        return {
            "id": "upload-1",
            "status": "ready",
            "media_filename": "movie.mkv",
            "subtitle_filename": None,
            "media_size_bytes": media.stat().st_size,
            "subtitle_size_bytes": None,
            "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            "subtitle_sha256": None,
            "job_id": None,
        }

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(cli, "_upload_artifact", upload)

    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(media),
            "--upload-id",
            "upload-1",
            "--i-have-rights",
        ],
    )

    assert result.exit_code == 0
    assert uploads == [("/v1/uploads/upload-1/media", media)]


def test_upload_resume_reuploads_legacy_session_without_digest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "movie.mkv"
    media.write_bytes(b"video")
    uploads: list[tuple[str, Path]] = []

    def request(method, path, *, api_url, payload=None, **kwargs):
        del api_url, payload, kwargs
        if method == "GET":
            return {
                "id": "upload-1",
                "status": "ready",
                "media_filename": "movie.mkv",
                "subtitle_filename": None,
                "media_size_bytes": media.stat().st_size,
                "subtitle_size_bytes": None,
                "job_id": None,
            }
        return {"id": "upload-1", "status": "ready_offline"}

    def upload(path, source, *, api_url):
        del api_url
        uploads.append((path, source))
        return {"id": "upload-1", "status": "ready"}

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(cli, "_upload_artifact", upload)

    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(media),
            "--upload-id",
            "upload-1",
            "--i-have-rights",
        ],
    )

    assert result.exit_code == 0
    assert uploads == [("/v1/uploads/upload-1/media", media)]


def test_upload_resume_finalized_session_is_idempotent_without_local_read(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requests: list[tuple[str, str]] = []

    def request(method, path, *, api_url, **kwargs):
        del api_url, kwargs
        requests.append((method, path))
        if method == "GET":
            return {
                "id": "upload-1",
                "status": "finalized",
                "media_filename": "movie.mkv",
                "subtitle_filename": None,
                "media_size_bytes": 123,
                "subtitle_size_bytes": None,
                "job_id": "upload-1",
            }
        return {"id": "upload-1", "status": "ready_offline"}

    monkeypatch.setattr(cli, "_request", request)
    monkeypatch.setattr(
        cli,
        "_upload_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("artifact uploaded")
        ),
    )
    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(tmp_path / "file-does-not-need-to-exist.mkv"),
            "--upload-id",
            "upload-1",
            "--i-have-rights",
        ],
    )

    assert result.exit_code == 0
    assert not isinstance(result.exception, AssertionError)
    assert requests == [
        ("GET", "/v1/uploads/upload-1"),
        ("POST", "/v1/uploads/upload-1/finalize"),
    ]


def test_upload_command_requires_language_for_manual_srt(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "upload",
            str(tmp_path / "movie.mp4"),
            "--subtitle",
            str(tmp_path / "movie.srt"),
            "--i-have-rights",
        ],
        color=False,
        terminal_width=200,
    )

    assert result.exit_code == 2
    plain_output = unstyle(result.output)
    assert "Phải chọn" in plain_output
    assert "source-language" in plain_output


def test_upload_rejects_missing_or_empty_media_before_creating_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def request(*args, **kwargs):
        del args, kwargs
        calls.append("api")
        raise AssertionError("API must not be called")

    monkeypatch.setattr(cli, "_request", request)
    missing = runner.invoke(
        cli.app,
        [
            "upload",
            str(tmp_path / "missing.mp4"),
            "--i-have-rights",
        ],
    )
    empty_path = tmp_path / "empty.mkv"
    empty_path.touch()
    empty = runner.invoke(
        cli.app,
        ["upload", str(empty_path), "--i-have-rights"],
    )

    assert missing.exit_code == 1
    assert empty.exit_code == 1
    assert calls == []
    assert not isinstance(missing.exception, AssertionError)
    assert not isinstance(empty.exception, AssertionError)


def test_raw_upload_helper_sets_length_and_streams_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"a" * (1024 * 1024 + 7))
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put(self, url, *, content, headers):
            captured["url"] = url
            captured["headers"] = headers
            captured["chunks"] = list(content)
            return httpx.Response(
                200,
                request=httpx.Request("PUT", url),
                json={"id": "upload-1", "status": "ready"},
            )

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    response = cli._upload_artifact(
        "/v1/uploads/upload-1/media",
        source,
        api_url="http://local",
    )

    assert response["status"] == "ready"
    assert captured["url"] == "http://local/v1/uploads/upload-1/media"
    assert captured["headers"]["Content-Length"] == str(source.stat().st_size)
    assert [len(chunk) for chunk in captured["chunks"]] == [1024 * 1024, 7]


def test_submit_waits_and_downloads_completed_video(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "dubbed.mp4"
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        cli,
        "_request",
        lambda *args, **kwargs: {"id": "job-1", "status": "downloading"},
    )
    monkeypatch.setattr(
        cli,
        "_watch_job",
        lambda *args, **kwargs: {"id": "job-1", "status": "completed"},
    )
    monkeypatch.setattr(
        cli,
        "_download_artifact",
        lambda path, output, **kwargs: downloads.append((path, output)),
    )

    result = runner.invoke(
        cli.app,
        [
            "submit",
            "--release-id",
            "release-1",
            "--i-have-rights",
            "--wait",
            "--output",
            str(destination),
        ],
    )

    assert result.exit_code == 0
    assert downloads == [("/v1/jobs/job-1/artifacts/video", destination)]


def test_jobs_list_forwards_repeated_status_filters(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def request(method, path, **kwargs):
        captured.update({"method": method, "path": path, **kwargs})
        return {"items": [], "count": 0}

    monkeypatch.setattr(cli, "_request", request)
    result = runner.invoke(
        cli.app,
        [
            "jobs",
            "list",
            "--status",
            "failed",
            "--status",
            "completed",
            "--limit",
            "25",
        ],
    )

    assert result.exit_code == 0
    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/jobs"
    assert ("status", "failed") in captured["params"]
    assert ("status", "completed") in captured["params"]
    assert ("limit", 25) in captured["params"]


def test_sse_parser_handles_comments_and_multiline_data() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://local/events"),
        content=(
            b": keep-alive\n\n"
            b"id: 7\nevent: job.status\n"
            b"data: {\"status\":\ndata: \"completed\"}\n\n"
        ),
    )

    events = list(cli._iter_sse(response))

    assert events == [
        {
            "id": "7",
            "event": "job.status",
            "data": '{"status":\n"completed"}',
        }
    ]


def test_artifact_download_resumes_part_file(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "result.mp4"
    partial = tmp_path / ".result.mp4.part"
    partial.write_bytes(b"abc")
    requests: list[dict[str, str] | None] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @contextmanager
        def stream(self, method, url, *, headers=None):
            del method, url
            requests.append(headers)
            yield httpx.Response(
                206,
                request=httpx.Request("GET", "http://local/result"),
                headers={"Content-Range": "bytes 3-5/6"},
                content=b"def",
            )

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    cli._download_artifact(
        "/artifact",
        destination,
        api_url="http://local",
        overwrite=False,
    )

    assert destination.read_bytes() == b"abcdef"
    assert requests == [{"Range": "bytes=3-"}]
    assert not partial.exists()


def test_model_profiles_are_complete(monkeypatch, tmp_path: Path) -> None:
    models = []
    all_ids = {model_id for values in cli._MODEL_PROFILES.values() for model_id in values}
    for index, model_id in enumerate(sorted(all_ids), start=1):
        models.append(
            {
                "id": model_id,
                "stage": "fixture",
                "repository": "owner/repo",
                "revision": str(index % 10) * 40,
                "path": f"models/{model_id}",
                "tree_sha256": str(index % 10) * 64,
                "files": [
                    {"path": "model.bin", "size": index, "sha256": "a" * 64}
                ],
            }
        )
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps({"schema_version": 1, "models": models}),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["models", "profiles", "--lock", str(lock)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [item["name"] for item in payload["profiles"]] == [
        "minimal",
        "balanced",
        "maximum",
    ]
    assert all(item["download_bytes"] > 0 for item in payload["profiles"])
    assert {
        item["name"]: item["minimum_vram_mib"] for item in payload["profiles"]
    } == {"minimal": 6144, "balanced": 8192, "maximum": 22528}


def _gpu_report(*gpus: NvidiaGpu):
    return type("GpuReport", (), {"gpus": gpus})()


def _gpu(name: str, *, vram_mib: int, capability: str) -> NvidiaGpu:
    return NvidiaGpu(
        uuid=f"GPU-{name}",
        name=name,
        driver_version="570.26",
        memory_total_mib=vram_mib,
        compute_capability=capability,
    )


def test_model_recommendation_uses_logical_cuda_zero_not_largest_host_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_gpu",
        lambda **_kwargs: _gpu_report(
            _gpu("Tesla T4", vram_mib=16_384, capability="7.5"),
            _gpu("NVIDIA A100", vram_mib=81_920, capability="8.0"),
        ),
    )

    result = runner.invoke(cli.app, ["models", "recommend"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "balanced"
    assert payload["vram_mib"] == 16_384
    assert payload["gpu"]["name"] == "Tesla T4"


def test_model_recommendation_forwards_installed_cuda_toolkit(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_inspect_gpu(**kwargs):
        captured.update(kwargs)
        return _gpu_report(
            _gpu("NVIDIA RTX 3090", vram_mib=24_576, capability="8.6")
        )

    monkeypatch.setenv("DUB_SELECTED_CUDA_TOOLKIT_VERSION", "12.6")
    monkeypatch.setattr(cli, "inspect_gpu", fake_inspect_gpu)

    result = runner.invoke(cli.app, ["models", "recommend"])

    assert result.exit_code == 0
    assert captured["expected_cuda_toolkit_version"] == "12.6"


def test_cmp_170hx_is_never_recommended_above_minimal(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_gpu",
        lambda **_kwargs: _gpu_report(
            _gpu("NVIDIA CMP 170HX", vram_mib=16_384, capability="8.0")
        ),
    )

    result = runner.invoke(cli.app, ["models", "recommend"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "minimal"
    assert payload["support_tier"] == "experimental"


def test_model_profile_install_rejects_vram_floor_before_download(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_gpu",
        lambda **_kwargs: _gpu_report(
            _gpu("Tesla T4", vram_mib=16_384, capability="7.5")
        ),
    )

    result = runner.invoke(
        cli.app,
        ["models", "install-profile", "maximum", "--yes"],
    )

    assert result.exit_code == 1
    assert "cần ít nhất 22528 MiB VRAM" in result.stderr


def test_cmp_170hx_cannot_install_balanced_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_gpu",
        lambda **_kwargs: _gpu_report(
            _gpu("NVIDIA CMP 170HX", vram_mib=16_384, capability="8.0")
        ),
    )

    result = runner.invoke(
        cli.app,
        ["models", "install-profile", "balanced", "--yes"],
    )

    assert result.exit_code == 1
    assert "chỉ được hỗ trợ thử nghiệm với profile minimal" in result.stderr


def test_structured_validation_error_is_human_readable() -> None:
    response = httpx.Response(
        422,
        request=httpx.Request("POST", "http://local/v1/jobs"),
        json={
            "detail": [
                {"loc": ["body", "release_id"], "msg": "Field required"},
            ]
        },
    )

    assert cli._error_message(response) == "body.release_id: Field required"
