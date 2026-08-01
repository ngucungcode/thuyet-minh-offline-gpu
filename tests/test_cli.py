from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

import dub_server.cli as cli


runner = CliRunner()


def test_help_exposes_complete_command_tree() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "submit",
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
    assert payload["models"] == {
        "asr": "asr-small",
        "translation": "mt-fast",
        "separation": "separation",
        "tts": "tts-fast",
    }


def test_run_requires_rights_before_calling_api(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API called")),
    )

    result = runner.invoke(cli.app, ["run", "--release-id", "release-1"])

    assert result.exit_code == 2
    assert not isinstance(result.exception, AssertionError)


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
