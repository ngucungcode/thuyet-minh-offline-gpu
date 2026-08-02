from __future__ import annotations

import json
import os
import stat
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

import httpx
import pytest

import dub_server.admin_integrations as admin_module
from dub_server.admin_integrations import (
    AdminIntegrationError,
    atomic_write_secret_bundle,
    atomic_write_secret_pair,
    can_manage_secret_bundle,
    delete_secret_bundle,
    delete_secret_pair,
    has_pending_secret_deletion,
)
from dub_server.api import _acquisition_opensubtitles_configuration, create_app
from dub_server.config import Settings
from dub_server.domain import MediaAsset, SubtitleSource
from dub_server.opensubtitles import DEFAULT_OPENSUBTITLES_API_ROOT


_ADMIN_HEADERS = {"X-Dub-Admin-Request": "1"}


def _settings(tmp_path: Path) -> Settings:
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir(mode=0o700)
    return Settings(
        database_path=tmp_path / "state" / "jobs.sqlite3",
        models_lock_path=tmp_path / "models.lock.json",
        models_dir=tmp_path / "models",
        incoming_dir=tmp_path / "incoming",
        jobs_dir=tmp_path / "jobs",
        output_dir=tmp_path / "output",
        gpu_report_path=tmp_path / "gpu-health.json",
        prowlarr_url="http://127.0.0.1:9696",
        prowlarr_api_key_file=secret_dir / "prowlarr_api_key",
        qbittorrent_password_file=None,
        opensubtitles_url="https://api.opensubtitles.com/api/v1",
        opensubtitles_api_key_file=secret_dir / "opensubtitles_api_key",
        opensubtitles_token_file=secret_dir / "opensubtitles_token",
        opensubtitles_base_url_file=secret_dir / "opensubtitles_base_url",
        acquisition_monitor_seconds=30.0,
    )


@asynccontextmanager
async def _client(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    client_host: str = "127.0.0.1",
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=False,
    ) as upstream:
        application = create_app(
            settings=settings,
            admin_http_client=upstream,
        )
        async with application.router.lifespan_context(application):
            api_transport = httpx.ASGITransport(
                app=application,
                client=(client_host, 49152),
            )
            async with httpx.AsyncClient(
                transport=api_transport,
                base_url="http://testserver",
            ) as client:
                yield client


@pytest.mark.asyncio
async def test_admin_endpoints_require_loopback_and_explicit_header(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("No upstream request was expected")

    async with _client(settings, unexpected) as client:
        missing_header = await client.get("/v1/admin/integrations")
    async with _client(settings, unexpected, client_host="203.0.113.20") as client:
        remote = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )
    async with _client(settings, unexpected) as client:
        allowed = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )

    assert missing_header.status_code == 403
    assert remote.status_code == 403
    assert missing_header.json()["detail"]["code"] == "admin_local_access_required"
    assert remote.json()["detail"]["code"] == "admin_local_access_required"
    assert allowed.status_code == 200
    assert allowed.headers["Cache-Control"] == "no-store, private"


@pytest.mark.asyncio
async def test_integration_status_is_redacted_and_reports_editability(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-top-secret\n", encoding="utf-8")
    settings.opensubtitles_api_key_file.write_text("subtitle-api-secret\n", encoding="utf-8")
    settings.opensubtitles_token_file.write_text("subtitle-token-secret\n", encoding="utf-8")
    settings.opensubtitles_base_url_file.write_text(
        "https://vip-api.opensubtitles.com/api/v1\n", encoding="utf-8"
    )

    async with _client(
        settings,
        lambda _request: httpx.Response(500),
    ) as client:
        response = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {
        "prowlarr": {
            "configured": True,
            "editable": False,
            "can_manage": False,
            "cleanup_pending": False,
            "can_delete": False,
        },
        "opensubtitles": {
            "configured": True,
            "editable": True,
            "can_manage": True,
            "cleanup_pending": False,
            "can_delete": True,
        },
    }
    encoded = response.text
    assert "prowlarr-top-secret" not in encoded
    assert "subtitle-api-secret" not in encoded
    assert "subtitle-token-secret" not in encoded


@pytest.mark.asyncio
async def test_integration_status_marks_unmounted_secret_store_read_only(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.opensubtitles_api_key_file = None
    settings.opensubtitles_token_file = None
    settings.opensubtitles_base_url_file = None

    async with _client(
        settings,
        lambda _request: httpx.Response(500),
    ) as client:
        status_response = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )
        configure_response = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": "api-key",
                "username": "viewer",
                "password": "password",
            },
        )

    assert status_response.json()["opensubtitles"] == {
        "configured": False,
        "editable": False,
        "can_manage": False,
        "cleanup_pending": False,
        "can_delete": False,
    }
    assert configure_response.status_code == 503
    assert configure_response.json()["detail"]["code"] == "secret_store_read_only"


@pytest.mark.asyncio
async def test_prowlarr_indexer_list_uses_allowlist_and_redacts_fields(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-top-secret\n", encoding="utf-8")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/api/v1/indexer"
        assert request.headers["X-Api-Key"] == "prowlarr-top-secret"
        return httpx.Response(
            200,
            json=[
                {
                    "id": 7,
                    "name": "Authorized Fixture",
                    "definitionName": "Generic Torznab",
                    "implementationName": "Cardigann",
                    "protocol": "torrent",
                    "privacy": "private",
                    "enable": True,
                    "supportsSearch": True,
                    "supportsRss": False,
                    "priority": 25,
                    "status": {
                        "disabledTill": None,
                        "mostRecentFailure": "2026-08-01T01:02:03Z",
                    },
                    "fields": [
                        {
                            "name": "password",
                            "privacy": "password",
                            "value": "INDEXER-PASSWORD-SECRET",
                        }
                    ],
                    "indexerUrls": [
                        "https://indexer.invalid/?apikey=URL-TOKEN-SECRET"
                    ],
                    "message": {"text": "MESSAGE-SECRET"},
                }
            ],
        )

    async with _client(settings, handler) as client:
        response = await client.get(
            "/v1/admin/prowlarr/indexers",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert len(seen) == 1
    assert response.json() == {
        "items": [
            {
                "id": 7,
                "name": "Authorized Fixture",
                "definition_name": "Generic Torznab",
                "implementation_name": "Cardigann",
                "protocol": "torrent",
                "privacy": "private",
                "enabled": True,
                "supports_search": True,
                "supports_rss": False,
                "priority": 25,
                "disabled_until": None,
                "most_recent_failure": "2026-08-01T01:02:03Z",
            }
        ],
        "count": 1,
    }
    for secret in (
        "prowlarr-top-secret",
        "INDEXER-PASSWORD-SECRET",
        "URL-TOKEN-SECRET",
        "MESSAGE-SECRET",
    ):
        assert secret not in response.text


@pytest.mark.asyncio
async def test_prowlarr_test_all_redacts_validation_failures(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-key\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/indexer/testall"
        assert request.extensions["timeout"]["read"] == 120.0
        return httpx.Response(
            400,
            json=[
                {
                    "id": 8,
                    "isValid": False,
                    "validationFailures": [
                        {
                            "propertyName": "password",
                            "errorMessage": "bad TEST-ALL-PASSWORD-SECRET",
                        }
                    ],
                },
                {"id": 9, "isValid": True, "validationFailures": []},
            ],
        )

    async with _client(settings, handler) as client:
        response = await client.post(
            "/v1/admin/prowlarr/test-all",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {
        "all_ok": False,
        "failed_count": 1,
        "results": [
            {"indexer_id": 8, "ok": False, "failure_count": 1},
            {"indexer_id": 9, "ok": True, "failure_count": 0},
        ],
    }
    assert "TEST-ALL-PASSWORD-SECRET" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_status", [200, 204])
async def test_prowlarr_test_all_accepts_empty_success_response(
    tmp_path: Path,
    upstream_status: int,
) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-key\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/indexer/testall"
        return httpx.Response(upstream_status)

    async with _client(settings, handler) as client:
        response = await client.post(
            "/v1/admin/prowlarr/test-all",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {
        "all_ok": True,
        "failed_count": 0,
        "results": [],
    }


@pytest.mark.asyncio
async def test_prowlarr_upstream_error_never_echoes_secret(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("PROWLARR-ERROR-SECRET\n", encoding="utf-8")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad PROWLARR-ERROR-SECRET")

    async with _client(settings, handler) as client:
        response = await client.get(
            "/v1/admin/prowlarr/indexers",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "prowlarr_auth_failed"
    assert "PROWLARR-ERROR-SECRET" not in response.text


@pytest.mark.asyncio
async def test_prowlarr_timeout_is_typed_and_retryable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-key\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async with _client(settings, handler) as client:
        response = await client.get(
            "/v1/admin/prowlarr/indexers",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "prowlarr_timeout",
        "message": "Prowlarr phản hồi quá thời gian cho phép",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_opensubtitles_login_writes_credentials_and_returned_api_root(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    api_key = "OPENSUBTITLES-API-SECRET"
    username = "fixture-user"
    password = "OPENSUBTITLES-PASSWORD-SECRET"
    token = "OPENSUBTITLES-TOKEN-SECRET"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/login":
            assert request.headers["Api-Key"] == api_key
            assert request.headers["User-Agent"] == settings.opensubtitles_user_agent
            assert json.loads(request.content) == {
                "username": username,
                "password": password,
            }
            return httpx.Response(
                200,
                json={"token": token, "base_url": "vip-api.opensubtitles.com"},
            )
        if request.url.path == "/api/v1/infos/user":
            assert request.url.host == "vip-api.opensubtitles.com"
            assert request.headers["Api-Key"] == api_key
            assert request.headers["Authorization"] == f"Bearer {token}"
            return httpx.Response(
                200,
                json={
                    "data": {
                        "allowed_downloads": 100,
                        "remaining_downloads": 91,
                        "downloads_count": 9,
                        "vip": True,
                        "level": "VIP",
                        "user_id": 123,
                    }
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async with _client(settings, handler) as client:
        response = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": api_key,
                "username": username,
                "password": password,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "authenticated": True,
        "restart_required": True,
        "quota": {
            "allowed_downloads": 100,
            "remaining_downloads": 91,
            "downloads_count": 9,
            "vip": True,
            "level": "VIP",
        },
    }
    assert len(requests) == 2
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    assert settings.opensubtitles_api_key_file.read_text(encoding="utf-8") == f"{api_key}\n"
    assert settings.opensubtitles_token_file.read_text(encoding="utf-8") == f"{token}\n"
    assert settings.opensubtitles_base_url_file.read_text(encoding="utf-8") == (
        "https://vip-api.opensubtitles.com/api/v1\n"
    )
    if os.name == "posix":
        assert stat.S_IMODE(settings.opensubtitles_api_key_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(settings.opensubtitles_token_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(settings.opensubtitles_base_url_file.stat().st_mode) == 0o600

    for secret in (api_key, username, password, token):
        assert secret not in response.text
    stored_files = b"".join(
        path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert password.encode() not in stored_files
    assert username.encode() not in stored_files
    database = settings.database_path.read_bytes()
    assert api_key.encode() not in database
    assert token.encode() not in database


@pytest.mark.asyncio
async def test_vip_api_root_is_used_by_acquisition_after_app_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dub_server.acquisition as acquisition_package
    from dub_server.acquisition.factory import (
        build_acquisition_service as real_build_acquisition_service,
    )

    settings = _settings(tmp_path)
    api_key = "VIP-API-KEY-SECRET"
    token = "VIP-TOKEN-SECRET"

    def admin_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/login":
            return httpx.Response(
                200,
                json={
                    "token": token,
                    "base_url": "vip-api.opensubtitles.com",
                },
            )
        if request.url.path == "/api/v1/infos/user":
            assert request.url.host == "vip-api.opensubtitles.com"
            return httpx.Response(200, json={"data": {"vip": True}})
        raise AssertionError(request.url)

    async with _client(settings, admin_handler) as client:
        configured = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": api_key,
                "username": "fixture-user",
                "password": "VIP-PASSWORD-SECRET",
            },
        )
    assert configured.status_code == 200

    assert settings.prowlarr_api_key_file is not None
    settings.prowlarr_api_key_file.write_text("prowlarr-key\n", encoding="utf-8")
    settings.qbittorrent_password_file = tmp_path / "secrets" / "qbittorrent-password"
    settings.qbittorrent_password_file.write_text("qbittorrent-password\n", encoding="utf-8")
    media_path = tmp_path / "incoming" / "Fixture.2026.mkv"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(b"\0" * (128 * 1024))
    media = MediaAsset(
        path=media_path,
        title="Fixture",
        duration_us=10_000_000,
        source_language="en",
        year=2026,
    )
    seen_urls: list[str] = []

    def acquisition_handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert request.url.host == "vip-api.opensubtitles.com"
        assert request.url.path == "/api/v1/subtitles"
        assert request.headers["Api-Key"] == api_key
        assert request.headers["Authorization"] == f"Bearer {token}"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "attributes": {
                            "language": "en",
                            "files": [
                                {"file_id": 42, "file_name": "Fixture.en.srt"}
                            ],
                        }
                    }
                ]
            },
        )

    class EmptyProbe:
        async def inspect(self, _media_path: Path) -> tuple[()]:
            return ()

        async def extract(
            self,
            _media_path: Path,
            _stream_index: int,
            _destination: Path,
        ) -> Path:
            raise AssertionError("No embedded subtitle should be extracted")

    captured_base_urls: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(acquisition_handler),
        follow_redirects=False,
    ) as acquisition_client:

        def rebuilt_factory(**kwargs: object):
            captured_base_urls.append(str(kwargs["opensubtitles_base_url"]))
            kwargs["client"] = acquisition_client
            kwargs["embedded_probe"] = EmptyProbe()
            return real_build_acquisition_service(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            acquisition_package,
            "build_acquisition_service",
            rebuilt_factory,
        )
        rebuilt_app = create_app(settings=settings)
        async with rebuilt_app.router.lifespan_context(rebuilt_app):
            candidates = await rebuilt_app.state.acquisition.find_subtitles(media)

    assert captured_base_urls == ["https://vip-api.opensubtitles.com/api/v1"]
    assert seen_urls
    assert any(item.source is SubtitleSource.OPENSUBTITLES for item in candidates)


def test_corrupt_persisted_api_root_disables_only_remote_subtitles(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    settings.opensubtitles_api_key_file.write_text("api-key\n", encoding="utf-8")
    settings.opensubtitles_token_file.write_text("token\n", encoding="utf-8")
    settings.opensubtitles_base_url_file.write_text(
        "https://attacker.invalid/api/v1\n",
        encoding="utf-8",
    )

    assert _acquisition_opensubtitles_configuration(settings) == (
        None,
        None,
        DEFAULT_OPENSUBTITLES_API_ROOT,
    )


@pytest.mark.asyncio
async def test_unofficial_legacy_api_root_is_reported_unconfigured(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    settings.opensubtitles_api_key_file.write_text("api-key\n", encoding="utf-8")
    settings.opensubtitles_token_file.write_text("token\n", encoding="utf-8")
    settings.opensubtitles_url = "https://attacker.invalid/api/v1"

    async with _client(settings, lambda _request: httpx.Response(500)) as client:
        response = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )

    assert response.status_code == 200
    assert response.json()["opensubtitles"]["configured"] is False
    assert not settings.opensubtitles_base_url_file.exists()


@pytest.mark.asyncio
async def test_opensubtitles_auth_failure_preserves_existing_secrets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    settings.opensubtitles_api_key_file.write_text("old-api-key\n", encoding="utf-8")
    settings.opensubtitles_token_file.write_text("old-token\n", encoding="utf-8")
    settings.opensubtitles_base_url_file.write_text(
        "https://api.opensubtitles.com/api/v1\n", encoding="utf-8"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="bad NEW-PASSWORD-SECRET and NEW-API-SECRET",
        )

    caplog.set_level("INFO")
    async with _client(settings, handler) as client:
        response = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": "NEW-API-SECRET",
                "username": "fixture-user",
                "password": "NEW-PASSWORD-SECRET",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "opensubtitles_auth_failed"
    assert "NEW-PASSWORD-SECRET" not in response.text
    assert "NEW-API-SECRET" not in response.text
    assert "NEW-PASSWORD-SECRET" not in caplog.text
    assert "NEW-API-SECRET" not in caplog.text
    assert settings.opensubtitles_api_key_file.read_text(encoding="utf-8") == "old-api-key\n"
    assert settings.opensubtitles_token_file.read_text(encoding="utf-8") == "old-token\n"
    assert settings.opensubtitles_base_url_file.read_text(encoding="utf-8") == (
        "https://api.opensubtitles.com/api/v1\n"
    )


@pytest.mark.asyncio
async def test_unofficial_login_base_url_does_not_overwrite_secret_bundle(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    existing = {
        settings.opensubtitles_api_key_file: "old-api-key\n",
        settings.opensubtitles_token_file: "old-token\n",
        settings.opensubtitles_base_url_file: (
            "https://api.opensubtitles.com/api/v1\n"
        ),
    }
    for path, value in existing.items():
        path.write_text(value, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/login"
        return httpx.Response(
            200,
            json={
                "token": "NEW-TOKEN-SECRET",
                "base_url": "https://attacker.invalid/api/v1",
            },
        )

    async with _client(settings, handler) as client:
        response = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": "NEW-API-SECRET",
                "username": "fixture-user",
                "password": "NEW-PASSWORD-SECRET",
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "opensubtitles_invalid_response"
    for path, value in existing.items():
        assert path.read_text(encoding="utf-8") == value
    for secret in (
        "NEW-TOKEN-SECRET",
        "NEW-API-SECRET",
        "NEW-PASSWORD-SECRET",
    ):
        assert secret not in response.text


@pytest.mark.asyncio
async def test_opensubtitles_5xx_is_typed_and_does_not_write_secrets(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream OPENSUBTITLES-5XX-SECRET")

    async with _client(settings, handler) as client:
        response = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": "OPENSUBTITLES-5XX-SECRET",
                "username": "fixture-user",
                "password": "fixture-password",
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "opensubtitles_unavailable",
        "message": "OpenSubtitles đang tạm thời không khả dụng",
        "retryable": True,
    }
    assert "OPENSUBTITLES-5XX-SECRET" not in response.text
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    assert not settings.opensubtitles_api_key_file.exists()
    assert not settings.opensubtitles_token_file.exists()
    assert not settings.opensubtitles_base_url_file.exists()


def test_atomic_secret_pair_rolls_back_first_file_when_second_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("old-first\n", encoding="utf-8")
    second.write_text("old-second\n", encoding="utf-8")
    real_replace = admin_module.os.replace
    replace_calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(admin_module.os, "replace", fail_second_replace)

    with pytest.raises(AdminIntegrationError) as caught:
        atomic_write_secret_pair(first, "new-first", second, "new-second")

    assert caught.value.code == "secret_store_unavailable"
    assert first.read_text(encoding="utf-8") == "old-first\n"
    assert second.read_text(encoding="utf-8") == "old-second\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_secret_bundle_rolls_back_when_third_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / name for name in ("api-key", "token", "base-url"))
    old_values = ("old-api-key\n", "old-token\n", "old-base-url\n")
    for path, value in zip(paths, old_values, strict=True):
        path.write_text(value, encoding="utf-8")
    real_replace = admin_module.os.replace
    replace_calls = 0

    def fail_third_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 3:
            raise OSError("simulated third replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(admin_module.os, "replace", fail_third_replace)

    with pytest.raises(AdminIntegrationError) as caught:
        atomic_write_secret_bundle(
            tuple(
                (path, value)
                for path, value in zip(
                    paths,
                    ("new-api-key", "new-token", "new-base-url"),
                    strict=True,
                )
            )
        )

    assert caught.value.code == "secret_store_unavailable"
    assert tuple(path.read_text(encoding="utf-8") for path in paths) == old_values
    assert not list(tmp_path.glob(".*.tmp"))


def test_secret_bundle_write_rejects_pending_delete_tombstone(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / name for name in ("api-key", "token", "base-url"))
    pending = tmp_path / ".token.delete-pending"
    pending.write_text("OLD-TOKEN-SECRET\n", encoding="utf-8")

    with pytest.raises(AdminIntegrationError) as caught:
        atomic_write_secret_bundle(
            tuple((path, f"new-{path.name}") for path in paths)
        )

    assert caught.value.code == "secret_store_unavailable"
    assert pending.read_text(encoding="utf-8") == "OLD-TOKEN-SECRET\n"
    assert not any(path.exists() for path in paths)


def test_delete_secret_pair_rolls_back_when_second_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first-secret\n", encoding="utf-8")
    second.write_text("second-secret\n", encoding="utf-8")
    real_replace = admin_module.os.replace
    replace_calls = 0

    def fail_second_rename(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated second rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(admin_module.os, "replace", fail_second_rename)

    with pytest.raises(AdminIntegrationError) as caught:
        delete_secret_pair(first, second)

    assert caught.value.code == "secret_store_unavailable"
    assert caught.value.retryable is True
    assert first.read_text(encoding="utf-8") == "first-secret\n"
    assert second.read_text(encoding="utf-8") == "second-secret\n"
    assert not (tmp_path / ".first.delete-pending").exists()
    assert not (tmp_path / ".second.delete-pending").exists()


def test_delete_secret_pair_retries_cleanup_after_second_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first-secret\n", encoding="utf-8")
    second.write_text("second-secret\n", encoding="utf-8")
    real_unlink = admin_module.os.unlink
    unlink_calls = 0

    def fail_second_unlink(path: Path) -> None:
        nonlocal unlink_calls
        unlink_calls += 1
        if unlink_calls == 2:
            raise OSError("simulated second unlink failure")
        real_unlink(path)

    monkeypatch.setattr(admin_module.os, "unlink", fail_second_unlink)

    with pytest.raises(AdminIntegrationError) as caught:
        delete_secret_pair(first, second)

    assert caught.value.code == "secret_cleanup_deferred"
    assert caught.value.retryable is True
    assert not first.exists()
    assert not second.exists()
    assert not (tmp_path / ".first.delete-pending").exists()
    assert (tmp_path / ".second.delete-pending").is_file()

    delete_secret_pair(first, second)

    assert not first.exists()
    assert not second.exists()
    assert not (tmp_path / ".first.delete-pending").exists()
    assert not (tmp_path / ".second.delete-pending").exists()


def test_delete_secret_bundle_rolls_back_when_third_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / name for name in ("api-key", "token", "base-url"))
    values = ("api-secret\n", "token-secret\n", "base-secret\n")
    for path, value in zip(paths, values, strict=True):
        path.write_text(value, encoding="utf-8")
    real_replace = admin_module.os.replace
    replace_calls = 0

    def fail_third_rename(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 3:
            raise OSError("simulated third rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(admin_module.os, "replace", fail_third_rename)

    with pytest.raises(AdminIntegrationError) as caught:
        delete_secret_bundle(paths)

    assert caught.value.code == "secret_store_unavailable"
    assert tuple(path.read_text(encoding="utf-8") for path in paths) == values
    assert not list(tmp_path.glob(".*.delete-pending"))


def test_delete_secret_bundle_retries_after_third_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / name for name in ("api-key", "token", "base-url"))
    for path in paths:
        path.write_text(f"{path.name}-secret\n", encoding="utf-8")
    real_unlink = admin_module.os.unlink
    unlink_calls = 0

    def fail_third_unlink(path: Path) -> None:
        nonlocal unlink_calls
        unlink_calls += 1
        if unlink_calls == 3:
            raise OSError("simulated third unlink failure")
        real_unlink(path)

    monkeypatch.setattr(admin_module.os, "unlink", fail_third_unlink)

    with pytest.raises(AdminIntegrationError) as caught:
        delete_secret_bundle(paths)

    assert caught.value.code == "secret_cleanup_deferred"
    assert not any(path.exists() for path in paths)
    assert has_pending_secret_deletion(paths) is True
    assert can_manage_secret_bundle(paths) is False

    delete_secret_bundle(paths)

    assert has_pending_secret_deletion(paths) is False
    assert not list(tmp_path.glob(".*.delete-pending"))


@pytest.mark.asyncio
async def test_delete_opensubtitles_requires_explicit_confirmation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    settings.opensubtitles_api_key_file.write_text("api-key\n", encoding="utf-8")
    settings.opensubtitles_token_file.write_text("token\n", encoding="utf-8")
    settings.opensubtitles_base_url_file.write_text(
        "https://vip-api.opensubtitles.com/api/v1\n", encoding="utf-8"
    )

    async with _client(
        settings,
        lambda _request: httpx.Response(500),
    ) as client:
        rejected = await client.request(
            "DELETE",
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={"confirm": "DELETE"},
        )
        deleted = await client.request(
            "DELETE",
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={"confirm": "DELETE_OPENSUBTITLES_CREDENTIALS"},
        )

    assert rejected.status_code == 422
    assert deleted.status_code == 200
    assert deleted.json() == {
        "configured": False,
        "authenticated": False,
        "restart_required": True,
        "quota": None,
    }
    assert not settings.opensubtitles_api_key_file.exists()
    assert not settings.opensubtitles_token_file.exists()
    assert not settings.opensubtitles_base_url_file.exists()


@pytest.mark.asyncio
async def test_pending_cleanup_is_reported_blocks_put_and_allows_delete_retry(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    assert settings.opensubtitles_api_key_file is not None
    assert settings.opensubtitles_token_file is not None
    assert settings.opensubtitles_base_url_file is not None
    paths = (
        settings.opensubtitles_api_key_file,
        settings.opensubtitles_token_file,
        settings.opensubtitles_base_url_file,
    )
    pending_token = settings.opensubtitles_token_file.with_name(
        f".{settings.opensubtitles_token_file.name}.delete-pending"
    )
    pending_token.write_text("OLD-TOKEN-SECRET\n", encoding="utf-8")

    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("PUT must not authenticate while cleanup is pending")

    async with _client(settings, unexpected) as client:
        integration_status = await client.get(
            "/v1/admin/integrations",
            headers=_ADMIN_HEADERS,
        )
        rejected = await client.put(
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={
                "api_key": "NEW-API-SECRET",
                "username": "fixture-user",
                "password": "NEW-PASSWORD-SECRET",
            },
        )
        cleaned = await client.request(
            "DELETE",
            "/v1/admin/opensubtitles",
            headers=_ADMIN_HEADERS,
            json={"confirm": "DELETE_OPENSUBTITLES_CREDENTIALS"},
        )

    assert integration_status.json()["opensubtitles"] == {
        "configured": False,
        "editable": False,
        "can_manage": False,
        "cleanup_pending": True,
        "can_delete": True,
    }
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "secret_cleanup_pending"
    assert "NEW-API-SECRET" not in rejected.text
    assert "NEW-PASSWORD-SECRET" not in rejected.text
    assert cleaned.status_code == 200
    assert not pending_token.exists()
    assert not any(path.exists() for path in paths)
