from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from dub_server.acquisition import (
    AcquisitionService,
    ProwlarrIndexerGateway,
    QBittorrentDownloadClient,
)
from dub_server.domain import (
    AcquisitionError,
    AcquisitionErrorCode,
    DownloadState,
    MediaKind,
    MediaQuery,
    ReleaseCandidate,
)


def _release(*, info_hash: str | None = "a" * 40) -> ReleaseCandidate:
    return ReleaseCandidate(
        release_id="release-1",
        title="Legal Test Fixture 2026",
        indexer_id=4,
        protocol="torrent",
        download_uri="magnet:?xt=urn:btih:" + "a" * 40,
        info_hash=info_hash,
    )


def test_prowlarr_search_is_generic_and_maps_candidates() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[
                {"title": "missing source", "indexerId": 3},
                {
                    "title": "Fixture 2026 1080p",
                    "guid": "provider-guid",
                    "indexerId": 9,
                    "protocol": "torrent",
                    "magnetUrl": "magnet:?xt=urn:btih:" + "b" * 40,
                    "size": 1024,
                    "seeders": 7,
                    "leechers": 2,
                    "categories": [{"id": 2000, "name": "Movies"}],
                },
            ],
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = ProwlarrIndexerGateway(
                base_url="http://prowlarr:9696", api_key="top-secret", client=client
            )
            result = await gateway.search(
                MediaQuery("Fixture", year=2026, media_kind=MediaKind.MOVIE)
            )
        assert len(result) == 1
        assert result[0].title == "Fixture 2026 1080p"
        assert result[0].info_hash == "b" * 40
        assert result[0].categories == ("Movies",)
        assert requests[0].headers["X-Api-Key"] == "top-secret"
        assert requests[0].url.params["query"] == "Fixture 2026"
        assert requests[0].url.params["type"] == "search"

    asyncio.run(scenario())


def test_prowlarr_auth_error_does_not_expose_secret() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(401, text="bad top-secret"))
        async with httpx.AsyncClient(transport=transport) as client:
            gateway = ProwlarrIndexerGateway(
                base_url="http://prowlarr:9696", api_key="top-secret", client=client
            )
            with pytest.raises(AcquisitionError) as caught:
                await gateway.search(MediaQuery("Fixture"))
        assert caught.value.retryable is False
        assert "top-secret" not in str(caught.value)

    asyncio.run(scenario())


def test_qbittorrent_add_status_files_and_cancel(tmp_path: Path) -> None:
    requests: list[tuple[str, str, dict[str, list[str]]]] = []
    added = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal added
        form = parse_qs(request.content.decode()) if request.content else {}
        requests.append((request.method, request.url.path, form))
        assert request.headers["Origin"] == "http://qbittorrent:8080"
        assert request.headers["Referer"] == "http://qbittorrent:8080"
        if request.url.path == "/api/v2/auth/login":
            assert form == {"username": ["dub"], "password": ["secret"]}
            return httpx.Response(200, text="Ok.", headers={"Set-Cookie": "SID=session"})
        if request.url.path == "/api/v2/app/version":
            return httpx.Response(200, text="v5.2.3")
        if request.url.path == "/api/v2/torrents/add":
            assert request.headers["Content-Type"].startswith("multipart/form-data;")
            assert b'name="urls"' in request.content
            assert b"magnet:" in request.content
            assert b'name="paused"' in request.content
            assert b"false" in request.content
            added = True
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/info":
            if not added:
                return httpx.Response(200, json=[])
            return httpx.Response(
                200,
                json=[
                    {
                        "hash": "a" * 40,
                        "state": "downloading",
                        "progress": 0.25,
                        "completed": 250,
                        "size": 1000,
                        "dlspeed": 50,
                        "eta": 15,
                    }
                ],
            )
        if request.url.path == "/api/v2/torrents/files":
            return httpx.Response(
                200,
                json=[{"name": "Fixture/video.mkv", "size": 1000, "progress": 0.25}],
            )
        if request.url.path == "/api/v2/torrents/setLocation":
            assert form["hashes"] == ["a" * 40]
            assert form["location"] == [tmp_path.resolve().as_posix()]
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/delete":
            assert form["deleteFiles"] == ["false"]
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/start":
            assert form["hashes"] == ["a" * 40]
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/stop":
            assert form["hashes"] == ["a" * 40]
            return httpx.Response(200, text="Ok.")
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            task = await downloads.add(
                _release(info_hash=None),
                tmp_path.resolve(),
            )
            status = await downloads.status(task.task_id)
            files = await downloads.files(task.task_id)
            await downloads.pause(task.task_id)
            await downloads.resume(task.task_id)
            await downloads.cancel(task.task_id)
        assert task.task_id == "a" * 40
        assert status.state is DownloadState.DOWNLOADING
        assert status.progress == 0.25
        assert files[0].relative_path == Path("Fixture", "video.mkv")
        assert sum(path == "/api/v2/auth/login" for _, path, _ in requests) == 1

    asyncio.run(scenario())


def test_qbittorrent_reuses_owned_torrent_at_new_job_path(
    tmp_path: Path,
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode()) if request.content else {}
        paths.append(request.url.path)
        if request.url.path == "/api/v2/auth/login":
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/app/version":
            return httpx.Response(200, text="v5.2.3")
        if request.url.path == "/api/v2/torrents/info":
            return httpx.Response(
                200,
                json=[{"hash": "a" * 40, "tags": "dub-old-job"}],
            )
        if request.url.path == "/api/v2/torrents/setLocation":
            assert form == {
                "hashes": ["a" * 40],
                "location": [tmp_path.resolve().as_posix()],
            }
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/start":
            return httpx.Response(200, text="Ok.")
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            task = await downloads.add(
                _release(info_hash=None),
                tmp_path.resolve(),
            )
        assert task.task_id == "a" * 40

    asyncio.run(scenario())
    assert "/api/v2/torrents/add" not in paths
    assert paths[-3:] == [
        "/api/v2/torrents/setLocation",
        "/api/v2/app/version",
        "/api/v2/torrents/start",
    ]


def test_qbittorrent_uses_canonical_hash_reported_for_tag(
    tmp_path: Path,
) -> None:
    added = False
    canonical = "b" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal added
        if request.url.path == "/api/v2/auth/login":
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/app/version":
            return httpx.Response(200, text="v5.2.3")
        if request.url.path == "/api/v2/torrents/info":
            if not added:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[{"hash": canonical}])
        if request.url.path == "/api/v2/torrents/add":
            added = True
            return httpx.Response(200, text="Ok.")
        if request.url.path in {
            "/api/v2/torrents/setLocation",
            "/api/v2/torrents/start",
        }:
            assert canonical.encode() in request.content
            return httpx.Response(200, text="Ok.")
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            task = await downloads.add(_release(info_hash="a" * 40), tmp_path.resolve())
        assert task.task_id == canonical

    asyncio.run(scenario())


def test_qbittorrent_reauthenticates_once_after_expired_session() -> None:
    login_count = 0
    status_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count, status_count
        if request.url.path == "/api/v2/auth/login":
            login_count += 1
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/torrents/info":
            status_count += 1
            if status_count == 1:
                return httpx.Response(403, text="Forbidden")
            return httpx.Response(
                200,
                json=[
                    {
                        "hash": "a" * 40,
                        "state": "downloading",
                        "progress": 0.5,
                        "completed": 500,
                        "size": 1000,
                    }
                ],
            )
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            result = await downloads.status("a" * 40)
        assert result.progress == 0.5

    asyncio.run(scenario())
    assert login_count == 2
    assert status_count == 2


def test_qbittorrent_uses_v4_pause_and_resume_endpoints() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/v2/auth/login":
            return httpx.Response(200, text="Ok.")
        if request.url.path == "/api/v2/app/version":
            return httpx.Response(200, text="v4.4.1")
        if request.url.path in {
            "/api/v2/torrents/pause",
            "/api/v2/torrents/resume",
        }:
            return httpx.Response(200, text="Ok.")
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            await downloads.pause("a" * 40)
            await downloads.resume("a" * 40)

    asyncio.run(scenario())
    assert paths.count("/api/v2/app/version") == 1
    assert paths[-2:] == [
        "/api/v2/torrents/pause",
        "/api/v2/torrents/resume",
    ]


def test_service_exposes_download_lifecycle_and_preserves_files_by_default() -> None:
    calls: list[tuple[str, object]] = []

    class FakeIndexer:
        async def search(self, query):
            return ()

    class FakeDownloads:
        async def add(self, release, save_path, *, paused=False):
            raise NotImplementedError

        async def status(self, task_id):
            calls.append(("status", task_id))
            return "status-result"

        async def files(self, task_id):
            calls.append(("files", task_id))
            return ("file-result",)

        async def pause(self, task_id):
            calls.append(("pause", task_id))

        async def resume(self, task_id):
            calls.append(("resume", task_id))

        async def cancel(self, task_id, *, delete_files=False):
            calls.append(("cancel", (task_id, delete_files)))

    class FakeSubtitles:
        async def find(self, media):
            return ()

        async def materialize(self, media, candidate, destination):
            raise NotImplementedError

    async def scenario() -> None:
        service = AcquisitionService(
            indexer=FakeIndexer(), downloads=FakeDownloads(), subtitles=FakeSubtitles()
        )
        assert await service.download_status("task") == "status-result"
        assert await service.download_files("task") == ("file-result",)
        await service.pause_download("task")
        await service.resume_download("task")
        await service.cancel_download("task")

    asyncio.run(scenario())
    assert calls == [
        ("status", "task"),
        ("files", "task"),
        ("pause", "task"),
        ("resume", "task"),
        ("cancel", ("task", False)),
    ]


def test_qbittorrent_rejects_unsafe_file_path(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth/login"):
            return httpx.Response(200, text="Ok.")
        return httpx.Response(200, json=[{"name": "../escape.mkv", "size": 1, "progress": 1}])

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            downloads = QBittorrentDownloadClient(
                base_url="http://qbittorrent:8080",
                username="dub",
                password="secret",
                client=client,
            )
            with pytest.raises(AcquisitionError) as caught:
                await downloads.files("a" * 40)
        assert caught.value.code is AcquisitionErrorCode.INVALID_RESPONSE

    asyncio.run(scenario())


def test_service_enforces_rights_before_download(tmp_path: Path) -> None:
    class FakeIndexer:
        async def search(self, query: MediaQuery) -> tuple[ReleaseCandidate, ...]:
            return (_release(),)

    class FakeDownloads:
        calls = 0

        async def add(self, release, save_path, *, paused=False):
            self.calls += 1
            raise AssertionError("must not be called")

        async def status(self, task_id):
            raise NotImplementedError

        async def files(self, task_id):
            raise NotImplementedError

        async def pause(self, task_id):
            raise NotImplementedError

        async def resume(self, task_id):
            raise NotImplementedError

        async def cancel(self, task_id, *, delete_files=False):
            raise NotImplementedError

    class FakeSubtitles:
        async def find(self, media):
            return ()

        async def materialize(self, media, candidate, destination):
            raise NotImplementedError

    async def scenario() -> None:
        downloads = FakeDownloads()
        service = AcquisitionService(
            indexer=FakeIndexer(), downloads=downloads, subtitles=FakeSubtitles()
        )
        releases = await service.search("Fixture", 2026, "movie")
        with pytest.raises(AcquisitionError) as caught:
            await service.start_download(
                releases[0], tmp_path.resolve(), rights_confirmed=False
            )
        assert caught.value.code is AcquisitionErrorCode.RIGHTS_CONFIRMATION_REQUIRED
        assert downloads.calls == 0

    asyncio.run(scenario())
