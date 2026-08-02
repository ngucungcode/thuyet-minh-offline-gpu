from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from dub_server.acquisition.subtitles import (
    CompositeSubtitleProvider,
    EmbeddedSubtitleTrack,
    FfprobeSubtitleProbe,
    compute_opensubtitles_hash,
    inspect_subtitle_bytes,
)
from dub_server.domain import (
    AcquisitionError,
    AcquisitionErrorCode,
    MediaAsset,
    SubtitleFormat,
    SubtitleSource,
)


SRT = b"""1
00:00:01,000 --> 00:00:02,500
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""


class FakeProbe:
    async def inspect(self, media_path: Path) -> tuple[EmbeddedSubtitleTrack, ...]:
        return (
            EmbeddedSubtitleTrack(2, "en", SubtitleFormat.SRT, "English"),
            EmbeddedSubtitleTrack(3, "vi", SubtitleFormat.SRT, "Vietnamese"),
        )

    async def extract(self, media_path: Path, stream_index: int, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(SRT)
        return destination


def _media(tmp_path: Path) -> MediaAsset:
    media_path = tmp_path / "Fixture.2026.mkv"
    media_path.write_bytes(b"\0" * (128 * 1024))
    return MediaAsset(
        path=media_path,
        title="Fixture",
        year=2026,
        duration_us=10_000_000,
        fps=24.0,
        source_language="eng",
    )


def _zip_payload(name: str, payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return output.getvalue()


def test_find_prefers_embedded_sidecar_and_exact_hash(tmp_path: Path) -> None:
    media = _media(tmp_path)
    (tmp_path / "Fixture.2026.en.srt").write_bytes(SRT)
    seen_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/subtitles"
        assert request.headers["Authorization"] == "Bearer token"
        assert request.headers["Api-Key"] == "secret"
        assert [key for key, _value in request.url.params.multi_items()] == [
            "languages",
            "moviebytesize",
            "moviehash",
        ]
        seen_params.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "subtitle-1",
                        "attributes": {
                            "language": "en",
                            "release": "Fixture.2026.1080p",
                            "fps": 24.0,
                            "from_trusted": True,
                            "foreign_parts_only": False,
                            "files": [{"file_id": 42, "file_name": "Fixture.en.srt"}],
                        },
                    }
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                embedded_probe=FakeProbe(),
            )
            candidates = await provider.find(media)
        by_source = {item.source: item for item in candidates}
        assert set(by_source) == {
            SubtitleSource.EMBEDDED,
            SubtitleSource.SIDECAR,
            SubtitleSource.OPENSUBTITLES,
        }
        assert all(item.high_confidence for item in by_source.values())
        assert by_source[SubtitleSource.EMBEDDED].language == "en"
        assert seen_params["moviebytesize"] == str(128 * 1024)
        assert len(seen_params["moviehash"]) == 16

    asyncio.run(scenario())


def test_vip_api_root_is_used_for_subtitle_search(tmp_path: Path) -> None:
    media = _media(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "vip-api.opensubtitles.com"
        assert request.url.path == "/api/v1/subtitles"
        return httpx.Response(200, json={"data": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                opensubtitles_base_url="vip-api.opensubtitles.com",
                embedded_probe=FakeProbe(),
            )
            await provider.find(media)

    asyncio.run(scenario())


def test_search_uses_opensubtitles_canonical_query_form(tmp_path: Path) -> None:
    media = _media(tmp_path)
    media.path.write_bytes(b"small fixture without an OpenSubtitles hash")
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert request.url.query.decode() == (
            "languages=en&query=fixture&type=movie&year=2026"
        )
        return httpx.Response(200, json={"data": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                embedded_probe=FakeProbe(),
            )
            await provider.find(media)

    asyncio.run(scenario())
    assert len(seen_urls) == 1


def test_unexpected_opensubtitles_redirect_is_not_followed(tmp_path: Path) -> None:
    media = _media(tmp_path)
    request_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_hosts.append(request.url.host)
        return httpx.Response(
            302,
            headers={"Location": "https://attacker.invalid/collect"},
            content=b"<html>redirect</html>",
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        ) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                embedded_probe=FakeProbe(),
            )
            with pytest.raises(AcquisitionError) as caught:
                await provider.find(media)
        assert caught.value.code is AcquisitionErrorCode.SUBTITLE_UNAVAILABLE
        assert caught.value.retryable is True
        assert "chuyển hướng" in caught.value.message_vi

    asyncio.run(scenario())
    assert request_hosts == ["api.opensubtitles.com"]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.opensubtitles.com/api/v1",
        "https://attacker.invalid/api/v1",
        "https://api.opensubtitles.com.attacker.invalid/api/v1",
        "https://api.opensubtitles.com:444/api/v1",
        "https://api.opensubtitles.com/api/v2",
        "https://api.opensubtitles.com/api/v1?token=secret",
        "https://user:pass@api.opensubtitles.com/api/v1",
    ],
)
def test_unofficial_opensubtitles_api_roots_are_rejected(base_url: str) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError, match="OpenSubtitles API URL"):
                CompositeSubtitleProvider(
                    client=client,
                    opensubtitles_base_url=base_url,
                )

    asyncio.run(scenario())


def test_remote_zip_is_validated_then_materialized_atomically(tmp_path: Path) -> None:
    media = _media(tmp_path)
    archive_payload = _zip_payload("nested/Fixture.en.srt", SRT)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/subtitles":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "attributes": {
                                "language": "en",
                                "files": [{"file_id": 42, "file_name": "Fixture.zip"}],
                            }
                        }
                    ]
                },
            )
        if request.url.path == "/api/v1/download":
            return httpx.Response(
                200,
                json={"link": "https://download.opensubtitles.test/file", "file_name": "Fixture.zip"},
            )
        if request.url.host == "download.opensubtitles.test":
            return httpx.Response(200, content=archive_payload)
        raise AssertionError(request.url)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                embedded_probe=FakeProbe(),
            )
            candidates = await provider.find(media)
            remote = next(item for item in candidates if item.source is SubtitleSource.OPENSUBTITLES)
            output = await provider.materialize(media, remote, tmp_path / "output" / "movie.vi")
        assert output.name == "movie.srt"
        assert output.read_bytes() == SRT
        assert not (output.parent / f".{output.name}.part").exists()

    asyncio.run(scenario())


def test_remote_zip_rejects_path_traversal(tmp_path: Path) -> None:
    media = _media(tmp_path)
    archive_payload = _zip_payload("../escape.srt", SRT)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/subtitles":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "attributes": {
                                "language": "en",
                                "files": [{"file_id": 42, "file_name": "Fixture.zip"}],
                            }
                        }
                    ]
                },
            )
        if request.url.path == "/api/v1/download":
            return httpx.Response(
                200,
                json={"link": "https://download.opensubtitles.test/file", "file_name": "Fixture.zip"},
            )
        return httpx.Response(200, content=archive_payload)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = CompositeSubtitleProvider(
                client=client,
                opensubtitles_api_key="secret",
                opensubtitles_token="token",
                embedded_probe=FakeProbe(),
            )
            candidates = await provider.find(media)
            remote = next(item for item in candidates if item.source is SubtitleSource.OPENSUBTITLES)
            with pytest.raises(AcquisitionError) as caught:
                await provider.materialize(media, remote, tmp_path / "output" / "movie")
        assert caught.value.code is AcquisitionErrorCode.SUBTITLE_ARCHIVE_UNSAFE
        assert not (tmp_path / "escape.srt").exists()

    asyncio.run(scenario())


def test_timing_sanity_rejects_cue_beyond_media_duration() -> None:
    invalid = b"1\n00:00:01,000 --> 00:01:00,000\nToo late\n"
    with pytest.raises(AcquisitionError):
        inspect_subtitle_bytes(invalid, SubtitleFormat.SRT, 10_000_000)


def test_local_subtitle_size_is_bounded_before_read(tmp_path: Path) -> None:
    oversized = tmp_path / "movie.en.srt"
    oversized.write_bytes(b"x" * 1025)

    with pytest.raises(AcquisitionError) as caught:
        from dub_server.acquisition.subtitles import inspect_subtitle_file

        inspect_subtitle_file(
            oversized,
            10_000_000,
            max_bytes=1024,
        )

    assert caught.value.code is AcquisitionErrorCode.SUBTITLE_ARCHIVE_UNSAFE


def test_embedded_subtitle_extraction_sets_ffmpeg_output_limit(
    tmp_path: Path,
) -> None:
    probe = FfprobeSubtitleProbe(max_output_bytes=4096)
    commands: list[tuple[str, ...]] = []

    async def fake_run(command: tuple[str, ...]) -> bytes:
        commands.append(command)
        return b""

    probe._run = fake_run  # type: ignore[method-assign]
    asyncio.run(probe.extract(tmp_path / "movie.mkv", 2, tmp_path / "subtitle"))

    fs_index = commands[0].index("-fs")
    assert commands[0][fs_index + 1] == "4096"


def test_srt_vtt_and_ass_timing_are_supported() -> None:
    vtt = b"WEBVTT\n\n00:01.000 --> 00:02.000\nHello\n"
    ass = b"[Events]\nDialogue: 0,0:00:01.00,0:00:02.20,Default,,0,0,0,,Hello\n"
    assert inspect_subtitle_bytes(SRT, SubtitleFormat.SRT, 10_000_000).cue_count == 2
    assert inspect_subtitle_bytes(vtt, SubtitleFormat.VTT, 10_000_000).cue_count == 1
    assert inspect_subtitle_bytes(ass, SubtitleFormat.ASS, 10_000_000).last_end_us == 2_200_000


def test_opensubtitles_hash_for_zero_fixture_is_file_size(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"\0" * (128 * 1024))
    assert compute_opensubtitles_hash(path) == f"{path.stat().st_size:016x}"
