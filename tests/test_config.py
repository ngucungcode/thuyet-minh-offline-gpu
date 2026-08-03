from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from dub_server.config import Settings


def test_settings_ignore_dotenv_in_working_directory(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Runtime configuration comes only from the prepared environment."""

    (tmp_path / ".env").write_text("DUB_API_PORT=65535\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DUB_API_PORT", raising=False)

    assert Settings().api_port == 8080

    monkeypatch.setenv("DUB_API_PORT", "9090")

    assert Settings().api_port == 9090


def test_upload_limits_are_bounded_and_configurable(
    monkeypatch: MonkeyPatch,
) -> None:
    for name in (
        "DUB_UPLOAD_MEDIA_MAX_BYTES",
        "DUB_UPLOAD_SUBTITLE_MAX_BYTES",
        "DUB_UPLOAD_SESSION_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = Settings()
    assert defaults.upload_media_max_bytes == 100 * 1024 * 1024 * 1024
    assert defaults.upload_subtitle_max_bytes == 16 * 1024 * 1024
    assert defaults.upload_session_ttl_seconds == 7 * 24 * 60 * 60

    monkeypatch.setenv("DUB_UPLOAD_MEDIA_MAX_BYTES", str(2 * 1024 * 1024))
    monkeypatch.setenv("DUB_UPLOAD_SUBTITLE_MAX_BYTES", "4096")
    monkeypatch.setenv("DUB_UPLOAD_SESSION_TTL_SECONDS", "3600")
    configured = Settings()
    assert configured.upload_media_max_bytes == 2 * 1024 * 1024
    assert configured.upload_subtitle_max_bytes == 4096
    assert configured.upload_session_ttl_seconds == 3600

    monkeypatch.setenv("DUB_UPLOAD_SESSION_TTL_SECONDS", "59")
    with pytest.raises(ValidationError):
        Settings()
