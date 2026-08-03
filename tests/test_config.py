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



@pytest.mark.parametrize(
    ("environment_name", "field_name", "minimum"),
    [
        ("DUB_UPLOAD_MEDIA_MAX_BYTES", "upload_media_max_bytes", 1024**2),
        ("DUB_UPLOAD_SUBTITLE_MAX_BYTES", "upload_subtitle_max_bytes", 1024),
        ("DUB_UPLOAD_SESSION_TTL_SECONDS", "upload_session_ttl_seconds", 60),
    ],
)
def test_upload_limits_reject_values_below_production_floor(
    monkeypatch: MonkeyPatch,
    environment_name: str,
    field_name: str,
    minimum: int,
) -> None:
    monkeypatch.setenv(environment_name, str(minimum))
    assert getattr(Settings(), field_name) == minimum

    monkeypatch.setenv(environment_name, str(minimum - 1))
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("environment_name", "field_name", "maximum"),
    [
        (
            "DUB_UPLOAD_MEDIA_MAX_BYTES",
            "upload_media_max_bytes",
            4 * 1024**4,
        ),
        (
            "DUB_UPLOAD_SUBTITLE_MAX_BYTES",
            "upload_subtitle_max_bytes",
            256 * 1024**2,
        ),
        (
            "DUB_UPLOAD_SESSION_TTL_SECONDS",
            "upload_session_ttl_seconds",
            90 * 24 * 60 * 60,
        ),
    ],
)
def test_upload_limits_reject_values_above_production_ceiling(
    monkeypatch: MonkeyPatch,
    environment_name: str,
    field_name: str,
    maximum: int,
) -> None:
    monkeypatch.setenv(environment_name, str(maximum))
    assert getattr(Settings(), field_name) == maximum

    monkeypatch.setenv(environment_name, str(maximum + 1))
    with pytest.raises(ValidationError):
        Settings()
