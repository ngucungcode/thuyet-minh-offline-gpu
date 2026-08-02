from __future__ import annotations

from pathlib import Path

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
