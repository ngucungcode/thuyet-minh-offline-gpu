from pathlib import Path

import pytest

from dub_server.volume_init import initialize_directories


def test_initialize_directories_creates_and_chowns(tmp_path: Path) -> None:
    target = tmp_path / "data" / "incoming"
    calls: list[tuple[Path, int, int]] = []

    initialize_directories(
        [target],
        uid=10001,
        gid=10001,
        chown=lambda path, uid, gid: calls.append((path, uid, gid)),
    )

    assert target.is_dir()
    assert calls == [(target, 10001, 10001)]


def test_initialize_directories_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="tuyệt đối"):
        initialize_directories([Path("relative")], uid=10001, gid=10001)


def test_initialize_directories_repairs_existing_descendant_ownership(
    tmp_path: Path,
) -> None:
    target = tmp_path / "models"
    nested = target / "model" / "weights.bin"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"fixture")
    calls: list[Path] = []

    initialize_directories(
        [target],
        uid=10002,
        gid=10003,
        chown=lambda path, _uid, _gid: calls.append(path),
    )

    assert nested in calls
    assert nested.parent in calls
    assert target in calls
