from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from dub_server.config import load_model_catalog
from dub_server.model_manager import install_model, load_catalog, verify_model
from dub_server.model_registry import (
    ModelRegistryError,
    ModelVerificationError,
    compute_tree_sha256,
    resolve_verified_model,
)


REVISION = "a" * 40


def _write_lock(
    tmp_path: Path,
    files: dict[str, bytes],
    *,
    model_id: str = "asr-test",
    stage: str = "asr",
    revision: str = REVISION,
    tree_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    records = []
    tree_records = []
    for relative_path, content in files.items():
        digest = hashlib.sha256(content).hexdigest()
        records.append(
            {"path": relative_path, "size": len(content), "sha256": digest}
        )
        tree_records.append((relative_path, len(content), digest))
    entry: dict[str, Any] = {
        "id": model_id,
        "stage": stage,
        "backend": "test",
        "repository": "owner/repository",
        "revision": revision,
        "tree_sha256": tree_sha256 or compute_tree_sha256(tree_records),
        "sha256": tree_sha256 or compute_tree_sha256(tree_records),
        "license": "MIT",
        "files": records,
    }
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8"
    )
    return lock, entry


def _write_model(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        destination = root.joinpath(*relative_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def test_load_catalog_accepts_schema_one(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps({"schema_version": 1, "models": [{"id": "x"}]}),
        encoding="utf-8",
    )
    assert load_catalog(lock)["models"][0]["id"] == "x"


def test_load_catalog_rejects_empty_catalog(tmp_path: Path) -> None:
    lock = tmp_path / "models.lock.json"
    lock.write_text(
        json.dumps({"schema_version": 1, "models": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty"):
        load_catalog(lock)


def test_tree_hash_is_deterministic_and_path_sensitive() -> None:
    first = compute_tree_sha256(
        [("b.bin", 2, "b" * 64), ("a.bin", 1, "a" * 64)]
    )
    second = compute_tree_sha256(
        [("a.bin", 1, "a" * 64), ("b.bin", 2, "b" * 64)]
    )
    changed = compute_tree_sha256(
        [("renamed.bin", 1, "a" * 64), ("b.bin", 2, "b" * 64)]
    )
    assert first == second
    assert first != changed


def test_worker_resolves_only_a_fully_verified_offline_model(tmp_path: Path) -> None:
    files = {"config.json": b"{}", "weights/model.bin": b"locked weights"}
    lock, _ = _write_lock(tmp_path, files)
    models_dir = tmp_path / "models"
    _write_model(models_dir / "asr-test", files)

    result = resolve_verified_model(lock, models_dir, "asr-test", "asr")

    assert result.model_id == "asr-test"
    assert result.stage == "asr"
    assert result.path == models_dir / "asr-test"
    assert len(result.tree_sha256) == 64


def test_worker_rejects_wrong_stage_without_network(tmp_path: Path) -> None:
    files = {"model.bin": b"data"}
    lock, _ = _write_lock(tmp_path, files)
    _write_model(tmp_path / "models" / "asr-test", files)

    with pytest.raises(ModelVerificationError, match="not mt"):
        resolve_verified_model(lock, tmp_path / "models", "asr-test", "mt")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unexpected", "allowlist mismatch"),
        ("wrong_size", "size mismatch"),
        ("wrong_hash", "SHA-256 mismatch"),
        ("extra_directory", "directory allowlist mismatch"),
    ],
)
def test_verification_rejects_any_tree_deviation(
    tmp_path: Path, mutation: str, message: str
) -> None:
    files = {"model.bin": b"locked"}
    lock, _ = _write_lock(tmp_path, files)
    model_dir = tmp_path / "models" / "asr-test"
    _write_model(model_dir, files)
    if mutation == "unexpected":
        (model_dir / "extra.txt").write_text("extra", encoding="utf-8")
    elif mutation == "wrong_size":
        (model_dir / "model.bin").write_bytes(b"different size")
    elif mutation == "wrong_hash":
        (model_dir / "model.bin").write_bytes(b"change")
    else:
        (model_dir / "empty").mkdir()

    with pytest.raises(ModelVerificationError, match=message):
        verify_model(lock, tmp_path / "models", "asr-test")


def test_verification_rejects_wrong_tree_digest(tmp_path: Path) -> None:
    files = {"model.bin": b"locked"}
    lock, _ = _write_lock(tmp_path, files, tree_sha256="f" * 64)
    _write_model(tmp_path / "models" / "asr-test", files)

    with pytest.raises(ModelVerificationError, match="tree SHA-256 mismatch"):
        verify_model(lock, tmp_path / "models", "asr-test")


def test_install_uses_exact_pin_allowlist_staging_and_atomic_publish(
    tmp_path: Path,
) -> None:
    files = {"config.json": b"{}", "nested/model.bin": b"weights"}
    lock, _ = _write_lock(tmp_path, files)
    models_dir = tmp_path / "models"
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        local_dir = Path(kwargs["local_dir"])
        assert local_dir.parent.parent == models_dir / ".staging"
        _write_model(local_dir, files)
        metadata = local_dir / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "config.json.metadata").write_text("local", encoding="utf-8")
        return str(local_dir)

    result = install_model(
        lock, models_dir, "asr-test", downloader=fake_snapshot_download
    )

    assert captured["repo_id"] == "owner/repository"
    assert captured["revision"] == REVISION
    assert captured["allow_patterns"] == ["config.json", "nested/model.bin"]
    assert captured["repo_type"] == "model"
    assert result.path == models_dir / "asr-test"
    assert not (result.path / ".cache").exists()
    assert verify_model(lock, models_dir, "asr-test").tree_sha256 == result.tree_sha256
    receipt = json.loads(
        (models_dir / ".verified" / "asr-test.json").read_text(encoding="utf-8")
    )
    assert receipt["revision"] == REVISION
    assert receipt["tree_sha256"] == result.tree_sha256
    listed = load_model_catalog(lock, models_dir).models[0]
    assert listed.installed is True
    assert listed.valid is True
    assert list((models_dir / ".staging").iterdir()) == []


def test_failed_install_never_publishes_partial_model(tmp_path: Path) -> None:
    files = {"model.bin": b"locked"}
    lock, _ = _write_lock(tmp_path, files)
    models_dir = tmp_path / "models"

    def fake_snapshot_download(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        _write_model(local_dir, files)
        (local_dir / "unexpected.txt").write_text("reject", encoding="utf-8")
        return str(local_dir)

    with pytest.raises(ModelVerificationError, match="allowlist mismatch"):
        install_model(lock, models_dir, "asr-test", downloader=fake_snapshot_download)

    assert not (models_dir / "asr-test").exists()
    assert list((models_dir / ".staging").iterdir()) == []


def test_install_never_downloads_over_or_deletes_a_valid_model(tmp_path: Path) -> None:
    files = {"model.bin": b"locked"}
    lock, _ = _write_lock(tmp_path, files)
    models_dir = tmp_path / "models"
    destination = models_dir / "asr-test"
    _write_model(destination, files)
    before = (destination / "model.bin").read_bytes()

    def forbidden_download(**_: Any) -> str:
        raise AssertionError("download must not run for a valid installed model")

    result = install_model(lock, models_dir, "asr-test", downloader=forbidden_download)

    assert result.path == destination
    assert (destination / "model.bin").read_bytes() == before


def test_install_preserves_an_existing_invalid_model(tmp_path: Path) -> None:
    files = {"model.bin": b"locked"}
    lock, _ = _write_lock(tmp_path, files)
    models_dir = tmp_path / "models"
    destination = models_dir / "asr-test"
    _write_model(destination, {"model.bin": b"corrupt"})

    def forbidden_download(**_: Any) -> str:
        raise AssertionError("invalid existing models must not be overwritten")

    with pytest.raises(ModelVerificationError, match="preserved"):
        install_model(lock, models_dir, "asr-test", downloader=forbidden_download)

    assert (destination / "model.bin").read_bytes() == b"corrupt"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_verification_rejects_symlink_artifacts(tmp_path: Path) -> None:
    content = b"locked"
    lock, _ = _write_lock(tmp_path, {"model.bin": content})
    external = tmp_path / "external.bin"
    external.write_bytes(content)
    model_dir = tmp_path / "models" / "asr-test"
    model_dir.mkdir(parents=True)
    try:
        os.symlink(external, model_dir / "model.bin")
    except OSError:
        pytest.skip("creating symlinks requires additional host privileges")

    with pytest.raises(ModelVerificationError, match="Symlink"):
        verify_model(lock, tmp_path / "models", "asr-test")


@pytest.mark.parametrize(
    "revision",
    ["PIN_REQUIRED", "A" * 40, "a" * 39, "g" * 40],
)
def test_install_requires_full_lowercase_commit_sha(
    tmp_path: Path, revision: str
) -> None:
    lock, _ = _write_lock(tmp_path, {"model.bin": b"x"}, revision=revision)

    with pytest.raises(ModelRegistryError, match="full lowercase 40-character"):
        install_model(
            lock,
            tmp_path / "models",
            "asr-test",
            downloader=lambda **_: "unused",
        )


@pytest.mark.parametrize("unsafe_path", ["../x", "/x", "a\\b", "*.bin", ".cache/x"])
def test_manifest_rejects_unsafe_or_non_exact_allowlist_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    lock, entry = _write_lock(tmp_path, {"model.bin": b"x"})
    entry["files"][0]["path"] = unsafe_path
    lock.write_text(
        json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8"
    )

    with pytest.raises(ModelRegistryError):
        install_model(
            lock,
            tmp_path / "models",
            "asr-test",
            downloader=lambda **_: "unused",
        )
