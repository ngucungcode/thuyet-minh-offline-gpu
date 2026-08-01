from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dub_server.sbom import (
    SbomError,
    build_cyclonedx_sbom,
    write_cyclonedx_sbom,
)


class FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        *,
        license_name: str = "MIT",
        summary: str = "",
    ) -> None:
        self.version = version
        self.metadata = {
            "Name": name,
            "License-Expression": license_name,
            "Summary": summary,
            "Home-page": "https://example.invalid/project",
        }


def _write_locks(root: Path) -> tuple[Path, Path]:
    models = root / "models.lock.json"
    native = root / "components.lock.json"
    models.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "tts-local",
                        "stage": "tts",
                        "backend": "test",
                        "repository": "owner/model",
                        "revision": "abc123",
                        "path": "tts/local",
                        "tree_sha256": "a" * 64,
                        "files": [
                            {"path": "model.bin", "size": 123, "sha256": "b" * 64}
                        ],
                        "license": "Apache-2.0",
                        "selectable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    native.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "components": {
                    "native-tool": {
                        "version": "1.2.3",
                        "repository": "https://example.invalid/native.git",
                        "url": "https://example.invalid/native.tar.gz",
                        "sha256": "c" * 64,
                        "license": "GPL-3.0-or-later",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return models, native


def test_build_sbom_inventories_python_models_and_native(tmp_path: Path) -> None:
    models, native = _write_locks(tmp_path)
    document = build_cyclonedx_sbom(
        models,
        native,
        distributions=[
            FakeDistribution("Example_Package", "2.0", summary="Ví dụ"),
            FakeDistribution("thuyet-minh-offline-gpu", "0.1.0", license_name="GPL-3.0-or-later"),
        ],
        generated_at=datetime(2026, 8, 1, 12, 30, tzinfo=UTC),
        serial_number="urn:uuid:00000000-0000-4000-8000-000000000001",
    )

    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    assert document["metadata"]["timestamp"] == "2026-08-01T12:30:00Z"
    assert document["metadata"]["component"]["version"] == "0.1.0"
    by_ref = {component["bom-ref"]: component for component in document["components"]}
    assert by_ref["python:example-package@2.0"]["purl"] == (
        "pkg:pypi/example-package@2.0"
    )
    model = by_ref["model:tts-local@abc123"]
    assert model["type"] == "machine-learning-model"
    assert model["hashes"] == [{"alg": "SHA-256", "content": "a" * 64}]
    assert {item["name"]: item["value"] for item in model["properties"]}[
        "thuyetminh:model:locked-size-bytes"
    ] == "123"
    native_component = by_ref["native:native-tool@1.2.3"]
    assert native_component["hashes"][0]["content"] == "c" * 64
    root_dependency = document["dependencies"][0]
    assert root_dependency["ref"] == "application:thuyet-minh-offline-gpu"
    assert root_dependency["dependsOn"] == sorted(by_ref)


def test_duplicate_python_distribution_is_deduplicated(tmp_path: Path) -> None:
    models, native = _write_locks(tmp_path)
    document = build_cyclonedx_sbom(
        models,
        native,
        distributions=[
            FakeDistribution("same.package", "1"),
            FakeDistribution("same-package", "1"),
        ],
    )
    assert sum(
        component["bom-ref"] == "python:same-package@1"
        for component in document["components"]
    ) == 1


def test_write_is_atomic_and_removes_partial_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, native = _write_locks(tmp_path)
    destination = tmp_path / "release" / "sbom.cdx.json"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("dub_server.sbom.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        write_cyclonedx_sbom(
            destination,
            models,
            native,
            distributions=[FakeDistribution("package", "1")],
        )

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(destination.parent.glob(".sbom.cdx.json.*.part")) == []


def test_invalid_lock_schema_is_rejected(tmp_path: Path) -> None:
    models, native = _write_locks(tmp_path)
    models.write_text('{"schema_version": 2, "models": []}', encoding="utf-8")
    with pytest.raises(SbomError, match="schema_version 1"):
        build_cyclonedx_sbom(models, native, distributions=[])
