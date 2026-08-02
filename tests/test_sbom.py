from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dub_server.sbom import (
    SbomError,
    _file_sha256,
    _npm_components,
    _read_web_lock,
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


def _write_web_lock(root: Path) -> Path:
    web = root / "package-lock.json"
    web.write_text(
        json.dumps(
            {
                "name": "web-test",
                "version": "1.0.0",
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "web-test", "version": "1.0.0"},
                    "node_modules/react": {
                        "version": "19.2.6",
                        "license": "MIT",
                        "resolved": "https://registry.npmjs.org/react/-/react-19.2.6.tgz",
                        "integrity": "sha256-" + "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE=",
                    },
                    "node_modules/tool": {
                        "version": "2.0.0",
                        "dev": True,
                        "license": "Apache-2.0",
                    },
                    "node_modules/parent/node_modules/@scope/pkg": {
                        "version": "3.1.0",
                        "license": "BSD-3-Clause",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return web


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
    assert native_component["licenses"] == [
        {"license": {"name": "GPL-3.0-or-later"}}
    ]
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


def test_legacy_license_labels_are_names_not_invalid_spdx_ids(tmp_path: Path) -> None:
    models, native = _write_locks(tmp_path)
    document = build_cyclonedx_sbom(
        models,
        native,
        distributions=[FakeDistribution("legacy-license", "1", license_name="BSD")],
    )
    component = next(
        item
        for item in document["components"]
        if item["bom-ref"] == "python:legacy-license@1"
    )
    assert component["licenses"] == [{"license": {"name": "BSD"}}]


def test_build_sbom_inventories_npm_runtime_and_build_dependencies(
    tmp_path: Path,
) -> None:
    models, native = _write_locks(tmp_path)
    web = _write_web_lock(tmp_path)
    document = build_cyclonedx_sbom(
        models,
        native,
        web_lock_path=web,
        distributions=[],
    )

    by_ref = {component["bom-ref"]: component for component in document["components"]}
    react = by_ref["npm:react@19.2.6"]
    assert react["purl"] == "pkg:npm/react@19.2.6"
    assert react["scope"] == "required"
    assert react["hashes"][0]["alg"] == "SHA-256"
    assert react["hashes"][0]["content"] == "61" * 32
    tool = by_ref["npm:tool@2.0.0"]
    assert tool["scope"] == "excluded"
    assert {item["name"]: item["value"] for item in tool["properties"]}[
        "thuyetminh:npm:scope"
    ] == "development"
    assert by_ref["npm:@scope/pkg@3.1.0"]["purl"] == (
        "pkg:npm/%40scope/pkg@3.1.0"
    )
    properties = {
        item["name"]: item["value"] for item in document["metadata"]["properties"]
    }
    assert properties["thuyetminh:web-lock-sha256"] == hashlib.sha256(
        web.read_bytes()
    ).hexdigest()


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


def test_release_sbom_tracks_complete_web_lock() -> None:
    project_root = Path(__file__).resolve().parents[1]
    web_lock = project_root / "web" / "package-lock.json"
    release = json.loads(
        (project_root / "release" / "sbom.cdx.json").read_text(encoding="utf-8")
    )
    expected_refs = {
        component["bom-ref"]
        for component in _npm_components(_read_web_lock(web_lock)["packages"])
    }
    actual_refs = {
        component["bom-ref"]
        for component in release["components"]
        if component["bom-ref"].startswith("npm:")
    }
    assert actual_refs == expected_refs
    properties = {
        item["name"]: item["value"] for item in release["metadata"]["properties"]
    }
    assert properties["thuyetminh:web-lock-sha256"] == _file_sha256(web_lock)
    for component in release["components"]:
        for item in component.get("licenses", []):
            assert "id" not in item.get("license", {}), component["bom-ref"]
