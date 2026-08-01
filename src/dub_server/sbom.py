"""CycloneDX SBOM generation without a runtime SBOM dependency.

The generator inventories the Python environment together with the two lock
manifests that describe downloaded models and native/system components.  It
does not inspect or download model payloads; the immutable lock metadata is
the source of truth for those components.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_NORMALIZE_NAME_RE = re.compile(r"[-_.]+")


class SbomError(RuntimeError):
    """Raised when a lock manifest cannot produce a trustworthy SBOM."""


def build_cyclonedx_sbom(
    models_lock_path: Path,
    native_lock_path: Path,
    *,
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
    generated_at: datetime | None = None,
    serial_number: str | None = None,
) -> dict[str, Any]:
    """Build a CycloneDX 1.6 document from local, read-only inputs."""

    models_lock = _read_lock(models_lock_path, expected_collection="models")
    native_lock = _read_lock(native_lock_path, expected_collection="components")
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    else:
        timestamp = timestamp.astimezone(UTC)

    distribution_list = tuple(
        importlib.metadata.distributions()
        if distributions is None
        else distributions
    )
    python_components = _python_components(distribution_list)
    model_components = _model_components(models_lock["models"])
    native_components = _native_components(native_lock["components"])
    components = sorted(
        [*python_components, *model_components, *native_components],
        key=lambda item: item["bom-ref"],
    )
    refs = [item["bom-ref"] for item in components]
    application_ref = "application:thuyet-minh-offline-gpu"

    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": serial_number or f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "component": {
                "type": "application",
                "bom-ref": application_ref,
                "name": "thuyet-minh-offline-gpu",
                "version": _application_version(distribution_list),
                "licenses": [{"license": {"id": "GPL-3.0-or-later"}}],
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "stdlib-sbom-generator",
                        "version": "1",
                    }
                ]
            },
            "properties": [
                {
                    "name": "thuyetminh:models-lock-sha256",
                    "value": _file_sha256(models_lock_path),
                },
                {
                    "name": "thuyetminh:native-lock-sha256",
                    "value": _file_sha256(native_lock_path),
                },
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": application_ref, "dependsOn": refs},
            *({"ref": ref, "dependsOn": []} for ref in refs),
        ],
    }


def write_cyclonedx_sbom(
    destination: Path,
    models_lock_path: Path,
    native_lock_path: Path,
    *,
    distributions: Iterable[importlib.metadata.Distribution] | None = None,
    generated_at: datetime | None = None,
    serial_number: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one UTF-8 CycloneDX JSON document."""

    document = build_cyclonedx_sbom(
        models_lock_path,
        native_lock_path,
        distributions=distributions,
        generated_at=generated_at,
        serial_number=serial_number,
    )
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return document


def _read_lock(path: Path, *, expected_collection: str) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SbomError(f"Không thể đọc lock manifest: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SbomError(f"Lock manifest không đúng schema_version 1: {path}")
    collection = raw.get(expected_collection)
    expected_type = list if expected_collection == "models" else dict
    if not isinstance(collection, expected_type):
        raise SbomError(
            f"Lock manifest thiếu collection {expected_collection}: {path}"
        )
    return raw


def _python_components(
    distributions: Iterable[importlib.metadata.Distribution],
) -> list[dict[str, Any]]:
    by_ref: dict[str, dict[str, Any]] = {}
    for distribution in distributions:
        metadata = distribution.metadata
        name = str(metadata.get("Name") or "").strip()
        version = str(getattr(distribution, "version", "") or "").strip()
        if not name or not version:
            continue
        normalized = _NORMALIZE_NAME_RE.sub("-", name).lower()
        purl = f"pkg:pypi/{quote(normalized, safe='-')}@{quote(version, safe='.+-')}"
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": f"python:{normalized}@{version}",
            "name": name,
            "version": version,
            "purl": purl,
            "properties": [
                {"name": "thuyetminh:source", "value": "python-distribution"}
            ],
        }
        description = str(metadata.get("Summary") or "").strip()
        if description:
            component["description"] = description
        license_name = str(
            metadata.get("License-Expression") or metadata.get("License") or ""
        ).strip()
        if license_name and license_name.upper() != "UNKNOWN":
            component["licenses"] = [_license(license_name)]
        homepage = str(metadata.get("Home-page") or "").strip()
        if homepage.startswith(("https://", "http://")):
            component["externalReferences"] = [
                {"type": "website", "url": homepage}
            ]
        by_ref[component["bom-ref"]] = component
    return list(by_ref.values())


def _model_components(models: list[Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in models:
        if not isinstance(raw, dict):
            raise SbomError("models.lock.json chứa model không hợp lệ")
        identifier = _required_text(raw, "id", "model")
        if identifier in seen:
            raise SbomError(f"ID model bị trùng trong lock manifest: {identifier}")
        seen.add(identifier)
        revision = str(raw.get("revision") or "unversioned")
        component: dict[str, Any] = {
            "type": "machine-learning-model",
            "bom-ref": f"model:{identifier}@{revision}",
            "name": identifier,
            "version": revision,
            "properties": _properties(
                raw,
                prefix="thuyetminh:model",
                keys=(
                    "stage",
                    "backend",
                    "repository",
                    "path",
                    "compute_type",
                    "minimum_vram_mib",
                    "quality_tier",
                    "selectable",
                ),
            ),
        }
        license_name = str(raw.get("license") or "").strip()
        if license_name:
            component["licenses"] = [_license(license_name)]
        digest = raw.get("tree_sha256") or raw.get("sha256")
        if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
            component["hashes"] = [
                {"alg": "SHA-256", "content": digest.lower()}
            ]
        repository = str(raw.get("repository") or "").strip()
        if repository:
            if repository.startswith(("https://", "http://")):
                url = repository
            else:
                url = f"https://huggingface.co/{repository}/tree/{quote(revision, safe='')}"
            component["externalReferences"] = [
                {"type": "distribution", "url": url}
            ]
        files = raw.get("files")
        if isinstance(files, list):
            total_size = sum(
                item.get("size", 0)
                for item in files
                if isinstance(item, dict)
                and isinstance(item.get("size"), int)
                and not isinstance(item.get("size"), bool)
            )
            component["properties"].extend(
                [
                    {
                        "name": "thuyetminh:model:file-count",
                        "value": str(len(files)),
                    },
                    {
                        "name": "thuyetminh:model:locked-size-bytes",
                        "value": str(total_size),
                    },
                ]
            )
        components.append(component)
    return components


def _native_components(components: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, raw in components.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise SbomError("components.lock.json chứa component không hợp lệ")
        version = str(
            raw.get("version")
            or raw.get("release")
            or raw.get("commit")
            or "system-managed"
        )
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": f"native:{name}@{version}",
            "name": name,
            "version": version,
            "properties": _properties(
                raw,
                prefix="thuyetminh:native",
                keys=(
                    "package",
                    "source",
                    "commit",
                    "cuda_version",
                    "cuda_architectures",
                    "runtime_path",
                ),
            ),
        }
        license_name = str(raw.get("license") or "").strip()
        if license_name:
            component["licenses"] = [_license(license_name)]
        digest = raw.get("sha256")
        if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
            component["hashes"] = [
                {"alg": "SHA-256", "content": digest.lower()}
            ]
        references: list[dict[str, str]] = []
        repository = str(raw.get("repository") or "").strip()
        url = str(raw.get("url") or "").strip()
        if repository.startswith(("https://", "http://")):
            references.append({"type": "vcs", "url": repository})
        if url.startswith(("https://", "http://")):
            references.append({"type": "distribution", "url": url})
        if references:
            component["externalReferences"] = references
        result.append(component)
    return result


def _application_version(
    distributions: Iterable[importlib.metadata.Distribution],
) -> str:
    for distribution in distributions:
        name = str(distribution.metadata.get("Name") or "")
        if _NORMALIZE_NAME_RE.sub("-", name).lower() == "thuyet-minh-offline-gpu":
            return str(distribution.version)
    try:
        return importlib.metadata.version("thuyet-minh-offline-gpu")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _license(value: str) -> dict[str, dict[str, str]]:
    # Lock manifests use SPDX identifiers. Python package metadata is less
    # predictable, so long/free-form values are represented as names.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,79}", value):
        return {"license": {"id": value}}
    return {"license": {"name": value[:512]}}


def _properties(
    raw: Mapping[str, Any],
    *,
    prefix: str,
    keys: Iterable[str],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        encoded = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else str(value).lower()
            if isinstance(value, bool)
            else str(value)
        )
        result.append({"name": f"{prefix}:{key.replace('_', '-')}", "value": encoded})
    return result


def _required_text(raw: Mapping[str, Any], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SbomError(f"{label} thiếu trường {key}")
    return value.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
