from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from dub_server import __version__
from dub_server.api import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _installer_assignment(name: str) -> str:
    source = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    match = re.search(
        rf'^{re.escape(name)}=(?:"([^"]+)"|([A-Za-z0-9_.$\{{\}}-]+))$',
        source,
        re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1) or match.group(2)


def test_product_versions_are_aligned() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    web = json.loads(
        (PROJECT_ROOT / "web" / "package.json").read_text(encoding="utf-8")
    )
    web_lock = json.loads(
        (PROJECT_ROOT / "web" / "package-lock.json").read_text(encoding="utf-8")
    )
    release_sbom = json.loads(
        (PROJECT_ROOT / "release" / "sbom.cdx.json").read_text(encoding="utf-8")
    )

    assert project["project"]["version"] == __version__
    assert web["version"] == __version__
    assert web_lock["version"] == __version__
    assert web_lock["packages"][""]["version"] == __version__
    assert release_sbom["metadata"]["component"]["version"] == __version__
    sbom_properties = {
        item["name"]: item["value"]
        for item in release_sbom["metadata"]["properties"]
    }
    for property_name, relative_path in (
        ("thuyetminh:models-lock-sha256", "config/models.lock.json"),
        ("thuyetminh:native-lock-sha256", "native/components.lock.json"),
        ("thuyetminh:web-lock-sha256", "web/package-lock.json"),
    ):
        expected_hash = hashlib.sha256(
            (PROJECT_ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert sbom_properties[property_name] == expected_hash
    assert _installer_assignment("INSTALLER_VERSION") == __version__
    assert _installer_assignment("SOURCE_REF") == "v${INSTALLER_VERSION}"
    assert create_app().version == __version__


def test_installer_defaults_to_safe_release_and_explicit_legacy_migration() -> None:
    assert _installer_assignment("MODEL_PROFILE") == "auto"
    assert _installer_assignment("MIGRATE_EXISTING") == "false"
    assert _installer_assignment("UPGRADE_EXISTING") == "false"
    assert _installer_assignment("COMPATIBLE_UPGRADE_FROM") == (
        "0.2.0 0.2.1 0.2.2 0.2.3 0.2.4 0.3.0"
    )
    assert _installer_assignment("ACCEPTANCE_MODE") == "basic"
    assert _installer_assignment("START_STACK") == "true"

    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "exec {prompt_fd}<>/dev/tty" in installer
    assert "--upgrade-existing" in installer
    assert "main() {" in installer
    assert installer.rstrip().endswith('main "$@"')


def test_deployment_templates_use_the_release_user_agent() -> None:
    expected = f"ThuyetMinhOfflineGPU v{__version__}"
    for relative_path in (".env.example", ".env.native.example", "compose.yaml"):
        template = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in template


def test_readme_uses_the_short_release_installer() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "https://github.com/ngucungcode/thuyet-minh-offline-gpu/"
        "releases/latest/download/install.sh | sudo bash"
    ) in readme
    assert "--profile auto --start --yes" not in readme


def test_installer_does_not_hash_profile_models_twice() -> None:
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")

    assert 'native-model.sh" install "${model_id}"' in installer
    assert 'native-model.sh" verify "${model_id}"' not in installer


def test_native_bootstrap_defaults_to_cached_smoke_checks() -> None:
    bootstrap = (PROJECT_ROOT / "scripts" / "native-bootstrap.sh").read_text(
        encoding="utf-8"
    )

    assert 'INSTALL_TEST_MODE="${DUB_INSTALL_TEST_MODE:-smoke}"' in bootstrap
    assert 'runtime_extras="managed-gpu,native"' in bootstrap
    assert 'runtime_extras="${runtime_extras},test"' in bootstrap
    assert 'if [[ "${INSTALL_TEST_MODE}" == full ]]' in bootstrap
    assert ' -m pip check' in bootstrap
    assert 'PIP_CACHE_DIR="${DUB_NATIVE_ROOT}/cache/pip"' in bootstrap
    assert "detected_build_jobs > 16" in bootstrap


def test_docker_build_cache_excludes_local_build_artifacts() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    locked_requirements = [
        line.strip()
        for line in (
            PROJECT_ROOT / "requirements" / "docker-gpu.lock"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_requirements = [
        *project["dependencies"],
        *(
            requirement
            for requirement in project["optional-dependencies"]["gpu"]
            if not requirement.startswith("torch==")
        ),
    ]

    assert "--mount=type=cache,target=/root/.cache/pip,sharing=locked" in dockerfile
    assert "PIP_NO_CACHE_DIR" not in dockerfile
    assert locked_requirements == expected_requirements
    assert dockerfile.index("COPY requirements/docker-gpu.lock") < dockerfile.index(
        "COPY src ./src"
    )
    assert dockerfile.index("--requirement requirements/docker-gpu.lock") < (
        dockerfile.index("COPY src ./src")
    )
    assert "pip install --break-system-packages --no-deps ." in dockerfile
    for ignored_path in (
        "**/node_modules/",
        ".artifacts/",
        ".pytest-tmp-*/",
        ".review-tmp-*/",
        "web/.next/",
        "web/.vinext/",
        "web/dist/",
    ):
        assert ignored_path in dockerignore
