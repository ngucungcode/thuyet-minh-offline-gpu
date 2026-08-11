from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from dub_server import __version__
from dub_server.api import create_app
from dub_server.gpu import (
    CUDA_TOOLKIT_MINIMUM_DRIVERS,
    SUPPORTED_CUDA_ARCHITECTURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CUDA_ARCHITECTURES = frozenset({70, 75, 80, 86, 89, 90})
EXPECTED_CUDA_TOOLKIT_VERSIONS = ("12.6", "12.8")
EXPECTED_CUDA_TOOLKIT_MINIMUM_DRIVERS = {
    "12.6": (560, 28, 3),
    "12.8": (570, 26),
}


def _installer_assignment(name: str) -> str:
    source = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    match = re.search(
        rf'^{re.escape(name)}=(?:"([^"]+)"|([A-Za-z0-9_.$\{{\}}-]+))$',
        source,
        re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1) or match.group(2)


def _cmake_cuda_architecture_tokens(source: str) -> tuple[str, ...]:
    match = re.search(
        r'-DCMAKE_CUDA_ARCHITECTURES=(?:"([^"]+)"|\'([^\']+)\'|([^\s\\]+))',
        source,
    )
    assert match is not None
    value = next(group for group in match.groups() if group is not None)
    variable = re.fullmatch(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
    if variable is not None:
        default = re.search(
            rf'^ARG {re.escape(variable.group(1))}=(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))$',
            source,
            re.MULTILINE,
        )
        assert default is not None, variable.group(1)
        value = next(group for group in default.groups() if group is not None)
    return tuple(value.split(";"))


def _cmake_cuda_architectures(source: str) -> tuple[int, ...]:
    architectures: list[int] = []
    for item in _cmake_cuda_architecture_tokens(source):
        token = re.fullmatch(r"([0-9]+)(?:-(?:real|virtual))?", item)
        assert token is not None, item
        architectures.append(int(token.group(1)))
    return tuple(architectures)


def _shell_case_architectures(source: str, variable: str) -> frozenset[int]:
    match = re.search(
        rf'case "\$\{{{re.escape(variable)}\}}" in\s*'
        r'([0-9]+(?:\|[0-9]+)*)\)',
        source,
    )
    assert match is not None, variable
    return frozenset(int(value) for value in match.group(1).split("|"))


def _model_profile_case_for_new_native_env(installer: str) -> str:
    prefix = installer[
        : installer.index('source "${PROJECT_ROOT}/scripts/native-common.sh"')
    ]
    expected_models = {
        "maximum": (
            "asr-faster-whisper-large-v3-turbo",
            "mt-gemma4-31b-q4",
            "tts-vieneu-v2",
        ),
        "balanced": (
            "asr-faster-whisper-small",
            "mt-gemma4-e2b-q4",
            "tts-vieneu-v2",
        ),
        "minimal": (
            "asr-faster-whisper-small",
            "mt-gemma4-e2b-q4",
            "tts-piper-vi-vais1000-medium",
        ),
    }
    for match in re.finditer(
        r'^[ \t]*case "\$\{MODEL_PROFILE\}" in[ \t]*\n(.*?)^[ \t]*esac[ \t]*$',
        prefix,
        re.MULTILINE | re.DOTALL,
    ):
        body = match.group(1)
        if all(
            all(model_id in body for model_id in model_ids)
            for model_ids in expected_models.values()
        ):
            for profile, model_ids in expected_models.items():
                branch = re.search(
                    rf'^\s*{profile}\)\s*(.*?)(?=^\s*(?:maximum|balanced|minimal|none)\)|\Z)',
                    body,
                    re.MULTILINE | re.DOTALL,
                )
                assert branch is not None, profile
                for model_id in model_ids:
                    assert model_id in branch.group(1), (profile, model_id)
            return body
    raise AssertionError("MODEL_PROFILE does not select native runtime defaults")


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
        "0.2.0 0.2.1 0.2.2 0.2.3 0.2.4 0.3.0 0.3.1 0.3.2 0.3.3"
        " 0.3.4 0.3.5"
    )
    assert _installer_assignment("ACCEPTANCE_MODE") == "basic"
    assert _installer_assignment("START_STACK") == "true"

    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "exec {prompt_fd}<>/dev/tty" in installer
    assert "--upgrade-existing" in installer
    assert "main() {" in installer
    assert installer.rstrip().endswith(
        'if [[ "${BASH_SOURCE[0]:-${0}}" == "${0}" ]]; then\n'
        '  main "$@"\n'
        "fi"
    )


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


def test_cuda_architecture_metadata_and_implementations_do_not_drift() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    native_lock = json.loads(
        (PROJECT_ROOT / "native" / "components.lock.json").read_text(
            encoding="utf-8"
        )
    )
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    llama_metadata = native_lock["components"]["llama_cpp"]

    locked = frozenset(llama_metadata["cuda_supported_architectures"])
    docker = frozenset(_cmake_cuda_architectures(dockerfile))
    docker_tokens = _cmake_cuda_architecture_tokens(dockerfile)
    gpu_preflight = frozenset(
        major * 10 + minor for major, minor in SUPPORTED_CUDA_ARCHITECTURES
    )
    native_installer = _shell_case_architectures(installer, "cuda_arch")

    assert locked == EXPECTED_CUDA_ARCHITECTURES
    assert docker == locked
    assert {
        int(value.removesuffix("-real"))
        for value in docker_tokens
        if value.endswith("-real")
    } == locked
    assert {
        int(value.removesuffix("-virtual"))
        for value in docker_tokens
        if value.endswith("-virtual")
    } == {90}
    assert gpu_preflight == locked
    assert native_installer == locked
    assert llama_metadata["cuda_default_build_architecture"] == 86
    assert llama_metadata["cuda_default_build_architecture"] in locked
    assert tuple(llama_metadata["cuda_supported_versions"]) == (
        EXPECTED_CUDA_TOOLKIT_VERSIONS
    )
    assert tuple(CUDA_TOOLKIT_MINIMUM_DRIVERS) == EXPECTED_CUDA_TOOLKIT_VERSIONS
    assert CUDA_TOOLKIT_MINIMUM_DRIVERS == EXPECTED_CUDA_TOOLKIT_MINIMUM_DRIVERS
    assert llama_metadata["cuda_version"] in EXPECTED_CUDA_TOOLKIT_VERSIONS
    assert "nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04@sha256:" in dockerfile
    assert "cuda_architectures" not in llama_metadata


def test_native_llama_build_can_be_overridden_and_is_keyed_by_architecture() -> None:
    installer = (PROJECT_ROOT / "scripts" / "install-llama-cpp.sh").read_text(
        encoding="utf-8"
    )

    assert ".components.llama_cpp.cuda_version" in installer
    assert ".components.llama_cpp.cuda_supported_versions" in installer
    assert ".components.llama_cpp.cuda_supported_architectures" in installer
    assert ".components.llama_cpp.cuda_default_build_architecture" in installer
    override_assignment = next(
        (
            line
            for line in installer.splitlines()
            if re.match(r'^LLAMA_CUDA_ARCH(?:ITECTURES)?="', line)
            and "DUB_LLAMA_CUDA_ARCHITECTURES" in line
        ),
        None,
    )
    assert override_assignment is not None
    assert ":-" in override_assignment
    internal_architecture_variable = override_assignment.split("=", 1)[0]
    assert (
        f'-DCMAKE_CUDA_ARCHITECTURES="${{{internal_architecture_variable}}}"'
        in installer
    )
    assert 'grep -Fxq -- "${architecture}"' in installer
    assert "LOCKED_LLAMA_CUDA_ARCHITECTURES" in installer
    assert "LOCKED_LLAMA_CUDA_VERSIONS" in installer
    assert 'LLAMA_CUDA_VERSION="${ACTUAL_NVCC_RELEASE}"' in installer

    label_assignment = re.search(
        rf'^([A-Z][A-Z0-9_]*ARCH[A-Z0-9_]*)="\$\{{{internal_architecture_variable}//;/_\}}"$',
        installer,
        re.MULTILINE,
    )
    assert label_assignment is not None
    label_name = label_assignment.group(1)
    assert (
        'LLAMA_TARGET="/usr/local/lib/llama.cpp-${LLAMA_RELEASE}'
        f'-cuda${{LLAMA_CUDA_VERSION}}-sm${{{label_name}}}-offline"'
        in installer
    )
    assert '--arg cuda_architectures "${LLAMA_CUDA_ARCHITECTURES}"' in installer
    assert ".cuda_architectures == $cuda_architectures" in installer
    assert '[[ -e "${LLAMA_TARGET}" || -L "${LLAMA_TARGET}" ]]' in installer
    assert 'mv -T -- "${STAGE_DIR}" "${LLAMA_TARGET}"' in installer
    assert 'ln -sfnT "${LLAMA_TARGET}" "${LLAMA_LINK}"' in installer


def test_migration_reads_native_lock_from_the_supplied_project_root() -> None:
    migration = (PROJECT_ROOT / "installer" / "migrate-legacy.sh").read_text(
        encoding="utf-8"
    )

    assert 'python3 - native/components.lock.json <<\'PY\'' in migration
    assert 'with Path(sys.argv[1]).open(encoding="utf-8") as handle:' in migration
    assert 'with Path("native/components.lock.json").open' not in migration


def test_installer_passes_selected_arch_and_writes_new_profile_defaults() -> None:
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")

    assert re.search(
        r'DUB_LLAMA_CUDA_ARCHITECTURES="\$\{cuda_arch\}"\s*\\?\s*'
        r'"\$\{PROJECT_ROOT\}/scripts/native-bootstrap\.sh"',
        installer,
    )
    assert installer.count('"sm_${cuda_arch}"') >= 2
    _model_profile_case_for_new_native_env(installer)

    env_start = installer.index('ENV_FILE="${PROJECT_ROOT}/.env.native"')
    env_end = installer.index('source "${PROJECT_ROOT}/scripts/native-common.sh"')
    new_env_contract = installer[env_start:env_end]
    for variable in (
        "DUB_DEFAULT_ASR_MODEL_ID",
        "DUB_DEFAULT_TRANSLATION_MODEL_ID",
        "DUB_DEFAULT_TTS_MODEL_ID",
        "CUDA_VISIBLE_DEVICES",
        "DUB_SELECTED_GPU_UUID",
        "DUB_SELECTED_CUDA_ARCHITECTURE",
        "DUB_SELECTED_CUDA_TOOLKIT_VERSION",
    ):
        assert variable in new_env_contract
    assert "migrate_legacy_model_default" in new_env_contract
    assert "Giữ cấu hình model tùy chỉnh" in new_env_contract


def test_native_installer_preflights_driver_architecture_and_cmp_profile() -> None:
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")

    assert "--query-gpu=driver_version" in installer
    assert "torch.cuda.get_device_properties(0)" in installer
    assert 'raw_device_uuid = getattr(properties, "uuid", None)' in installer
    assert 'device_uuid.casefold().startswith("gpu-")' in installer
    assert 'device_uuid = f"GPU-{device_uuid[4:]}"' in installer
    assert 'device_uuid = f"GPU-{device_uuid}"' in installer
    assert "hỗ trợ CUDA toolkit 12.6 hoặc 12.8" in installer
    assert re.search(
        r'12\.6\)\s*minimum_driver_major=560\s*'
        r'minimum_driver_minor=28\s*'
        r'minimum_driver_patch=3\s*'
        r'minimum_driver_version="560\.28\.03"',
        installer,
    )
    assert re.search(
        r'12\.8\)\s*minimum_driver_major=570\s*'
        r'minimum_driver_minor=26\s*'
        r'minimum_driver_patch=0\s*'
        r'minimum_driver_version="570\.26"',
        installer,
    )
    assert "cuda_toolkit_version=%s" in installer
    assert "gpu_cuda_toolkit_version" in installer
    assert "CUDACXX=/usr/local/cuda/bin/nvcc" in installer
    assert "--native-receipt /usr/local/lib/llama.cpp/build-receipt.json" in installer
    assert _shell_case_architectures(installer, "cuda_arch") == (
        EXPECTED_CUDA_ARCHITECTURES
    )
    assert "torch.cuda.get_device_capability(0)" in installer
    assert "torch.float16" in installer
    assert "left @ left" in installer
    assert 'gpu_support_tier="experimental"' in installer
    assert re.search(
        r'if \[\[ "\$\{gpu_support_tier\}" == "experimental" \]\]; then\s*'
        r'MODEL_PROFILE="minimal"',
        installer,
    )
    assert "CMP 170HX hiện chỉ hỗ trợ profile minimal hoặc none" in installer
    assert "torch.cuda.get_device_properties(0)" in installer
    assert "PyTorch không trả về UUID GPU logical 0 hợp lệ" in installer
    assert "--gpu-device INDEX|UUID" in installer
    assert 'export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"' in installer
    assert installer.index('existing_native_env="${INSTALL_DIR}/.env.native"') < (
        installer.index('torch_gpu_report="$(python3')
    )
    assert 'persisted_gpu_uuid="$(sed -n' in installer
    assert 'export CUDA_VISIBLE_DEVICES="${persisted_gpu_device}"' in installer


def test_docker_selects_one_logical_gpu_and_uses_minimal_safe_defaults() -> None:
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    expected_defaults = {
        "DUB_DEFAULT_ASR_MODEL_ID": "asr-faster-whisper-small",
        "DUB_DEFAULT_TRANSLATION_MODEL_ID": "mt-gemma4-e2b-q4",
        "DUB_DEFAULT_SEPARATION_MODEL_ID": "separation-tiger-dnr",
        "DUB_DEFAULT_TTS_MODEL_ID": "tts-piper-vi-vais1000-medium",
    }

    assert 'device_ids: ["${DUB_GPU_DEVICE_ID:-0}"]' in compose
    assert "count: 1" not in compose
    assert "NVIDIA_VISIBLE_DEVICES" not in compose
    assert "DUB_GPU_DEVICE_ID=0" in example
    for key, value in expected_defaults.items():
        assert f"{key}={value}" in dockerfile
        assert f"{key}: ${{{key}:-{value}}}" in compose
        assert f"{key}={value}" in example


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
        *project["optional-dependencies"]["gpu"],
    ]
    release_python_requirements = [
        line.strip()
        for line in (
            PROJECT_ROOT / "requirements" / "release-python.lock"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_release_requirements = [
        *project["dependencies"],
        *project["optional-dependencies"]["gpu"],
        *project["optional-dependencies"]["native"],
    ]

    assert "--mount=type=cache,target=/root/.cache/pip,sharing=locked" in dockerfile
    assert "PIP_NO_CACHE_DIR" not in dockerfile
    assert locked_requirements == expected_requirements
    assert release_python_requirements == expected_release_requirements
    assert dockerfile.index("COPY requirements/docker-gpu.lock") < dockerfile.index(
        "COPY src ./src"
    )
    assert dockerfile.index("--requirement requirements/docker-gpu.lock") < (
        dockerfile.index("COPY src ./src")
    )
    assert "pip install --break-system-packages --no-deps ." in dockerfile
    assert 'assert "sm_70" in flags' in dockerfile
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
