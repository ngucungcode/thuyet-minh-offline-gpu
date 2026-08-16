from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ROOT = PROJECT_ROOT / "windows"


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_windows_native_entrypoints_are_present_and_do_not_evaluate_env_code() -> None:
    scripts = {
        name: _read(f"windows/{name}")
        for name in ("common.ps1", "preflight.ps1", "install.ps1", "stack.ps1", "dub.ps1")
    }

    for name, source in scripts.items():
        assert "Set-StrictMode" in source, name
        assert "Invoke-Expression" not in source, name
        assert "DownloadString" not in source, name
    common = scripts["common.ps1"]
    assert "^[A-Za-z_][A-Za-z0-9_]*$" in common
    assert "[Environment]::SetEnvironmentVariable" in common
    assert "[IO.Path]::IsPathRooted" in common
    assert '"127.0.0.1"' in common
    assert '"http://127.0.0.1:8080"' in common
    assert '"Lib\\site-packages\\torch\\lib"' in common
    assert 'DUB_API_HOST -ne "127.0.0.1"' in common
    assert "DUB_API_PORT phai nam trong khoang 1..65535" in common


def test_windows_installer_is_locked_gpu_specific_and_fail_closed() -> None:
    installer = _read("windows/install.ps1")
    manifest = json.loads(_read("native/components.lock.json"))

    assert r"native\components.lock.json" in installer
    assert "Sync-LockedRepository" in installer
    assert '"fetch" "--depth" "1" "origin" $Commit' in installer
    assert "Get-FileHash -Algorithm SHA256" in installer
    assert "cuda_supported_architectures" in installer
    assert '$env:CUDACXX = Require-Command "nvcc.exe"' in installer
    assert "nvcc.exe --list-gpu-arch" in installer
    assert '"-DCMAKE_CUDA_ARCHITECTURES=" + $Gpu.Architecture' in installer
    assert '"12.8" = "570.65"' in installer
    assert '$gpu.Capability -eq "12.0"' in installer
    assert '"torch==2.8.0"' in installer
    assert '"https://download.pytorch.org/whl/cu128"' in installer
    assert "torch.cuda.get_arch_list()" in installer
    assert "x@x" in installer
    assert set(manifest["components"]["llama_cpp"]["cuda_supported_architectures"]) >= {
        75,
        86,
        89,
        120,
    }


def test_windows_stack_validates_process_identity_and_stays_on_loopback() -> None:
    stack = _read("windows/stack.ps1")

    assert "Get-ValidatedProcess" in stack
    assert "$process.Path" in stack
    assert "StartTime.ToUniversalTime().Ticks" in stack
    assert "taskkill.exe /PID $process.Id /T /F" in stack
    assert '"-m", "dub_server.worker"' in stack
    assert '"-m", "uvicorn", "dub_server.api:app"' in stack
    assert "DUB_API_HOST" in stack
    assert "Prowlarr" not in stack
    assert "qBittorrent" not in stack


def test_windows_defaults_and_documentation_match_local_upload_scope() -> None:
    example = _read(".env.windows.example")
    docs = _read("docs/windows-native.md")
    gitignore = _read(".gitignore")
    gitattributes = _read(".gitattributes")

    assert "DUB_API_HOST=127.0.0.1" in example
    assert "DUB_API_URL=http://127.0.0.1:8080" in example
    assert "DUB_DEFAULT_TTS_MODEL_ID=tts-piper-vi-vais1000-medium" in example
    assert "!.env.windows.example" in gitignore
    assert ".venv-windows/" in gitignore
    assert "*.ps1 text eol=crlf" in gitattributes
    for expected in (
        "Windows 10 22H2",
        "local-upload",
        "RTX 20",
        "RTX 30",
        "RTX 40",
        "RTX 50",
        ".\\windows\\install.ps1",
        ".\\windows\\stack.ps1 start",
    ):
        assert expected in docs


def test_windows_runtime_uses_executable_suffixes() -> None:
    gpu = _read("src/dub_server/gpu.py")
    phase4 = _read("src/dub_server/phase4_stage.py")

    assert '"bin" / "nvcc.exe"' in gpu
    assert '"piper.exe" if os.name == "nt" else "piper"' in phase4
