from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from test_sbom import _write_locks


def test_script_resolves_web_lock_outside_project_cwd(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    models, native = _write_locks(tmp_path)
    output = tmp_path / "report" / "sbom.cdx.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "generate-sbom.py"),
            "--models-lock",
            str(models),
            "--native-lock",
            str(native),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    refs = {component["bom-ref"] for component in document["components"]}
    assert "npm:next@16.2.12" in refs
    assert "npm:react@19.2.6" in refs
    assert "npm:vinext@0.0.50" in refs
