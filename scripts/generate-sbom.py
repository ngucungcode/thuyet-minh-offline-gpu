#!/usr/bin/env python3
"""Generate the release CycloneDX SBOM from the deployed environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dub_server.sbom import write_cyclonedx_sbom


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo CycloneDX SBOM cục bộ")
    parser.add_argument(
        "--models-lock",
        type=Path,
        default=Path("config/models.lock.json"),
    )
    parser.add_argument(
        "--native-lock",
        type=Path,
        default=Path("native/components.lock.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/reports/sbom.cdx.json"),
    )
    args = parser.parse_args()
    document = write_cyclonedx_sbom(
        args.output,
        args.models_lock,
        args.native_lock,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "components": len(document["components"]),
                "serial_number": document["serialNumber"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
