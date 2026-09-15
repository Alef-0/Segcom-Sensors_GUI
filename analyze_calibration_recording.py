#!/usr/bin/env python3
"""Inspect a QR calibration recording, then create its final analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from calibration.recording_display import DEFAULT_INTRINSICS, run_recording_display


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_final_analysis(output: Path) -> dict:
    destination = output / "final_analysis"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "calibration.final_analysis",
            str(output),
            "--output-directory",
            str(destination),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return json.loads(
        (destination / "final_analysis.json").read_text(encoding="utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Calibration recording folder")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS,
        help=f"Camera intrinsics JSON (default: {DEFAULT_INTRINSICS})",
    )
    arguments = parser.parse_args()

    output = run_recording_display(arguments.folder, arguments.intrinsics)
    if output is None:
        print(
            "Final analysis skipped because this window session did not finish "
            "creating analysis files.",
            flush=True,
        )
        return

    report = _run_final_analysis(output)
    print(
        f"Final analysis saved in {report['output_directory']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
