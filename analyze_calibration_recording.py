#!/usr/bin/env python3
"""Inspect a QR calibration recording, then create its timing reports and graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from calibration.quantitative_analysis import matplotlib_environment
from calibration.recording_display import DEFAULT_INTRINSICS, run_recording_display


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_distance_analysis(recording: Path, output: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "calibration.distance_analysis",
            str(recording),
            "--analysis-directory",
            str(output),
            "--output-directory",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=matplotlib_environment(),
        check=True,
    )
    return json.loads((output / "distance_analysis.json").read_text(encoding="utf-8"))


def _run_quantitative_analysis(output: Path) -> dict:
    subprocess.run(
        [sys.executable, "-m", "calibration.quantitative_analysis", str(output)],
        cwd=PROJECT_ROOT,
        env=matplotlib_environment(),
        check=True,
    )
    return json.loads((output / "calibration_verdict.json").read_text(encoding="utf-8"))


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
            "Distance analysis skipped because this window session did not finish "
            "creating analysis files.",
            flush=True,
        )
        return

    report = _run_distance_analysis(arguments.folder, output)
    verdict = _run_quantitative_analysis(output)
    print(
        f"Distance analysis saved in {report['output_directory']}",
        flush=True,
    )
    print(
        f"Quantitative verdict saved in {verdict['output_directory']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
