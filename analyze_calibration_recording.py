#!/usr/bin/env python3
"""Inspect QR evidence; optionally evaluate a previously frozen anchor offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from calibration.recording_display import DEFAULT_INTRINSICS, run_recording_display


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_anchor_analysis(output: Path, offset_file: Path) -> dict:
    destination = output / "pts_anchor_analysis"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "analyze_pts_anchor.py"),
            "evaluate",
            str(output),
            "--offset-file",
            str(offset_file.resolve()),
            "--output-directory",
            str(destination),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return json.loads(
        (destination / "pts_anchor_analysis.json").read_text(encoding="utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Calibration recording folder")
    parser.add_argument("--offset-file", type=Path,
                        help="Previously calibrated offset from independent laboratory streams")
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
            "Anchor evaluation was not rerun: this window session did not complete "
            "a fresh decode. Saved results can be reviewed without rerunning analysis.",
            flush=True,
        )
        return

    if arguments.offset_file is None:
        print(f"QR evidence saved in {output}. Use analyze_pts_anchor.py calibrate for laboratory "
              "data, or evaluate --offset-file for an independent test stream.", flush=True)
        return
    report = _run_anchor_analysis(output, arguments.offset_file)
    print(
        f"Anchor evaluation saved in {report['output_directory']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
