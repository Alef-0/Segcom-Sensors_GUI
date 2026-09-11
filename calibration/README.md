# QR camera timing calibration

This folder contains the QR display, recording inspection, and quantitative
analysis used to estimate the camera timing correction. The measured target is
the host-anchored camera PTS minus the newest decoded QR marker matched to the
display journal.

That target is a software timing reference. It is not a direct measurement of
physical exposure time, and it does not by itself prove camera-to-radar
alignment.

## Components

- `display_qt.py` creates one persistent Qt/OpenGL window. It advances
  timestamped QR codes through a selectable 4, 6, 8, 9, 10, or 12-cell grid at
  the selected monitor's refresh rate, keeps the selected number of recent QR
  codes visible,
  underlines the newest code, and records swap timing in
  `display_timestamps.jsonl`. Press `P` to
  pause or resume; `Q`, Escape, and Ctrl+C close it.
- `display.py` provides the selectable Pygame/SDL clock with the same grid,
  visible-code, snake-order, newest-marker, monitor-index, and journal behavior.
  It retains Pygame's paced presentation loop and supports `P`, `Q`, and Escape.
- `qr.py` creates the 12-digit monotonic-millisecond QR payloads and provides
  QReader decoding, per-cell retries, and snake-grid ordering.
- `recording_display.py` opens the recording inspection window. It decodes the
  undistorted frames, allows PTS, NTP, and QR values to be corrected, validates
  every readable value against the display journal, selects the latest valid
  readable QR, and reports camera-grid position disagreements without rejecting
  an otherwise unambiguous journal match. Missing or unreadable QR codes do not
  reject a frame when another valid QR is available. The default undistortion
  alpha is `0.25`. Full-folder scans use two isolated frame workers, each with
  its own CUDA-capable QReader model; within each frame, recovery crops are sent
  through QRDet in groups of four. Every configured cell is covered by the
  full-frame pass or a missing-cell retry. The image cache is bounded to eight
  frames so long recordings do not retain every original and undistorted image
  in memory.
- `quantitative_analysis.py` compares correction strategies on clean evidence,
  writes the verdict, and creates all graphs with Matplotlib. PNG is the default;
  vector SVG copies are optional.
- `intrinsics.json` contains the default camera matrix, distortion coefficients,
  and calibration image size used for undistortion.
- `../analyze_calibration_recording.py` is the normal entry point. It opens the
  recording window first and runs the quantitative analyzer after the window
  has created its two source files and closed.

## Recording and analysis workflow

1. Start the QR calibration from the application. The recording begins after
   the configured three-second delay. Keep the calibration display visible for
   the complete recording, then close it so its journal is flushed.
2. Open **Visualization** for the recording, or start the root analyzer from a
   terminal.
3. Review the initially decoded frame. Correct editable PTS, NTP, or QR values
   only when the recording visibly supports the correction.
4. Select **GO — DECODE FULL FOLDER**. Each frame is shown as it finishes
   decoding. Frames without any valid readable QR are skipped; other readable
   values remain usable even when several grid cells are unreadable. When the
   scan finishes, it automatically creates
   `calibration_analysis.json` and `calibration_frames.csv` in a sibling folder
   named `<recording>_analysis`.
5. Close the inspection window. The root analyzer then creates the quantitative
   verdict and graphs from those saved files.

Run the complete workflow with:

```bash
python3 analyze_calibration_recording.py /path/to/recording
```

Use another intrinsic calibration when needed:

```bash
python3 analyze_calibration_recording.py /path/to/recording \
  --intrinsics /path/to/intrinsics.json
```

To regenerate only the verdict and graphs from existing analysis files:

```bash
python3 -m calibration.quantitative_analysis /path/to/recording_analysis
```

Add `--svg` when running this standalone command to also create vector copies:

```bash
python3 -m calibration.quantitative_analysis \
  /path/to/recording_analysis --svg
```

The application and `analyze_calibration_recording.py` intentionally use the
PNG-only default.

The QR display can also be started directly:

```bash
# Qt/OpenGL
python3 -m calibration.display_qt --list-screens
python3 -m calibration.display_qt --screen 1
python3 -m calibration.display_qt --screen 1 --grid-qrs 12 --visible-qrs 8
python3 -m calibration.display_qt --screen 1 --windowed --width 1280 --height 720

# Pygame/SDL
python3 -m calibration.display --screen 1
python3 -m calibration.display --screen 1 --grid-qrs 12 --visible-qrs 8
python3 -m calibration.display --screen 1 --windowed --width 1280 --height 720
```

## Required recording inputs

The recording folder must contain:

- captured image files referenced by the camera journal;
- `camera_timestamps.jsonl` or `camera_timestamps.json`;
- `display_timestamps.jsonl` from the matching QR display session.

`camera_timing_session.json` is optional, but it is needed when camera PTS must
be converted to the host monotonic clock through a recorded stream epoch.

The analyzer reads source images without modifying them. Undistortion is applied
only to the in-memory image used for inspection and decoding.

## Quantitative strategies

Only rows marked `accepted_clean` with `Clean` replacement timing and a finite
offset are used. The first 70% of clean frames is the training portion; the
later 30% is a chronological holdout that is not used to fit parameters.

The report compares five strategies:

- **Current fixed 109 ms** uses the existing correction without fitting.
- **Calibrated fixed median** uses the training median. A median minimizes total
  absolute error and is resistant to occasional large offsets.
- **Calibrated fixed mean** uses the training mean. A mean minimizes squared
  error but is more sensitive to outliers.
- **PTS-step median** groups the current camera PTS interval into 5 ms buckets
  and uses the training median for the matching bucket.
- **Six-step PTS history** is a least-squares linear regression using the current
  PTS interval and the five previous intervals. It does not use QR values,
  future frames, display index, or elapsed recording time as predictors.

For each frame, the signed residual is:

```text
observed QR-derived offset - predicted correction
```

The **absolute residual** removes its direction:

```text
abs(observed QR-derived offset - predicted correction)
```

For example, residuals of `+6 ms` and `-6 ms` both have an absolute residual of
`6 ms`. MAE is the average absolute residual. The P95 absolute residual is the
value met or improved upon by 95% of evaluated frames, so it exposes uncommon
large errors that an average can hide.

## Generated files

The inspection window creates:

- `calibration_analysis.json` — complete per-frame evidence and scan summary;
- `calibration_frames.csv` — the same per-frame evidence in tabular form.

The quantitative analyzer then creates:

- `calibration_verdict.md` — readable recommendation and strategy comparison;
- `calibration_verdict.json` — metrics, fitted parameters, provenance hashes,
  histogram policy, and machine-readable verdict;
- `calibration_strategy_predictions.csv` — observed offsets, split labels, and
  every strategy prediction for each clean frame;
- `calibration_offset_timeline.png` — clean and excluded offsets over frame
  order;
- `calibration_residual_cdf.png` — absolute-residual distributions on the
  chronological holdout;
- `calibration_offset_histogram.png` — the clean observed-offset distribution;
- `calibration_fixed_residual_histogram.png` — one panel per fixed strategy;
- `calibration_pts_residual_histogram.png` — one panel per PTS-based strategy.

With `--svg`, the quantitative analyzer also creates an `.svg` copy beside each
PNG. Existing SVG files from an earlier run are not deleted when the analyzer is
later run without the flag.

Histogram panels never overlap strategies. Each panel uses the full range of
its own data and Freedman-Diaconis bin spacing constrained to 8–18 succinct
bins. Residual ranges are symmetric around zero.

## Interpreting the verdict safely

- Exclude timing-suspect rows and rows whose QR removal timing is unknown. A QR
  is removed after the configured number of newer visible codes; missing that
  replacement evidence is not a clean transition.
- A recommended correction replaces the existing `109 ms` subtraction; it is
  never added to it.
- The chronological holdout tests a later portion of the same recording. It is
  useful for comparison, but it is not independent session validation.
- Do not enable a learned dynamic correction from one recording. Preselect the
  strategy and confirm it on a later, independently recorded calibration first.
- Keep the software-marker result separate from claims about physical exposure
  timing, RTSP transport delay, or radar alignment.

## Dependencies and checks

The workflow uses OpenCV, Pillow, Pygame, NumPy, `qrcode`, `qreader`, and
Matplotlib. The quantitative entry point checks for an installed NumPy and
Matplotlib pair that can import together before rendering graphs.

Run the focused non-visual checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_qr_calibration.py tests/test_calibration_workflow.py
```

These checks validate data handling and orchestration. Confirm the real display,
camera framing, editable controls, and generated graph readability manually on
the target system.
