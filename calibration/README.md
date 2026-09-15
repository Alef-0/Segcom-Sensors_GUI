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
  codes visible, retains unchanged cells while repainting only changed regions,
  predicts the next swap timestamp for the new QR, underlines the newest code,
  keeps each cached QR as an unscaled module pixmap that is enlarged with
  nearest-neighbor drawing, and records both predicted and actual swap timing in
  `display_timestamps.jsonl`. Press `P` to
  pause or resume; `Q`, Escape, and Ctrl+C close it.
- `display.py` provides the selectable Pygame/SDL clock with the same grid,
  visible-code, snake-order, newest-marker, monitor-index, predicted-presentation
  timestamp, and journal behavior. It updates only changed cells, draws each QR
  through one scaled surface by default, prepares the next predicted QR during
  pacing slack, retains Pygame's paced presentation loop, and supports `P`, `Q`,
  and Escape. Its final busy-wait duration is configurable when reduced CPU use
  is more important than minimum scheduling jitter.
- Both display loops disable Python's automatic cyclic garbage collector only
  while presenting frames and never force a collection. Pygame updates its QR
  rectangle and reusable QR surfaces in place; Qt retains only the configured
  visible QR matrices and pixmaps. The display journal streams frame rows to
  disk and keeps only running summary counters in memory.
- `qr.py` creates the 12-digit monotonic-millisecond QR payloads and provides
  QReader decoding, per-cell retries, and snake-grid ordering. Both display
  backends choose the lowest-penalty QR mask automatically by default and expose
  an experimental fixed-mask option for timing and camera-readability tests.
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
  in memory. It also reconstructs the software presentation interval in which
  each decoded newest QR was active, lists intervening display events between
  camera frames, and classifies stable, boundary-transition, stale, and future
  observations. Frame-arrival intervals remain separate transport diagnostics.
- `quantitative_analysis.py` compares correction strategies on clean evidence,
  writes the verdict, and creates all graphs with Matplotlib. PNG is the default;
  vector SVG copies are optional.
- `distance_analysis.py` maps every decoded QR to its predicted and observed
  software flip, fits an interval-aware host-arrival model, and estimates an
  observation timestamp for frames with and without readable QR anchors.
- `final_analysis.py` is the current integrated estimator evaluation. It uses
  raw journals, segment-mapped media time, strict maximum/median interval goals,
  explicit stream holdouts, and causal model inputs.
- `intrinsics.json` contains the default camera matrix, distortion coefficients,
  and calibration image size used for undistortion.
- `../analyze_calibration_recording.py` is the normal entry point. It opens the
  recording window first and runs only the final analyzer after the window has
  created its saved QR evidence and closed.

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
   `calibration_analysis.json`, `calibration_frames.csv`, and
   `display_presentations.csv` in a sibling folder named
   `<recording>_analysis`.
5. Close the inspection window. The root analyzer then creates the final
   interval-constrained report and per-model graphs from those saved files.

Run the complete workflow with:

```bash
python3 analyze_calibration_recording.py /path/to/recording
```

Use another intrinsic calibration when needed:

```bash
python3 analyze_calibration_recording.py /path/to/recording \
  --intrinsics /path/to/intrinsics.json
```

To regenerate final analysis from explicit existing QR analysis folders:

```bash
python3 -m calibration.final_analysis /path/to/recording_analysis \
  --output-directory /path/to/final_analysis
```

The older distance analysis can still be run explicitly:

```bash
python3 -m calibration.distance_analysis /path/to/recording
```

The separate quantitative strategy analyzer can still be run explicitly:

```bash
python3 -m calibration.quantitative_analysis /path/to/recording_analysis
```

The QR display can also be started directly:

```bash
# Qt/OpenGL
python3 -m calibration.display_qt --list-screens
python3 -m calibration.display_qt --screen 1
python3 -m calibration.display_qt --screen 1 --grid-qrs 12 --visible-qrs 8
python3 -m calibration.display_qt --screen 1 --windowed --width 1280 --height 720
python3 -m calibration.display_qt --screen 1 --timestamp-mode paint-start
python3 -m calibration.display_qt --screen 1 --qr-mask-pattern 3

# Pygame/SDL
python3 -m calibration.display --screen 1
python3 -m calibration.display --screen 1 --grid-qrs 12 --visible-qrs 8
python3 -m calibration.display --screen 1 --windowed --width 1280 --height 720
python3 -m calibration.display --screen 1 --timestamp-mode paint-start
python3 -m calibration.display --screen 1 --qr-mask-pattern 3
python3 -m calibration.display --screen 1 --spin-wait-us 250
```

`--qr-mask-pattern` accepts `0` through `7`. A fixed mask avoids the automatic
eight-mask scoring pass, but it must be compared with automatic selection using
real calibration recordings before it becomes a recording default. The selected
mask policy is written to the display journal. Pygame's default
`--spin-wait-us 1000` preserves the established timing behavior; reducing it can
save CPU at the cost of additional wake-up jitter. Use `monitor_hz_tests.py` to
compare these settings on an explicitly selected monitor.

## Required recording inputs

The recording folder must contain:

- captured image files referenced by the camera journal;
- `camera_timestamps.jsonl` or `camera_timestamps.json`;
- `display_timestamps.jsonl` from the matching QR display session.

`camera_timing_session.json` is optional, but it is needed when camera PTS must
be converted to the host monotonic clock through a recorded stream epoch.

The analyzer reads source images without modifying them. Undistortion is applied
only to the in-memory image used for inspection and decoding.

## Final analysis strategies

`analyze_calibration_recording.py` now launches only
`calibration.final_analysis` after the inspection window. The final analyzer
reconstructs the media reference from saved `media_monotonic_ns` or from the
recorded pipeline-zero anchor plus segment-mapped `running_time_ns`; raw PTS is
used only for legacy recordings that lack running-time data.

The primary model sequence is deliberately simple:

- **Model A: constant interval correction** fits one correction against the
  permissible QR intervals.
- **Model B: cadence state** uses only causal mapped-running-time gaps. Arrival
  state remains separate: `q=A-M`, and estimated arrival delay is `q+c`.
- **Model C: regularized interval history** uses causal mapped-running-time
  history and minimizes distance outside the training intervals. It advances
  beyond simpler models only when their inner chronological checks fail.

Midpoint and nearest-neighbor strategies remain labeled comparison baselines.
QR identities, cells, display indices, recording identity, future frames, and
absolute recording time are not runtime predictors. Application-arrival gaps,
queue occupancy, and `A-M` are retained as diagnostics but cannot move the
capture estimate merely because downstream delivery was delayed.

The primary pass rule is strict: maximum absolute interval error must be below
10 ms and median absolute interval error below 5 ms. A single defensibly scored
frame at or above 10 ms fails the evaluation. MAE, P95, P99, the number and
proportion at or above 10 ms, unscorable frames, invalid estimates, interval
widths, and distance to the farther interval endpoint are supplementary report
fields. A finite test result is evidence for those recordings, not a guarantee
for future road conditions.

## Legacy standalone quantitative strategies

Every accepted frame with finite ordered presentation bounds is retained;
timing flags and partial QR readability remain explicit diagnostics. Legacy
reports without interval bounds fall back only to clean finite offsets. The
first 70% of usable frames is the training portion and the later 30% is a
chronological holdout that is not used to fit parameters.

The report compares these strategies:

- **Current fixed 87.348 ms** uses the provisional default without fitting.
- **Calibrated fixed median** uses the training median. A median minimizes total
  absolute error and is resistant to occasional large offsets.
- **Calibrated fixed mean** uses the training mean. A mean minimizes squared
  error but is more sensitive to outliers.
- **PTS-step median** groups the current camera PTS interval into 5 ms buckets
  and uses the training median for the matching bucket.
- **PTS cadence state** selects a compact one-, two-, or three-step categorical
  5 ms cadence state inside the training portion.
- **Selected-length PTS history** compares regularized linear histories from one
  through six PTS steps on an inner chronological validation split, then refits
  the selected length on the complete training portion.
- **Regularized six-step PTS history** preserves the explicit six-step candidate
  for comparison. No PTS strategy uses QR values, future frames, display index,
  generation labels, or elapsed recording time as predictors.

Readability is scored separately for full and partial grids. Each prediction
row also contains the selected interval shifted by one and two measured display
periods. These newer-generation alternatives expose the consequence of a
missing latest QR, but the analyzer never chooses an alternative by minimizing
its residual and never supplies these alternatives to a live predictor.

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
- `calibration_frames.csv` — the same per-frame evidence in tabular form;
- `display_presentations.csv` — every predicted marker and observed software
  presentation return, including display index, cell, interval, prediction
  error, and timing issues.

The root analyzer creates an analysis-local `final_analysis/` directory with:

- `final_analysis.json` and `final_analysis.md` — timing contract, input audit,
  frozen models, strict metrics, limitations, and verdict;
- `final_analysis_frames.csv` — every camera frame, including unscorable rows;
- `final_analysis_predictions.csv` — every valid/invalid estimate, QR interval,
  interval residual, estimated capture time, `A-M`, and estimated arrival delay;
- `final_analysis_results.csv` — one row per model/evaluation pair;
- `final_analysis_residual_cdf.png` — held-out absolute interval errors;
- `final_analysis_overview.png` — strict maximum and median results;
- `final_analysis_interval_error_histograms.png` — one signed interval-error
  histogram for each model family;
- `final_analysis_evidence.png` — QR readability and contradiction audit.

The older distance, quantitative, and cross-recording modules remain available
as explicit standalone tools for comparison, but the root launcher does not run
or combine them automatically.

With `--svg`, the final analyzer also creates an `.svg` copy beside each PNG.
The legacy quantitative analyzer keeps the same opt-in SVG behavior. Final
interval-error histogram panels never overlap model families, use symmetric
signed ranges around zero, and mark both strict ±10 ms boundaries.

## Interpreting the verdict safely

- Review timing-suspect and partial-readability subsets separately. They remain
  usable when their recorded presentation interval is complete, but they must
  not be mistaken for fully readable evidence.
- A recommended correction replaces the configured subtraction; it is
  never added to it.
- The chronological holdout tests a later portion of the same recording. It is
  useful for comparison, but it is not independent session validation.
- Do not enable a learned dynamic correction from one recording. Preselect the
  strategy and confirm it on a later, independently recorded calibration first.
- Final acceptance uses entire restarted stream groups as holdouts. Unknown or
  overlapping stream identities cannot count as independent validation.
  Restart both the camera stream and calibration display when collecting
  calibration, selection, and untouched evaluation evidence.
- Keep the software-marker result separate from claims about physical exposure
  timing, RTSP transport delay, or radar alignment.
- Treat each decoded newest QR as an interval constraint: its state is active
  from its software presentation return until the following presentation. The
  host-anchored segment-running-time interval is primary; arrival-time fields include
  transport, buffering, decoding, and callback delay and are diagnostic only.
- `presentation_return_ns` is sampled at Qt `frameSwapped` or immediately after
  `pygame.display.flip()` returns. It does not measure monitor processing,
  physical scanout, photon output, exposure duration, or rolling shutter.

## Dependencies and checks

The workflow uses OpenCV, Pillow, Pygame, NumPy, `qrcode`, `qreader`, and
Matplotlib. The final-analysis entry point checks for an installed NumPy and
Matplotlib pair that can import together before rendering graphs.

Run the focused non-visual checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_qr_calibration.py \
  tests/test_final_analysis.py tests/test_camera_pipeline_policy.py \
  tests/test_recording_changes.py tests/test_calibration_workflow.py
```

These checks validate data handling and orchestration. Confirm the real display,
camera framing, editable controls, and generated graph readability manually on
the target system.
