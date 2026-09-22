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
- `final_analysis.py` now contains only journal reconstruction, stream grouping,
  and conditional interval scoring. Model fitting and its old CLI were retired.
- `../analyze_pts_anchor.py` calibrates one laboratory offset and evaluates it on
  independent streams without test-time recalibration.
- `quantitative_analysis.py` is the pre-branch legacy comparison tool, restored
  without the branch's added models. It is outside the anchor workflow and its
  historical default must not be treated as the current calibration.
- `intrinsics.json` supplies camera intrinsics for undistortion.
- `../analyze_calibration_recording.py` opens the inspection window, saving QR
  evidence. It evaluates only when given a previously frozen `--offset-file`.

## Recording and analysis workflow

1. Start the QR calibration from the application. The recording begins after
   the configured three-second delay. Keep the calibration display visible for
   the complete recording, then close it so its journal is flushed.
2. Open **Visualization** for the recording, or start the root analyzer from a
   terminal.
3. If `<recording>_analysis/calibration_analysis.json` exists, the window first
   loads the saved QR values by image filename and any compatible saved boxes.
   Browsing these results does not run QReader or overwrite files. Missing frames
   in a partial analysis are labeled as having no saved results. Otherwise the
   current image is decoded as before. Correct editable PTS, NTP, or QR values
   only when the recording visibly supports the correction.
4. Select **GO — DECODE FULL FOLDER** to explicitly start a fresh decode,
   replacing the saved analysis when the scan is saved. Each frame is shown as it finishes
   decoding. Frames without any valid readable QR are skipped; other readable
   values remain usable even when several grid cells are unreadable. When the
   scan finishes, it automatically creates
   `calibration_analysis.json`, `calibration_frames.csv`, and
   `display_presentations.csv` in a sibling folder named
   `<recording>_analysis`.
5. Close the inspection window after a completed scan. Saved evidence is ready
   for laboratory calibration or independent evaluation. The GUI does not fit
   an offset automatically. Reviewing saved results does not rewrite them.

```bash
python3 analyze_calibration_recording.py /path/to/recording
python3 analyze_calibration_recording.py /path/to/recording \
  --intrinsics /path/to/intrinsics.json --offset-file /path/to/offset-v1.json
python3 analyze_pts_anchor.py calibrate /path/to/lab_analysis \
  --minimum-stream-age-seconds 30 --output /path/to/offset-v1.json
python3 analyze_pts_anchor.py evaluate /path/to/test_analysis \
  --offset-file /path/to/offset-v1.json --output-directory /path/to/anchor-report
```

See [ANCHOR_EXPERIMENT.md](ANCHOR_EXPERIMENT.md) for the predeclared protocol.

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

## Evidence quality in anchor evaluation

The newest decoded QR is not necessarily the newest displayed QR. The evidence
loader separates readable identities from usable timing evidence:

- `usable_conditional`: saved detections have no known clipping, unresolved
  regions, identity conflict, geometry disagreement, or visibility contradiction.
  This still assumes there is no completely undetected newer code; it does not
  establish physical exposure timing.
- `usable_newest_readable_with_artifacts`: the newest successfully decoded QR
  supplies the interval. Older generations, unreadable regions and clipped QR
  regions remain warnings; they do not discard this interval. This assumes the
  newest readable QR represents the observed state, even if a newer unreadable
  QR may exist. The artifact's physical cause is not inferred.
- `potentially_missing_newer_generation`: conflicting identity, manual override,
  or invalid screen mapping prevents a narrow interval.
- `suspected_stale_visual_state`: the same newest readable QR persists across
  consecutive camera images longer than its journal interval plus one normal
  display period (at least two periods). The whole repeated run is excluded
  from timing fitting/scoring, retained in diagnostics and the camera-frame
  denominator. `temporal_evidence` records its duration and threshold. Checks
  reset on missing reads or timing discontinuities and respect journal pauses.
  This cannot distinguish a held display, repeated camera/DVR content, or a
  newer unreadable QR; brief holds below the threshold can escape detection.
- `unknown_pixel_evidence`: legacy evidence lacks the saved detection audit.
- `no_reference`: there is no uniquely matched QR.

Every frame remains in the report and coverage denominator. Identity/geometry
failures retain diagnostic intervals without using them for training or scoring.
Artifact warnings are exported separately from exclusions; their scores remain
conditional and cannot verify every frame. No generation is chosen by minimizing prediction error, and no
frame is removed based on its residual. Rerun the decoding stage to populate
`qr_evidence` for old recordings; running final analysis alone cannot recreate it.
The JSON stores every detection's box, original-image coordinates, confidence,
payload, journal identity, clipping status and overlap group. An unreadable retry
overlapping a readable detection is not counted as a separate missing QR.

Stream grouping uses the recorded pipeline base time when available, so jitter
in sampled anchors cannot turn one stream into independent sessions. Recording
attaches to the running pipeline and retains its anchor. Only an actual stream
restart establishes a new stream. Native clock identity is compared instead of
temporary Python wrapper IDs. Timing discontinuities remain explicit diagnostics.

Optional screen mapping uses `analysis_screen_geometry.json` in the recording
directory. It changes position checks, not QR identities. Coordinates must refer
to the **undistorted** image at the selected alpha, ordered top-left, top-right,
bottom-right, bottom-left. For example (replace with measured coordinates):

```json
{
  "default": {
    "image_size": [1920, 1080],
    "undistortion_alpha": 0.25,
    "corners": [[300, 150], [1600, 180], [1650, 950], [250, 920]]
  },
  "frames": {}
}
```

`frames` may map an exact journal filename, such as `images/camera_000421.jpg`,
to an overriding entry with the same fields (or `null` to mark it unavailable).
Without mapping, position checks retain the legacy camera-image grid and the
report records geometry as unverified. Invalid polygons, changed image size/alpha,
clipped screen corners and mapped-cell/journal disagreement flag the evidence.
Movement that happens to preserve cell assignments is not automatically detectable;
recalibrate the corners after movement. Fully cropped codes cannot be recovered.

## Generated files

The inspection window saves `calibration_analysis.json`, `calibration_frames.csv`,
and `display_presentations.csv`. Calibration writes a versioned offset JSON with
its estimator, age cutoff, source hashes, stream identities and per-stream results.
It refuses to overwrite an existing offset file.

Evaluation writes `pts_anchor_analysis.json`, `pts_anchor_analysis.md`, and
`pts_anchor_predictions.csv`. Every saved camera frame has a prediction row,
including missing predictions and unscorable evidence. Rows include QR interval
bounds, signed error, diagnostic intervals, evidence warnings and exclusions.
The JSON also retains dropped/rejected frame counts from recording metadata.
There is no model selection, target-prefix calibration, or automatic live update.

The strict goals are maximum interval error <10 ms and median <5 ms. A passing
subset does not verify unscorable or unsaved frames. Software display returns
are not exposure or photon timestamps. Any eventual configured correction
replaces the existing subtraction; never add a second correction.

## Dependencies and checks

The workflow uses OpenCV, Pillow, Pygame, NumPy, `qrcode`, `qreader`, and
Matplotlib. The final-analysis entry point checks for an installed NumPy and
Matplotlib pair that can import together before rendering graphs.

Run the focused non-visual checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q tests/test_qr_calibration.py \
  tests/test_pts_anchor.py tests/test_calibration_evidence.py \
  tests/test_final_analysis.py tests/test_camera_pipeline_policy.py \
  tests/test_recording_changes.py tests/test_calibration_workflow.py
```

These checks validate data handling and orchestration. Confirm the real display,
camera framing and editable controls manually on
the target system.
