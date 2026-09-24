# Calibration and processing code map

Updated 2026-09-23 after the package rewrite. The normal import paths point to the rewritten code.

## Calibration

Calibration displays monotonic-time QR markers, records display and camera journals, then reviews saved evidence or explicitly decodes a recording for offline timing analysis. QR and software presentation records provide conditional timing evidence. They do not measure exposure, panel scanout, or physical camera-to-radar alignment.

### Recording and display flow

1. `main.py` prepares a recording folder and creates shared stop, first-QR, and recording-wait events.
2. `calibration.display_qt` or `calibration.display` renders a rotating QR grid and writes the display journal. The red strip occupies its own reserved area above the QR cells, is painted after the QR content, and its on/off state is recorded per frame. It remains red while the recording-wait event is set; `main.py` clears that event when recording starts, or when the countdown completes without a recording or startup fails.
3. The first presented QR starts the configured seven-second delay. The camera recorder then starts saving frames and timing metadata.
4. `analyze_calibration_recording.py` opens the review window. Existing results are reviewed without decoding or writing; the operator's GO action starts a fresh scan and saves a sibling analysis folder.
5. `analyze_pts_anchor.py` uses `calibration.anchor_analysis` to inspect raw journals and evaluate frozen offsets across stream sessions.

### Modules

| Path | Responsibility |
| --- | --- |
| `calibration/__init__.py` | Exposes the calibration recording delay. |
| `calibration/qr.py` | QR payload and matrix generation, grid layouts, QReader setup, full-frame decoding, and bounded retries for missing cells. |
| `calibration/display_common.py` | Shared display settings, frame pacing and swap timing, journal writing, and layout helpers. |
| `calibration/display_qt.py` | Qt/OpenGL QR clock. Repaints the complete frame, then paints the recording strip last. |
| `calibration/display.py` | Pygame/SDL QR clock with matching grid, journal, and recording-strip behavior. |
| `calibration/evidence.py` | Pixel, screen-geometry, conflict, clipping, and repeated-state evidence checks. |
| `calibration/recording_display.py` | Read-only saved-result viewer and explicit fresh decoder; calculates presentation intervals and exports per-frame reports. |
| `calibration/anchor_analysis.py` | Reconstructs timing and stream evidence for the frozen-offset workflow; it is a helper module, not a separate report generator. |
| `calibration/quantitative_analysis.py` | Descriptive summaries and plots for an existing analysis output. Its results are diagnostic. |
| `calibration/scheduler_priority.py` | Best-effort process scheduling priority helper. |
| `calibration/intrinsics.json` | Camera matrix, distortion coefficients, and calibration image size for undistortion. |

Timing interpretation: segment-mapped running time anchored to the pipeline monotonic clock is the camera media reference. Application arrival is kept separate as a diagnostic. A payload repeated in the same display journal is treated as ambiguous; proximity does not establish its identity. Software presentation returns are interval boundaries, not measured photons or exposure timestamps. Frozen-offset evaluation requires independent stream sessions, median error below 5 ms, and maximum error below 10 ms.

## Processing

Processing contains the normal radar/camera recording format, playback controllers, graph filters, radar plots, and camera projection. `main.py`, `sensors/camera/`, and `sensors/radar/` connect these services to the GUI and live workers.

### Modules

| Path | Responsibility |
| --- | --- |
| `processing/recording/paths.py` | Defines recording folders and resolves image/point-cloud paths across supported layouts. |
| `processing/recording/point_cloud_recorder.py` | Encodes cluster/object PCD data, queues radar writes, tracks timestamps, associates camera frames, and coordinates radar sessions. |
| `processing/recording/point_cloud_reader.py` | Reads current and legacy PCD schemas into radar model objects. |
| `processing/recording/camera_snapshot_recorder.py` | Writes camera images and per-frame timing metadata on a worker thread. |
| `processing/recording/camera_telemetry.py` | Writes camera timing journals, session epochs, events, and summaries. |
| `processing/recording/manual_snapshot.py` | Saves a manual radar/camera snapshot into the standard recording structure. |
| `processing/playback/playback.py` | Loads recording entries and runs timed radar/camera playback. |
| `processing/playback/snapshot_playback.py` | Browses saved snapshots and supports manual snapshot saving. |
| `processing/visualization/filter_schema.py` | Shared filter keys, option values, and display colors. |
| `processing/visualization/graph_filter.py` | Applies radar quality, state, RCS, and coordinate filters. |
| `processing/visualization/graph_draw.py` | Draws and supports inspection of the top-down radar graph. |
| `processing/visualization/radar_camera_transformer.py` | Validates calibration coefficients and projects radar points into camera coordinates. |
| `processing/visualization/transposition.py` | Builds and draws the live radar-on-camera overlay while discarding stale frames. |

The established recording structure is `recording_<channel>_<timestamp>/point_cloud/`, `images/`, `recording.json`, and `timestamps.json`. Camera timing corrections remain explicit configuration; these modules do not infer physical exposure alignment from radar pairing.
