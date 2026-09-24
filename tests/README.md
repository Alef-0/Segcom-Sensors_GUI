# Test suite map

The suite is grouped into category folders so you can run only the tests near
the function you change. Run it from the repository root with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Disabling automatic plugin loading prevents unrelated system-wide pytest
plugins from affecting this project.

## Areas covered

| Test file | Main coverage |
| --- | --- |
| `messaging/test_connection_packages.py` | ARS40X packet extraction, scaling, merge behavior, and configuration bit layout |
| `messaging/test_message_module_split.py` | Compatibility exports after cluster/object message separation |
| `messaging/test_object_message_update.py` | Extended object fields, missing optional values, and object filter behavior |
| `messaging/test_object_recording_update.py` | Object metadata and recording behavior |
| `processing/visualization/test_graph_filter.py` | Dynamic, quality, ambiguity, invalid-state, and RCS filtering |
| `processing/recording/test_point_cloud_recorder.py` | Cluster/object PCD schemas, writer behavior, and recording sessions |
| `processing/recording/test_changes.py` | Camera/radar pairing, metadata, frame-rate selection, calibration journals, and playback loading |
| `calibration/workflow/test_workflow.py` | Fake-process display startup, pipeline/monitor/QR-count selections, recording destinations, delayed capture and cancellation |
| `calibration/workflow/test_scheduler_priority.py` | Best-effort calibration process priority behavior |
| `calibration/qr/test_functions.py` | QR payloads, grid classification, QReader helpers, and contrast detection |
| `calibration/display/test_rendering.py` | Qt/Pygame rendering, display journals, and frame pacing |
| `calibration/recording/test_analyzer.py` | Frame analysis, contrast fallback, and grid-cell selection |
| `calibration/recording/test_review.py` | Saved analysis, review behavior, and image overlays |
| `calibration/display/test_timeline.py` | Display journal timing and presentation intervals |
| `calibration/recording/test_scan.py` | Folder scans, frame acceptance, and report output |
| `calibration/analysis/test_quantitative.py` | Calibration verdicts and graph artifacts |
| `calibration/recording/test_launcher.py` | Command-line routing for calibration analysis |
| `calibration/qr/test_evidence.py` | Saved QR evidence checks, session reconstruction, stream grouping, and stale-run diagnostics |
| `calibration/analysis/test_anchor_analysis.py` | Strict frozen-offset interval scoring |
| `calibration/analysis/test_pts_anchor.py` | Clock-only prediction, independent-stream checks, and frozen-offset evaluation |
| `processing/recording/test_manual_snapshot.py` | Snapshot folder validation, indexes, metadata, and cleanup after failure |
| `processing/playback/test_snapshot_playback.py` | Paired-entry filtering, stepping, rendering controls, and copy-current-pair behavior |
| `camera/test_pipeline_policy.py` | Decoder choice, pipeline structure, host-anchored PTS, reference clocks, transport counters, and capture callbacks |
| `camera/test_recording_restart.py` | Recording restart behavior and pipeline/decoder state reuse |
| `processing/visualization/test_radar_camera_transposition.py` | Radar-to-camera projection and transposition behavior |

## Category folders

- `calibration/qr/` — QR utilities, contrast detection, and recorded QR evidence
- `calibration/display/` — calibration display rendering and its timing journal
- `calibration/recording/` — recording analysis, review, scanning, and launch behavior
- `calibration/analysis/` — quantitative and clock-anchor analysis
- `calibration/workflow/` — calibration process startup and scheduling
- `camera/` — camera pipeline and recording restart behavior
- `processing/` — recording, playback, and visualization behavior
- `messaging/` — packet decoding, message compatibility, and object updates

## What the suite does not prove

The tests primarily use temporary folders, fake point clouds, synthetic CAN
payloads, fake GStreamer samples, mocked clocks, and fake GUI/process objects.
Passing tests do not establish:

- access to the real radar gateway or DVR;
- correct physical A/B/C channel placement;
- actual GStreamer plugin or hardware-decoder availability;
- the negotiated RTSP transport or real packet loss;
- sustained 30 FPS capture and JPEG writing;
- fullscreen calibration appearance or QR readability through the camera;
- the correctness of the operational camera-delay value for a new session;
- clean interaction among the real FreeSimpleGUI, OpenCV, GTK, and Qt/OpenGL
  windows.

Those behaviors require separate checks on the target installation. UI checks
should be completed by the project owner or from screenshots they deliberately
provide.

## Focused runs

Run a full behavior category when changing a related function:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/calibration/qr
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/calibration/display
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/calibration/recording
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/calibration/analysis
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/calibration/workflow
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/camera
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/processing
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/messaging
```

For a single behavior, pytest also accepts the individual test node, such as
`tests/calibration/qr/test_functions.py::TestQRFunctions::test_contrast_detection_finds_content_in_saved_recording_frames`.

Calibration image inputs live under `calibration/fixtures/`. The synthetic PNG
images cover empty and low-contrast cases; `recording_frames/` contains stable
copies of two frames from `recordings/camera_calibration_3/images/`. Tests load
them through `calibration.support` and do not depend on mutable recording data.
