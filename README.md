# Segcom Sensors GUI

Segcom Sensors GUI is a desktop tool for operating a three-group Continental
ARS40X radar setup together with cameras and the GPS exposed by a network DVR.
It provides live monitoring, radar configuration, synchronized radar/camera
recording, manual paired snapshots, playback, and a camera-delay calibration
workflow.

This README describes the current working tree. The hardware addresses,
credentials, timing values, and physical group layout are deployment-specific
and should be confirmed on the real Segcom installation.

## Main capabilities

- Connect to a TCP gateway that forwards CAN packets from radar groups A, B,
  and C.
- Decode radar configuration, cluster, quality, object, extended-object, and
  collision-warning messages.
- Send runtime or non-volatile configuration changes to one or all radars.
- Display the selected radar group as a filtered top-down point plot.
- Open the corresponding DVR camera stream over RTSP, with automatic decoder
  selection for desktop NVIDIA, Jetson, or CPU decoding.
- Record PCD radar frames and JPEG camera frames into synchronized recording
  folders.
- Capture a single camera/radar pair into a snapshot folder.
- Play complete recordings at their recorded cadence, or inspect paired
  snapshots one frame at a time.
- Poll the DVR GPS endpoint and open the last position in Google Maps.
- Record camera channel 4 while displaying monotonic QR markers for
  latency and clock-drift analysis.
- Convert recorded PCD trees to CSV while preserving images and metadata.

## Deployment assumptions

The current source contains fixed addresses and credentials:

| Device or service | Current endpoint | Used by |
| --- | --- | --- |
| Radar CAN gateway | `192.168.1.101:2323` over TCP | `sensors/radar/connection_communication.py` |
| DVR RTSP cameras | `192.168.1.108:554`, channels 1-4 | `sensors/camera/camera_gstreamer.py` |
| DVR GPS status | `http://192.168.1.108/cgi-bin/positionManager.cgi?action=getStatus` | `sensors/gps/gps_connection.py` |

The DVR username and password are also embedded in the camera and GPS source.
Do not publish a deployment copy of this repository without first deciding how
those credentials should be handled.

The source treats channels 1, 2, and 3 as groups A, B, and C respectively.
The user interface labels those positions LEFT A, MIDDLE B, and RIGHT C.
Camera channel 4 is reserved for calibration.

## Running the application

Use Python 3 from the repository root:

```bash
python3 main.py
```

Python packages are listed in `requirements.txt`. The host also needs the
native GStreamer runtime and plugins required by the selected H.264 decoder,
plus a working display backend for FreeSimpleGUI, OpenCV, and Pygame.

The optional `SEGCOM_CAMERA_DECODER` environment variable accepts `auto`,
`rtx`, `orin`, or `cpu`. Automatic selection prefers Jetson decoding on a
Jetson, desktop NVIDIA decoding elsewhere, and keeps the CPU decoder as the
fallback.

## Runtime architecture

`main.py` owns the FreeSimpleGUI window and starts workers with Python's
`spawn` multiprocessing context:

```text
FreeSimpleGUI process
├── radar worker: TCP/CAN input, decoding, plot, PCD recording
├── camera worker: RTSP/GStreamer input, display, JPEG recording
├── GPS worker: DVR position polling and map link
├── recording playback worker
└── snapshot playback worker
```

The GUI sends commands to each worker through a dedicated pipe. Workers return
state, progress, warnings, and errors through one bounded status queue. A
shared shutdown event coordinates normal termination. The calibration display
is started only when requested and runs in its own process.

`application_core.py` contains the common event loop and record/playback orchestration.
`main.py` extends that behavior with calibration and snapshot-playback modes.
Likewise, `interface_core.py` contains the common window and state logic while
`menu_configurations.py` adds the newer controls.

## Live radar flow

The radar worker opens a non-blocking TCP connection to the CAN gateway. Each
gateway packet is 23 bytes and contains the CAN ID, eight data bytes, a source
timestamp, and a channel number.

The code recognizes these main ARS40X messages:

- `0x200`: configuration command sent to the radar.
- `0x201`: configuration/state response shown in the GUI.
- `0x600`, `0x60A`: start markers for cluster and object frames.
- `0x701`, `0x702`: cluster general and quality data.
- `0x60B` through `0x60E`: object general, quality, extended, and warning data.

A new `0x600` or `0x60A` closes the preceding logical frame. The completed
frame can then be plotted, retained briefly for manual snapshot matching, and
queued for recording. Plot filters include distance, RCS, dynamic property,
false-alarm probability, ambiguity state, and invalid-state flags.

## Live camera and timestamp flow

The camera worker builds one GStreamer pipeline with a tee:

- The display branch is intentionally leaky and holds only the newest frame so
  the live view does not accumulate delay.
- The full-resolution capture branch uses a bounded 30-buffer pipeline queue
  and a separate bounded image-writer queue.

The RTSP source currently allows TCP or UDP negotiation, requests RTCP and
reference timestamp metadata when supported, and defaults to 145 ms of
GStreamer jitter-buffer latency. That 145 ms value controls buffering; it is
not itself a measured end-to-end correction.

Saved-frame time starts from buffer PTS mapped through the GStreamer segment
and pipeline clock to a stable host-time anchor. The capture callback records
application arrival before pulling or converting the sample; later processing
timestamps and capture-queue occupancy remain separate diagnostics. Camera NTP
metadata is observational and never moves the segment-mapped media time.
Invalid or non-forward timing is rejected from image recording but retained as
a timing event with the information available at rejection.

The separate camera latency adjustment provisionally defaults to 87.348 ms. It is subtracted
when associating a camera observation with radar time and is recorded in
metadata. It does not replace or configure the RTSP jitter buffer. The current
repeated recordings keep their interval-aware fixed candidates within 2 ms, but
they share one camera stream epoch. Keep the value provisional until it passes
a separately restarted camera/display session, and recheck it after changes to
the DVR, decoder, network path, or capture setup.

See `sensors/camera/README.md` for the pipeline and timestamp policy in more detail.

## Recording and snapshots

Starting a recording creates one folder per selected radar group:

```text
recording_A_YYYYMMDD_HHMMSS_microseconds/
├── images/
│   └── camera_000001.jpg
├── point_cloud/
│   └── frame_000001.pcd
├── recording.json
└── timestamps.json
```

Radar frames are written as PCD point clouds inside `point_cloud/`. Camera
frames are written as JPEGs inside `images/` in every selected group folder.
`recording.json` is the authoritative ordered association between radar frames
and camera frames; `timestamps.json` keeps the older filename-to-radar-time
mapping. The loader also accepts older JSON that refers to loose root-level
files, and can resolve those bare references if the files were moved into the
new subfolders.

For each camera frame, the recorder subtracts the configured camera delay and
matches the result to the closest still-unpaired radar frame. The metadata
retains the camera time, applied delay, and residual synchronization error.

A manual snapshot first captures a valid-timestamp camera frame for the chosen
group. The radar worker then selects the closest completed radar frame from a
three-second history. The pair is rejected if the residual difference exceeds
500 ms. Successful snapshots use the same PCD/JPEG and JSON contracts as a
normal recording, so they can be played and converted by the same tools.

See `processing/recording/README.md` for file schemas and overload handling.

## Operating modes

Live monitoring, normal playback, snapshot playback, and calibration camera
mode are coordinated as mutually exclusive uses of the camera/radar displays.
When playback or calibration needs the devices, the GUI first stops active
recording and closes conflicting live workers before starting the requested
mode.

Normal playback follows recorded timestamps and supports restart and five
second seeks. Snapshot playback can restrict the list to entries that contain
both image and PCD data, pause, step backward or forward, and save the current
pair into another snapshot folder.

## Calibration

The Calibration tab lets you require the desktop NVIDIA, ARM/Jetson, or CPU
camera pipeline, select either the Qt/OpenGL or Pygame/SDL fullscreen clock,
and choose its monitor.
The selected pipeline is validated before the clock starts and does not silently
fall back to another decoder. Both clock implementations accept a 4, 6, 8, 9,
10, or 12-code grid and independently control how many of its latest QR codes
remain visible. They present QR timestamps in a snake path through the grid,
show the display index beside each timestamp, and underline the newest marker.
Press **P** to pause/resume, or use **Q** or Escape to close either clock; the Qt
clock also handles Ctrl+C directly. Paused markers are excluded from timing
analysis.

The Visualization tab opens the single analyzer with the project's copied
camera intrinsics, an undistorted image, and alpha 0.25 by default. QReader
detects every QR bounding box, retries grid cells without a readable result,
and orders detections by grid cell. When an existing sibling `_analysis` folder
contains saved results, opening the viewer loads those QR values and compatible
boxes for review without decoding or writing files. Otherwise the first frame is
decoded when the viewer opens. Fresh full-folder decoding begins only after **GO — DECODE FULL FOLDER** is
pressed, and each completed frame is shown. A frame is accepted when at least
one readable QR matches its journal; camera-grid position disagreements remain
visible as warnings. Unreadable and invalid detections are retained as
diagnostics while the latest journal-matched QR supplies the offset. The PTS,
NTP, and configured grid values are editable without automatically starting a
scan. Finishing the full scan writes `calibration_analysis.json` and
`calibration_frames.csv` to a sibling
`<recording-folder-name>_analysis` directory. The launcher now saves evidence only
unless you supply `--offset-file` from a separate laboratory calibration. With
that file, it evaluates one frozen correction after a fresh scan. Reviewing
saved results does not rewrite evidence or reports.

```bash
python3 analyze_calibration_recording.py /path/to/calibration-recording

# Establish c0 using laboratory streams only, before examining the test stream
python3 analyze_pts_anchor.py calibrate /path/to/lab_analysis \
  --minimum-stream-age-seconds 30 --output /path/to/offset-v1.json

# Restart the stream, record independently, decode, then evaluate unchanged c0
python3 analyze_pts_anchor.py evaluate /path/to/test_analysis \
  --offset-file /path/to/offset-v1.json --output-directory /path/to/anchor-report
```

The sole estimate is `pipeline_zero_monotonic_ns + running_time_ns - c0`.
Each pipeline epoch supplies its clock anchor; the offset stays fixed across
independent tests. QR intervals only score test predictions. Maximum interval
error must be below 10 ms and median below 5 ms. Missing evidence remains in
the denominator; these software intervals do not establish physical exposure time.
See [the experiment protocol](calibration/ANCHOR_EXPERIMENT.md) for the offset
strategy, startup experiment, scope of this simplification, and next steps.

## CSV conversion

Convert a recording tree with:

```bash
python3 convert_to_csv.py /path/to/recordings
```

The tool creates a sibling folder named `<source> - CSV`, converts every PCD
file to CSV, copies images, updates `.pcd` references in Segcom metadata, and
writes a value dictionary beside converted point clouds. It refuses to replace
an existing output folder and removes a partial output tree if conversion
fails.

## Tests

The tests use `pytest` and avoid real hardware by replacing the DVR, CAN
gateway, GStreamer samples, filesystem writers, and GUI objects with focused
fakes. In this environment, disable unrelated globally installed pytest
plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Unit tests provide static and simulated evidence only. Camera optics,
fullscreen behavior, decoder availability, network timing, radar traffic, and
the complete GUI workflow still require checks on the actual installation.

See `tests/README.md` for the test-area map.

## Source map

| Path | Responsibility |
| --- | --- |
| `main.py` | Current application entry point and mode orchestration |
| `application_core.py` | Shared GUI event, recording, playback, and shutdown behavior |
| `menu_configurations.py` | Current window layout and UI state extensions |
| `interface_core.py` | Shared GUI layout and state transitions |
| `sensors/` | Radar, RTSP camera, timestamp, and GPS integrations |
| `processing/` | Plotting, filtering, recording, PCD reading, snapshots, and playback |
| `calibration/` | QR display/decoding, recording viewer, anchor evidence/scoring, and camera intrinsics |
| `analyze_calibration_recording.py` | Runs the recording viewer; optionally evaluates a frozen offset |
| `convert_to_csv.py` | Recursive PCD-to-CSV export |
| `content/` | ARS40X technical-documentation extracts |
| `recordings/` | Generated recording data, kept outside source packages |
| `snapshots/` | Generated or manually assembled snapshot data |
| `tests/` | Automated tests, kept outside source packages |

## Points to confirm with the project owner

The code supports the following interpretation, but these product-level facts
are not independently proven by source alone:

- LEFT/MIDDLE/RIGHT and A/B/C are the intended physical channel assignments.
- The gateway packet timestamp is intentionally ignored in favor of host
  receipt time for radar frame recording.
- Camera channel 4 is always the calibration camera.
- The provisional 87.348 ms camera adjustment still needs confirmation on an
  independent recording before it becomes the final deployment default.
- Snapshot matching should continue to allow up to 500 ms residual error.
- Recording camera frames into every selected radar folder is the desired data
  duplication model.
