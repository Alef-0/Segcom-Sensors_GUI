# Legacy Calibration & Camera Timing System (Archive Documentation)

## 1. Overview & Historical Context

Between August and September 2026, extensive research and development was conducted in this repository to solve a critical sensor fusion challenge: **accurately measuring and compensating for end-to-end latency between the RTSP/H.264 camera stream and the host system clock**.

In automotive sensor fusion (combining Continental ARS408-21 radar clusters/objects with live camera feeds), temporal misalignment degrades spatial projection accuracy. If a camera frame arriving at monotonic time $T$ actually captured the scene at $T - \Delta t$, overlaying current radar targets produces visual lag, parallax errors, and inaccurate object correlation.

To measure this latency without expensive hardware trigger boxes, an optical screen-to-camera calibration pipeline was conceived, developed, and iterated through multiple generations. This document preserves the architectural designs, experimental methodologies, mathematical models, key findings, and code structure of that effort before its removal from the production application.

---

## 2. Architecture & Evolutionary Phases

```
+-------------------------------------------------------------------------------------------------+
|                                 Evolution of Calibration Machinery                              |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|  Phase 1: Screen Clock (Aug 2026)                                                              |
|  [CALIBRATION/calibration_screen_clock.py]                                                      |
|  - Fullscreen Tkinter/Pygame millisecond counter.                                               |
|  - Manual visual inspection of recorded frames to gauge delay.                                  |
|  - Limitations: Rolling shutter tear, motion blur, tedious manual verification.                 |
|                                         |                                                       |
|                                         v                                                       |
|  Phase 2: QR Matrix & Dual Renderers (Early Sep 2026)                                          |
|  [calibration/display_qt.py] & [calibration/display.py]                                         |
|  - Rotating QR codes encoding monotonic timestamps and generation counters.                     |
|  - Configurable grids (4, 6, 8, 9, 10, 12 cells) to handle monitor spatial distribution.        |
|  - Dual engines: VSync-locked Qt/OpenGL and SDL/Pygame.                                         |
|  - JSONL display journals logging exact monotonic timestamp of each buffer swap (`swapBuffers`).|
|                                         |                                                       |
|                                         v                                                       |
|  Phase 3: Automated Decoding & Offline Review (Mid Sep 2026)                                    |
|  [calibration/qr.py] & [calibration/recording_display.py]                                       |
|  - QReader / OpenCV automatic QR extraction, adaptive thresholding, and contrast recovery.     |
|  - Dedicated Tkinter GUI (`analyze_calibration_recording.py`) to inspect frames and decode.    |
|  - Quantitative statistical reports: histograms, CDF curves, median offsets.                    |
|                                         |                                                       |
|                                         v                                                       |
|  Phase 4: Transposition Tab Milestone (Sep 10, 2026 - Commit 70054a6)                           |
|  [processing/visualization/transposition.py] & [menu_configurations.py]                         |
|  - Calibrated 3D-to-2D projection matrices (`camera_matrixes.json`).                            |
|  - Group B radar targets projected directly onto live camera stream.                            |
|  - Stale-data dropping and low-latency IPC queue (`put_latest` / `get_latest`).                 |
|                                         |                                                       |
|                                         v                                                       |
|  Phase 5: Anchor Experiments & Interval Intersection (Sep 11 - Sep 24, 2026)                    |
|  [calibration/qr_timing.py], [calibration/anchor_analysis.py], [analyze_pts_anchor.py]           |
|  - Frozen PTS clock-anchor offset model: mapping GStreamer `GstSegment` running time to host   |
|    monotonic time via pipeline clock anchor ($M = \text{anchor} + R$).                          |
|  - Joint visibility lifetime intervals: intersecting visible QR lifetimes per frame.            |
|  - Bidirectional constraint propagation across continuous clock segments.                       |
|  - Frame-level evidence scoring, screen-corner perspective mapping, and conflict rejection.     |
+-------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Component Breakdown

### 3.1 Display Generators (`display_qt.py` & `display.py`)
* **Qt/OpenGL Clock (`display_qt.py`)**: Subclassed `QOpenGLWidget` configured for double-buffering and hardware VSync (`format.setSwapInterval(1)`). Repainted rotating QR cells on each frame tick, writing display journal entries (`display_timestamps.jsonl`) with the exact `monotonic_ns` timestamp recorded immediately after `swapBuffers()`. Included a top recording-wait strip that remained red while awaiting camera worker recording synchronization.
* **Pygame/SDL Clock (`display.py`)**: Cross-platform fallback implementation using `pygame.time.Clock` and `pygame.display.flip()`.
* **Screen Pacing & Diagnostic Helpers (`display_common.py`)**: Abstracted journal writing, frame pacing, monitor refresh interval calculations, and common grid layouts.

### 3.2 QR Generation, Detection & Timing Models (`qr.py` & `qr_timing.py`)
* **Payload Generation**: Encoded compact string payloads containing monotonic timestamp, display generation counter, and cell position identifier.
* **Cell Grids**: Supported 4, 6, 8, 9, 10, and 12-cell configurations. Multiple cells allowed distinguishing between new frame presentation and persistent scanout.
* **Decoder Pipeline**: Integrated `QReader` (backed by YOLO / PyZbar / OpenCV). Implemented multi-pass contrast recovery for washed-out or low-exposure frames.
* **Visibility Interval Intersection (`qr_timing.py`)**:
  - Rather than assuming a single QR code represents an instantaneous point in time, each decoded QR was mapped to its valid display presentation window $[t_{\text{start}}, t_{\text{end}})$.
  - Multiple visible QRs in a single camera frame were intersected: $[t_{\text{min}}, t_{\text{max}}] = \bigcap [t_{\text{start}, i}, t_{\text{end}, i}]$.
  - A two-pass forward and backward constraint propagation algorithm tightened interval bounds using neighboring frames in continuous clock segments, halting at camera resets, dropouts, or contradictory reads.

### 3.3 Evidence Validation & Frozen Anchor Analysis (`evidence.py`, `anchor_analysis.py`, `analyze_pts_anchor.py`)
* **Evidence Validation**: Checked bounding boxes, clipping at frame boundaries, perspective distortions, screen geometry validity, and conflicting reads.
* **Frozen PTS Clock Anchor (`ANCHOR_EXPERIMENT.md`)**:
  - Formulated the media timestamp model:
    $$M = \text{pipeline\_zero\_monotonic\_ns} + R(\text{PTS})$$
    where $R(\text{PTS})$ is the running time computed by mapping the buffer PTS through the active `GstSegment`.
  - A frozen offset $c_0$ was fitted across training streams:
    $$T_{\text{predicted}} = M - c_0$$
  - The offset was evaluated on held-out test streams against the QR visibility intervals to determine whether a fixed constant offset could reliably replace per-frame visual decoding.

### 3.4 Reviewer & Statistical Evaluation (`recording_display.py`, `quantitative_analysis.py`)
* **Interactive Viewer (`recording_display.py`)**: A 1,600+ line Tkinter application capable of stepping through recordings frame-by-frame, visualizing bounding boxes, showing QR payload data, displaying uncertainty windows, and launching multiprocess batch decoders.
* **Quantitative Analysis (`quantitative_analysis.py`)**: Computed error residuals, cumulative distribution functions (CDF), offset stability over time, and output formal acceptance verdicts (`calibration_verdict.json`).

### 3.5 Sensor & Pipeline Integrations
* **Dedicated Channel 4**: Camera channel 4 was reserved in `sensors/camera/camera_gstreamer.py` specifically for calibration mode.
* **Elevated Priority (`scheduler_priority.py`)**: Applied `os.sched_setscheduler` (SCHED_RR / SCHED_FIFO) and process niceness adjustments to prevent OS scheduler jitter from distorting display swap timestamps.

---

## 4. Key Findings & Technical Limitations

1. **Software vs. Physical Timing Boundaries**:
   - The display journal recorded software return times from `swapBuffers()` or `glFinish()`. However, the physical photons on the LCD/OLED panel depend on hardware scanout cadence, panel response time ($GtG$), and monitor internal buffering. Software timestamps represented an upper or lower boundary rather than true optical emission time.
2. **Rolling Shutter and Exposure Artifacts**:
   - The security camera's CMOS sensor operates with a rolling shutter. Top rows and bottom rows of the sensor expose at different physical instants (often spanning 15–33 ms). This caused partial QR reads, torn codes, and apparent timing conflicts between top and bottom grid cells.
3. **Decodability Under Motion**:
   - Rapidly cycling QR codes at 60 Hz or 144 Hz created motion blur and partial temporal superposition on the camera sensor, drastically reducing automated decoding yield (frequently dropping below 50% readable frames without aggressive contrast post-processing).
4. **Codebase Bloat and Divergence**:
   - The calibration subsystem grew into over 7,400 lines of code across `calibration/`, `tests/calibration/`, and auxiliary scripts.
   - It introduced heavy dependencies (`PyQt6`/`PySide6`, `pygame`, `qreader`, `pyzbar`, `matplotlib`, `tkinter`) that were completely unnecessary for the core mission: a responsive desktop GUI for recording radar point clouds and video in vehicle trials.
   - For real-world radar-camera fusion, an empirical, laboratory-fixed offset ($\approx 109\text{--}145\text{ ms}$) or static intrinsic/extrinsic matrix projection was sufficient.

---

## 5. What Is Preserved: The Transposition Feature

While the runtime calibration experiments and display generators are retired, **the ultimate goal of the calibration work—the Transposition Feature—is retained and fully operational**:

* **`processing/visualization/transposition.py`**: High-performance, low-latency overlay engine that projects 3D radar coordinates (Continental ARS408 Group B) onto 2D camera pixels in real time.
* **`processing/visualization/radar_camera_transformer.py`**: Mathematical transformation validating camera intrinsics, distortion coefficients, and 4x4 extrinsic transformation matrices.
* **`camera_matrixes.json`**: Static configuration storing verified camera matrix, distortion coefficients, and radar-to-camera rotation/translation vectors.
* **Transposition GUI Tab**: Located under **Radar controls -> Transposition**, providing live toggle buttons, distance cutoff filtering, and real-time status indication without any dependency on the legacy calibration modules.
