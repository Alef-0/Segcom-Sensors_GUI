# QR calibration

The package displays monotonic-time QR markers, records display and camera timing journals, and reviews calibration recordings.

## Flow

- `display_qt.py` is the Qt/OpenGL clock; `display.py` is the Pygame/SDL clock. Both write the same journal format and support the configured QR grid.
- A reserved strip above the QR grid is red while the application waits to start camera recording. Each display backend paints it after the QR content and records its state in the display journal. For a scheduled calibration recording, the application clears the shared wait event only after the camera worker confirms recording is active. If no recording is scheduled or startup fails, the strip clears at countdown completion.
- `recording_display.py` opens saved results read-only. Opening a frame without saved results does not decode it. Select **GO · Decode recording** to scan the folder: bright content in a screen cell triggers timestamp decoding, and contrast detections remain visible in the evidence when a timestamp cannot be read. Empty frames skip the QR reader; the status line reports progress while the current image remains displayed.

Camera media time comes from segment-mapped running time and its pipeline monotonic anchor. Application arrival remains a separate diagnostic. Display swap returns are software timing boundaries; physical exposure and panel scanout are not measured.

The independent frozen-offset procedure and acceptance limits are in [ANCHOR_EXPERIMENT.md](ANCHOR_EXPERIMENT.md). Its helper code lives in `anchor_analysis.py` and is used by the root `analyze_pts_anchor.py` command.

Run the display through the application or with `python3 -m calibration.display_qt --help`. Open a saved recording with `python3 analyze_calibration_recording.py <recording-folder>`.
