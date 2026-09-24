# Calibration test images

The test fixtures include both generated edge cases and stable copies of real
frames from a calibration recording. Tests never read from the replaceable
recording directory, so cleaning or replacing a recording cannot change their
results.

- `images/empty_screen.png` is a dark, empty camera frame.
- `images/bright_qr_cell_0.png` is the timestamp QR `000012345678` on a dark
  background in grid cell 0.
- `images/low_contrast_cell.png` has a dim patch in cell 0 that should not
  count as screen content.
- `recording_frames/camera_000019.jpg` and
  `recording_frames/camera_000020.jpg` are unmodified 1920x1080 captures copied
  from `recordings/camera_calibration_3/images/`. They are successive frames
  used in the earlier QR evidence recovery discussion. The contrast test crops
  the laptop panel in memory (x=175..1831, y=28..1074) to avoid counting the
  brighter room around the screen.

`calibration.support.image_fixture()` loads the synthetic PNGs, and
`calibration.support.recording_frame_fixture()` loads the recorded JPEGs.
