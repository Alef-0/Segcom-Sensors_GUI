# Visualization and filtering

This package filters and renders radar observations.

- `filter_schema.py` defines the shared filter fields, defaults, and value
  normalization used by the interface and radar worker.
- `graph_filter.py` applies range, quality, state, and classification filters
  to cluster and object observations.
- `graph_draw.py` renders the accepted points and radar context with OpenCV.
- `radar_camera_transformer.py` loads the intrinsic, distortion, and extrinsic
  matrices and projects radar coordinates into camera pixels.
- `transposition.py` carries only the newest filtered group-B radar frame and
  draws valid projected points on the group-B camera display.

The live transposition uses `dist_long` as forward X, `dist_latitude` as
lateral Y, and a ground-plane Z of zero. Points behind the camera, outside the
image, or older than 0.5 seconds are not drawn.

The package does not acquire sensor data or save recordings. Those concerns
remain in `sensors/radar/` and `processing/recording/` respectively.


Calibration image inspection is documented in
[calibration/inspection](../../calibration/inspection/README.md).
