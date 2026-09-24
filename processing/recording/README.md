# Recording

- `paths.py` defines and resolves the image, point-cloud, and metadata paths.
- `point_cloud_recorder.py` writes cluster and object PCD records from a bounded queue and associates camera frames with radar timestamps.
- `point_cloud_reader.py` reads current and legacy PCD schemas into radar model objects.
- `camera_snapshot_recorder.py` saves images and per-frame timing records off the capture callback.
- `camera_telemetry.py` writes JSONL timing journals, stream epochs, events, and summaries.
- `manual_snapshot.py` saves a manually requested radar/camera pair in the normal recording layout.

The active recording layout uses `point_cloud/`, `images/`, `recording.json`, and `timestamps.json`. Calibration camera timing journals are additional evidence and remain distinct from the radar recording timestamp index.
