# Sensor processing

This package groups radar/camera recording, playback, and visualization services used by `main.py` and the sensor workers.

- `recording/` writes and reads camera images, radar PCD frames, timestamp journals, and manual snapshots.
- `playback/` loads recordings and saved snapshots for timed or stepwise review.
- `visualization/` applies radar filters, draws the top-down graph, projects radar points into camera coordinates, and renders live overlays.

The normal recording layout is `recording_<channel>_<timestamp>/point_cloud/`, `images/`, `recording.json`, and `timestamps.json`. Camera timing offsets are explicit configuration; radar association is not physical exposure validation.

See the subpackage READMEs for the module map.
