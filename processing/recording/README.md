# Capture and recording

This package defines the persistent radar/camera recording format and the
workers that create, read, and copy paired observations.

## Files

- `point_cloud_recorder.py` writes cluster or object frames as PCD and manages
  per-group recording sessions.
- `camera_snapshot_recorder.py` writes asynchronous JPEG sequences and detailed
  calibration timing evidence.
- `point_cloud_reader.py` restores supported PCD schemas as typed radar points.
- `manual_snapshot.py` appends one radar/image pair to a compatible folder.
- `paths.py` defines the current data directories and legacy JSON path lookup.
- `__init__.py` exports the main recorder, reader, and snapshot classes.

## Normal recording folders

Each selected radar group gets its own folder named
`recording_<A|B|C>_<timestamp>`. A folder may contain:

| File | Meaning |
| --- | --- |
| `point_cloud/frame_NNNNNN.pcd` | One complete cluster or object radar frame |
| `images/camera_NNNNNN.jpg` | One camera frame; its number follows the camera sequence |
| `recording.json` | Ordered radar records plus optional paired camera fields |
| `timestamps.json` | Compatibility mapping from PCD filename to radar time |

`recording.json` records these fields for each radar frame:

```json
{
  "point_cloud": "point_cloud/frame_000001.pcd",
  "recorded_at": "ISO-8601 radar time",
  "frame_type": "cluster or object",
  "camera_frame": "images/camera_000001.jpg or null",
  "camera_recorded_at": "ISO-8601 camera time or null",
  "camera_delay_ms": 87.348,
  "synchronization_error_ms": 0.0
}
```

JSON updates use a temporary file followed by replacement so readers do not
normally observe a partially written document. Metadata is flushed at most
about once per second during recording and forced at shutdown.

## PCD schemas

Cluster frames contain ID, longitudinal/lateral distance and velocity, dynamic
property, RCS, false-alarm category, ambiguity state, and invalid-state code.

Current object frames add RMS/quality values, measurement state, probability
of existence, acceleration, object class, orientation, size, and collision
region flags. `PointCloudReader` also accepts the earlier object schema that
ends after probability of existence. Missing floating values are stored as
NaN; missing integer qualities use `0xFFFFFFFF`.

## Camera-to-radar association

Camera frames and radar frames are produced independently. When the radar
recorder receives a saved camera notification, it computes:

```text
target radar time = camera captured time - configured camera delay
synchronization error = chosen radar time - target radar time
```

The camera frame is attached to the closest radar record that does not already
have a camera frame. Pending camera notifications wait until a new-enough radar
frame exists, or are resolved against available records when recording stops.

The provisional default delay is 87.348 ms. Metadata stores the actual value
used so that a recording remains interpretable if the setting changes later.

## Queue and failure behavior

Radar PCD writing uses a bounded queue of 64 frames per selected group. A full
queue becomes a recording error and stops the session; radar frames are not
silently discarded.

The JPEG writer uses a bounded queue of eight selected frames. A full queue
drops that image, increments an explicit counter, and reports a warning while
the recording continues. Invalid-timestamp rejections are counted separately.
Unusual PTS gaps are retained as candidates but are not claimed as losses.

## Manual snapshots

`ManualSnapshotWriter` creates or appends a synchronized PCD/JPEG pair using
the normal `recording.json` and `timestamps.json` format. The destination must:

- already exist;
- contain both metadata files or neither one;
- contain no unrelated JSON files.

Indexes are chosen after scanning both existing files and metadata, so gaps do
not cause older pairs to be overwritten. If writing fails, newly created PCD
and JPEG files are removed before the error is returned.

In live capture, the radar worker keeps three seconds of complete frames and
selects the one nearest the delayed camera time. A residual error over 500 ms
rejects the snapshot.

## Playback loading

`load_recording_entries()` prefers `recording.json`, supplements it from
`timestamps.json`, and finally discovers unreferenced PCD and `camera_*` image
files. It resolves both current paths such as `point_cloud/frame_000001.pcd`
and legacy bare names such as `frame_000001.pcd`; missing files are skipped,
and file modification time is the fallback when metadata has no timestamp.
Entries are sorted by effective recorded time.

Normal playback follows the original intervals, draws any PCD frame, displays
any image, and supports restart or five-second seeks. Mixed and single-modality
entries are allowed.

Snapshot playback can require paired PCD/image entries. It supports automatic
advance, pause, previous/next stepping, live filter changes, and saving the
currently displayed pair to a snapshot destination.

## Calibration-only files

A channel-4 calibration folder contains an `images/` directory with JPEGs plus:

| File | Purpose |
| --- | --- |
| `camera_timestamps.jsonl` | Append-only timing journal, one object per saved frame |
| `camera_timing_session.json` | Stable recording settings and per-restart clock anchors |
| `camera_timing_events.jsonl` | Sparse RTCP, jitter-buffer, and timing warning events |
| `camera_recording_summary.json` | Start/stop, saved/drop counts, and transport totals |
| `display_timestamps.jsonl` | Display geometry and per-marker sample/submission/flip timing, including warm-up |

The JSONL journal is the single canonical frame manifest. A compact row stores
the image path (new recordings use `images/camera_*.jpg`; legacy bare names are
also accepted), stream epoch, PTS, running time, host receipt clocks, raw and
interpreted reference timestamp, host-anchored media time, adjusted exposure
estimate, save time, and flags. Repeated values such as decoder choice,
pipeline latency, the 87.348 ms adjustment, and pipeline-zero anchors live in the
session file.

`media_unix_ns` is computed from the moment the pipeline clock was anchored to
the host plus the frame running time. `estimated_exposure_unix_ns` subtracts the
configured application adjustment exactly once. Reference NTP remains an
independent observation and does not correct either value.

The summary distinguishes confirmed frames not saved (writer overflow or
invalid timing) from unusual PTS-gap candidates. `num_lost`, `num_late`,
`num_duplicates`, and retransmission fields are aggregated RTP jitter-buffer
packet counters; per-pipeline values are retained in
`transport_stats_by_epoch` so a restart does not overwrite earlier evidence.
They must not be interpreted as decoded-frame counts.

The display journal is written by the calibration display process, separately
from camera telemetry. It starts when the fullscreen view opens, before the
three-second camera-recording delay. The camera session names this journal;
analysis refuses to silently fall back to legacy decoding if that required
file is missing. Close the QR display before analyzing, so its buffered
tail and final summary are flushed. Interrupted recordings may have a partial
last line; analysis ignores only an incomplete final JSON line, never an
internal gap or reordered display frame.

## Compatibility rules

- Keep `recording.json` and `timestamps.json` readable together; manual
  snapshots and both playback modes depend on them.
- New JSON references use `point_cloud/` and `images/`; readers also accept
  legacy root-level references and root-level files.
- Keep support for cluster, legacy-object, and current-object PCD schemas unless
  old recordings are intentionally retired.
- Do not treat the numeric suffixes of `frame_*.pcd` and `camera_*.jpg` as proof
  of synchronization. Use `recording.json`.
- Preserve raw times and the applied delay when changing synchronization logic;
  derived associations alone cannot be recalculated later.
- Historical `camera_timestamps.json` and verbose JSONL rows remain readable by
  `calibration/analysis/recording.py`; new recordings do not duplicate the
  journal into a JSON array.
