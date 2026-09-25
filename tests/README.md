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
| `processing/recording/test_changes.py` | Camera/radar pairing, metadata, frame-rate selection, and playback loading |
| `processing/recording/test_manual_snapshot.py` | Snapshot folder validation, indexes, metadata, and cleanup after failure |
| `processing/playback/test_snapshot_playback.py` | Paired-entry filtering, stepping, rendering controls, and copy-current-pair behavior |
| `camera/test_pipeline_policy.py` | Decoder choice, pipeline structure, host-anchored PTS, reference clocks, transport counters, and capture callbacks |
| `camera/test_recording_restart.py` | Recording restart behavior and pipeline/decoder state reuse |
| `processing/visualization/test_radar_camera_transposition.py` | Radar-to-camera projection and transposition behavior |

## Category folders

- `camera/` — camera pipeline and recording restart behavior
- `processing/` — recording, playback, and visualization behavior
- `messaging/` — packet decoding, message compatibility, and object updates

## What the suite does not prove

The tests primarily use temporary folders, fake point clouds, synthetic CAN
payloads, fake GStreamer samples, mocked clocks, and fake GUI/process objects.
Passing tests do not establish:

- access to the real radar gateway or DVR;
- CAN bus bit timing or hardware filtering;
- native GStreamer element availability or RTSP network latency;
- physical camera rolling-shutter exposure;
- GPS serial port access;
- physical display appearance.
