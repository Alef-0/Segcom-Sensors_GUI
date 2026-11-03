# Test suite map

The test suite is organized by behavior rather than mirroring every source
module. Run it from the repository root with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Disabling automatic plugin loading prevents unrelated system-wide pytest
plugins from affecting this project.

## Areas covered

| Test file | Main coverage |
| --- | --- |
| `test_connection_packages.py` | ARS40X packet extraction, scaling, merge behavior, and configuration bit layout |
| `test_message_module_split.py` | Compatibility exports after cluster/object message separation |
| `test_object_message_update.py` | Extended object fields, missing optional values, and object filter behavior |
| `test_graph_filter.py` | Dynamic, quality, ambiguity, invalid-state, and RCS filtering |
| `test_point_cloud_recorder.py` | Cluster/object PCD schemas, writer behavior, and recording sessions |
| `test_object_recording_update.py` | Extended object field serialization into and deserialization from PCD files |
| `test_radar_camera_transposition.py` | Radar Cartesian 3D to camera pixel 2D projection and overlay bounding |
| `test_recording_changes.py` | Camera/radar pairing, metadata, frame-rate selection, and playback loading |
| `test_manual_snapshot.py` | Snapshot folder validation, indexes, metadata, and cleanup after failure |
| `test_snapshot_playback.py` | Paired-entry filtering, stepping, rendering controls, and copy-current-pair behavior |
| `test_camera_pipeline_policy.py` | Decoder choice, pipeline structure, host-anchored PTS, reference clocks, transport counters, and capture callbacks |

## What the suite does not prove

The tests primarily use temporary folders, fake point clouds, synthetic CAN
payloads, fake GStreamer samples, mocked clocks, and fake GUI/process objects.
Passing tests do not establish:

- access to the real radar gateway or DVR;
- correct physical A/B/C channel placement;
- actual GStreamer plugin or hardware-decoder availability;
- the negotiated RTSP transport or real packet loss;
- sustained 30 FPS capture and JPEG writing;
- the correctness of the operational camera-delay value for a new session;
- clean interaction among the real FreeSimpleGUI, OpenCV, GTK, and Qt/OpenGL
  windows.

Those behaviors require separate checks on the target installation. UI checks
should be completed by the project owner or from screenshots they deliberately
provide.
