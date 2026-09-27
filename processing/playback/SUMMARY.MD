# Processing Playback Package Summary

This package manages loading, navigation, and playback of recorded multi-sensor datasets and manual snapshots.

## Modules

### [playback.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/playback.py)
Loads radar PCD and camera recording sessions from disk and provides a stateful playback engine.
- [load_recording_entries](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/playback.py#L42): Discovers recording files, reads metadata indices (`recording.json`, `timestamps.json`), and sorts timestamped entries.
- [PlaybackController](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/playback.py#L90): Controls step-by-step and timed real-time playback of synchronized camera and radar frames.

### [snapshot_playback.py](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/snapshot_playback.py)
Provides interactive navigation and review for saved manual snapshot pairs.
- [SnapshotPlaybackController](file:///media/alef/Everything/SEGCOM/Segcom-Sensors_GUI/processing/playback/snapshot_playback.py#L22): Discovers saved snapshot pairs, renders paired frames, and facilitates single-shot manual snapshot captures.
