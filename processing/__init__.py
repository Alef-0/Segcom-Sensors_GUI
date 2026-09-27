"""Processing package exposing recording, playback, and visualization services."""

def __getattr__(name: str):
    if name in ("PlaybackController", "SnapshotPlaybackController", "PlaybackEntry", "load_recording_entries", "load_snapshot_entries"):
        import sensors.playback as _sp
        return getattr(_sp, name)
    if name in ("CameraRecorder", "CameraSnapshotRecorder", "RadarRecorder", "RadarRecordingSession"):
        import sensors.recording as _sr
        return getattr(_sr, name)
    if name in ("SnapshotWriter", "ManualSnapshotWriter"):
        import sensors.snapshot as _ss
        return getattr(_ss, name)
    if name == "PointCloudReader":
        from sensors.auxiliary import PointCloudReader
        return PointCloudReader
    raise AttributeError(f"module 'processing' has no attribute '{name}'")

__all__ = [
    "CameraRecorder",
    "CameraSnapshotRecorder",
    "ManualSnapshotWriter",
    "PlaybackController",
    "PlaybackEntry",
    "PointCloudReader",
    "RadarRecorder",
    "RadarRecordingSession",
    "SnapshotPlaybackController",
    "SnapshotWriter",
    "load_recording_entries",
    "load_snapshot_entries",
]

