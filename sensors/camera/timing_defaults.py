"""Shared camera timestamp-correction defaults."""

# Provisional interval-aware correction derived from recordings/01_calibration.
# Revalidate it on an independent recording before treating it as final.
DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS = 87.348
DEFAULT_CAMERA_CALIBRATION_VERSION = "provisional-fixed-87.348ms-v1"
DEFAULT_CAMERA_TIMESTAMP_CORRECTION_SECONDS = (
    DEFAULT_CAMERA_TIMESTAMP_CORRECTION_MS / 1000.0
)
