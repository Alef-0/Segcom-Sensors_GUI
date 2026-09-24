# Frozen camera clock offset

This protocol calibrates one candidate media-to-QR offset and evaluates it unchanged on independent camera streams. The result is software timing evidence; it does not measure sensor exposure or establish physical accuracy.

## Timing model

For each camera frame, map PTS through its `GstSegment` to running time `R`. The camera journal's pipeline clock anchor gives `M = pipeline_zero_monotonic_ns + R`. A frozen offset `c0` predicts `T = M - c0`. Application callback arrival is reported separately and is not used as the media timestamp.

The QR journal provides a conditional presentation interval rather than an exact exposure time. Calibration uses interval midpoints from readable, journal-matched evidence. QR values do not enter the runtime prediction formula.

## Collect and calibrate

1. Record camera, decoder, display, QR grid, and timing settings. Keep laboratory and evaluation recordings on separate stream sessions; include pipeline-base identity in the journals.
2. Collect continuous laboratory recordings across explicit stream restarts. Keep startup frames and cold or warm reconnect status visible in the source data.
3. Choose the minimum stream age and initial-frame exclusion before inspecting QR results. Defaults are 30 seconds and 100 frames per recording. These are experimental cutoffs, not proof of camera stabilization.
4. Calibrate from lab analysis folders. The tool forms a median interval-midpoint offset per independent stream, then takes the median of those stream offsets. It refuses duplicate journals and will not overwrite the output artifact.

```bash
python3 analyze_pts_anchor.py calibrate \
  /path/to/lab_1_analysis /path/to/lab_2_analysis /path/to/lab_3_analysis \
  --minimum-stream-age-seconds 30 --exclude-initial-frames 100 \
  --output recordings/anchor-offset-v2.json
```

## Evaluate the frozen candidate

Use held-out sessions from separate stream identities. The primary report scores every saved camera frame that has a defensible QR interval. It reports missing or unscorable frames and applies strict per-session goals: maximum interval error `<10 ms` and median `<5 ms`.

```bash
python3 analyze_pts_anchor.py evaluate \
  /path/to/test_1_analysis /path/to/test_2_analysis \
  --offset-file recordings/anchor-offset-v2.json \
  --output-directory recordings/anchor-evaluation-v2
```

The report's optional filtered score excludes the first configured camera frames and test frames whose QR-derived offset midpoint falls outside the training range. Because that selection uses test QR values, it is diagnostic only; the all-frame score remains primary. A passing software score does not establish physical exposure accuracy or deployment readiness.
