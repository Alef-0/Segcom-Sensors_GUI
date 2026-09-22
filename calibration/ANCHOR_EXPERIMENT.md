# One clock anchor and one frozen offset

## What this experiment tests

For frame i, use:

    M_i = X_b + (R_i - X_a)
    T_i = M_i - c0

R_i is buffer PTS mapped through its GstSegment into pipeline running time.
X_a and X_b are paired readings of pipeline running time and host monotonic
time. The stored pipeline_zero_monotonic_ns is X_b - X_a. Clock sampling brackets
the pipeline read with monotonic reads, uses their midpoint and records half
the bracket width as sampling uncertainty. The anchor is frozen until a genuine
clock/base-time change; recording start does not recreate the pipeline.

c0 is a distinct media-to-visual-reference correction in nanoseconds. A clock
pair does not reveal this camera/DVR/PTS offset. It must be measured separately.
The runtime formula needs no QR, arrival-based fit, cadence model or training
history. Laboratory QR calibration is still calibration, not a calibration-free
method. The current live correction remains provisional and is not replaced
automatically by this experiment.

## Finding c0 before the test

1. Fix the camera/DVR exposure and processing settings, stream configuration,
   decoder, display rate, grid, visible count and timestamp mode. Record their
   values. Use the same settings in laboratory and evaluation runs.
2. Assign complete independent stream sessions to laboratory or test roles
   before looking at results. Multiple recordings from one running pipeline
   stay together. Record camera/device power-on age separately; stream age does
   not establish device warm-up age.
3. Collect continuous laboratory recordings from as close to stream startup as
   practical through at least 90 seconds. Repeat over at least three explicit
   stream restarts and separate camera/DVR cold starts from warm reconnects.
   Keep the QR display running for the full record, with no gaps between clips.
4. Initially predeclare a 30-second minimum stream age for fitting c0. This is
   an experimental cutoff, not evidence that stabilization takes 30 seconds.
   Keep the earlier frames for startup diagnostics. Missing stream-age telemetry
   cannot silently substitute a recording-relative age.
5. For each eligible laboratory frame, the display interval [S_n, S_(n+1)]
   yields a correction interval [M_i-S_(n+1), M_i-S_n]. Take its midpoint,
   then the median across all eligible frames of each independent stream group.
   Take the median of those stream medians, giving independent streams equal
   weight. Identity, geometry, stale-image and missing-evidence checks are kept;
   residual size is never an exclusion rule.
6. Save a versioned offset artifact before analyzing held-out streams. It records
   c0, the age cutoff, source hashes, stream keys, per-recording distributions,
   and per-stream medians. The tool refuses to overwrite it. A single laboratory
   stream can create a candidate, but cannot establish generalization.

```bash
python3 analyze_pts_anchor.py calibrate \
  /path/to/lab_1_analysis /path/to/lab_2_analysis /path/to/lab_3_analysis \
  --minimum-stream-age-seconds 30 --output recordings/anchor-offset-v1.json

python3 analyze_pts_anchor.py evaluate \
  /path/to/test_1_analysis /path/to/test_2_analysis \
  --offset-file recordings/anchor-offset-v1.json \
  --output-directory recordings/anchor-evaluation-v1
```

The evaluator rejects shared calibration/test stream identities and duplicate
camera journals. Unknown stream identity or missing pipeline-base identity also
blocks evaluation. This enforces
the recorded pipeline boundary, not independent physical camera conditions;
the operator must document device restarts and unchanged settings. All test
frames, including startup frames, are evaluated. No initial test QR or test
prefix changes c0, and no live setting is written.

## Next steps and decision rule

1. Inspect the continuous laboratory offset trace in predeclared 0-10, 10-30,
   30-60 and 60-90 second windows. Compare per-window and per-stream medians,
   P95 and ranges; distinguish gradual settling from abrupt changes. If the
   cutoff or estimator changes, version the protocol and collect fresh test
   sessions instead of tuning against the existing holdout.
2. Freeze c0 with the command above. Collect at least three held-out stream
   sessions on another run/day with the same settings. Keep both cold-start
   and warm-reconnect results visible; do not pool away a failing session.
3. Score every saved frame for which a defensible conditional QR interval exists.
   Require maximum distance outside the interval <10 ms and median <5 ms in
   each test session. Report P95, inside-interval percentage, failures and
   unscorable/missing frames separately. Missing evidence is not success.
4. Investigate extremes using saved evidence, stream age, clock mapping
   differences and recording drop/rejection counts. Review source camera images
   manually when needed. Do not silently discard an outlier because it disagrees
   with the fixed correction. Software presentation returns do not measure
   physical scanout, photons or sensor exposure.
5. If offset changes within a stream or independent sessions require incompatible
   offsets, reject the universal constant for those conditions. Investigate
   source timestamp semantics and independently anchored capture timing; do not
   hide that failure with per-test QR recalibration. Only after passing fresh
   validation should c0 replace the existing configured subtraction once.

Earlier exploratory results used each recording's own initial QR window
(roughly 108, 79, 85 and 85 ms). Those are not deployment offsets. Two recordings
from one stream differed by about 30 ms, with a recording gap between them;
the new continuous-startup experiment must explain that difference. Reusing
these already examined recordings can check software, not provide prospective
validation.

The implementation replay check calibrated only `calibration_qr_2remain_2_analysis`
with the 30-second cutoff, producing 78.890893 ms. Applied unchanged to
`4remain_1` and `4remain_2`, it scored 485/488 and 478/482 frames, with medians
1.084/1.198 ms, P95 17.018/18.112 ms, and maxima 135.215/28.606 ms. Both fail
the strict maximum goal. Local artifacts are under
`recordings/anchor_simplification_check/`; they are exploratory software checks,
not a validated deployment profile.

## Simplification scope

The branch was created at 3684396. The experiment retains its useful capture,
segment/clock mapping, telemetry, QR display/decoding and evidence safeguards.
It preserves the subsequent recording-start and native-clock-identity fixes.
Display preparation, bounded decoding and scheduling support remain as
measurement infrastructure. Unrelated recording compatibility changes remain.

The integrated history/cadence model fitting, model selection, distance/cross-
recording estimators and model-transfer/prefix-recalibration scripts are retired.
final_analysis.py now only reconstructs evidence and scores intervals. The
pre-branch standalone quantitative tool is restored to its original sections
and is outside this experiment. Historical reports remain untouched.

The normal viewer now exports evidence without fitting anything. Supplying
--offset-file explicitly runs the independent anchor evaluation after a fresh
decode. Saved-result review still does not rewrite results. The protocol adds
no automatic live correction or claimed physical accuracy.
