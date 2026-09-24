# Camera subsystem

The camera subsystem connects to one DVR RTSP channel at a time, displays the
latest decoded image, produces timestamped full-resolution frames for
recording, and temporarily changes channels for manual snapshots or
calibration.

## Files

- `camera_gstreamer.py` owns connection state, GStreamer lifecycle, display,
  recording commands, manual snapshots, retry behavior, and GUI status events.
- `camera_pipeline.py` selects a decoder and builds the pipeline string.
- `camera_timebase.py` validates PTS and maps it onto host Unix/monotonic time.
- `camera_reference_clock.py` observes declared frame reference clocks, RTCP
  sender reports, and RTP jitter-buffer counters.
- `camera_pipeline_policy.py` is a compatibility facade for older imports.

## Channel behavior

Channels 1-3 correspond to radar/camera groups A-C. The selected live group
starts as channel 2. A manual snapshot can restart the pipeline on another
group, capture one frame, and then restore the previous channel. Calibration
mode moves to channel 4 and closes it when calibration ends.

The RTSP URL and credentials are currently constructed directly in
`GStreamerPipeline.create_url()`.

## Decoder selection

The pipeline policy recognizes three backends:

| Name | Main decoder path | Intended platform |
| --- | --- | --- |
| `rtx` | `nvh264dec` plus CUDA conversion/download | Desktop NVIDIA GPU |
| `orin` | `nvv4l2decoder` plus `nvvidconv` | NVIDIA Jetson |
| `cpu` | `avdec_h264` plus `videoconvert` | Software fallback |

Only backends whose required GStreamer elements exist are attempted. Each
backend gets a five-second first-frame timeout. If an attempt produces no frame,
the camera immediately advances to the next available backend. After the final
backend, three failed attempts close the camera.

Set `SEGCOM_CAMERA_DECODER=rtx`, `orin`, or `cpu` to request a path explicitly.
Hardware requests still fall back to CPU when their elements are missing.
The explicit pipeline selected in the Calibration tab is stricter: missing
plugins or a pipeline construction/startup failure is shown as an error instead
of falling back to a different decoder. The ARM/Jetson option is rejected on a
computer that is not detected as an NVIDIA Jetson, even if similarly named
plugins happen to be installed.

## Pipeline shape

The source pipeline is conceptually:

```text
rtspsrc -> H.264 depay/parse -> selected decoder -> BGR -> tee
                                                        ├-> latest-frame display
                                                        └-> full-resolution capture
```

The display branch has a one-frame leaky queue and a one-frame dropping
appsink. The capture branch has a bounded 30-buffer queue and a non-dropping
appsink. After that appsink, `CameraSnapshotRecorder` has a separate eight-item
writer queue. This keeps the live view responsive while making recording loss
observable and preventing unbounded memory growth.

The current `rtspsrc` options are:

- `latency=<configured milliseconds>`, default 145;
- `protocols=tcp+udp`, allowing transport negotiation;
- `buffer-mode=1`, the sender-clock-slave mode;
- `do-retransmission=true`;
- RTCP, drop-on-latency, and reference timestamp metadata when those properties
  exist in the installed plugin.

These settings do not demonstrate which negotiated transport was selected or
whether TCP retransmissions occurred. That requires runtime GStreamer logs,
packet capture, or operating-system network telemetry.

## Timestamp policy

Every capture sample must have a valid, strictly increasing PTS that can be
mapped through its segment to pipeline running time. The first usable frame
anchors pipeline running time to host realtime and monotonic time. The pipeline
clock read is bracketed by two host-monotonic samples; the half-window is stored
as mapping-sample uncertainty. A clock, base-time, or segment change starts a
new mapping/segment revision instead of silently reusing the prior history.
The policy retains the clock object and compares its native identity; a new
Python wrapper for the same clock does not reset the anchor or frame history.

The capture appsink callback records `application_arrival_*` before pulling the
sample or converting it to an image. Separate pull-complete, conversion-complete,
and timestamp-policy fields expose work added after that boundary. The legacy
`received_*` fields retain their old post-conversion meaning in schema version 3.
Capture-queue occupancy is a diagnostic; application arrival is not described
as network arrival.

The resulting legacy `captured_at` still comes from the fixed host anchor plus
segment-mapped frame running time, so downstream code continues to subtract the
configured correction exactly once. The timing journal additionally carries an
observation-only estimated capture time and `estimated_arrival_delay = A - (M-c)`.
Its uncertainty covers only the host/pipeline clock sampling window; camera
processing, display scanout, and the physical exposure reference remain
unverified. A frame reference timestamp is converted only when its caps
explicitly identify an NTP or Unix clock. It is retained as diagnostic evidence
and never moves the segment-derived image time. Unknown reference clocks remain raw
and are flagged instead of being guessed.

Each saved camera row keeps the independent observations needed to recompute the
relationship: raw PTS, segment running time, the early application-arrival
boundary, later processing timestamps, queue occupancy, declared reference-clock
data, mapped media time, provisional estimate, and flags. Mapping anchors are
stored by pipeline restart, mapping revision, and segment revision in the session
file rather than repeated in every row.

Invalid-timing frames and writer-queue drops are not saved as images, but their
reason and available timing fields are retained in `camera_timing_events.jsonl`
for both normal and calibration recordings.
A mapped running-time interval above 1.75 nominal frame periods is reported as
an unusual-gap candidate, not as confirmed frame loss.
This avoids misclassifying the observed alternating 20/40 ms cadence and its
occasional approximately 50 ms interval. Confirmed transport evidence comes
from jitter-buffer counters and messages.

## Latency values

Two settings have different purposes:

- Pipeline latency, default 145 ms, configures the GStreamer RTSP jitter
  buffer and causes a pipeline restart when changed while connected.
- Latency adjustment, provisional default 87.348 ms, is an application-level camera-to-radar
  alignment offset used by recording, snapshots, playback snapshots, and
  calibration metadata.

Changing one does not automatically derive the other. The adjustment must be
validated from a calibration recording for the current stream session and
hardware path.

## Recording and loss reporting

Starting a recording attaches the image writer to the running RTSP pipeline.
The preview, decoder reference frames, stream epoch, and timing anchor continue
across recording boundaries. The calibration session records
`pipeline_restarted_for_recording: false` and the current `stream_epoch_at_start`.
Reconnects and camera configuration changes still rebuild the pipeline.

The camera recorder can select any integer number of frames from each nominal
set of 30. Selection uses an accumulator, so rates lower than 30 are spread
through the incoming sequence.

Three distinct conditions are retained:

- unusual PTS-gap candidates, which are diagnostic and not counted as loss;
- frames rejected because timing was invalid;
- selected frames dropped because the JPEG writer queue was full.

RTCP sender reports are written as sparse events. Jitter-buffer packet loss,
late packets, duplicates, and retransmission counters are aggregated into the
final calibration summary. These are RTP packet counters, not decoded-frame
counts. Warnings are rate-limited for the GUI. See
`../../processing/recording/README.md` for the output contract.

## Hardware verification boundary

Tests exercise decoder ordering, pipeline text, timestamp math, retry paths,
and callbacks with fakes. They do not verify that the installed GStreamer
plugins negotiate the expected DVR transport, that hardware decoding works,
or that the live stream sustains 30 FPS without loss.
