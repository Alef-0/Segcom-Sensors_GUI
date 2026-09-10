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

Only backends whose required GStreamer elements exist are attempted. A stream
that cannot produce a first frame within five seconds is retried, then moves to
the next backend when available. After three failed pipeline attempts, the
camera is reported closed.

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
anchors pipeline running time to host realtime while a monotonic clock tracks
host-clock movement.

The resulting `captured_at` always comes from the fixed host anchor plus frame
running time. A frame reference timestamp is converted only when its caps
explicitly identify an NTP or Unix clock. It is retained as diagnostic evidence
and never moves the PTS-derived image time. Unknown reference clocks remain raw
and are flagged instead of being guessed.

Each calibration row keeps the independent observations needed to recompute
the relationship: PTS, segment running time, host Unix and monotonic receipt,
declared reference-clock data, PTS-derived media time, and flags. Epoch anchors
are stored once per pipeline restart in the session file rather than repeated
in every row.

Invalid PTS frames are not recorded. A PTS interval above 1.75 nominal frame
periods is reported as an unusual-gap candidate, not as confirmed frame loss.
This avoids misclassifying the observed alternating 20/40 ms cadence and its
occasional approximately 50 ms interval. Confirmed transport evidence comes
from jitter-buffer counters and messages.

## Latency values

Two settings have different purposes:

- Pipeline latency, default 145 ms, configures the GStreamer RTSP jitter
  buffer and causes a pipeline restart when changed while connected.
- Latency adjustment, default 109 ms, is an application-level camera-to-radar
  alignment offset used by recording, snapshots, playback snapshots, and
  calibration metadata.

Changing one does not automatically derive the other. The adjustment must be
validated from a calibration recording for the current stream session and
hardware path.

## Recording and loss reporting

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
