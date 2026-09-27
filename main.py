import argparse
from multiprocessing import get_context
import signal
import FreeSimpleGUI as sg

from menu_controls import MenuControls
from menu_functions import WorkerPipes, RuntimeState, run_event_loop, shutdown_workers
from processing.playback.playback import playback_main
from processing.playback.snapshot_playback import snapshot_playback_main
from sensors.camera.camera_gstreamer import gstreamer_main
from sensors.gps.gps_connection import main as gps_main
from sensors.radar.connection_main import create_connection_communication


def parse_arguments():
    """Parse CLI options for the SEGCOM sensor interface."""
    parser = argparse.ArgumentParser(description="SEGCOM Sensors GUI Application")
    parser.add_argument("--font-size", type=int, default=12, help="UI font size (default: 12)")
    parser.add_argument("--font-family", type=str, default="Helvetica", help="UI font family (default: Helvetica)")
    return parser.parse_args()


def main(args=None):
    """Spawn worker processes and run the GUI event loop."""
    args = parse_arguments() if args is None else args
    sg.set_options(font=(args.font_family, args.font_size))
    ctx = get_context("spawn")
    shutdown_event = ctx.Event()

    signal.signal(signal.SIGINT, lambda _s, _f: shutdown_event.set())
    signal.signal(signal.SIGTERM, lambda _s, _f: shutdown_event.set())

    all_queue, trans_channel = ctx.Queue(128), ctx.Queue(1)
    rx_radar, tx_radar = ctx.Pipe()
    rx_cam, tx_cam = ctx.Pipe()
    rx_gps, tx_gps = ctx.Pipe()
    rx_pb, tx_pb = ctx.Pipe()
    rx_snp, tx_snp = ctx.Pipe()
    pipes = WorkerPipes(radar=tx_radar, cam=tx_cam, gps=tx_gps, playback=tx_pb, snapshot=tx_snp)

    controls = MenuControls(font=(args.font_family, args.font_size))
    _, values = controls.read()
    runtime = RuntimeState(process_context=ctx)

    processes = [
        ctx.Process(target=create_connection_communication, args=(values, rx_radar, all_queue, shutdown_event, trans_channel)),
        ctx.Process(target=gstreamer_main, args=(rx_cam, all_queue, shutdown_event, trans_channel)),
        ctx.Process(target=gps_main, args=(rx_gps, all_queue, shutdown_event)),
        ctx.Process(target=playback_main, args=(rx_pb, all_queue, shutdown_event, values)),
        ctx.Process(target=snapshot_playback_main, args=(rx_snp, all_queue, shutdown_event, values)),
    ]
    for proc in processes:
        proc.start()

    try:
        run_event_loop(controls, all_queue, runtime, pipes, shutdown_event)
    finally:
        try:
            shutdown_workers(processes, pipes, controls, shutdown_event)
        finally:
            trans_channel.close()


if __name__ == "__main__":
    main()
