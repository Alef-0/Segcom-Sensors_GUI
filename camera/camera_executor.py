import gi
import numpy as np
import cv2 as cv
import signal

# gi.require_version('Gst', '1.0')
# gi.require_version('GstApp', '1.0')
# from gi.repository import Gst, GstApp, GLib

# Initialize GStreamer
loop = True

import socket
import zenoh    
import time

def tcp_ping(host: str, port: int, timeout: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def signal_handler(sig, frame):
    print("Terminating Gstreamer Camera")
    global loop; loop = False

def threat_message(sample : zenoh.Sample, session : zenoh.Session):
    load = sample.payload.to_string(); topic = sample.key_expr
    print("RECEIVED SOMETHING", topic, load)

def camera_main():
    print("Camera Started")
    
    pub_sub = zenoh.open(zenoh.Config())
    sub = pub_sub.declare_subscriber("GUI/CAMERA", lambda x: threat_message(x, pub_sub))
    
    global loop
    try:
        while loop:
            pass
            print("STILL ON LOOP")
            time.sleep(1)
    except: pass
        

if __name__ == "__main__": 
    camera_main()