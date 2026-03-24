from menu_configurations import Configurations
import FreeSimpleGUI as sg
import signal
import zenoh

from multiprocessing import Process
from gps.gps_executor import gps_main
from camera.camera_executor import camera_main
import multiprocessing as mp



def deal_answers(x : zenoh.Sample, config : Configurations):
    topic, values = str(x.key_expr), x.payload.to_string()
    print("NEW COMMUNICATION", topic, values)        
    
    if      topic.endswith("conn_gps"): config.change_connection_gps(values == "True")
    elif    topic.endswith("gps_text"): config.window["gps_text"].update(values)

def main():
    # It needs to start everything in the beggining
    gps_process = Process(target=gps_main, daemon=True)
    gps_process.start()
    camera_process = Process(target=camera_main, daemon=True)
    camera_process.start()
    
    # mp.set_start_method('spawn')
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)

    font = ("Helvetica", 12)
    sg.set_options(font=font)

    config = Configurations()
    event, values = config.read()

    pub_sub = zenoh.open(zenoh.Config())
    sub = pub_sub.declare_subscriber("GUI/MAIN/**", lambda sample: deal_answers(sample, config))
    loop = True
    
    while loop:
        # print(gps_process.is_alive())
        event, values = config.read()
        if event == sg.WINDOW_CLOSED: loop = False; break

        # Parte do Menu
        if event.startswith("conn_"):
            if      event.endswith("gps"): pub_sub.put("GUI/GPS", event)
            elif    event.endswith("cam"): pub_sub.put("GUI/CAMERA", event)
        elif event == "gps_maps": pub_sub.put("GUI/GPS", event)
        if event == sg.TIMEOUT_EVENT: pass
    
    # print("LEFT EVERYTHING BEHIND")
    pub_sub.close()
    config.window.close()
        

if __name__ == "__main__": main()