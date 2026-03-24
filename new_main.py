from menu_configurations import Configurations
import FreeSimpleGUI as sg
import signal
import zenoh

from multiprocessing import Process
from gps.gps_executor import gps_main
import multiprocessing as mp


loop = True

def signal_handler(sig, frame):
    print("Finalizing menu by interruption")
    global loop
    loop = False

def deal_answers(x : zenoh.Sample, config : Configurations):
    topic, values = str(x.key_expr), x.payload.to_string()
    print("NEW COMMUNICATION", topic, values)        
    if topic.endswith("conn_gps"): config.change_connection_gps(values == "True")
    elif topic.endswith("gps_text"): config.window["gps_text"].update(values)


def main():
    # mp.set_start_method('spawn')
    signal.signal(signal.SIGINT, signal_handler)

    font = ("Helvetica", 12)
    sg.set_options(font=font)

    config = Configurations()
    event, values = config.read()

    gps_process = Process(target=gps_main, daemon=True)
    gps_process.start()

    pub_sub = zenoh.open(zenoh.Config())
    sub = pub_sub.declare_subscriber("GUI/MAIN/**", lambda sample: deal_answers(sample, config))
    global loop; 
    
    while loop:
        # print(gps_process.is_alive())
        event, values = config.read()
        if event == sg.WINDOW_CLOSED: loop = False; break

        # Parte do Menu
        if event.startswith("conn_"):
            if event.endswith("gps"): 
                pub_sub.put("GUI/GPS", event)
        elif event == "gps_maps": pub_sub.put("GUI/GPS", event)
        if event == sg.TIMEOUT_EVENT: pass
        

if __name__ == "__main__": main()