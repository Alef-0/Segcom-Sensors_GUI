from connection.connection_packages import read_201_radar_state as r201
from connection.connection_packages import read_701_cluster_list as r701
from connection.connection_packages import read_702_quality_info as r702
from connection.connection_packages import create_200_radar_configuration as c200
from connection.connection_packages import Clusters_messages

from connection.connection_communication import Can_Connection, can_data
from graph.graph_filter import Filter_graph
from graph.graph_draw import Graph_radar

from multiprocessing.connection import Connection
from multiprocess.synchronize import Event
from multiprocessing import Process, Manager, Queue
import re
import cv2 as cv

from time import sleep

def threat_201_message(channel, bytes, pool : Queue):
    MaxDistanceCfg, RadarPowerCfg, OutputTypeCfg, RCS_Treshold, SendQualityCfg, _ = r201(bytes)
    
    distance = MaxDistanceCfg * 2
    radar = ["STANDARD", "-3db TX", "-6db TX", "-9db TX"][RadarPowerCfg]
    output = ["None", "Objects", "Clusters"][OutputTypeCfg]
    rcs = ["Standard", "High Sensitivity"][RCS_Treshold]
    quality = ["No", "Ok"][SendQualityCfg]

    # Mudar para o dicionário real
    dicio = {}

    dicio[f'DISTANCE_{channel}'] = distance
    dicio[f'RPW_{channel}'] = radar
    dicio[f'OUT_{channel}'] = output
    dicio[f'RCS_{channel}'] = rcs
    dicio[f'EXT_{channel}'] = quality
    pool.put_nowait(("message_201", dicio))

def send_configuration_message(dic : dict, connection : Can_Connection, save_volatile):
    values = []
    values.append(dic['CHECK_DISTANCE'])
    values.append(int(dic["DISTANCE"] / 2) )
    values.append(dic["CHECK_RPW"])
    values.append(["STANDARD", "-3dB Tx gain", "-6dB Tx gain", "-9dB Tx gain"].index(dic['RPW']))
    values.append(dic["CHECK_OUT"])
    values.append(["NONE", "OBJECT", "CLUSTERS"].index(dic["OUT"]))
    values.append(dic["CHECK_RCS"])
    values.append(["STANDARD", "HIGH SENSITIVITY"].index(dic['RCS']))
    values.append(1)
    values.append(dic['CHECK_QUALITY'])

    data = c200(*values, save_volatile)
    print("Enviando algo")

    # Long way
    if dic['send_1'] or dic['send_all']:
        message = connection.packet_struct.pack(8, 0, 0x200, 0, data.to_bytes(8), 1)
        connection.send_message(message)
    if dic['send_2'] or dic['send_all']:
        message = connection.packet_struct.pack(8, 0, 0x200, 0, data.to_bytes(8), 2)
        connection.send_message(message)
    if dic['send_3'] or dic['send_all']:
        message = connection.packet_struct.pack(8, 0, 0x200, 0, data.to_bytes(8), 3)
        connection.send_message(message)

def create_connection_communication(initial_dict : dict, pipe : Connection, pool : Queue):
    radar_choice = [int(x[-1]) for x in initial_dict.keys() if (x.startswith("choose_") and initial_dict[x] == True)][0] # Só deveria haver 1
    
    connection = Can_Connection()
    message_collection = Clusters_messages()
    graph = Graph_radar()
    filter = Filter_graph(initial_dict)

    while True:
        sleep(0.01)
        try:
            if pipe.poll(): 
                event, values = pipe.recv()
                match event:
                    case "conn_radar":                  
                        print("Trying to change connection")
                        connection.change_connection()
                        pool.put(("change_radar", connection.connected))
                        cv.destroyAllWindows()
                    case "Send":
                        if connection.connected: send_configuration_message(values, connection, False)
                    case "save_nvm":
                        if connection.connected: send_configuration_message(values, connection, True)
                    case "choose":
                        print("Changing visualization to ", values)
                        radar_choice = values
                    case s if re.match(r"^filter", s):  
                        print("Updating Filter")
                        filter.update_values(event, values)
                    case _: pass
        except KeyboardInterrupt:
            # Serve para o cntrl C não interromper tudo
            break
        
        # Working normally
        if (not connection.connected): continue
        connection.read_chunk()
        while (connection.can_create_can()):
            message  = connection.create_package()
            print(message)
            # continue
        # Tratar da Conexão
            if message.canId == 0x201: threat_201_message(message.canChannel, message.canData, pool)
            if message.canChannel != radar_choice: continue 
            match message.canId:
                # case 0x201: threat_201_message(message.canChannel, message.canData, config) # Deactivated for filtering id = 2        
                case 0x600: 
                    # print(message_collection.dyn, len(message_collection.dyn))
                    x, y, colors = filter.filter_points(message_collection)
                    graph.show_points(x,y, colors) 
                    message_collection.clear()
                    # if teste_id_701 and teste_id_702: print(max(teste_id_701), max(teste_id_702))
                    # teste_id_701.clear(); teste_id_702.clear()
                case 0x701:
                    things = r701(message.canData); #teste_id_701.append(things[0])
                    message_collection.fill_701(things)
                case 0x702:
                    things = r702(message.canData); #teste_id_702.append(things[0])
                    message_collection.fill_702(things)