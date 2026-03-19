import struct
import numpy as np
import socket
import time

g_CANData_fmt = "<BBIQ8sB"  # little-endian, packed
g_CANData_size = struct.calcsize(g_CANData_fmt)  # 23 bytes


class Can_Connection:
    CANData_fmt =   g_CANData_fmt
    CANData_size =  g_CANData_size
    
    def __init__(self):        
        self.connected = False
        self.sock = None
        self.data = b""
        self.packet_struct = struct.Struct(self.CANData_fmt)
        
    def change_connection(self):
        print("Changing Connection State")
        if self.connected:
            self.connected = False
            self.data = b""
            self.sock.close()
            self.sock = None
            return 
        
        while True:
            try:
                # Create TCP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                # Connect to server
                sock.connect(("192.168.1.101", 2323))
                # At this point, the connection is established and you can use sock.send / sock.recv
                print("Connected to server")
                self.sock = sock
                break
            except socket.error as e:
                print(f"Socket error: {e}; Connection failed")
                return
                
        self.connected = True
        self.sock.setblocking(False)
        self.sock.settimeout(0)


    def read_chunk(self, max = 64000):
        if not self.sock: print("Not connected?")
        try:
            chunk = self.sock.recv(max)
            if not chunk: print("Can't read anything, something went wrong")
            self.data += chunk
        except BlockingIOError: 
            pass

    def can_create_can(self): return len(self.data) >= self.CANData_size

    def create_package(self):
        new_can = self.data[:self.CANData_size]
        self.data = self.data[self.CANData_size:]
        return can_data(new_can)

    def send_message(self, raw : bytes): 
        sent = self.sock.send(raw)
        time.sleep(0.5) # necessário para a vector registar o pacote
        
        

class can_data:
    CANData_fmt =  g_CANData_fmt
    CANData_size = g_CANData_size

    def __init__(self,raw: bytes):
        if len(raw) < self.CANData_size: return
        self.raw =raw
        self.dlc, self.flags, self.canId,self.timestamp, self.canData, self.canChannel = struct.unpack(self.CANData_fmt, raw[:self.CANData_size])

    def __repr__(self):
        return f"""{hex(self.canId)}, in {self.timestamp / 1e9:0.6f} from channel {self.canChannel}: {self.canData}"""