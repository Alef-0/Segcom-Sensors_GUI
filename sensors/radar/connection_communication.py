import socket
import struct
import time

CAN_DATA_FMT = "<BBIQ8sB"  # little-endian: dlc, flags, can_id, timestamp, payload, channel
CAN_DATA_SIZE = struct.calcsize(CAN_DATA_FMT)  # 23 bytes
GATEWAY_HOST = "192.168.1.101"
GATEWAY_PORT = 2323


class Can_Connection:
    """TCP transport connection to the CAN-Ethernet gateway."""
    CANData_fmt = CAN_DATA_FMT
    CANData_size = CAN_DATA_SIZE

    def __init__(self):
        self.connected = False
        self.sock = None
        self.data = b""
        self.packet_struct = struct.Struct(self.CANData_fmt)

    def change_connection(self):
        """Toggle TCP connection to the radar CAN gateway."""
        if self.connected:
            self.connected = False
            self.data = b""
            if self.sock:
                self.sock.close()
                self.sock = None
            return

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((GATEWAY_HOST, GATEWAY_PORT))
            sock.setblocking(False)
            self.sock = sock
            self.connected = True
        except socket.error as exc:
            print(f"Socket error: {exc}; Connection failed")

    def read_chunk(self, max_bytes: int = 64000):
        """Read pending stream bytes into internal buffer."""
        if not self.sock:
            return
        try:
            chunk = self.sock.recv(max_bytes)
            if chunk:
                self.data += chunk
        except (BlockingIOError, socket.error):
            pass

    def can_create_can(self) -> bool:
        """Return True if at least one complete 23-byte CAN frame is buffered."""
        return len(self.data) >= self.CANData_size

    def create_package(self):
        """Consume and decode one 23-byte CAN packet from the buffer."""
        new_can = self.data[:self.CANData_size]
        self.data = self.data[self.CANData_size:]
        return can_data(new_can)

    def send_message(self, raw: bytes):
        """Send raw CAN packet to gateway and pause for hardware registration."""
        if self.sock:
            self.sock.send(raw)
            time.sleep(0.5)  # Required for Vector interface to register packet


class can_data:
    """Decoded 23-byte CAN-Ethernet gateway record."""
    CANData_fmt = CAN_DATA_FMT
    CANData_size = CAN_DATA_SIZE

    def __init__(self, raw: bytes):
        self.raw = raw
        if len(raw) >= self.CANData_size:
            (
                self.dlc, self.flags, self.canId,
                self.timestamp, self.canData, self.canChannel,
            ) = struct.unpack(self.CANData_fmt, raw[:self.CANData_size])
        else:
            self.dlc = self.flags = self.canId = self.timestamp = self.canChannel = 0
            self.canData = b""

    def __repr__(self) -> str:
        return f"{hex(self.canId)}, in {self.timestamp / 1e9:0.6f} from channel {self.canChannel}: {self.canData}"


CanConnection = Can_Connection
CanData = can_data