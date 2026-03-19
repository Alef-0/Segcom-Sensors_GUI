import socket

sock = None

try:
    # Create TCP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    # Connect to server
    sock.connect(("192.168.1.101", 2323))
    # At this point, the connection is established and you can use sock.send / sock.recv
    print("Connected to server")
except Exception as e: print(e)

if sock:
    while True:
        try:
            chunk = sock.recv(1000)
            if not chunk: print("Can't read anything, something went wrong")
            print(chunk)
        except BlockingIOError: 
            pass