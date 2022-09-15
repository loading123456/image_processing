import asyncio
from email import message
import socket
import sys
import os

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the port
serverAddress = ('localhost', 2021)
sock.bind(serverAddress)


print('Starting up on')


# Listen for incoming connections

while True:
    # Wait for a connection
    print('Waiting for a connection')
    data, address = sock.recvfrom(4096)

    if data:
        if data.decode() == '1.png':
            print('Sending data back to the client')
            sock.sendto("1.png".encode(), address)
            sock.sendto("2.png".encode(), address)
        print(data.decode())
            
