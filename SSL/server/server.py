import socket
import ssl

# Server configuration
HOST = '127.0.0.1'
PORT = 12345
CERTFILE = 'server.crt'
KEYFILE = 'server.key'

# Create SSL context
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)

# Create a TCP/IP socket and ipv4
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the address and port
server_socket.bind((HOST, PORT))
# Listen for incoming connections
server_socket.listen()

print("Server is listening...")

# Accept a connection
client_socket, client_address = server_socket.accept()
with context.wrap_socket(client_socket, server_side=True) as ssl_socket:
    print("Server connected.")  

    while True:
        # Receive data from the client
        data = ssl_socket.recv(1024)
        print(data)
        if not data:
            break
        print("Received from client:", data.decode())

        # Get user input from terminal
        message = input("Type a message to send to client (or type 'exit' to quit): ")
        if message.lower() == 'exit':
            break

        # Send the message to the client
        ssl_socket.sendall(message.encode())

# Close the sockets
ssl_socket.close()
server_socket.close()