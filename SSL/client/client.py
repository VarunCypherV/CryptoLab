import socket
import ssl

# Server configuration
HOST = '127.0.0.1'
PORT = 12345
CERTFILE = 'client.crt'
KEYFILE = 'client.key'

# Create SSL context
context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)

context.check_hostname = False 
context.load_verify_locations(cafile="server.crt")  # Adding server's self-signed certificate to trust store

# Connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
with context.wrap_socket(client_socket, server_hostname=HOST) as ssl_socket:
    print("Connected to server.")

    while True:
        # Get user input from terminal
        message = input("Type a message to send to server (or type 'exit' to quit): ")
        if message.lower() == 'exit':
            break

        # Send the message to the server
        ssl_socket.sendall(message.encode())

        # Receive data from the server
        data = ssl_socket.recv(1024)
        print("Received from server:", data.decode())

# Close the socket
client_socket.close()