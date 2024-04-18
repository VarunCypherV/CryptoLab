import socket
import ssl

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 12345
CERT_FILE = 'server.crt'  
KEY_FILE = 'server.key'   

def server_ssl():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)
    print(f"Server is listening on {SERVER_HOST}:{SERVER_PORT}...")
    connection, address = server_socket.accept()
    print(f"Connection established from: {address}")
    secure_connection = ssl_context.wrap_socket(connection, server_side=True)
    data = secure_connection.recv(1024)
    print(f"Received from client: {data.decode()}")
    secure_connection.sendall("Hello from the server!".encode())
    secure_connection.close()
    server_socket.close()

if __name__ == "__main__":
    server_ssl()
