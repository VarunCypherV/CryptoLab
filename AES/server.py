import socket
from AES import AES128Decryption
import pickle

def main():
    # Server configuration
    host = '127.0.0.1'
    port = 12345
    
    # Creating a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Binding the socket to the address
    server_socket.bind((host, port))
    
    # Listening for incoming connections
    server_socket.listen(1)
    print("Server is listening...")
    
    # Accepting a connection
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")
    
    # Receiving the data from the client
    data = client_socket.recv(4096)
    
    # Unpickling the received data to get the encrypted message and the key
    encrypted_message, key = pickle.loads(data)
    print("Received encrypted message:", encrypted_message)
    print("Received key:", key)
    
    # Decrypting the ciphertext using the received key
    decrypted_message = AES128Decryption(encrypted_message, key)
    print("Decrypted plaintext is (in bytes): ", bytes(decrypted_message))
    
    # Closing the connection
    client_socket.close()

if __name__ == "__main__":
    main()
