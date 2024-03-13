import socket
import json
from DESlab1 import DESDecryption

def server_main():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to a specific address and port
    server_address = ('localhost', 12345)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(1)
    print("Server is waiting for a connection...")

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)

    # Receive encrypted text from the client
    recv_data = client_socket.recv(1024).decode('utf-8')
    print(recv_data)

    # Deserialize the JSON string into a Python dictionary
    recv = json.loads(recv_data)

    # Access the 'key' attribute from the dictionary
    key = recv['key']
    encrypted_text = recv['ciphertext']
    isPaddingRequired = recv['isPaddingRequired']

    # Perform DES decryption
    decrypted_text = DESDecryption(key, encrypted_text, isPaddingRequired)

    # Print the decrypted text
    print("Decrypted plaintext is:", decrypted_text)

    # Close the sockets
    client_socket.close()
    server_socket.close()

if __name__ == '__main__':
    server_main()
