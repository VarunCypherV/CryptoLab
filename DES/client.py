import socket
import json
from DESlab1 import DESEncryption 

def client_main():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    server_address = ('localhost', 12345)
    client_socket.connect(server_address)
    print("Connected to server.")

    # Taking inputs from the user
    plaintext = input("Enter the message to be encrypted: ")
    key = input("Enter a key of 8 length (64-bits) (characters or numbers only): ")

    # Checking if key is valid or not
    if len(key) != 8:
        print("Invalid Key. Key should be of 8 length (8 bytes).")
        return

    # Determining if padding is required
    isPaddingRequired = (len(plaintext) % 8 != 0)

    # Encryption
    ciphertext = DESEncryption(key, plaintext, isPaddingRequired)

    # Create a dictionary to store the data
    data = {
        'key': key,
        'ciphertext': ciphertext,
        'isPaddingRequired': isPaddingRequired,
    }
    print(data)
    # Serialize the dictionary to a JSON string
    data_json = json.dumps(data)

    # Sending the JSON string to the server
    client_socket.send(data_json.encode('utf-8'))

    # Close the socket
    client_socket.close()

if __name__ == '__main__':
    client_main()


#SAMPLE INPUT KEY : ABCD1234
#SAMPLE INPUT PT: hello there
