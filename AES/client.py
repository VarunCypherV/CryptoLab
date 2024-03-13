import socket
from AES import AES128Encryption
import pickle

def main():
    # Server configuration
    host = '127.0.0.1'
    port = 12345

    # Creating a socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connecting to the server
    client_socket.connect((host, port))

    # Encrypting the plaintext
    print()
    plaintext = input('Enter the plaintext : ').encode()
    key = input('Enter key (must be less than 16 symbols and should consist of only alphabets & numbers) : ')
    print()

    # Checking if key is invalid
    if len(key) > 16:
        print('Invalid Key. Key too long.')
        return

    # Checking if key is invalid
    for symbol in key:
        if ord(symbol) > 0xff:
            print('Invalid Key. Please use only Latin alphabet and numbers.')
            return

    # Encrypting the plaintext using AES
    encryptedMessage = AES128Encryption(plaintext, key)
    print("Encrypted ciphertext is (in bytes) : ", encryptedMessage)

    data = pickle.dumps((encryptedMessage, key))

    # Sending the ciphertext and key to the server
    client_socket.send(data)
    client_socket.send(key.encode())

    # Closing the connection
    client_socket.close()

if __name__ == "__main__":
    main()


#Two One Nine Two : Plaintext
#Thats my Kung Fu : CipherText