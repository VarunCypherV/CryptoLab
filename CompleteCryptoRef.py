#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# THE ULTIMATE CRYPTOGRAPHY JUPYTER NOTEBOOK 
# THANK ME LATER WITH A MILKSHAKE 👉👈 
# 
# -VARUN

# ### LIST OF EXPS COVERED
# 
# - CLIENT SERVER
# 
# 
# - DES
# 
# - AES
# 
# - RSA
# 
# - MAN IN MIDDLE DFH
# 
# - MD5
# 
# - SHA1
# 
# - DSS
# 
# - SSL
# 
# - WEB APP
# 
# - CLASSICAL SUBS AND TRANSPOS ENCRYPTION
#     - CAESAR X
#     
#     - PLAYFAIR X
#     
#     - VIGNERE  X
#     
#     - HILL CIPHER X
#     
#     - VERMAN CIPHER X
#     
#     - RAIL FENCE CIPHER X

# ## CLASSICAL ENCRYPTION

# ### CAESAR

# In[35]:


MAP = {chr(ord('A')+i):i  for i in range(26)}
print(MAP_D)
def caesarE(p,k):
    c=""
    for i in p :
        c+= chr((MAP[i] + k) % 26 +ord('A'))#(p+k)mod26
    return c

def caesarD(c,k):
    p=""
    for i in c :
        p+= chr((MAP[i] - k) % 26 +ord('A'))#(p+k)mod26
    return p

c = caesarE("MEET",3)
p = caesarD("PHHW",3)
print(c)
print(p)


# ### PLAYFAIR CIPHER

# In[108]:


MATRIX = [["" for i in range(5)]for i in range(5)]
MATRIX
MAP={chr(ord('A')+i):False for i in range(26)} #used or not

def playfairmatrix(p):
    k=0
    unique_characters = set(p)
    limit=len(unique_characters)

    for i in range(5):
        for j in range(5):
            if k<limit and MATRIX[i][j]=="" and MAP[p[k]]==False:
                MATRIX[i][j]=p[k]
                MAP[p[k]]=True
                k+=1
                continue # so that k+=1 at last p doesnt trigger next block so it gets replaced error
            if k>=limit :
                for d in MAP:
                    if MAP[d]==False and d!="J": 
                        MATRIX[i][j]=d
                        MAP[d]=True
                        break
    return MATRIX

def formtable(p):
    for i in range(1,len(p)):
        if p[i-1]==p[i] and i%2==1:
            prevv=p[:i]
            nextt=p[i:]
            p=prevv+'X'+nextt
    if (len(p)%2!=0):
        p+='X'
    l=[p[i]+p[i+1] for i in range(0,len(p),2)]
    return l

def find(char, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == char:
                return i, j
    return -1, -1 


def playfairE(p):
    matrix=playfairmatrix("MONARCHY")
    l=formtable(p)
    c=""
    print(l)
    print(matrix)
    for i in l:
        m = i[0]
        n = i[1]
        # find indices of m and n in the matrix
        mi, mj = find(m, matrix)
        ni, nj = find(n, matrix)
        if mi == ni:  # same row
            # move to the right with wrap-around
            c += matrix[mi][(mj + 1) % 5]
            c += matrix[ni][(nj + 1) % 5]
        elif mj == nj:  # same column
            # move down with wrap-around
            c += matrix[(mi + 1) % 5][mj]
            c += matrix[(ni + 1) % 5][nj]
        else:  # rectangle case
            c += matrix[mi][nj]
            c += matrix[ni][mj]
    return c

res=playfairE("BALLOON")
print(res)


# ### HILL CIPHER

# In[134]:


MAP = {chr(ord('A')+i):i  for i in range(26)}
def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return None
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += (A[i][k] * B[k][j])%26
    return result

def hillE(p,kl,k):
    c=[]
    for l in range(0,len(p),kl):
        for i in p[l:l+kl]:
            c+= chr((MAP[i] + kl) % 26 +ord('A'))#(p+k)mod26
        r=matrix_multiply(c,k)
    return c

m=[[17,17,5],[21,18,21],[2,2,19]]
c = hillE("PAYMOREMONEY",3,m)
print(c)


# In[163]:


MAP = {chr(ord('A')+i):i  for i in range(26)}
MAPD = {i : chr(ord('A')+i)  for i in range(26)}
def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return None
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += (A[i][k] * B[k][j])
                result[i][j] %= 26 
    return result

def hillE(p, block_size, k):
    c = []
    res=""
    for l in range(0, len(p), block_size):
        block = p[l:l + block_size]
        print(block)
        block_indices = [MAP[char] for char in block]
        print(block_indices)
        encrypted_block_indices = matrix_multiply([block_indices], k)
        print(encrypted_block_indices)
        encrypted_block = "".join(MAPD[x] for x in encrypted_block_indices[0])
        print(encrypted_block)
        c.append(encrypted_block)
    return ''.join(c)

m = [[17, 17, 5], [21, 18, 21], [2, 2, 19]]
c = hillE("PAYMOREMONEY", 3, m)
print(c)


# In[182]:


import numpy as np

MAP = {chr(ord('A') + i): i for i in range(26)}
MAPD = {i: chr(ord('A') + i) for i in range(26)}

def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return None
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += (A[i][k] * B[k][j])
                result[i][j] %= 26
    return result

def matrix_inverse(A):
    det = int(round(np.linalg.det(A)))
    det_inv = pow(det, -1, 26)  # Computing modular inverse of det
    adjugate = np.round(det_inv * np.linalg.inv(A) * det) % 26
    return adjugate.astype(int)

def hillD(p, block_size, k):
    c = []
    inv = matrix_inverse(k)
    for l in range(0, len(p), block_size):
        block = p[l:l + block_size]
        block_indices = [MAP[char] for char in block]
        encrypted_block_indices = matrix_multiply([block_indices], inv)
        encrypted_block = "".join(MAPD[x] for x in encrypted_block_indices[0])
        c.append(encrypted_block)
    return ''.join(c)

m = np.array([[17, 17, 5], [21, 18, 21], [2, 2, 19]])
c = hillD("RRLMWBKASPDH", 3, m)
print("Decrypted text:", c)


# ### VIGNERE CIPHER

# In[290]:


def vigenere_encrypt(plain_text, key):
    key = key.upper()
    key_length = len(key)
    encrypted_text = ""
    for i in range(len(plain_text)):
        char = plain_text[i]
        if char.isalpha():
            key_char = key[i % key_length]
            shift = ord(key_char) - 65  # ASCII value of 'A' is 65
            if char.isupper():
                encrypted_text += chr((ord(char) + shift - 65) % 26 + 65)
            else:
                encrypted_text += chr((ord(char) + shift - 97) % 26 + 97)
        else:
            encrypted_text += char
    return encrypted_text

def vigenere_decrypt(encrypted_text, key):
    key = key.upper()
    key_length = len(key)
    decrypted_text = ""
    for i in range(len(encrypted_text)):
        char = encrypted_text[i]
        if char.isalpha():
            key_char = key[i % key_length]
            shift = ord(key_char) - 65  # ASCII value of 'A' is 65
            if char.isupper():
                decrypted_text += chr((ord(char) - shift - 65) % 26 + 65)
            else:
                decrypted_text += chr((ord(char) - shift - 97) % 26 + 97)
        else:
            decrypted_text += char
    return decrypted_text

# Example usage:
plain_text = "Hello, World!"
key = "KEY"
encrypted_text = vigenere_encrypt(plain_text, key)
print("Encrypted Text:", encrypted_text)
decrypted_text = vigenere_decrypt(encrypted_text, key)
print("Decrypted Text:", decrypted_text)


# ### VERMAN CIPHER

# In[296]:


def vernam_encrypt(plain_text, key):
    key = (key * (len(plain_text) // len(key) + 1))[:len(plain_text)]
    encrypted_text = ""
    for i in range(len(plain_text)):
        encrypted_char = chr(ord(plain_text[i]) ^ ord(key[i]))
        encrypted_text += encrypted_char
    return encrypted_text

def vernam_decrypt(encrypted_text, key):
    # Repeat the key to match the length of the encrypted text
    key = (key * (len(encrypted_text) // len(key) + 1))[:len(encrypted_text)]
    decrypted_text = ""
    for i in range(len(encrypted_text)):
        decrypted_char = chr(ord(encrypted_text[i]) ^ ord(key[i]))
        decrypted_text += decrypted_char
    return decrypted_text

# Example usage:
plain_text = "Hello, World!"
key = "RANDOM"
encrypted_text = vernam_encrypt(plain_text, key)
print("Encrypted Text:", encrypted_text)
decrypted_text = vernam_decrypt(encrypted_text, key)
print("Decrypted Text:", decrypted_text)


# ### RAIL FENCE CIPHER

# In[214]:


def railfenceE(p):
    x=[p[i] for i in range(0,len(p),2)]
    y=[p[i] for i in range(1,len(p),2)]
    return "".join(x+y)
res=railfenceE("NESOACADEMY")
print(res)

def railfenceD(p):
    x=[p[i] for i in range(0,len(p)//2+1)]
    y=[p[i] for i in range(len(p)//2+1,len(p))]
    print(x,y)
    res=[]
    for i in range(len(p)//2+1):
        if(i<len(x) and i<len(y)):
            res+=x[i]+y[i]
        elif(i<len(x)):
            res+=x[i]
        elif(i<len(y)):
            res+=y[i]
        else:
            pass
    return res
res2=railfenceD(res)
print(res2)


# ### ROWCOL CIPHER

# In[261]:


import random

def get_random_alphabet():
    return random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


MATRIX = [["" for i in range(5)]for i in range(5)]
MATRIX
MAP={chr(ord('A')+i):False for i in range(26)}

def rowcolmatrix(p):
    k=0
    for i in range(5):
        for j in range(5):
            if k<len(p) and MATRIX[i][j]=="":
                MATRIX[i][j]=p[k]
                k+=1
                continue
            if k>=len(p) :
                MATRIX[i][j]=get_random_alphabet()
                break
    return MATRIX
print(rowcolmatrix("KILLCORONAVIRUS"))

def RowColE(p,k):
    matrix=rowcolmatrix(p)
    res=[]
    for i in k:
        for j in range(len(matrix[i-1])):
            res+=matrix[j][i-1]
    return res
print(RowColE("KILLCORONAVIRUS",[3,4,1,2,5]))


# ## END OF CLASSICAL ENCRYPTIONG

# ## TERROR CODES BEGIN

# ## CLIENT SERVER

# In[ ]:


# server
import socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
port = 12345
server_socket.bind((host, port))
server_socket.listen(5)

print("Server listening on {}:{}".format(host, port))

while True:
    client_socket, addr = server_socket.accept()
    print('Got connection from', addr)
    data = client_socket.recv(1024).decode()
    if data:
        print("Message from client:", data)
        client_socket.send(data.encode())
    client_socket.close()

    
##CLIENT

import socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
port = 12345
client_socket.connect((host, port))

message = "Hello, server!"
client_socket.send(message.encode())

response = client_socket.recv(1024).decode()
print("Response from server:", response)

client_socket.close()


# ### RSA

# In[ ]:


import random

# Euclid for divisor
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Extended Euclid for MI
def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi
    while e > 0:
        temp1 = temp_phi // e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2
        x = x2 - temp1 * x1
        y = d - temp1 * y1
        x2 = x1
        x1 = x
        d = y1
        y1 = y
    if temp_phi == 1:
        return d + phi

def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num ** 0.5) + 2, 2):
        if num % n == 0:
            return False
    return True

def generate_prime():
    prime_candidate = 0
    while not is_prime(prime_candidate):
        prime_candidate = random.randint(100, 1000)  # Adjust the range as per your requirements
    return prime_candidate

def generate_key_pair():
    p = generate_prime()
    q = generate_prime()
    n = p * q
    phi = (p - 1) * (q - 1)
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
    d = multiplicative_inverse(e, phi)
    return ((e, n), (d, n))
def encrypt(pk, plaintext):
    # Unpack the key into it's components
    key, n = pk
    # Convert each letter in the plaintext to numbers based on the character using a^b mod m
    cipher = [pow(ord(char), key, n) for char in plaintext]
    # Return the array of bytes
    return cipher


def decrypt(pk, ciphertext):
    # Unpack the key into its components
    key, n = pk
    # Generate the plaintext based on the ciphertext and key using a^b mod m
    aux = [str(pow(char, key, n)) for char in ciphertext]
    # Return the array of bytes as a string
    plain = [chr(int(char2)) for char2 in aux]
    return ''.join(plain)

if __name__ == '__main__':
    public, private = generate_key_pair()
    print(" - Your public key is ", public, " and your private key is ", private)

    message = input(" - Enter a message to encrypt with your public key: ")
    encrypted_msg = encrypt(public, message)

    print(" - Your encrypted message is: ", ''.join(map(lambda x: str(x), encrypted_msg)))
    print(" - Decrypting message with private key ", private, " . . .")
    print(" - Your message is: ", decrypt(private, encrypted_msg))

#Sample Input : hello
#HASH : 2977587856797527975287276
#Reconstructed : hello


# ### DSS

# In[ ]:


def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y

def mod_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

def digital_signature_sign(p, q, g, x, k, message):
    # Calculate H(M)
    H_M = message
    
    # Signing
    r1 = pow(g, k, p) % q
    s1 = (mod_inverse(k, q) * (H_M + x * r1)) % q
    return r1, s1

def digital_signature_verify(p, q, g, y, message, r1, s1):
    # Verification
    w = mod_inverse(s1, q)
    u1 = (message * w) % q
    u2 = (r1 * w) % q
    
    # Calculate v
    v = (pow(g, u1, p) * pow(y, u2, p)) % q
    
    # Check if v equals r1
    if v == r1:
        return True
    else:
        return False

# Parameters
p = 283
q = 47
g = 60
x = 24
k = 5
message = 41

# Signing
r1, s1 = digital_signature_sign(p, q, g, x, k, message)
print("r1:", r1)
print("s1:", s1)

# Verification
y = pow(g, x, p)
is_verified = digital_signature_verify(p, q, g, y, message, r1, s1)
print("Signature verified:", is_verified)

# Signing another message
message2 = 55
r2, s2 = digital_signature_sign(p, q, g, x, k, message2)
print("r2:", r2)
print("s2:", s2)

# Verification of the second message
y2 = pow(g, x, p)
is_verified2 = digital_signature_verify(p, q, g, y2, message2, r2, s2)
print("Signature verified for the second message:", is_verified2)


# ### MAN IN MIDDLE DFH

# In[ ]:


# Input prime number and primitive root
prime = int(input("Enter the prime number to be considered: "))
root = int(input("Enter the primitive root: "))
 
# Party1 chooses a secret number
alicesecret = int(input("Enter a secret number for Party1: "))
 
# Party2 chooses a secret number
bobsecret = int(input("Enter a secret number for Party2: "))
print("\n")
 
# Attacker intercepts Party1's public key
attacker_public_to_bob = (root ** alicesecret) % prime
print("Attacker intercepts Party1's public key intended for Party2:", attacker_public_to_bob)
 
# Attacker intercepts Party2's public key
attacker_public_to_alice = (root ** bobsecret) % prime
print("Attacker intercepts Party2's public key intended for Party1:", attacker_public_to_alice)
print("\n")
 
# Attacker can now compute their own shared keys with Party1 and Party2
attacker_key_with_alice = (attacker_public_to_alice ** alicesecret) % prime
attacker_key_with_bob = (attacker_public_to_bob ** bobsecret) % prime
 
# Both Party1 and Party2 now believe they are communicating securely with each other,
# but in reality, the attacker is intercepting and possibly altering the communication.
print("Attacker calculates the shared key with Party1:", attacker_key_with_alice)
print("Attacker calculates the shared key with Party2:", attacker_key_with_bob)
print("\n")
print("Both Alice and Bob are unaware that the attacker has intercepted the communication.")


#Primitve Root of 19 = 2, 3, 10, 13, 14, 15


# ### SSL
# 

# In[ ]:


###CLIENT
import socket
import ssl

HOST = '127.0.0.1'
PORT = 12345
CERTFILE = 'client.crt'
KEYFILE = 'client.key'

''' RUN SEPERATELY IN GIT BASH OR SOME OPENSSL 
openssl req -x509 -nodes -newkey rsa:2048 -keyout server.key -out server.crt -days 365
openssl genpkey -algorithm RSA -out client.key
openssl req -new -key client.key -out client.csr
openssl x509 -req -in client.csr -CA server.crt -CAkey server.key -out client.crt -days 365
'''

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)

context.check_hostname = False 
context.load_verify_locations(cafile="server.crt") 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
with context.wrap_socket(client_socket, server_hostname=HOST) as ssl_socket:
    print("Connected to server.")

    while True:
        message = input("Type a message to send to server (or type 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        ssl_socket.sendall(message.encode())
        data = ssl_socket.recv(1024)
        print("Received from server:", data.decode())

client_socket.close()

### SERVER

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


# ### WEB APP

# In[ ]:


const express = require('express');
const bodyParser = require('body-parser');
const jwt = require('jsonwebtoken');

const app = express();
const port = 3000;
const secretKey = 'MY-SECRET-KEY-AVXZGHCASJ';

app.use(bodyParser.json());


const users = [
    { id: 1, username: 'user1', password: 'password1' },
    { id: 2, username: 'user2', password: 'password2' }
];

app.post('/login', (req, res) => {
    const { username, password } = req.body;
    const user = users.find(u => u.username === username && u.password === password);
    if (user) {
        const token = jwt.sign({ userId: user.id }, secretKey, { expiresIn: '1h' });
        res.json({ token });
    } else {
        res.status(401).json({ message: 'Invalid username or password' });
    }
});

app.get('/protected', verifyToken, (req, res) => {
    jwt.verify(req.token, secretKey, (err, authData) => {
        if (err) {
            res.sendStatus(403);
        } else {
            res.json({ message: 'Welcome to the protected route!', authData });
        }
    });
});


function verifyToken(req, res, next) {
    const bearerHeader = req.headers['authorization'];
    if (typeof bearerHeader !== 'undefined') {
        const bearerToken = bearerHeader.split(' ')[1];
        req.token = bearerToken;
        next();
    } else {
        res.sendStatus(403);
    }
}


app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});


# ## I RATHER KILL MYSELF CODES

# ### DES
# 

# In[ ]:


def main():
    print()
    # Taking inputs from the user
    plaintext = input("Enter the message to be encrypted : ")
    key = input("Enter a key of 8 length (64-bits) (characters or numbers only) : ")
    print()

    # Checking if key is valid or not
    if len(key) != 8:
        print("Invalid Key. Key should be of 8 length (8 bytes).")
        return
    # Determining if padding is required
    isPaddingRequired = (len(plaintext) % 8 != 0)
    # Encryption
    ciphertext = DESEncryption(key, plaintext, isPaddingRequired)
    # Decryption
    plaintext = DESDecryption(key, ciphertext, isPaddingRequired)

    # Printing result
    print()
    print("Encrypted Ciphertext is : %r " % ciphertext)
    print("Decrypted plaintext is  : ", plaintext)
    print()

# Permutation Matrix used after each SBox substitution for each round
eachRoundPermutationMatrix = [
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
]

# Final Permutation Matrix for data after 16 rounds
finalPermutationMatrix = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
]

# Initial Permutation Matrix for data
initialPermutationMatrix = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]
#Expand matrix to get a 48bits matrix of datas to apply the xor with Ki
expandMatrix = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

# Matrix used for shifting after each round of keys
SHIFT = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# Permutation matrix for key
keyPermutationMatrix1 = [
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
]

# Permutation matrix for shifted key to get next key
keyPermutationMatrix2 = [
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
]

# Sboxes used in the DES Algorithm
SboxesArray = [
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    ],

    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    ],

    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    ],

    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    ],

    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    ],

    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    ],

    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    ],

    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ]
]

def DESEncryption(key, text, padding):
    # Adding padding if required
    if padding == True:
        text = addPadding(text)
    # Encryption
    ciphertext = DES(text, key, padding, True)
    # Returning ciphertext
    return ciphertext

def DESDecryption(key, text, padding):
    # Decryption
    plaintext = DES(text, key, padding, False)
    # Removing padding if required
    if padding == True:
        # Removing padding and returning plaintext
        return removePadding(plaintext)
    # Returning plaintext
    return plaintext



def DES(text, key, padding, isEncrypt):
    """Function to implement DES Algorithm."""
    # Initializing variables required
    isDecrypt = not isEncrypt
    # Generating keys
    keys = generateKeys(key)
    # Splitting text into 8 byte blocks
    plaintext8byteBlocks = nSplit(text, 8)
    result = []

    # For all 8-byte blocks of text
    for block in plaintext8byteBlocks:
        # Convert the block into bit array
        block = stringToBitArray(block)
        # Do the initial permutation
        block = permutation(block, initialPermutationMatrix)
        # Splitting block into two 4 byte (32 bit) sized blocks
        leftBlock, rightBlock = nSplit(block, 32)
        temp = None

        # Running 16 identical DES Rounds for each block of text
        for i in range(16):
            # Expand rightBlock to match round key size(48-bit)
            expandedRightBlock = expand(rightBlock, expandMatrix)
            # Xor right block with appropriate key
            if isEncrypt == True:
                # For encryption, starting from first key in normal order
                temp = xor(keys[i], expandedRightBlock)
            elif isDecrypt == True:
                # For decryption, starting from last key in reverse order
                temp = xor(keys[15 - i], expandedRightBlock)
            # Sbox substitution Step
            temp = SboxSubstitution(temp)
            # Permutation Step
            temp = permutation(temp, eachRoundPermutationMatrix)
            # XOR Step with leftBlock
            temp = xor(leftBlock, temp)
            # Blocks swapping
            leftBlock = rightBlock
            rightBlock = temp
        # Final permutation then appending result
        result += permutation(rightBlock + leftBlock, finalPermutationMatrix)
    # Converting bit array to string
    finalResult = bitArrayToString(result)
    return finalResult



def generateKeys(key):
    # Inititalizing variables required
    keys = []
    key = stringToBitArray(key)
    # Initial permutation on key
    key = permutation(key, keyPermutationMatrix1)
    # Split key in to (leftBlock->LEFT), (rightBlock->RIGHT)
    leftBlock, rightBlock = nSplit(key, 28)
    # 16 rounds of keys
    for i in range(16):
        # Do left shifting (different for different rounds)
        leftBlock, rightBlock = leftShift(leftBlock, rightBlock, SHIFT[i])
        # Merge them
        temp = leftBlock + rightBlock
        # Permutation on shifted key to get next key
        keys.append(permutation(temp, keyPermutationMatrix2))
    # Return generated keys
    return keys



def SboxSubstitution(bitArray):
    """Function to substitute all the bytes using Sbox."""
    # Split bit array into 6 sized chunks
    # For Sbox indexing
    blocks = nSplit(bitArray, 6)
    result = []

    for i in range(len(blocks)):
        block = blocks[i]
        # Row number to be obtained from first and last bit
        row = int( str(block[0]) + str(block[5]), 2 )
        # Getting column number from the 2,3,4,5 position bits
        column = int(''.join([str(x) for x in block[1:-1]]), 2)
        # Taking value from ith Sbox in ith round
        sboxValue = SboxesArray[i][row][column]
        # Convert the sbox value to binary
        binVal = binValue(sboxValue, 4)
        # Appending to result
        result += [int(bit) for bit in binVal]
    # Returning result
    return result









#UTILITY FUNCTIONS

def addPadding(text):
    """Function to add padding according to PKCS5 standard."""
    # Determining padding length
    paddingLength = 8 - (len(text) % 8)
    # Adding paddingLength number of chr(paddingLength) to text
    text += chr(paddingLength) * paddingLength
    # Returning text
    return text

def removePadding(data):
    """Function to remove padding from plaintext according to PKCS5."""
    # Getting padding length
    paddingLength = ord(data[-1])
    # Returning data with removed padding
    return data[ : -paddingLength]

def expand(array, table):
    """Function to expand the array using table."""
    # Returning expanded result
    return [array[element - 1] for element in table]

def permutation(array, table):
    """Function to do permutation on the array using table."""
    # Returning permuted result
    return [array[element - 1] for element in table]

def leftShift(list1, list2, n):
    """Function to left shift the arrays by n."""
    # Left shifting the two arrays
    return list1[n:] + list1[:n], list2[n:] + list2[:n]

def nSplit(list, n):
    """Function to split a list into chunks of size n."""
    # Chunking and returning the array of chunks of size n
    # and last remainder
    return [ list[i : i + n] for i in range(0, len(list), n)]

def xor(list1, list2):
    """Function to return the XOR of two lists."""
    # Returning the xor of the two lists
    return [element1 ^ element2 for element1, element2 in zip(list1,list2)]

def binValue(val, bitSize):
    """Function to return the binary value as a string of given size."""
    binVal = bin(val)[2:] if isinstance(val, int) else bin(ord(val))[2:]
    # Appending with required number of zeros in front
    while len(binVal) < bitSize:
        binVal = "0" + binVal
    # Returning binary value
    return binVal


def stringToBitArray(text):
    """Funtion to convert a string into a list of bits."""
    # Initializing variable required
    bitArray = []
    for letter in text:
        # Getting binary (8-bit) value of letter
        binVal = binValue(letter, 8)
        # Making list of the bits
        binValArr = [int(x) for x in list(binVal)]
        # Apending the bits to array
        bitArray += binValArr
    # Returning answer
    return bitArray


def bitArrayToString(array):
    """Function to convert a list of bits to string."""
    # Chunking array of bits to 8 sized bytes
    byteChunks = nSplit(array, 8)
    # Initializing variables required
    stringBytesList = []
    stringResult = ''
    # For each byte
    for byte in byteChunks:
        bitsList = []
        for bit in byte:
            bitsList += str(bit)
        # Appending byte in string form to stringBytesList
        stringBytesList.append(''.join(bitsList))
    # Converting each stringByte to char (base 2 int conversion first)
    # and then concatenating
    result = ''.join([chr(int(stringByte, 2)) for stringByte in stringBytesList])
    # Returning result
    return result

if __name__ == '__main__':
    main()


# ### AES
# 

# In[ ]:


# Number of columns in state
noOfColumns = 4
# Number of rounds in AES 128 cycle
noOfRounds = 10
# Key Length (in 32-bit words)
noOfWordsInKey = 4

def main():

    # Taking inputs from user
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
            print('Invalid Key. Please use only latin alphabet and numbers.')
            return

    # Encrypting message 16 bytes at a time
    encryptedBytes = []
    temp = []
    for byte in plaintext:
        temp.append(byte)
        if len(temp) == 16:
            Encrypted16Bytes = AES128Encryption(temp, key)
            encryptedBytes.extend(Encrypted16Bytes)
            temp = []

    # If plaintext is not a multiple of 16 bytes
    emptySpaces = None
    if 0 < len(temp) < 16:
            emptySpaces = 16 - len(temp)
            for i in range(emptySpaces - 1):
                temp.append(0)
            temp.append(1)
            Encrypted16Bytes = AES128Encryption(temp, key)
            encryptedBytes.extend(Encrypted16Bytes)

    # Encrypted message is now compulsorily a multiple of 16 bytes
    # Because padding is added.

    encryptedMessage = bytes(encryptedBytes)

    # Printed encrypted message
    print("Encrypted ciphertext is (in bytes) : ", encryptedMessage)

    # Decrypted 16 Bytes at a time
    decryptedBytes = []
    temp = []
    for byte in encryptedMessage:
        temp.append(byte)
        if len(temp) == 16:
            Decrypted16Bytes = AES128Decryption(temp, key)
            decryptedBytes.extend(Decrypted16Bytes)
            temp = []

    # Removing padding at the end
    if emptySpaces != None:
        decryptedMessage = bytes(decryptedBytes[ : -emptySpaces])
    else:
        decryptedMessage = bytes(decryptedBytes)

    # Printed decrypted message
    print("Decrypted plaintext is (in bytes) : ", decryptedMessage)
    print()

def AES128Encryption(plaintextBytes, key):
    """Function to implement AES Encryption Algorithm. Number of bytes should be equal to 16."""

    # Initializing state
    state = [[] for j in range(4)]

    # Adding values into state from plaintextBytes array into state matrix
    for row in range(4):
        for column in range(noOfColumns):
            state[row].append(plaintextBytes[row + 4 * column])

    # Expanding key
    keySchedule = keyExpansion(key)

    # Round 0, Adding round key only
    round = 0
    state = addRoundKey(state, keySchedule, round)

    # Performing round 1 to last round - 1 round of AES
    for round in range(1, noOfRounds):
        state = subBytes(state, False)
        state = shiftRows(state, False)
        state = mixColumns(state, False)
        state = addRoundKey(state, keySchedule, round)

    # Last round of AES
    state = subBytes(state, False)
    state = shiftRows(state, False)
    state = addRoundKey(state, keySchedule, round + 1)

    # Computing output
    output = [None for i in range(4 * noOfColumns)]
    for row in range(4):
        for column in range(noOfColumns):
            output[row + 4 * column] = state[row][column]

    # Returning output
    return output


def AES128Decryption(ciphertextBytes, key):
    """Function to implement AES Decryption Algorithm. Number of bytes should be equal to 16."""

    # Initializing state
    state = [[] for i in range(noOfColumns)]

    # Adding values into state from ciphertextBytes array into state matrix
    for row in range(4):
        for column in range(noOfColumns):
            state[row].append(ciphertextBytes[row + 4 * column])

    # Expanding key
    keySchedule = keyExpansion(key)

    # Starting from last round (in reverse, because decryption)
    state = addRoundKey(state, keySchedule, noOfRounds)

    # Performing rounds from last round -1 to first in reverse
    round = noOfRounds - 1
    for round in range(round, 0, -1):
        state = shiftRows(state, True)
        state = subBytes(state, True)
        state = addRoundKey(state, keySchedule, round)
        state = mixColumns(state, True)

    round -= 1

    # Round 0 in reverse
    state = shiftRows(state, True)
    state = subBytes(state, True)
    state = addRoundKey(state, keySchedule, round)

    # Computing output
    output = [None for i in range(4 * noOfColumns)]
    for row in range(4):
        for column in range(noOfColumns):
            output[row + 4 * column] = state[row][column]

    # Returning output
    return output

# Precomputed Sbox array constants
Sbox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

# Precomputed inverse Sbox array constants
inverseSbox = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

def subBytes(state, inverse):
    """Function to use Sbox or Inverse Sbox to determine subBytes value in the algorithm"""

    box = Sbox
    if inverse == True:
        box = inverseSbox

    # Substituting bytes according to box
    for i in range(len(state)):
        for j in range(len(state[i])):

            # Formula for computing sbox index row and column
            row = state[i][j] // 0x10
            column = state[i][j] % 0x10

            # 16 * round + column, because box is an array, not a matrix
            boxElement = box[16 * row + column]
            # Substituting value to state
            state[i][j] = boxElement

    # Returning state
    return state


def shiftRows(state, inverse):
    """Function to shift rows in the state matrix as done in the AES algorithm."""

    count = 1

    # Right shift for decryption
    if inverse == True:
        for i in range(1, noOfColumns):
            state[i] = rightShift(state[i], count)
            count += 1
    # Left shift for decryption
    else:
        for i in range(1, noOfColumns):
            state[i] = leftShift(state[i], count)
            count += 1

    # Returning state
    return state

def leftShift(array, count):
    """Left shift an array count times."""

    # Returning left shifted array
    return array[count:] + array[:count]


def rightShift(array, count):
    """Right shift an array count times."""

    # Returning right shifted array
    return array[-count:] + array[:-count]


def mixColumns(state, inverse):
    """Mixing columns with the Galois field attributes as done in the AES algorithm.
       Multiplication of matrices takes place here."""

    # Matrix multiplication as done in the AES Algorithm
    for i in range(noOfColumns):

        # Matrix multiplication for decryption
        if inverse == True:
            s0 = multiplyBy0x0e(state[0][i]) ^ multiplyBy0x0b(state[1][i]) ^ multiplyBy0x0d(state[2][i]) ^ multiplyBy0x09(state[3][i])
            s1 = multiplyBy0x09(state[0][i]) ^ multiplyBy0x0e(state[1][i]) ^ multiplyBy0x0b(state[2][i]) ^ multiplyBy0x0d(state[3][i])
            s2 = multiplyBy0x0d(state[0][i]) ^ multiplyBy0x09(state[1][i]) ^ multiplyBy0x0e(state[2][i]) ^ multiplyBy0x0b(state[3][i])
            s3 = multiplyBy0x0b(state[0][i]) ^ multiplyBy0x0d(state[1][i]) ^ multiplyBy0x09(state[2][i]) ^ multiplyBy0x0e(state[3][i])
        # Matrix multiplication for encryption
        else:
            s0 = multiplyBy0x02(state[0][i]) ^ multiplyBy0x03(state[1][i]) ^ state[2][i] ^ state[3][i]
            s1 = state[0][i] ^ multiplyBy0x02(state[1][i]) ^ multiplyBy0x03(state[2][i]) ^ state[3][i]
            s2 = state[0][i] ^ state[1][i] ^ multiplyBy0x02(state[2][i]) ^ multiplyBy0x03(state[3][i])
            s3 = multiplyBy0x03(state[0][i]) ^ state[1][i] ^ state[2][i] ^ multiplyBy0x02(state[3][i])

        # Storing values into state
        state[0][i] = s0
        state[1][i] = s1
        state[2][i] = s2
        state[3][i] = s3

    # Returning state
    return state

# RCON matrix constants for key exansion as done in AES Algorithm
RCON = [
    [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
]

def keyExpansion(key):
    """Funtion to make list of keys for addRoundKey."""

    # Initializing key symbols array
    keySymbols = [ord(symbol) for symbol in key]

    # Padding if key is less than 16 symbols
    if len(keySymbols) < 4 * noOfWordsInKey:
        for i in range(4 * noOfWordsInKey - len(keySymbols)):
            keySymbols.append(0x01)

    # Making cipher key (base of key schedule)
    keySchedule = [[] for i in range(4)]
    for row in range(4):
        for column in range(noOfWordsInKey):
            keySchedule[row].append(keySymbols[row + 4 * column])

    # Continuing to key schedule
    for column in range(noOfWordsInKey, noOfColumns * (noOfRounds + 1)):

        if column % noOfWordsInKey == 0:
            # Shifting last - 1 column
            temp = [keySchedule[row][column - 1] for row in range(1, 4)]
            temp.append(keySchedule[0][column - 1])

            # Changing elements using Sbox
            for j in range(len(temp)):

                # Formulas to compute sbox row and column
                SboxRow = temp[j] // 0x10
                SboxColumn = temp[j] % 0x10

                # 16 * round + column, because box is an array, not a matrix
                SboxElement = Sbox[16 * SboxRow + SboxColumn]

                # Substituting value into temp
                temp[j] = SboxElement

            # XORing 3 columns as required
            for row in range(4):
                s = (keySchedule[row][column - 4]) ^ (temp[row]) ^ (RCON[row][int(column / noOfWordsInKey - 1)])
                keySchedule[row].append(s)

        else:
            # XORing two columns
            for row in range(4):
                s = keySchedule[row][column - 4] ^ keySchedule[row][column - 1]
                keySchedule[row].append(s)

    # Returning key schedule
    return keySchedule


def addRoundKey(state, keySchedule, round):
    """That transformation combines State and KeySchedule together. Xor
    of State and RoundSchedule(part of KeySchedule).
    """

    for column in range(noOfWordsInKey):

        # XORing as required as a part of KeySchedule
        s0 = state[0][column] ^ keySchedule[0][noOfColumns * round + column]
        s1 = state[1][column] ^ keySchedule[1][noOfColumns * round + column]
        s2 = state[2][column] ^ keySchedule[2][noOfColumns * round + column]
        s3 = state[3][column] ^ keySchedule[3][noOfColumns * round + column]

        # Storing values into state
        state[0][column] = s0
        state[1][column] = s1
        state[2][column] = s2
        state[3][column] = s3

    # Returning state
    return state

def multiplyBy0x02(number):
    """Function to multiply by 0x02 in Galois Space."""

    # Multiplying by 0x02 in Galois Space
    if number < 0x80:
        result = (number << 1)
    else:
        result = (number << 1) ^ 0x1b

    result = result % 0x100

    # Returning result
    return result


def multiplyBy0x03(number):
    """Function to multiply by 0x03 in Galois Space."""
    # Breaking into simpler parts
    return (multiplyBy0x02(number) ^ number)


def multiplyBy0x09(number):
    """Function to multiply by 0x09 in Galois Space."""
    # Breaking into simpler parts
    return multiplyBy0x02(multiplyBy0x02(multiplyBy0x02(number))) ^ number


def multiplyBy0x0b(number):
    """Function to multiply by 0x0b in Galois Space."""
    # Breaking into simpler parts
    return multiplyBy0x02(multiplyBy0x02(multiplyBy0x02(number))) ^ multiplyBy0x02(number) ^ number


def multiplyBy0x0d(number):
    """Function to multiply by 0x0d in Galois Space."""
    # Breaking into simpler parts
    return multiplyBy0x02(multiplyBy0x02(multiplyBy0x02(number))) ^ multiplyBy0x02(multiplyBy0x02(number)) ^ number


def multiplyBy0x0e(number):
    """Function to multiply by 0x0e in Galois Space."""
    # Breaking into simpler parts
    return multiplyBy0x02(multiplyBy0x02(multiplyBy0x02(number))) ^ multiplyBy0x02(multiplyBy0x02(number)) ^ multiplyBy0x02(number)

if __name__ == '__main__':
    main()


# ### MD5
# 

# In[297]:


import math

# Initialize variables
s = [
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
]  # Initial constants for MD5
K = [
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
    0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
    0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
    0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
    0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
    0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
    0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
    0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
    0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C,
    0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
    0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05,
    0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
    0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039,
    0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
    0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1,
    0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391
]  # Constants for rounds

# Left-rotate function
def left_rotate(x, c):
    return ((x << c) | (x >> (32 - c))) & 0xFFFFFFFF

# Helper functions
def F(X, Y, Z):
    return (X & Y) | ((~X) & Z)

def G(X, Y, Z):
    return (X & Z) | (Y & (~Z))

def H(X, Y, Z):
    return X ^ Y ^ Z

def I(X, Y, Z):
    return Y ^ (X | (~Z))

# MD5 round functions
def round1(a, b, c, d, k, s, i, X):
    return (b + left_rotate((a + F(b, c, d) + X[k] + i) & 0xFFFFFFFF, s)) & 0xFFFFFFFF

def round2(a, b, c, d, k, s, i, X):
    return (b + left_rotate((a + G(b, c, d) + X[k] + i) & 0xFFFFFFFF, s)) & 0xFFFFFFFF

def round3(a, b, c, d, k, s, i, X):
    return (b + left_rotate((a + H(b, c, d) + X[k] + i) & 0xFFFFFFFF, s)) & 0xFFFFFFFF

def round4(a, b, c, d, k, s, i, X):
    return (b + left_rotate((a + I(b, c, d) + X[k] + i) & 0xFFFFFFFF, s)) & 0xFFFFFFFF

# Pad the message
def pad_message(message):
    original_len = len(message)
    message += b'\x80'
    while len(message) % 64 != 56:
        message += b'\x00'
    message += original_len.to_bytes(8, byteorder='little')
    return message

# MD5 hashing function
def md5_hash(message):
    message = pad_message(message)
    a, b, c, d = s

    for chunk_start in range(0, len(message), 64):
        chunk = message[chunk_start:chunk_start + 64]

        # Break chunk into 16 words
        X = [int.from_bytes(chunk[i:i + 4], byteorder='little') for i in range(0, 64, 4)]

        # Initialize hash values for this chunk
        A, B, C, D = a, b, c, d

        # Main loop
        for i in range(64):
            if 0 <= i < 16:
                A = round1(A, B, C, D, i, [7, 12, 17, 22][i % 4], K[i], X)
            elif 16 <= i < 32:
                A = round2(A, B, C, D, (5 * i + 1) % 16, [5, 9, 14, 20][i % 4], K[i], X)
            elif 32 <= i < 48:
                A = round3(A, B, C, D, (3 * i + 5) % 16, [4, 11, 16, 23][i % 4], K[i], X)
            else:
                A = round4(A, B, C, D, (7 * i) % 16, [6, 10, 15, 21][i % 4], K[i], X)

            A, B, C, D = D, (B + A) & 0xFFFFFFFF, B, C

        # Update hash values
        a = (a + A) & 0xFFFFFFFF
        b = (b + B) & 0xFFFFFFFF
        c = (c + C) & 0xFFFFFFFF
        d = (d + D) & 0xFFFFFFFF

    # Produce the final hash value
    return '{:08x}{:08x}{:08x}{:08x}'.format(a, b, c, d)

# Example usage:
input_string = "Hello, World!"
md5_hash = md5_hash(input_string.encode('utf-8'))
print("MD5 hash of '{}' is: {}".format(input_string, md5_hash))


# ### SHA1

# In[298]:


import struct

# Rotate left function
def left_rotate(n, b):
    return ((n << b) | (n >> (32 - b))) & 0xffffffff

# SHA-1 constants
SHA1_CONSTANTS = [
    0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6
]

# SHA-1 functions
def sha1(message):
    # Initialize variables
    h0 = 0x67452301
    h1 = 0xEFCDAB89
    h2 = 0x98BADCFE
    h3 = 0x10325476
    h4 = 0xC3D2E1F0

    # Pre-processing
    original_byte_len = len(message)
    original_bit_len = original_byte_len * 8
    message += b'\x80'
    while len(message) % 64 != 56:
        message += b'\x00'
    message += struct.pack('>Q', original_bit_len)

    # Process message in 512-bit blocks
    for i in range(0, len(message), 64):
        chunk = message[i:i+64]
        words = [0] * 80

        # Break chunk into 16 words
        for j in range(16):
            words[j] = struct.unpack('>I', chunk[j*4:j*4+4])[0]

        # Extend to 80 words
        for j in range(16, 80):
            words[j] = left_rotate((words[j-3] ^ words[j-8] ^ words[j-14] ^ words[j-16]), 1)

        # Initialize hash value for this chunk
        a = h0
        b = h1
        c = h2
        d = h3
        e = h4

        # Main loop
        for j in range(80):
            if j < 20:
                f = (b & c) | ((~b) & d)
                k = SHA1_CONSTANTS[0]
            elif j < 40:
                f = b ^ c ^ d
                k = SHA1_CONSTANTS[1]
            elif j < 60:
                f = (b & c) | (b & d) | (c & d)
                k = SHA1_CONSTANTS[2]
            else:
                f = b ^ c ^ d
                k = SHA1_CONSTANTS[3]

            temp = left_rotate(a, 5) + f + e + k + words[j] & 0xffffffff
            e = d
            d = c
            c = left_rotate(b, 30)
            b = a
            a = temp

        # Add this chunk's hash to result so far
        h0 = (h0 + a) & 0xffffffff
        h1 = (h1 + b) & 0xffffffff
        h2 = (h2 + c) & 0xffffffff
        h3 = (h3 + d) & 0xffffffff
        h4 = (h4 + e) & 0xffffffff

    # Produce the final hash value
    return '{:08x}{:08x}{:08x}{:08x}{:08x}'.format(h0, h1, h2, h3, h4)

# Example usage:
input_string = "Hello, World!"
sha1_hash = sha1(input_string.encode('utf-8'))
print("SHA-1 hash of '{}' is: {}".format(input_string, sha1_hash))


# In[ ]:





# In[ ]:




