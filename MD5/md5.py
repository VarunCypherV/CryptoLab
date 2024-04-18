import math

ALPHABETS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

A = [int("1", 16), int("23", 16), int("45", 16), int("67", 16)][::-1]
B = [int("89", 16), int("ab", 16), int("cd", 16), int("ef", 16)][::-1]
C = [int("fe", 16), int("dc", 16), int("ba", 16), int("98", 16)][::-1]
D = [int("76", 16), int("54", 16), int("32", 16), int("10", 16)][::-1]

A = "".join([bin(i)[2:].rjust(8, "0") for i in A])
B = "".join([bin(i)[2:].rjust(8, "0") for i in B])
C = "".join([bin(i)[2:].rjust(8, "0") for i in C])
D = "".join([bin(i)[2:].rjust(8, "0") for i in D])

K = [] 
for i in range(0, 64):
    K.append(bin(math.floor(2 ** 32 * abs(math.sin(i + 1))))[2:].rjust(32, "0"))

SHIFTS = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
          5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
          4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
          6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21]

non_linear_process = {
    0: lambda x, y, z: bitwise_or(bitwise_and(x, y), bitwise_and(bitwise_not(x), z)),
    1: lambda x, y, z: bitwise_or(bitwise_and(x, z), bitwise_and(y, bitwise_not(z))),
    2: lambda x, y, z: bitwise_xor(bitwise_xor(x, y), z),
    3: lambda x, y, z: bitwise_xor(y, bitwise_or(y, bitwise_not(z)))
}

def bitwise_and(a, b):
    return "".join(map(lambda x: "1" if x[0] == x[1] and x[0] == "1" else "0", zip(a, b)))

def bitwise_or(a, b):
    return "".join(map(lambda x: "1" if x[0] == "1" or x[1] == "1" else "0", zip(a, b)))

def bitwise_not(a):
    return "".join(map(lambda x: "1" if x[0] == "0" else "0", a))

def bitwise_xor(a, b):
    return "".join(map(lambda x: "1" if x[0] != x[1] else "0", zip(a, b)))

def get_binary(word):
    binary_word = ""
    for letter in word:
        binary_word = binary_word + bin(ALPHABETS.index(letter))[2:].rjust(8, "0")
    extra_length = 512 - (len(binary_word) % 512 + 64)
    return binary_word + "1" + (extra_length - 1) * "0" + bin(len(binary_word))[2:].rjust(64, "0")

def get_md5_block(user_input):
    global A, B, C, D
    sub_blocks = [user_input[i:i + 32] for i in range(0, len(user_input), 32)]
    for round in range(4):  # 16 operations per round
        for sub_round in range(16):
            if 0 <= round * 16 + sub_round <= 15:
                k = sub_round
            elif 16 <= round * 16 + sub_round <= 31:
                k = (5 * (round * 16 + sub_round) + 1) % 16
            elif 32 <= round * 16 + sub_round <= 47:
                k = (3 * (round * 16 + sub_round) + 5) % 16
            else:
                k = (7 * (round * 16 + sub_round)) % 16
            intermediate = bitwise_and(A, non_linear_process[round](B, C, D))
            intermediate = bitwise_and(intermediate, K[round * 16 + sub_round])
            intermediate = bitwise_and(intermediate, sub_blocks[k])
            shift = SHIFTS[round * 16 + sub_round]
            intermediate = intermediate[shift:] + intermediate[:shift]
            intermediate = bitwise_or(intermediate, B)
            A, B, C, D = D, intermediate, B, C

def get_md5(user_input):
    start = 0
    for _ in range(len(user_input) // 512):
        get_md5_block(user_input[start: start + 512])
        start += 512
    return D + C + B + A

user_input = get_binary(input("Enter text: "))
print("Hash is:", hex(int(get_md5(user_input), 2)))


##SAMPLE INPUT : qwerty
#SAMPLE OUTPUT : 0xefcfab8fefcfab8fefcfab8fefcfab8f