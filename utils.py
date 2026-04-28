import random

def text_to_binary(text):
    return [int(b) for char in text for b in format(ord(char), '08b')]

def binary_to_text(bits):
    chars = []
    for i in range(0, len(bits) - len(bits) % 8, 8):
        byte = bits[i:i+8]
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def shuffle_bits(H, seed=42):
    HS = H.copy()
    random.seed(seed)
    random.shuffle(HS)
    return HS

def unshuffle_bits(HS, seed=42):
    n = len(HS)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    
    H = [None] * n
    for original_pos, shuffled_pos in zip(indices, range(n)):
        H[original_pos] = HS[shuffled_pos]
    return H