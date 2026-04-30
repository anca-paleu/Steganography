import random

def text_to_bits(text: str) -> list[int]:
    return [int(b) for char in text for b in format(ord(char), '08b')]

def bits_to_text(bits: list[int]) -> str:
    chars = []
    for i in range(0, len(bits) - len(bits) % 8, 8):
        byte = bits[i:i + 8]
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def strip_lsb(pixel_value: int, n: int) -> int:
    return pixel_value - (pixel_value % (2 ** n))

def strip_lsb_image(image, n: int):
    import numpy as np
    return image - (image % (2 ** n))

def shuffle(sequence: list, seed: int) -> list:
    result = sequence.copy()
    random.seed(seed)
    random.shuffle(result)
    return result

def unshuffle(shuffled: list, seed: int) -> list:
    n = len(shuffled)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    original = [None] * n
    for original_pos, shuffled_pos in zip(indices, range(n)):
        original[original_pos] = shuffled[shuffled_pos]
    return original