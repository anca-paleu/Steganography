from Crypto.Cipher import AES, ChaCha20
from Crypto.Util.Padding import pad, unpad

AES_KEY = b"ThisIsASecretKey"
AES_IV  = b"InitVectorForCBC"

CHACHA_KEY   = b"ThisIsASecretKeyForChaCha20Use!!"
CHACHA_NONCE = b"NonceVal"


def aes_encrypt_text(text: str, key: bytes = AES_KEY, iv: bytes = AES_IV) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(text.encode('utf-8'), AES.block_size)
    return cipher.encrypt(padded)


def aes_decrypt_text(ciphertext: bytes, key: bytes = AES_KEY, iv: bytes = AES_IV) -> str:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = cipher.decrypt(ciphertext)
    return unpad(padded, AES.block_size).decode('utf-8')


def aes_decrypt_partial(ciphertext: bytes, key: bytes = AES_KEY, iv: bytes = AES_IV) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.decrypt(ciphertext)


def chacha20_encrypt_text(text: str, key: bytes = CHACHA_KEY, nonce: bytes = CHACHA_NONCE) -> bytes:
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(text.encode('utf-8'))


def chacha20_decrypt_bytes(ciphertext: bytes, key: bytes = CHACHA_KEY, nonce: bytes = CHACHA_NONCE) -> bytes:
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.decrypt(ciphertext)


def bytes_to_bit_string(data: bytes) -> str:
    return data.decode('latin-1')


def bit_string_to_bytes(text: str) -> bytes:
    return text.encode('latin-1')