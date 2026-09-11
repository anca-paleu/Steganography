import os
import cv2
import matplotlib.pyplot as plt

from embedding import embed
from extraction_corrected import extract_corrected
from crypto import (
    aes_encrypt_text, aes_decrypt_partial,
    chacha20_encrypt_text, chacha20_decrypt_bytes,
    bytes_to_bit_string, bit_string_to_bytes,
)
import metrics as m
from config import COVER_DIR, N_BITS, LONG_MESSAGE, IMAGE_NAME_MAP


def _load_cover_images(cover_dir: str):
    extensions = ('.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    filenames = sorted(f for f in os.listdir(cover_dir) if f.lower().endswith(extensions))

    results = []
    for filename in filenames:
        path = os.path.join(cover_dir, filename)
        cover = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if cover is None:
            print(f"  [!] Could not read {filename}, skipping.")
            continue
        cover = cv2.resize(cover, (512, 512))
        display_name = IMAGE_NAME_MAP.get(filename, os.path.splitext(filename)[0].capitalize())
        results.append((filename, display_name, cover, path))

    return results


def _render_table(rows, col_headers, title):
    all_rows = [col_headers] + rows

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=all_rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')

    plt.title(title, pad=20, fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


def _verify_aes(extracted_secret: str, secret_text: str) -> bool:
    recovered_bytes = bit_string_to_bytes(extracted_secret)
    aligned_length  = (len(recovered_bytes) // 16) * 16
    aligned_bytes   = recovered_bytes[:aligned_length]

    decrypted_prefix = aes_decrypt_partial(aligned_bytes)
    expected_prefix  = secret_text.encode('utf-8')[:aligned_length]
    return decrypted_prefix == expected_prefix


def _verify_chacha20(extracted_secret: str, secret_text: str) -> bool:
    recovered_bytes  = bit_string_to_bytes(extracted_secret)
    decrypted_prefix = chacha20_decrypt_bytes(recovered_bytes)
    expected_prefix  = secret_text.encode('utf-8')[:len(recovered_bytes)]
    return decrypted_prefix == expected_prefix


def show_metrics_table_encrypted(algorithm: str,
                                  cover_dir: str = COVER_DIR,
                                  secret_text: str = LONG_MESSAGE,
                                  n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    if not images:
        print("[!] No images found.")
        return

    print(f"Computing metrics for {algorithm.upper()} ... please wait.\n")

    if algorithm == "aes":
        ciphertext = aes_encrypt_text(secret_text)
        verify_fn  = _verify_aes
    elif algorithm == "chacha20":
        ciphertext = chacha20_encrypt_text(secret_text)
        verify_fn  = _verify_chacha20
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    encoded_secret = bytes_to_bit_string(ciphertext)

    rows = []
    for filename, display_name, cover, path in images:
        print(f"  Processing {display_name} ...")
        stego, tc, msg_len = embed(path, encoded_secret, n_bits)
        extracted_secret, restored = extract_corrected(stego, tc, n_bits, msg_len)

        try:
            text_match = verify_fn(extracted_secret, secret_text)
        except Exception:
            text_match = False

        rows.append([
            display_name,
            f"{m.embedding_capacity(msg_len, tc):.4f}",
            f"{m.psnr(cover, restored):.2f}",
            f"{m.ssim(cover, restored):.4f}",
            f"{m.entropy(cover):.4f}",
            f"{m.entropy(restored):.4f}",
            f"{m.correlation(cover, restored):.6f}",
            f"{m.cosine_similarity(cover, restored):.6f}",
            "OK" if text_match else "FAIL",
        ])

    headers = ["Image", "EC", "PSNR (dB)", "SSIM",
               "Entropy C", "Entropy R", "Correlation", "Cosine Sim.", "Text Match"]
    _render_table(rows, headers,
                  f"Performance metrics with {algorithm.upper()}-encrypted secret message")


if __name__ == "__main__":
    print("=== AES ===")
    show_metrics_table_encrypted("aes")

    print("\n=== ChaCha20 ===")
    show_metrics_table_encrypted("chacha20")