import cv2
import numpy as np

from utils import text_to_bits, strip_lsb_image, shuffle
from config import CANNY_LOW, CANNY_HIGH, SOBEL_THRESH, IMAGE_SIZE


def _build_edge_map(blue_channel):
    edge_canny = cv2.Canny(blue_channel, CANNY_LOW, CANNY_HIGH)

    sobel_x = cv2.Sobel(blue_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blue_channel, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    _, edge_sobel = cv2.threshold(sobel_magnitude, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    edge_sobel = np.uint8(edge_sobel)

    combined = cv2.bitwise_or(edge_canny, edge_sobel)

    dilation_mask = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(combined, dilation_mask, iterations=1)
    return dilated


def _ring_coords(center_row, center_col):
    r, c = center_row, center_col
    return [
        (r-1, c-1), (r-1, c), (r-1, c+1),
        (r,   c+1),
        (r+1, c+1), (r+1, c), (r+1, c-1),
        (r,   c-1),
    ]


def _local_pattern(channel, center_row, center_col, ring):
    center_value = channel[center_row, center_col]
    return [1 if channel[r, c] < center_value else 0 for (r, c) in ring]



def embed(cover_path: str, secret_text: str, n_bits: int):
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover_bgr = cv2.resize(cover_bgr, IMAGE_SIZE)

    blue  = cover_bgr[:, :, 0].copy()
    green = cover_bgr[:, :, 1].copy()
    red   = cover_bgr[:, :, 2].copy()

    smoothed_blue = strip_lsb_image(blue, n_bits)
    edge_map      = _build_edge_map(smoothed_blue)

    message_bits   = text_to_bits(secret_text)
    message_length = len(message_bits)

    stego_blue = blue.copy()
    tc         = np.zeros_like(blue, dtype=np.uint8)

    bit_index = 0
    rows, cols = smoothed_blue.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):

            if bit_index >= message_length:
                stego_bgr = cv2.merge([stego_blue, green, red])
                return stego_bgr, tc, message_length

            ring    = _ring_coords(i, j)
            pattern = _local_pattern(smoothed_blue, i, j, ring)

            edge_positions = [(r, c) for (r, c) in ring if edge_map[r, c] == 255]
            edge_pattern   = [pattern[k] for k, (r, c) in enumerate(ring)
                              if edge_map[r, c] == 255]

            if not edge_positions:
                continue

            capacity       = min(len(edge_positions), message_length - bit_index)
            edge_positions = edge_positions[:capacity]
            edge_pattern   = edge_pattern[:capacity]
            block_bits     = message_bits[bit_index:bit_index + capacity]

            xor_bits = [edge_pattern[k] ^ block_bits[k] for k in range(capacity)]
            shuffled = shuffle(xor_bits, seed=bit_index)

            for k, (r, c) in enumerate(edge_positions):
                pixel = int(stego_blue[r, c])
                if shuffled[k] == 1:
                    if pixel % 2 == 0:
                        stego_blue[r, c] = pixel + 1
                        tc[r, c] = 1
                    else:
                        tc[r, c] = 0
                else:
                    if pixel % 2 != 0:
                        stego_blue[r, c] = pixel - 1
                        tc[r, c] = 1
                    else:
                        tc[r, c] = 0

            bit_index += capacity

    stego_bgr = cv2.merge([stego_blue, green, red])
    return stego_bgr, tc, message_length


def embed_all_images(cover_dir: str, stego_dir: str, secret_text: str, n_bits: int):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(stego_dir, exist_ok=True)

    extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir)
                        if f.lower().endswith(extensions))

    if not filenames:
        print(f"[!] No images found in: {cover_dir}")
        return []

    results = []
    for filename in filenames:
        cover_path = os.path.join(cover_dir, filename)
        stego_path = os.path.join(stego_dir, f"stego_{filename}")

        print(f"  Embedding into {filename} ...")
        stego_bgr, _, _ = embed(cover_path, secret_text, n_bits)
        cv2.imwrite(stego_path, stego_bgr)

        display_name = filename.split('.')[0].capitalize()
        stego_rgb = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2RGB)
        results.append((display_name, stego_rgb))

    n      = len(results)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    for i, (name, stego_img) in enumerate(results):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(f"({chr(97 + i)}) {name}")
        plt.imshow(stego_img)
        plt.axis('off')

    plt.suptitle("Stego images – Blue Channel Method", fontweight='bold')
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    from config import COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS

    print("Running batch embedding (Blue Channel Method) ...")
    embed_all_images(COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS)