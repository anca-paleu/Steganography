"""
Extraction – Blue Channel Method
==================================
Extrage mesajul ascuns din canalul Blue al imaginii stego
și restaurează imaginea originală.
"""

import cv2
import numpy as np

from utils import bits_to_text, strip_lsb_image, unshuffle
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
    return cv2.dilate(combined, dilation_mask, iterations=1)


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


def extract(stego_bgr, tc_matrix, n_bits: int, message_length: int):
    """
    Extrage mesajul din canalul Blue și restaurează imaginea.

    Parameters
    ----------
    stego_bgr      : np.ndarray  – imaginea stego color (BGR)
    tc_matrix      : np.ndarray  – matricea de modificări
    n_bits         : int
    message_length : int         – numărul de biți de extras

    Returns
    -------
    text           : str         – mesajul recuperat
    restored_bgr   : np.ndarray  – imaginea restaurată (BGR)
    """
    # Extragem canalele
    stego_blue = stego_bgr[:, :, 0]
    green      = stego_bgr[:, :, 1]
    red        = stego_bgr[:, :, 2]

    smoothed_blue = strip_lsb_image(stego_blue, n_bits)
    edge_map      = _build_edge_map(smoothed_blue)

    extracted_bits = []
    restored_blue  = stego_blue.copy()

    bit_index = 0
    rows, cols = smoothed_blue.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):

            if bit_index >= message_length:
                restored_bgr = cv2.merge([restored_blue, green, red])
                return bits_to_text(extracted_bits), restored_bgr

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

            is_sequence = [int(tc_matrix[r, c]) ^ (int(stego_blue[r, c]) % 2)
                           for (r, c) in edge_positions]

            shuffled_h = [is_sequence[k] ^ int(tc_matrix[edge_positions[k][0],
                                                          edge_positions[k][1]])
                          for k in range(capacity)]

            h = unshuffle(shuffled_h, seed=bit_index)

            for k in range(capacity):
                extracted_bits.append(h[k] ^ edge_pattern[k])

            # Restaurare canal Blue
            for k, (r, c) in enumerate(edge_positions):
                sl_val = edge_pattern[k]
                tc_val = int(tc_matrix[r, c])
                s_val  = int(stego_blue[r, c])

                if sl_val == 1:
                    restored_blue[r, c] = np.clip(s_val + tc_val, 0, 255)
                else:
                    restored_blue[r, c] = np.clip(s_val - tc_val, 0, 255)

            bit_index += capacity

    restored_bgr = cv2.merge([restored_blue, green, red])
    return bits_to_text(extracted_bits), restored_bgr


if __name__ == "__main__":
    import os
    from config import COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS
    from embedding import embed

    test_image = os.path.join(COVER_DIR, 'lena_color.tiff')

    print("Embedding (Blue Channel) ...")
    stego_bgr, tc, msg_len = embed(test_image, SHORT_MESSAGE, N_BITS)

    print("Extracting (Blue Channel) ...")
    text, restored = extract(stego_bgr, tc, N_BITS, msg_len)

    print(f"Recovered message : {text[:80]} ...")
    print(f"Recovery success  : {text.strip() == SHORT_MESSAGE.strip()}")