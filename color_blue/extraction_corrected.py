import cv2
import numpy as np

from utils import bits_to_text, strip_lsb_image, unshuffle
from extraction import _build_edge_map, _ring_coords, _local_pattern


def extract_corrected(stego_bgr, tc_matrix, n_bits: int, message_length: int):
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

            for k, (r, c) in enumerate(edge_positions):
                tc_val = int(tc_matrix[r, c])
                s_val  = int(stego_blue[r, c])

                if s_val % 2 != 0:
                    restored_blue[r, c] = np.clip(s_val - tc_val, 0, 255)
                else:
                    restored_blue[r, c] = np.clip(s_val + tc_val, 0, 255)

            bit_index += capacity

    restored_bgr = cv2.merge([restored_blue, green, red])
    return bits_to_text(extracted_bits), restored_bgr


if __name__ == "__main__":
    import os
    from config import COVER_DIR, SHORT_MESSAGE, N_BITS
    from embedding import embed

    test_image = os.path.join(COVER_DIR, 'lena_color.tiff')

    print("Embedding (Blue Channel) ...")
    stego_bgr, tc, msg_len = embed(test_image, SHORT_MESSAGE, N_BITS)

    print("Extracting (Blue Channel, corrected) ...")
    text, restored = extract_corrected(stego_bgr, tc, N_BITS, msg_len)

    cover_bgr = cv2.imread(test_image, cv2.IMREAD_COLOR)
    cover_bgr = cv2.resize(cover_bgr, (512, 512))

    print(f"Recovered message : {text[:80]} ...")
    print(f"Recovery success  : {text.strip() == SHORT_MESSAGE.strip()}")
    print(f"Cover reconstruit identic : {bool(np.array_equal(cover_bgr, restored))}")