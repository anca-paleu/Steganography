import cv2
import numpy as np
from utils import bits_to_text, strip_lsb_image, unshuffle
from extraction import _build_edge_map, _ring_coords, _local_pattern

def _extract_channel_corrected(stego_channel, tc_matrix, n_bits: int, max_bits_to_extract: int):
    smoothed = strip_lsb_image(stego_channel, n_bits)
    edge_map = _build_edge_map(smoothed)
    extracted_bits = []
    restored = stego_channel.copy()

    bit_index = 0
    rows, cols = smoothed.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):
            if bit_index >= max_bits_to_extract:
                return extracted_bits, restored

            ring = _ring_coords(i, j)
            pattern = _local_pattern(smoothed, i, j, ring)
            edge_positions = [(r, c) for (r, c) in ring if edge_map[r, c] == 255]
            edge_pattern = [pattern[k] for k, (r, c) in enumerate(ring) if edge_map[r, c] == 255]

            if not edge_positions:
                continue

            capacity = min(len(edge_positions), max_bits_to_extract - bit_index)
            edge_positions = edge_positions[:capacity]
            edge_pattern = edge_pattern[:capacity]

            is_sequence = [int(tc_matrix[r, c]) ^ (int(stego_channel[r, c]) % 2) for (r, c) in edge_positions]
            shuffled_h = [is_sequence[k] ^ int(tc_matrix[edge_positions[k][0], edge_positions[k][1]]) for k in range(capacity)]
            h = unshuffle(shuffled_h, seed=bit_index)

            for k in range(capacity):
                extracted_bits.append(h[k] ^ edge_pattern[k])

            for k, (r, c) in enumerate(edge_positions):
                tc_val = int(tc_matrix[r, c])
                s_val  = int(stego_channel[r, c])

                if s_val % 2 != 0:
                    restored[r, c] = np.clip(s_val - tc_val, 0, 255)
                else:
                    restored[r, c] = np.clip(s_val + tc_val, 0, 255)

            bit_index += capacity

    return extracted_bits, restored

def extract_rgb_corrected(stego_image, tc_rgb, n_bits: int, total_message_length: int):
    b, g, r = cv2.split(stego_image)
    tc_b, tc_g, tc_r = tc_rgb
    remaining_length = total_message_length

    bits_b, rest_b = _extract_channel_corrected(b, tc_b, n_bits, remaining_length)
    remaining_length -= len(bits_b)

    bits_g, rest_g = _extract_channel_corrected(g, tc_g, n_bits, remaining_length)
    remaining_length -= len(bits_g)

    bits_r, rest_r = _extract_channel_corrected(r, tc_r, n_bits, remaining_length)

    all_bits = bits_b + bits_g + bits_r
    restored_img = cv2.merge([rest_b, rest_g, rest_r])

    return bits_to_text(all_bits), restored_img

if __name__ == "__main__":
    import os
    from config import COVER_DIR, SHORT_MESSAGE, N_BITS
    from embedding import embed_rgb

    test_image = os.path.join(COVER_DIR, 'lena_color.tiff')
    print("Embedding RGB ...")
    stego, tc_rgb, msg_len = embed_rgb(test_image, SHORT_MESSAGE, N_BITS)

    print("Extracting RGB (corrected) ...")
    text, restored = extract_rgb_corrected(stego, tc_rgb, N_BITS, msg_len)

    cover = cv2.imread(test_image, cv2.IMREAD_COLOR)
    cover = cv2.resize(cover, (512, 512))

    print(f"Recovered message : {text[:80]}")
    print(f"Recovery success  : {text.strip() == SHORT_MESSAGE.strip()}")
    print(f"Cover reconstruit identic : {bool(np.array_equal(cover, restored))}")