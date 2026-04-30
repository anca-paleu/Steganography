import cv2
import numpy as np
from utils import bits_to_text, strip_lsb_image, unshuffle
from config import CANNY_LOW, CANNY_HIGH, SOBEL_THRESH, IMAGE_SIZE

def _build_edge_map(image):
    edge_canny = cv2.Canny(image, CANNY_LOW, CANNY_HIGH)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    _, edge_sobel = cv2.threshold(sobel_magnitude, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    edge_sobel = np.uint8(edge_sobel)

    combined = cv2.bitwise_or(edge_canny, edge_sobel)
    dilation_mask = np.ones((3, 3), np.uint8)
    return cv2.dilate(combined, dilation_mask, iterations=1)

def _ring_coords(center_row, center_col):
    r, c = center_row, center_col
    return [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c+1), (r+1, c+1), (r+1, c), (r+1, c-1), (r, c-1)]

def _local_pattern(image, center_row, center_col, ring):
    center_value = image[center_row, center_col]
    return [1 if image[r, c] < center_value else 0 for (r, c) in ring]

def _extract_channel(stego_channel, tc_matrix, n_bits: int, max_bits_to_extract: int):
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
                sl_val = edge_pattern[k]
                tc_val = int(tc_matrix[r, c])
                s_val = int(stego_channel[r, c])

                if sl_val == 1:
                    restored[r, c] = np.clip(s_val + tc_val, 0, 255)
                else:
                    restored[r, c] = np.clip(s_val - tc_val, 0, 255)

            bit_index += capacity

    return extracted_bits, restored

def extract_rgb(stego_image, tc_rgb, n_bits: int, total_message_length: int):
    b, g, r = cv2.split(stego_image)
    tc_b, tc_g, tc_r = tc_rgb
    remaining_length = total_message_length

    # Extragem succesiv din canale
    bits_b, rest_b = _extract_channel(b, tc_b, n_bits, remaining_length)
    remaining_length -= len(bits_b)

    bits_g, rest_g = _extract_channel(g, tc_g, n_bits, remaining_length)
    remaining_length -= len(bits_g)

    bits_r, rest_r = _extract_channel(r, tc_r, n_bits, remaining_length)

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

    print("Extracting RGB ...")
    text, restored = extract_rgb(stego, tc_rgb, N_BITS, msg_len)

    print(f"Recovered message : {text[:80]} ...")
    print(f"Recovery success  : {text.strip() == SHORT_MESSAGE.strip()}")