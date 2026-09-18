import cv2
import numpy as np

from utils import bits_to_text, strip_lsb_image, unshuffle
from config import CANNY_LOW, CANNY_HIGH, SOBEL_THRESH, IMAGE_SIZE


def _bgr_to_ycocg(image_bgr):
    b = image_bgr[:, :, 0].astype(np.int16)
    g = image_bgr[:, :, 1].astype(np.int16)
    r = image_bgr[:, :, 2].astype(np.int16)

    co  = r - b
    tmp = b + (co >> 1)
    cg  = g - tmp
    y   = tmp + (cg >> 1)

    return y, co, cg


def _ycocg_to_bgr(y, co, cg):
    y  = y.astype(np.int16)
    co = co.astype(np.int16)
    cg = cg.astype(np.int16)

    tmp = y - (cg >> 1)
    g   = cg + tmp
    b   = tmp - (co >> 1)
    r   = b + co

    restored_bgr = np.zeros((*y.shape, 3), dtype=np.uint8)
    restored_bgr[:, :, 0] = np.clip(b, 0, 255)
    restored_bgr[:, :, 1] = np.clip(g, 0, 255)
    restored_bgr[:, :, 2] = np.clip(r, 0, 255)
    return restored_bgr


def _reconstruct_bgr_single(y_val, co_val, cg_val):
    tmp = y_val - (cg_val >> 1)
    g = cg_val + tmp
    b = tmp - (co_val >> 1)
    r = b + co_val
    return b, g, r


def _in_range(*values):
    return all(0 <= v <= 255 for v in values)


def _is_risky(target_idx, y_val, co_val, cg_val):
    values = [y_val, co_val, cg_val]
    pixel = values[target_idx]

    candidate = pixel + 1 if pixel % 2 == 0 else pixel - 1

    trial = values.copy()
    trial[target_idx] = candidate
    b, g, r = _reconstruct_bgr_single(*trial)
    return not _in_range(b, g, r)


def _build_edge_map(y_channel):
    y_uint8 = np.clip(y_channel, 0, 255).astype(np.uint8)
    edge_canny = cv2.Canny(y_uint8, CANNY_LOW, CANNY_HIGH)

    sobel_x = cv2.Sobel(y_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(y_uint8, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    _, edge_sobel = cv2.threshold(sobel_magnitude, SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    edge_sobel = np.uint8(edge_sobel)

    combined = cv2.bitwise_or(edge_canny, edge_sobel)
    dilation_mask = np.ones((3, 3), np.uint8)
    return cv2.dilate(combined, dilation_mask, iterations=1)


def _ring_coords(center_row, center_col):
    r, c = center_row, center_col
    return [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c+1), (r+1, c+1), (r+1, c), (r+1, c-1), (r, c-1)]


def _local_pattern(image_ref, center_row, center_col, ring):
    center_value = image_ref[center_row, center_col]
    return [1 if image_ref[r, c] < center_value else 0 for (r, c) in ring]


def _filter_safe_positions(target_idx, tc_target, y_arr, co_arr, cg_arr, edge_positions, edge_pattern):
    safe_positions = []
    safe_pattern   = []
    for k, (r, c) in enumerate(edge_positions):
        if int(tc_target[r, c]) == 1:
            safe_positions.append((r, c))
            safe_pattern.append(edge_pattern[k])
            continue

        y_val, co_val, cg_val = int(y_arr[r, c]), int(co_arr[r, c]), int(cg_arr[r, c])
        if not _is_risky(target_idx, y_val, co_val, cg_val):
            safe_positions.append((r, c))
            safe_pattern.append(edge_pattern[k])

    return safe_positions, safe_pattern


def _extract_from_channel(stego, tc, edge_positions, edge_pattern, bit_index):
    capacity = len(edge_positions)

    is_sequence = [int(tc[r, c]) ^ (int(stego[r, c]) % 2) for (r, c) in edge_positions]
    shuffled_h  = [is_sequence[k] ^ int(tc[edge_positions[k][0], edge_positions[k][1]])
                   for k in range(capacity)]
    h = unshuffle(shuffled_h, seed=bit_index)

    bits = [h[k] ^ edge_pattern[k] for k in range(capacity)]

    for k, (r, c) in enumerate(edge_positions):
        sl_val = edge_pattern[k]
        tc_val = int(tc[r, c])
        s_val  = int(stego[r, c])

        if sl_val == 1:
            stego[r, c] = s_val + tc_val
        else:
            stego[r, c] = s_val - tc_val

    return bits


def extract_ycbcr(stego_image, tc_ycc, n_bits: int, total_message_length: int):
    y, co, cg = _bgr_to_ycocg(stego_image)
    tc_y, tc_co, tc_cg = tc_ycc

    y_uint8  = np.clip(y, 0, 255).astype(np.uint8)
    y_smooth = strip_lsb_image(y_uint8, n_bits)
    edge_map = _build_edge_map(y_smooth)

    restored_y, restored_co, restored_cg = y.copy(), co.copy(), cg.copy()
    stego_channels = [restored_y, restored_co, restored_cg]
    tc_channels    = [tc_y, tc_co, tc_cg]

    extracted_bits = []
    channel_turn = 0
    bit_index = 0
    rows, cols = y_smooth.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):

            if bit_index >= total_message_length:
                restored_bgr = _ycocg_to_bgr(restored_y, restored_co, restored_cg)
                return bits_to_text(extracted_bits), restored_bgr

            ring    = _ring_coords(i, j)
            pattern = _local_pattern(y_smooth, i, j, ring)

            edge_positions = [(r, c) for (r, c) in ring if edge_map[r, c] == 255]
            edge_pattern   = [pattern[k] for k, (r, c) in enumerate(ring)
                              if edge_map[r, c] == 255]

            if not edge_positions:
                continue

            target_idx = channel_turn % 3
            channel_turn += 1

            edge_positions, edge_pattern = _filter_safe_positions(
                target_idx, tc_channels[target_idx], y, co, cg, edge_positions, edge_pattern)

            if not edge_positions:
                continue

            capacity       = min(len(edge_positions), total_message_length - bit_index)
            edge_positions = edge_positions[:capacity]
            edge_pattern   = edge_pattern[:capacity]

            bits = _extract_from_channel(stego_channels[target_idx], tc_channels[target_idx],
                                          edge_positions, edge_pattern, bit_index)
            extracted_bits.extend(bits)

            bit_index += capacity

    restored_bgr = _ycocg_to_bgr(restored_y, restored_co, restored_cg)
    return bits_to_text(extracted_bits), restored_bgr


if __name__ == "__main__":
    import os
    from config import COVER_DIR, SHORT_MESSAGE, N_BITS
    from embedding import embed_ycbcr

    test_image = os.path.join(COVER_DIR, 'lena_color.tiff')
    stego, tc_ycc, msg_len = embed_ycbcr(test_image, SHORT_MESSAGE, N_BITS)
    text, restored = extract_ycbcr(stego, tc_ycc, N_BITS, msg_len)

    print(f"Recovered message : {text[:80]} ...")
    print(f"Recovery success  : {text.strip() == SHORT_MESSAGE.strip()}")