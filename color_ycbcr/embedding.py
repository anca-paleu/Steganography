import cv2
import numpy as np

from utils import text_to_bits, strip_lsb_image, shuffle
from config import CANNY_LOW, CANNY_HIGH, SOBEL_THRESH, IMAGE_SIZE


def _bgr_to_ycocg(cover_bgr):
    b = cover_bgr[:, :, 0].astype(np.int16)
    g = cover_bgr[:, :, 1].astype(np.int16)
    r = cover_bgr[:, :, 2].astype(np.int16)

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

    stego_bgr = np.zeros((*y.shape, 3), dtype=np.uint8)
    stego_bgr[:, :, 0] = np.clip(b, 0, 255)
    stego_bgr[:, :, 1] = np.clip(g, 0, 255)
    stego_bgr[:, :, 2] = np.clip(r, 0, 255)
    return stego_bgr


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


def _filter_safe_positions(target_idx, y_arr, co_arr, cg_arr, edge_positions, edge_pattern):
    safe_positions = []
    safe_pattern   = []
    for k, (r, c) in enumerate(edge_positions):
        y_val, co_val, cg_val = int(y_arr[r, c]), int(co_arr[r, c]), int(cg_arr[r, c])
        if not _is_risky(target_idx, y_val, co_val, cg_val):
            safe_positions.append((r, c))
            safe_pattern.append(edge_pattern[k])
    return safe_positions, safe_pattern


def _embed_into_channel(target, tc, edge_positions, edge_pattern, block_bits, bit_index):
    capacity = len(edge_positions)
    xor_bits = [edge_pattern[k] ^ block_bits[k] for k in range(capacity)]
    shuffled = shuffle(xor_bits, seed=bit_index)

    for k, (r, c) in enumerate(edge_positions):
        pixel = int(target[r, c])
        if shuffled[k] == 1:
            if pixel % 2 == 0:
                target[r, c] = pixel + 1
                tc[r, c] = 1
            else:
                tc[r, c] = 0
        else:
            if pixel % 2 != 0:
                target[r, c] = pixel - 1
                tc[r, c] = 1
            else:
                tc[r, c] = 0


def embed_ycbcr(cover_path: str, secret_text: str, n_bits: int):
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover_bgr = cv2.resize(cover_bgr, IMAGE_SIZE)

    y, co, cg = _bgr_to_ycocg(cover_bgr)

    y_uint8  = np.clip(y, 0, 255).astype(np.uint8)
    y_smooth = strip_lsb_image(y_uint8, n_bits)
    edge_map = _build_edge_map(y_smooth)

    message_bits   = text_to_bits(secret_text)
    message_length = len(message_bits)

    stego_y, stego_co, stego_cg = y.copy(), co.copy(), cg.copy()
    tc_y  = np.zeros(y.shape, dtype=np.uint8)
    tc_co = np.zeros(co.shape, dtype=np.uint8)
    tc_cg = np.zeros(cg.shape, dtype=np.uint8)

    stego_channels = [stego_y, stego_co, stego_cg]
    tc_channels    = [tc_y, tc_co, tc_cg]

    channel_turn = 0
    bit_index = 0
    rows, cols = y_smooth.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):

            if bit_index >= message_length:
                stego_bgr = _ycocg_to_bgr(stego_y, stego_co, stego_cg)
                return stego_bgr, (tc_y, tc_co, tc_cg), bit_index

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
                target_idx, stego_y, stego_co, stego_cg, edge_positions, edge_pattern)

            if not edge_positions:
                continue

            capacity       = min(len(edge_positions), message_length - bit_index)
            edge_positions = edge_positions[:capacity]
            edge_pattern   = edge_pattern[:capacity]
            block_bits     = message_bits[bit_index:bit_index + capacity]

            _embed_into_channel(stego_channels[target_idx], tc_channels[target_idx],
                                 edge_positions, edge_pattern, block_bits, bit_index)

            bit_index += capacity

    stego_bgr = _ycocg_to_bgr(stego_y, stego_co, stego_cg)
    return stego_bgr, (tc_y, tc_co, tc_cg), bit_index


def embed_all_images(cover_dir: str, stego_dir: str, secret_text: str, n_bits: int):
    import os
    os.makedirs(stego_dir, exist_ok=True)
    extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir) if f.lower().endswith(extensions))

    for filename in filenames:
        cover_path = os.path.join(cover_dir, filename)
        stego_path = os.path.join(stego_dir, f"stego_{filename}")
        print(f"  Embedding YCoCg into {filename} ...")
        stego, _, _ = embed_ycbcr(cover_path, secret_text, n_bits)
        cv2.imwrite(stego_path, stego)


if __name__ == "__main__":
    from config import COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS
    print("Running batch YCoCg embedding ...")
    embed_all_images(COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS)