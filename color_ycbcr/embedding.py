import cv2
import numpy as np
from utils import text_to_bits, strip_lsb_image, shuffle
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

def _local_pattern(image_ref, center_row, center_col, ring):
    center_value = image_ref[center_row, center_col]
    return [1 if image_ref[r, c] < center_value else 0 for (r, c) in ring]

def _embed_channel_with_ref(target_channel, ref_y_smooth, edge_map, message_bits):
    message_length = len(message_bits)
    stego = target_channel.copy()
    tc = np.zeros_like(target_channel, dtype=np.uint8)
    bit_index = 0
    rows, cols = ref_y_smooth.shape

    for i in range(1, rows - 1, 3):
        for j in range(1, cols - 1, 3):
            if bit_index >= message_length:
                return stego, tc, bit_index

            ring = _ring_coords(i, j)
            pattern = _local_pattern(ref_y_smooth, i, j, ring)
            edge_positions = [(r, c) for (r, c) in ring if edge_map[r, c] == 255]
            edge_pattern = [pattern[k] for k, (r, c) in enumerate(ring) if edge_map[r, c] == 255]

            if not edge_positions:
                continue

            capacity = min(len(edge_positions), message_length - bit_index)
            edge_positions = edge_positions[:capacity]
            edge_pattern = edge_pattern[:capacity]
            block_bits = message_bits[bit_index:bit_index + capacity]

            xor_bits = [edge_pattern[k] ^ block_bits[k] for k in range(capacity)]
            shuffled = shuffle(xor_bits, seed=bit_index)

            for k, (r, c) in enumerate(edge_positions):
                pixel = int(stego[r, c])
                if shuffled[k] == 1:
                    if pixel % 2 == 0:
                        stego[r, c] = np.clip(pixel + 1, 0, 255)
                        tc[r, c] = 1
                    else:
                        tc[r, c] = 0
                else:
                    if pixel % 2 != 0:
                        stego[r, c] = np.clip(pixel - 1, 0, 255)
                        tc[r, c] = 1
                    else:
                        tc[r, c] = 0

            bit_index += capacity

    return stego, tc, bit_index

def embed_ycbcr(cover_path: str, secret_text: str, n_bits: int):
    cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover = cv2.resize(cover, IMAGE_SIZE)
    
    img_ycc = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycc)

    y_smooth = strip_lsb_image(y, n_bits)
    edge_map = _build_edge_map(y_smooth)

    message_bits = text_to_bits(secret_text)

    stego_y, tc_y, emb_y = _embed_channel_with_ref(y, y_smooth, edge_map, message_bits)
    stego_cr, tc_cr, emb_cr = _embed_channel_with_ref(cr, y_smooth, edge_map, message_bits[emb_y:])
    stego_cb, tc_cb, emb_cb = _embed_channel_with_ref(cb, y_smooth, edge_map, message_bits[emb_y + emb_cr:])

    total_embedded = emb_y + emb_cr + emb_cb
    stego_ycc = cv2.merge([stego_y, stego_cr, stego_cb])
    stego_bgr = cv2.cvtColor(stego_ycc, cv2.COLOR_YCrCb2BGR)
    tc_ycc = (tc_y, tc_cr, tc_cb)

    return stego_bgr, tc_ycc, total_embedded

def embed_all_images(cover_dir: str, stego_dir: str, secret_text: str, n_bits: int):
    import os
    os.makedirs(stego_dir, exist_ok=True)
    extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir) if f.lower().endswith(extensions))

    results = []
    for filename in filenames:
        cover_path = os.path.join(cover_dir, filename)
        stego_path = os.path.join(stego_dir, f"stego_{filename}")
        print(f"  Embedding YCbCr into {filename} ...")
        stego, _, _ = embed_ycbcr(cover_path, secret_text, n_bits)
        cv2.imwrite(stego_path, stego)

if __name__ == "__main__":
    from config import COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS
    print("Running batch YCbCr embedding ...")
    embed_all_images(COVER_DIR, STEGO_DIR, SHORT_MESSAGE, N_BITS)