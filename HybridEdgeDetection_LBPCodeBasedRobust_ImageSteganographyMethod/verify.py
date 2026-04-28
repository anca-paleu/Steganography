import os
import cv2
import numpy as np

from config import COVER_DIR, STEGO_DIR


def verify_pixel_differences(cover_path: str, stego_path: str):
    if not os.path.exists(cover_path):
        print(f"[!] Cover image not found: {cover_path}")
        return
    if not os.path.exists(stego_path):
        print(f"[!] Stego image not found: {stego_path}")
        return

    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

    if cover is None or stego is None:
        print("[!] OpenCV could not read one of the images.")
        return

    diff              = cover.astype(np.int16) - stego.astype(np.int16)
    changed_pixels    = int(np.sum(diff != 0))
    total_pixels      = cover.size
    change_percentage = (changed_pixels / total_pixels) * 100
    max_diff          = int(np.max(np.abs(diff)))

    print(f"Cover image : {cover_path}")
    print(f"Stego image : {stego_path}")
    print(f"Total pixels      : {total_pixels:,}")
    print(f"Changed pixels    : {changed_pixels:,}")
    print(f"Percentage changed: {change_percentage:.4f}%")
    print(f"Max pixel delta   : {max_diff}")
    print()

    if changed_pixels == 0:
        print("WARNING: Images are 100% identical. "
              "The message was not embedded (check the embedding algorithm).")
    else:
        print("OK: Images differ — embedding was applied correctly.")


def verify_all(cover_dir: str = COVER_DIR, stego_dir: str = STEGO_DIR):
    extensions = ('.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    cover_files = sorted(f for f in os.listdir(cover_dir)
                         if f.lower().endswith(extensions))

    if not cover_files:
        print(f"[!] No cover images found in {cover_dir}")
        return

    for filename in cover_files:
        cover_path = os.path.join(cover_dir, filename)
        stego_path = os.path.join(stego_dir, f"stego_{filename}")
        print(f"--- {filename} ---")
        verify_pixel_differences(cover_path, stego_path)


if __name__ == "__main__":
    cover = os.path.join(COVER_DIR, 'lena_color.tiff')
    stego = os.path.join(STEGO_DIR, 'stego_lena_color.tiff')

    print("=== Single image verification ===\n")
    verify_pixel_differences(cover, stego)
    print("=== Batch verification ===\n")
    verify_all()