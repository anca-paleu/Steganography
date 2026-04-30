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

    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    stego_bgr = cv2.imread(stego_path, cv2.IMREAD_COLOR)

    if cover_bgr is None or stego_bgr is None:
        print("[!] OpenCV could not read one of the images.")
        return

    cover_bgr = cv2.resize(cover_bgr, (512, 512))


    for ch_idx, ch_name in enumerate(['Blue', 'Green', 'Red']):
        diff = cover_bgr[:, :, ch_idx].astype(np.int16) - \
               stego_bgr[:, :, ch_idx].astype(np.int16)
        changed = int(np.sum(diff != 0))
        print(f"  {ch_name} channel – changed pixels: {changed:,}")

    diff_blue = cover_bgr[:, :, 0].astype(np.int16) - \
                stego_bgr[:, :, 0].astype(np.int16)

    changed_pixels    = int(np.sum(diff_blue != 0))
    total_pixels      = cover_bgr[:, :, 0].size
    change_percentage = (changed_pixels / total_pixels) * 100
    max_diff          = int(np.max(np.abs(diff_blue)))

    print(f"\nCover image : {cover_path}")
    print(f"Stego image : {stego_path}")
    print(f"Total pixels (Blue)   : {total_pixels:,}")
    print(f"Changed pixels (Blue) : {changed_pixels:,}")
    print(f"Percentage changed    : {change_percentage:.4f}%")
    print(f"Max pixel delta       : {max_diff}")
    print()

    if changed_pixels == 0:
        print("WARNING: Blue channel is 100% identical – embedding did not work!")
    else:
        print("OK: Blue channel differs – embedding applied correctly.")
        print("OK: Green and Red channels should be 0 (unmodified).")


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

    print("=== Single image verification (Blue Channel Method) ===\n")
    verify_pixel_differences(cover, stego)

    print("=== Batch verification ===\n")
    verify_all()