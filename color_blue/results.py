import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from embedding import embed
import metrics as m
from config import COVER_DIR, N_BITS, LONG_MESSAGE, IMAGE_NAME_MAP


def _load_cover_images(cover_dir: str):
    extensions = ('.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir)
                        if f.lower().endswith(extensions))

    results = []
    for filename in filenames:
        path      = os.path.join(cover_dir, filename)
        cover_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if cover_bgr is None:
            print(f"  [!] Could not read {filename}, skipping.")
            continue
        cover_bgr = cv2.resize(cover_bgr, (512, 512))
        display_name = IMAGE_NAME_MAP.get(filename,
                           os.path.splitext(filename)[0].capitalize())
        results.append((filename, display_name, cover_bgr, path))

    return results


def _render_table(rows, col_headers, title):
    all_rows = [col_headers] + rows

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=all_rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')

    plt.title(title, pad=20, fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()



def show_metrics_table(cover_dir: str = COVER_DIR,
                       secret_text: str = LONG_MESSAGE,
                       n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    if not images:
        print("[!] No images found.")
        return

    print("Computing metrics (Blue Channel Method) ... please wait.\n")
    rows = []

    for filename, display_name, cover_bgr, path in images:
        print(f"  Processing {display_name} ...")

        stego_bgr, tc, msg_len = embed(path, secret_text, n_bits)

        cover_blue = cover_bgr[:, :, 0]
        stego_blue = stego_bgr[:, :, 0]

        rows.append([
            display_name,
            f"{m.embedding_capacity(msg_len, tc):.4f}",
            f"{m.psnr(cover_blue, stego_blue):.2f}",
            f"{m.ssim(cover_blue, stego_blue):.4f}",
            f"{m.entropy(cover_blue):.4f}",
            f"{m.entropy(stego_blue):.4f}",
            f"{m.correlation(cover_blue, stego_blue):.6f}",
            f"{m.cosine_similarity(cover_blue, stego_blue):.6f}",
        ])

    headers = ["Image", "EC", "PSNR (dB)", "SSIM",
               "Entropy C", "Entropy S", "Correlation", "Cosine Sim."]
    _render_table(rows, headers,
                  "Performance metrics – Blue Channel Method (metrics on Blue channel)")


def show_metrics_table_full_image(cover_dir: str = COVER_DIR,
                                  secret_text: str = LONG_MESSAGE,
                                  n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    if not images:
        print("[!] No images found.")
        return

    print("Computing full-image metrics (Blue Channel Method) ... please wait.\n")
    rows = []

    for filename, display_name, cover_bgr, path in images:
        print(f"  Processing {display_name} ...")

        stego_bgr, tc, msg_len = embed(path, secret_text, n_bits)

        cover_gray = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2GRAY)

        rows.append([
            display_name,
            f"{m.embedding_capacity(msg_len, tc):.4f}",
            f"{m.psnr(cover_gray, stego_gray):.2f}",
            f"{m.ssim(cover_gray, stego_gray):.4f}",
            f"{m.entropy(cover_gray):.4f}",
            f"{m.entropy(stego_gray):.4f}",
            f"{m.correlation(cover_gray, stego_gray):.6f}",
            f"{m.cosine_similarity(cover_gray, stego_gray):.6f}",
        ])

    headers = ["Image", "EC", "PSNR (dB)", "SSIM",
               "Entropy C", "Entropy S", "Correlation", "Cosine Sim."]
    _render_table(rows, headers,
                  "Performance metrics – Blue Channel Method (full image, grayscale)")


def show_ttest_table(cover_dir: str = COVER_DIR,
                     secret_text: str = LONG_MESSAGE,
                     n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    if not images:
        print("[!] No images found.")
        return

    print("Computing t-test values (Blue Channel Method) ... please wait.\n")
    rows = []

    for filename, display_name, cover_bgr, path in images:
        print(f"  Processing {display_name} ...")
        stego_bgr, _, _ = embed(path, secret_text, n_bits)

        cover_blue = cover_bgr[:, :, 0]
        stego_blue = stego_bgr[:, :, 0]

        rows.append([display_name, f"{m.t_test(cover_blue, stego_blue):.4f}"])

    _render_table(rows, ["Image Name", "t-Test Value"],
                  "t-Test values – Blue Channel Method")


if __name__ == "__main__":
    print("=== Metrics table (Blue channel only) ===")
    show_metrics_table()

    print("\n=== Metrics table (full image) ===")
    show_metrics_table_full_image()

    print("\n=== t-Test table ===")
    show_ttest_table()