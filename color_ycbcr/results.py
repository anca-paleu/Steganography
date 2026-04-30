import os
import cv2
import matplotlib.pyplot as plt

from embedding import embed_ycbcr 
import metrics as m
from config import COVER_DIR, N_BITS, LONG_MESSAGE, IMAGE_NAME_MAP

def _load_cover_images(cover_dir: str):
    extensions = ('.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir) if f.lower().endswith(extensions))
    results = []
    for filename in filenames:
        path = os.path.join(cover_dir, filename)
        cover = cv2.imread(path, cv2.IMREAD_COLOR)
        if cover is None:
            continue
        cover = cv2.resize(cover, (512, 512))
        display_name = IMAGE_NAME_MAP.get(filename, os.path.splitext(filename)[0].capitalize())
        results.append((filename, display_name, cover, path))
    return results

def _render_table(rows, col_headers, title):
    all_rows = [col_headers] + rows
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis('tight'); ax.axis('off')
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

def show_metrics_table(cover_dir: str = COVER_DIR, secret_text: str = LONG_MESSAGE, n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    rows = []
    for filename, display_name, cover, path in images:
        print(f"  Processing {display_name} ...")
        stego, tc_ycc, msg_len = embed_ycbcr(path, secret_text, n_bits)
        rows.append([
            display_name,
            f"{m.embedding_capacity(msg_len, tc_ycc):.4f}",
            f"{m.psnr(cover, stego):.2f}",
            f"{m.ssim(cover, stego):.4f}",
            f"{m.entropy(cover):.4f}",
            f"{m.entropy(stego):.4f}",
            f"{m.correlation(cover, stego):.6f}",
            f"{m.cosine_similarity(cover, stego):.6f}",
        ])
    _render_table(rows, ["Image", "EC", "PSNR (dB)", "SSIM", "Entropy C", "Entropy S", "Correlation", "Cosine Sim."], "Performance metrics (YCbCr)")

def show_ttest_table(cover_dir: str = COVER_DIR, secret_text: str = LONG_MESSAGE, n_bits: int = N_BITS):
    images = _load_cover_images(cover_dir)
    rows = []
    for filename, display_name, cover, path in images:
        stego, _, _ = embed_ycbcr(path, secret_text, n_bits)
        rows.append([display_name, f"{m.t_test(cover, stego):.4f}"])
    _render_table(rows, ["Image Name", "t-Test Value"], "Table 4. t-Test values (YCbCr)")

if __name__ == "__main__":
    show_metrics_table()
    show_ttest_table()