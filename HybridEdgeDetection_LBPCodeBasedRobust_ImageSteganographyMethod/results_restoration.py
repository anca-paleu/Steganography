import os
import cv2
import matplotlib.pyplot as plt

from embedding import embed
from extraction import extract
from extraction_corrected import extract_corrected
import metrics as m
from config import COVER_DIR, N_BITS, LONG_MESSAGE, IMAGE_NAME_MAP


def _load_cover_images(cover_dir: str):
    extensions = ('.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    filenames  = sorted(f for f in os.listdir(cover_dir)
                        if f.lower().endswith(extensions))

    results = []
    for filename in filenames:
        path  = os.path.join(cover_dir, filename)
        cover = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if cover is None:
            print(f"  [!] Could not read {filename}, skipping.")
            continue
        cover = cv2.resize(cover, (512, 512))
        display_name = IMAGE_NAME_MAP.get(filename,
                           os.path.splitext(filename)[0].capitalize())
        results.append((filename, display_name, cover, path))

    return results


def _render_table(rows, col_headers, title):
    all_rows = [col_headers] + rows

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.55 + 1.5)))
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


def _run_restoration_table(extract_fn, title,
                            cover_dir: str, secret_text: str, n_bits: int):
    images = _load_cover_images(cover_dir)
    if not images:
        print("[!] No images found.")
        return

    print("Computing metrics ... please wait.\n")
    rows = []

    for filename, display_name, cover, path in images:
        print(f"  Processing {display_name} ...")
        stego, tc, msg_len = embed(path, secret_text, n_bits)
        _, restored = extract_fn(stego, tc, n_bits, msg_len)

        rows.append([
            display_name,
            f"{m.embedding_capacity(msg_len, tc):.4f}",
            f"{m.psnr(cover, restored):.2f}",
            f"{m.ssim(cover, restored):.4f}",
            f"{m.entropy(cover):.4f}",
            f"{m.entropy(restored):.4f}",
            f"{m.correlation(cover, restored):.6f}",
            f"{m.cosine_similarity(cover, restored):.6f}",
        ])

    headers = ["Image", "EC", "PSNR (dB)", "SSIM",
               "Entropy C", "Entropy R", "Correlation", "Cosine Sim."]
    _render_table(rows, headers, title)


def show_restored_metrics_table_as_article(cover_dir: str = COVER_DIR,
                                           secret_text: str = LONG_MESSAGE,
                                           n_bits: int = N_BITS):
    _run_restoration_table(extract,
                            "Restoration metrics — as-in-article formula (Cover vs Restored)",
                            cover_dir, secret_text, n_bits)


def show_restored_metrics_table_corrected(cover_dir: str = COVER_DIR,
                                          secret_text: str = LONG_MESSAGE,
                                          n_bits: int = N_BITS):
    _run_restoration_table(extract_corrected,
                            "Restoration metrics — corrected formula (Cover vs Restored)",
                            cover_dir, secret_text, n_bits)


if __name__ == "__main__":
    print("=== As-in-article restoration ===")
    show_restored_metrics_table_as_article()

    print("\n=== Corrected restoration ===")
    show_restored_metrics_table_corrected()