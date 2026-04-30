"""
Histograms – Blue Channel Method
===================================
PDH și histograma de intensitate, calculate pe canalul Blue.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from embedding import embed
from config import IMAGE_SIZE, N_BITS, MASSIVE_MESSAGE


def _compute_pdh(channel):
    """Pixel Difference Histogram pe un singur canal (2D array)."""
    img_int = channel.astype(np.int16)
    differences = img_int[:, :-1] - img_int[:, 1:]
    frequencies, bin_edges = np.histogram(differences, bins=511, range=(-255, 255))
    return frequencies, bin_edges[:-1]


def plot_pdh(cover_path: str, image_name: str,
             secret_text: str = MASSIVE_MESSAGE, n_bits: int = N_BITS):
    print(f"[PDH] Computing for {image_name} ...")

    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover_bgr = cv2.resize(cover_bgr, IMAGE_SIZE)
    cover_blue = cover_bgr[:, :, 0]

    stego_bgr, _, _ = embed(cover_path, secret_text, n_bits)
    stego_blue = stego_bgr[:, :, 0]

    freq_cover, bins_cover = _compute_pdh(cover_blue)
    freq_stego, bins_stego = _compute_pdh(stego_blue)

    marker_indices = list(range(0, len(bins_cover), 12))
    if 255 not in marker_indices:
        marker_indices.append(255)
        marker_indices.sort()

    plt.figure(figsize=(10, 6))

    plt.plot(bins_cover, freq_cover,
             linestyle='-', color='black', linewidth=0.8,
             marker='s', markerfacecolor='none', markersize=5,
             markevery=marker_indices,
             label=f'Cover {image_name} (Blue)')

    plt.plot(bins_stego, freq_stego,
             linestyle='-', color='black', linewidth=0.8,
             marker='o', markerfacecolor='none', markersize=5,
             markevery=marker_indices,
             label=f'Stego {image_name} (Blue)')

    plt.title(f'Pixel Difference Histogram (PDH) – Blue Channel – {image_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Pixel difference', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency of occurrence', fontsize=12, fontweight='bold')
    plt.xlim([-600, 600])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend(loc='center left')
    plt.tight_layout()

    print(f"  -> Displaying PDH for {image_name}. Close the window to continue.")
    plt.show(block=True)


def plot_intensity_histogram(cover_path: str, image_name: str,
                             secret_text: str = MASSIVE_MESSAGE, n_bits: int = N_BITS):
    print(f"[Intensity] Computing for {image_name} ...")

    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover_bgr = cv2.resize(cover_bgr, IMAGE_SIZE)
    cover_blue = cover_bgr[:, :, 0]

    stego_bgr, _, _ = embed(cover_path, secret_text, n_bits)
    stego_blue = stego_bgr[:, :, 0]

    plt.figure(figsize=(10, 6))

    plt.hist(cover_blue.flatten(), bins=256, range=[0, 256],
             color='blue', alpha=0.6, density=True,
             label=f'Cover – Blue channel ({image_name})')

    plt.hist(stego_blue.flatten(), bins=256, range=[0, 256],
             color='red', alpha=0.6, density=True,
             label=f'Stego – Blue channel ({image_name})')

    plt.title(f'Histogram of Blue Channel Pixel Intensities – {image_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim([0, 255])
    plt.legend(loc='upper right')
    plt.tight_layout()

    print(f"  -> Displaying intensity histogram for {image_name}. Close the window to continue.")
    plt.show(block=True)


if __name__ == "__main__":
    import os
    from config import COVER_DIR

    images = {
        'Lena'   : os.path.join(COVER_DIR, 'lena_color.tiff'),
        'Baboon' : os.path.join(COVER_DIR, '4.2.03.tiff'),
        'Wheel'  : os.path.join(COVER_DIR, '5.2.08.tiff'),
        'F16'    : os.path.join(COVER_DIR, '4.2.05.tiff'),
    }

    print("=== Generating histogram figures – Blue Channel Method ===\n")

    plot_pdh(images['Lena'],   'Lena')
    plot_pdh(images['Baboon'], 'Baboon')

    plot_intensity_histogram(images['Wheel'], 'Wheel')
    plot_intensity_histogram(images['F16'],   'F16')

    print("\n=== Done. ===")