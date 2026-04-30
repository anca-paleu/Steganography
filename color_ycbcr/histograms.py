import cv2
import numpy as np
import matplotlib.pyplot as plt

from embedding import embed_rgb
from config import IMAGE_SIZE, N_BITS, MASSIVE_MESSAGE

def _compute_pdh(image):
    img_int = image.astype(np.int16)
    flat_img = img_int.flatten()
    differences = flat_img[:-1] - flat_img[1:]
    frequencies, bin_edges = np.histogram(differences, bins=511, range=(-255, 255))
    return frequencies, bin_edges[:-1]

def plot_pdh(cover_path: str, image_name: str, secret_text: str = MASSIVE_MESSAGE, n_bits: int = N_BITS):
    print(f"[PDH] Computing RGB for {image_name} ...")
    cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover = cv2.resize(cover, IMAGE_SIZE)
    stego, _, _ = embed_rgb(cover_path, secret_text, n_bits)

    freq_cover, bins_cover = _compute_pdh(cover)
    freq_stego, bins_stego = _compute_pdh(stego)

    marker_indices = list(range(0, len(bins_cover), 12))
    if 255 not in marker_indices:
        marker_indices.append(255)
        marker_indices.sort()

    plt.figure(figsize=(10, 6))
    plt.plot(bins_cover, freq_cover, linestyle='-', color='black', linewidth=0.8,
             marker='s', markerfacecolor='none', markersize=5, markevery=marker_indices, label=f'Cover {image_name}')
    plt.plot(bins_stego, freq_stego, linestyle='-', color='black', linewidth=0.8,
             marker='o', markerfacecolor='none', markersize=5, markevery=marker_indices, label=f'Stego {image_name}')

    plt.title(f'Pixel Difference Histogram (PDH) RGB — {image_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel difference', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency of occurrence', fontsize=12, fontweight='bold')
    plt.xlim([-600, 600])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend(loc='center left')
    plt.tight_layout()
    plt.show(block=True)

def plot_intensity_histogram(cover_path: str, image_name: str, secret_text: str = MASSIVE_MESSAGE, n_bits: int = N_BITS):
    print(f"[Intensity] Computing RGB for {image_name} ...")
    cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    cover = cv2.resize(cover, IMAGE_SIZE)
    stego, _, _ = embed_rgb(cover_path, secret_text, n_bits)

    plt.figure(figsize=(10, 6))
    plt.hist(cover.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.6, density=True, label=f'Cover Image {image_name}')
    plt.hist(stego.flatten(), bins=256, range=[0, 256], color='red', alpha=0.6, density=True, label=f'Stego Image {image_name}')

    plt.title('Histogram of Pixel Intensities (RGB)', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim([0, 255])
    plt.legend(loc='upper right')
    plt.tight_layout()
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
    plot_pdh(images['Lena'], 'Lena')
    plot_intensity_histogram(images['Wheel'], 'Wheel')