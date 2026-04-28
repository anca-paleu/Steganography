import numpy as np
from skimage.metrics import structural_similarity as _ssim


def embedding_capacity(message_length: int, tc_matrix) -> float:
    changed_pixels = int(np.sum(tc_matrix == 1))
    if changed_pixels == 0:
        return 0.0
    return message_length / changed_pixels


def psnr(cover, stego) -> float:
    c = cover.astype(np.float64)
    s = stego.astype(np.float64)
    mse = np.mean((c - s) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)

def ssim(cover, stego) -> float:
    return _ssim(cover, stego, data_range=255)


def entropy(image) -> float:
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    probabilities = hist / np.sum(hist)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))

def correlation(cover, stego) -> float:
    return float(np.corrcoef(cover.flatten(), stego.flatten())[0, 1])

def cosine_similarity(cover, stego) -> float:
    c = cover.flatten().astype(np.float64)
    s = stego.flatten().astype(np.float64)
    numerator   = np.sum(c * s)
    denominator = np.sqrt(np.sum(c ** 2)) * np.sqrt(np.sum(s ** 2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def t_test(cover, stego) -> float:
    D   = cover.astype(np.float64) - stego.astype(np.float64)
    N   = D.size
    sum_d  = np.sum(D)
    sum_d2 = np.sum(D ** 2)

    variance = (sum_d2 - (sum_d ** 2 / N)) / (N - 1)
    if variance <= 0:
        return 0.0

    mean_d = sum_d / N
    return float(abs(mean_d / np.sqrt(variance)))