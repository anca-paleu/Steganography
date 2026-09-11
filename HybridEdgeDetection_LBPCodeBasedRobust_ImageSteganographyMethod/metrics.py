import numpy as np
import cv2

def embedding_capacity(embedded_bits: int, tc_matrix) -> float:
    changed_pixels = int(np.sum(tc_matrix == 1))
    if changed_pixels == 0:
        return 0.0
    return embedded_bits / changed_pixels


def psnr(cover, stego) -> float:
    c = cover.astype(np.float64)
    s = stego.astype(np.float64)
    mse = np.mean((c - s) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse)

def ssim(cover, stego) -> float:

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    c = cover.astype(np.float64)
    s = stego.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(c, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(s, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(c ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(s ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(c * s, -1, window)[5:-5, 5:-5] - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / den

    return float(np.mean(ssim_map))


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
    return float(abs(mean_d / np.sqrt(variance)) * np.sqrt(N))