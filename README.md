# Hybrid Edge Detection & LBP Code-Based Image Steganography

## Description
This repository contains a 1:1 Python implementation of the image steganography method described in the research paper: **"A Novel Hybrid Edge Detection and LBP Code-Based Robust Image Steganography Method"**.

The project demonstrates a high-capacity, robust data-hiding technique that combines hybrid edge detection (logical OR of multiple detectors) with Local Binary Pattern (LBP) codes to minimize image distortion while maximizing security.

## Original Paper Reference
All credit for the algorithm design, methodology, and scientific research belongs to the original authors:

* **Title:** A Novel Hybrid Edge Detection and LBP Code-Based Robust Image Steganography Method
* **Authors:** Habiba Sultana, A. H. M. Kamal, Gahangir Hossain, and Muhammad Ashad Kabir
* **Publication:** *Future Internet* 2023, 15(3), 108; MDPI.
* **DOI:** [https://doi.org/10.3390/fi15030108](https://doi.org/10.3390/fi15030108)

This repository serves as an independent implementation to facilitate testing, verification, and further study of the proposed method.

## Project Structure
* `config.py` - Global parameters and configuration settings.
* `embedding.py` - Implementation of the data embedding (hiding) process.
* `extraction.py` - Implementation of the data extraction process.
* `metrics.py` - Functions for calculating performance metrics (PSNR, SSIM, MSE).
* `histograms.py` - Utility to generate and compare histograms of cover vs. stego images.
* `utils.py` - Helper functions for image processing and bit manipulation.
* `verify.py` / `results.py` - Scripts used to validate results against the paper's findings.
* `cover_images/` - Directory containing original test images (e.g., Lena, Baboon, Boat).
* `stego_images/` - Directory where generated stego images are saved.

## Prerequisites
To run this project, you need the following Python packages:

```bash
pip install numpy opencv-python scikit-image matplotlib
