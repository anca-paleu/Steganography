import os
import cv2
import csv
import numpy as np
from embedding import embed 
import metrics as m
from config import LONG_MESSAGE, N_BITS

def process_large_datasets(base_dir: str, csv_filename: str):
    headers = [
        "Dataset", "Class", "Image Name", 
        "EC (Capacity)", "PSNR (dB)", "SSIM", 
        "Entropy Cover", "Entropy Stego", "Correlation", 
        "Cosine Sim", "t-Test"
    ]
    
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for root, dirs, files in os.walk(base_dir):
            cls = os.path.basename(root)
            dataset = os.path.basename(os.path.dirname(root))
            
            image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            
            if not image_files:
                continue 
                
            print(f"\n--> Processing Dataset: [{dataset}] | Class: [{cls}] ({len(image_files)} images)")
            
            for img_name in image_files:
                img_path = os.path.join(root, img_name)
                
                cover = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if cover is None:
                    continue
                
                cover = cv2.resize(cover, (512, 512)) 
                
                try:
                    stego, tc, msg_len = embed(img_path, LONG_MESSAGE, N_BITS)
                    
                    ec       = m.embedding_capacity(msg_len, tc)
                    psnr_val = m.psnr(cover, stego)
                    ssim_val = m.ssim(cover, stego)
                    ent_c    = m.entropy(cover)
                    ent_s    = m.entropy(stego)
                    corr     = m.correlation(cover, stego)
                    cos_sim  = m.cosine_similarity(cover, stego)
                    ttest    = m.t_test(cover, stego)
                    
                    writer.writerow([
                        dataset, cls, img_name, 
                        f"{ec:.4f}", f"{psnr_val:.2f}", f"{ssim_val:.4f}", 
                        f"{ent_c:.4f}", f"{ent_s:.4f}", f"{corr:.6f}", 
                        f"{cos_sim:.6f}", f"{ttest:.4f}"
                    ])
                    
                except Exception as e:
                    print(f"    [!] Error processing {img_name}: {e}")

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\anca\Desktop\Licenta\Articolul 2 - criptare\imagini"
    
    CSV_OUTPUT = os.path.join("steganography", "results.csv")
    
    print(f"=== Starting image processing from: {DATASET_PATH} ===")
    process_large_datasets(DATASET_PATH, CSV_OUTPUT)
    print(f"\n=== Processing completed successfully! ===")
    print(f"Results saved to: {CSV_OUTPUT}")