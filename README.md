# rePPG: Rewriting Photoplethysmography Signal to Video

## Introduction

**rePPG** is a deep learning-based framework that generates a new synthetic image by inserting a desired rPPG signal while preserving the original facial image.
The method is designed to preserve the facial biometric signals (e.g., heart rate) while maintaining the visual quality.

This project is an official implementation of the paper **"Rewriting Photoplethysmography Signal to Video"**.

## Features
- Modify rPPG signal while maintaining original face image
- Controllable by separating Appearance Feature and PPG Feature
- Validate inserted signal through various rPPG extraction algorithms
- Performance verified on PURE and UBFC datasets
- A new approach for **Privacy-preserving biometric video processing**

---
## Results

### Quantitative Evaluation
```python
## rPPG Embedding Accuracy Results
results = {
    "Method": ["OMIT", "LGI", "ICA", "CHROM", "POS", "PBV", "GREEN"],
    "MAE ↓": [1.10, 1.30, 2.30, 3.00, 6.45, 26.82, 42.20],
    "RMSE ↓": [5.02, 5.48, 10.56, 8.90, 15.16, 39.63, 53.08],
    "PTE6 ↑": [95.00, 94.17, 92.50, 87.50, 77.31, 42.86, 20.00],
    "PEARSON ↑": [0.9371, 0.9384, 0.2079, 0.8461, 0.4317, 0.1504, 0.2915]
}

## PSNR & SSIM Results
visual_quality = {
    "Dataset": ["PURE", "UBFC"],
    "PSNR ↑": [24.61, 20.35],
    "SSIM ↑": [0.638, 0.630]
}

df_visual = pd.DataFrame(visual_quality)
print(df_visual)

## Intsallation

### Install Dependencies
```bash
git clone https://github.com/your-repository/rePPG.git
cd rePPG
pip install -r requirements.txt
