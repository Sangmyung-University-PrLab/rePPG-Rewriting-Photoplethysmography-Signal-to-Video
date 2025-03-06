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

### Generated Signal Qaulity

| Method | MAE ↓ | RMSE ↓ | PTE6 ↑ | PEARSON ↑ |
|--------|-------|--------|--------|-----------|
| OMIT   | 1.10  | 5.02   | 95.00  | 0.9371    |
| LGI    | 1.30  | 5.48   | 94.17  | 0.9384    |
| ICA    | 2.30  | 10.56  | 92.50  | 0.2079    |
| CHROM  | 3.00  | 8.90   | 87.50  | 0.8461    |
| POS    | 6.45  | 15.16  | 77.31  | 0.4317    |
| PBV    | 26.82 | 39.63  | 42.86  | 0.1504    |
| GREEN  | 42.20 | 53.08  | 20.00  | 0.2915    |

### Generated Visual Quality

| Dataset | PSNR ↑ | SSIM ↑ |
|---------|--------|--------|
| PURE    | 24.61  | 0.638  |
| UBFC    | 20.35  | 0.630  |

## Intsallation

### Install Dependencies
```bash
git clone https://github.com/your-repository/rePPG.git
cd rePPG
pip install -r requirements.txt
