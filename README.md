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

## Usage

### Data

To run this project, you will need face video data (`sample_face.npy`) and cPPG signal data (`sample_sig.npy`).
These files must be created and saved in the `./sample_data/` directory.

#### Data Format
- `face.npy`: A `numpy` array of Shape `(N, H, W, 3)`, containing face video data with N frames (e.g. `(1873, 256, 256, 3)`)
- `cppg.npy`: A `numpy` array containing rPPG and cPPG signals, with the following dictionary structure
'''python
{
"cppg_signal": np.array([...]), # An array of cPPG signals of length N
"cppg_bpm": np.array([...]) # An array of cPPG BPMs of length N
}

### Pretrained Weight
To run this project, you will need face pretrained weight (`pretrained.pt`).
These files must be created and saved in the `./weight/` directory.
