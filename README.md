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

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ (CUDA support available)
- OpenCV
- NumPy
- SciPy
- Matplotlib

### Install Dependencies
```bash
git clone https://github.com/your-repository/rePPG.git
cd rePPG
pip install -r requirements.txt
