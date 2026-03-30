# AI-Enhanced Multi-Modal Secure Steganography System
## Project Planning Document

**Project:** B.Tech Major Project (Project-II), AY 2025-26
**Institution:** Delhi Technological University, Dept. of Computer Science Engineering
**Team:** Naman Padiyar (2K22/CO/297), Nipun Rawat (2K22/CO/311), Piyush Thalwal (2K22/CO/328)

---

## 1. Project Overview

A multi-modal steganography system that embeds AES-256-GCM encrypted data into images, audio, and video using both classical and state-of-the-art deep learning techniques. **Primary focus: Video Steganography.**

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    React + Tailwind Frontend              │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │  Image    │  │  Audio   │  │  Video (Primary)      │  │
│  │  Encode/  │  │  Encode/ │  │  Encode/Decode        │  │
│  │  Decode   │  │  Decode  │  │  + Metrics Dashboard  │  │
│  └────┬─────┘  └────┬─────┘  └───────────┬───────────┘  │
└───────┼──────────────┼────────────────────┼──────────────┘
        │              │                    │
        ▼              ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Backend (REST API)              │
│  /api/image/*     /api/audio/*      /api/video/*         │
└───────┬──────────────┬────────────────────┬──────────────┘
        │              │                    │
        ▼              ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                    Core Algorithm Layer                   │
│                                                          │
│  ┌─────────────┐  ┌──────────────────────────────────┐  │
│  │ AES-256-GCM │  │  Steganography Methods:          │  │
│  │ Encryption  │──│  • LSB (baseline)                 │  │
│  │ + SHA-256   │  │  • DCT (Quantization Index Mod.)  │  │
│  │ + PBKDF2    │  │  • DWT (Multi-level wavelet)      │  │
│  └─────────────┘  │  • U-Net++ Attention (DL)         │  │
│                   │  • HiDDeN Adversarial (DL)        │  │
│                   │  • Invertible Neural Network (DL)  │  │
│                   └──────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Evaluation Metrics:                              │   │
│  │  PSNR, SSIM, MS-SSIM, MSE, SNR, LPIPS, BER      │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Techniques Used (Modern 2025 State-of-the-Art)

### 3.1 Classical Methods (Baseline + Transform Domain)

| Method | Domain | Technique | Key Innovation |
|--------|--------|-----------|----------------|
| **LSB** | Spatial | Multi-bit (1-4 bits/channel) | Randomized pixel order via seed key |
| **DCT** | Frequency | Quantization Index Modulation on 8x8 blocks | Adaptive strength based on block texture complexity |
| **DWT** | Frequency | Multi-level Haar/Daubechies decomposition | Embed in detail coefficients with magnitude-adaptive strength |

### 3.2 Deep Learning Methods (Advanced)

| Model | Architecture | Key Features |
|-------|-------------|--------------|
| **Attention U-Net++** | Encoder-Decoder with dense skip connections | CBAM attention, ConvNeXt blocks, multi-scale message injection, residual learning |
| **HiDDeN Adversarial** | Encoder + Decoder + WGAN-GP Discriminator | Spectral normalization, noise layer (JPEG/Gaussian/Crop/Blur/Codec), frequency loss |
| **Invertible Neural Network** | Normalizing flow with affine coupling layers | Haar wavelet lifting, mathematically invertible, 3D temporal attention for video |

### 3.3 Video-Specific Techniques (PRIMARY FOCUS)

| Technique | Description |
|-----------|-------------|
| **Motion-Compensated Embedding** | Optical flow (Farneback) identifies stable regions; embed only in low-motion areas |
| **Temporal Consistency** | 3D CNN temporal attention ensures no visible flicker across frames |
| **Frame Selection Strategy** | Skip frames (every Nth) + stability-based filtering |
| **Codec-Robust Training** | Noise layer simulates H.264/H.265 compression during training |
| **Multi-scale Decomposition** | Haar wavelet lifting + invertible blocks in frequency domain |

### 3.4 Security

| Component | Technique |
|-----------|-----------|
| **Encryption** | AES-256-GCM (authenticated encryption — provides confidentiality + integrity) |
| **Key Derivation** | PBKDF2-HMAC-SHA256, 600,000 iterations (OWASP 2024 recommendation) |
| **Integrity** | SHA-256 hash verification |
| **Anti-Steganalysis** | Frequency-domain loss minimizes spectral anomalies |

### 3.5 Training Pipeline

- **Optimizer:** AdamW with weight decay
- **Scheduler:** Cosine annealing with warm restarts
- **Precision:** Automatic Mixed Precision (AMP) for speed
- **Losses:** MSE + MS-SSIM + LPIPS (perceptual) + BCE (message) + Frequency + WGAN-GP (adversarial)
- **Tracking:** Weights & Biases (wandb) experiment dashboard
- **Noise Layer:** Random selection from: JPEG, Gaussian noise, crop, blur, codec simulation, identity

---

## 4. Project Directory Structure

```
steganography-project/
├── config/
│   ├── __init__.py
│   └── settings.py                 # Global config (dataclasses)
├── core/
│   ├── encryption/
│   │   ├── aes_cipher.py           # AES-256-GCM + PBKDF2
│   │   └── integrity.py            # SHA-256 hashing
│   ├── image/
│   │   ├── lsb.py                  # Image LSB
│   │   ├── dct_stego.py            # Image DCT + QIM
│   │   └── dwt_stego.py            # Image DWT (multi-level)
│   ├── audio/
│   │   ├── lsb.py                  # Audio LSB
│   │   └── dwt_stego.py            # Audio DWT (Daubechies-4)
│   ├── video/
│   │   ├── frame_utils.py          # PyAV I/O, optical flow, frame selection
│   │   ├── lsb.py                  # Video LSB + motion comp.
│   │   ├── dct_stego.py            # Video DCT + temporal
│   │   └── dwt_stego.py            # Video DWT + frame selection
│   └── metrics/
│       └── evaluate.py             # PSNR, SSIM, MSE, SNR, BER, plots
├── models/
│   ├── layers.py                   # CBAM, ConvNeXt, noise layers
│   ├── losses.py                   # Combined loss functions
│   ├── train.py                    # Training pipeline (HiDDeN, U-Net, INN)
│   ├── unet/
│   │   └── encoder_decoder.py      # Attention U-Net++ model
│   ├── hidden/
│   │   └── hidden_model.py         # HiDDeN + WGAN-GP discriminator
│   └── invertible/
│       └── inn_model.py            # INN with Haar wavelet + 3D temporal
├── api/
│   └── main.py                     # FastAPI backend
├── frontend/                       # React + Tailwind + Vite
│   ├── src/
│   │   ├── components/             # Navbar, FileUpload, MetricsDisplay
│   │   ├── pages/                  # Home, Image, Audio, Video pages
│   │   └── utils/                  # API client
│   └── package.json
├── scripts/
│   ├── download_test_data.py       # Test data downloader
│   └── run_demo.py                 # CLI demo script
├── data/                           # Test images, audio, video
├── requirements.txt
└── PLANNING.md                     # This document
```

---

## 5. Data Flow

### Encoding (Hide)
```
Secret Message
    ↓
AES-256-GCM Encrypt (password → PBKDF2 → key)
    ↓
Encrypted Bytes + SHA-256 Hash
    ↓
Select Steganography Method (LSB / DCT / DWT / DL)
    ↓
[For Video: Extract Frames → Optical Flow → Select Stable Frames]
    ↓
Embed in Cover Media
    ↓
Output: Stego Media + Quality Metrics
```

### Decoding (Reveal)
```
Stego Media
    ↓
Select Same Method
    ↓
[For Video: Extract Frames → Same Frame Selection]
    ↓
Extract Embedded Data
    ↓
AES-256-GCM Decrypt (validates authentication tag)
    ↓
Original Secret Message (+ integrity verified)
```

---

## 6. Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **PSNR** | > 35 dB | Higher = less visual distortion |
| **SSIM** | > 0.97 | 1.0 = identical structural similarity |
| **MS-SSIM** | > 0.95 | Multi-scale perceptual quality |
| **MSE** | < 10 | Lower = less pixel error |
| **SNR** | > 30 dB | Signal quality |
| **LPIPS** | < 0.05 | Deep perceptual similarity (lower = better) |
| **BER** | 0% | Bit error rate in message recovery |

---

## 7. Build Phases & Milestones

### Phase 1: Core Algorithms (Week 1-2)
- [x] Project structure and configuration
- [x] AES-256-GCM encryption module
- [x] Image steganography (LSB, DCT, DWT)
- [x] Audio steganography (LSB, DWT)
- [x] Video steganography (LSB, DCT, DWT + motion compensation)
- [x] Metrics evaluation module
- [ ] Unit tests for all core modules
- **Milestone:** CLI demo working with all classical methods

### Phase 2: Deep Learning Models (Week 3-5)
- [x] Neural network layers (attention, ConvNeXt, noise layers)
- [x] U-Net++ Attention encoder-decoder
- [x] HiDDeN adversarial model with WGAN-GP
- [x] Invertible Neural Network with Haar wavelet
- [x] Training pipeline with AMP + wandb
- [x] Loss functions (MSE + MS-SSIM + LPIPS + frequency + adversarial)
- [ ] Train U-Net model on image dataset
- [ ] Train HiDDeN model with discriminator
- [ ] Train INN model for image-in-image hiding
- [ ] Fine-tune for video (temporal attention)
- **Milestone:** DL models producing PSNR > 35 dB with BER = 0%

### Phase 3: Web Application (Week 6-7)
- [x] FastAPI backend with all endpoints
- [x] React frontend with Tailwind CSS
- [x] File upload, encode/decode UI
- [x] Metrics visualization (charts)
- [ ] Install frontend dependencies and test
- [ ] End-to-end integration testing
- **Milestone:** Working web application with file upload → encode → decode

### Phase 4: Evaluation & Report (Week 8-9)
- [ ] Comprehensive benchmarking across all methods
- [ ] Generate comparison charts (methods × metrics × modalities)
- [ ] Video steganography robustness testing (compression, noise)
- [ ] Prepare project report with results and analysis
- **Milestone:** Complete evaluation report with charts and tables

---

## 8. How to Run

### Backend
```bash
cd steganography-project
pip install -r requirements.txt
python scripts/download_test_data.py     # Download test data
python scripts/run_demo.py               # Run CLI demo
uvicorn api.main:app --reload --port 8000  # Start API server
```

### Frontend
```bash
cd frontend
npm install
npm run dev    # Starts at http://localhost:5173
```

### Train Deep Learning Models
```python
from models.hidden import HiDDeNSteganography
from models.train import train_hidden, StegoImageDataset
from torch.utils.data import DataLoader

model = HiDDeNSteganography(msg_length=128)
dataset = StegoImageDataset("data/images", image_size=256)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
train_hidden(model, loader, epochs=100, device="cuda")
```

---

## 9. Key Research References

1. Zhu et al., "HiDDeN: Hiding Data With Deep Networks" (ECCV 2018) — Adversarial steganography
2. Baluja, "Hiding Images in Plain Sight" (NeurIPS 2017) — Deep encoder-decoder
3. Lu et al., "Robust Video Steganography Based on Multi-scale Decomposition and Invertible Networks" (2025) — INN for video
4. StegaStamp (Tancik et al., 2020) — Robust learned steganography
5. Nature Scientific Reports 2025 — Adaptive network steganography with DCGAN for video
