# AI-Enhanced Multi-Modal Secure Steganography System
## Complete Project Flow Documentation

**Institution:** Delhi Technological University  
**Department:** Computer Science Engineering  
**Project Type:** B.Tech Major Project (Project-II), AY 2025-26  
**Team:** Naman Padiyar | Nipun Rawat | Piyush Thalwal

---

## 1. HIGH-LEVEL OVERVIEW

This is a B.Tech major project that implements **multi-modal steganography** (hiding encrypted data in images, audio, and video) using both classical algorithms and deep learning. The primary focus is **video steganography** with 6 encoding methods across 3 modalities.

---

## 2. SYSTEM ARCHITECTURE

```
User ↓
React Frontend (Vite + Tailwind)
        ↓
FastAPI Backend (/api/image, /api/audio, /api/video endpoints)
        ↓
Encryption Layer (AES-256-GCM)
        ↓
Core Steganography Algorithms
        ↓
Output Media + Quality Metrics
```

---

## 3. COMPLETE DATA FLOW

### ENCODING (HIDING) PROCESS:

1. **User Input** → Message text + Cover media (image/audio/video) + Password + Method choice

2. **Encryption** → 
   - Generate random salt (16 bytes)
   - Derive key from password: PBKDF2-HMAC-SHA256 (600k iterations per OWASP)
   - Encrypt message with AES-256-GCM (provides both confidentiality + integrity)
   - Output: `[salt(16B) | nonce(12B) | auth_tag(16B) | ciphertext]`

3. **Steganography** → Embed encrypted bytes into media using selected method:
   - **LSB**: Hide in least significant bits of pixel/sample values
   - **DCT**: Hide in discrete cosine transform coefficients  
   - **DWT**: Hide in wavelet transform detail coefficients
   - **Deep Learning** (U-Net, HiDDeN, INN): Neural network encodes data end-to-end

4. **For Video**: Extract frames → Use optical flow to detect motion → Embed only in stable regions → Temporal consistency checking

5. **Output** → Stego media file + Quality metrics (PSNR, SSIM, LPIPS, etc.)

### DECODING (REVEALING) PROCESS:

1. **User Input** → Stego media + Password + Same method used

2. **Extract Encrypted Data** → Reverse steganography process to retrieve bytes

3. **Decryption** → 
   - Parse salt, nonce, auth_tag from extracted data
   - Derive same key from password + salt
   - Decrypt with AES-256-GCM (validates authentication tag = integrity check)

4. **Output** → Original message (or error if tampered)

---

## 4. AVAILABLE METHODS & IMPLEMENTATIONS

### IMAGE STEGANOGRAPHY (`/core/image/`)

| Method | File | Technique | Key Feature |
|--------|------|-----------|-------------|
| **LSB** | `lsb.py` | Embed in pixel LSBs | Randomized pixel order via seed |
| **DCT** | `dct_stego.py` | Quantization Index Modulation on 8×8 blocks | Adaptive strength (α=10) |
| **DWT** | `dwt_stego.py` | Multi-level Haar/Daubechies wavelet | Embed in detail coefficients |

### AUDIO STEGANOGRAPHY (`/core/audio/`)

| Method | File | Technique | Key Feature |
|--------|------|-----------|-------------|
| **LSB** | `lsb.py` | Embed in PCM sample LSBs | 16-bit audio samples |
| **DWT** | `dwt_stego.py` | Daubechies-4 wavelet (4 levels) | Adaptive strength (α=0.02) |

### VIDEO STEGANOGRAPHY (`/core/video/`) — PRIMARY FOCUS

| Method | File | Technique | Key Feature |
|--------|------|-----------|-------------|
| **LSB** | `lsb.py` | Frame-by-frame LSB embedding | Motion compensation optional |
| **DCT** | `dct_stego.py` | DCT on each frame + temporal awareness | 8×8 block-based |
| **DWT** | `dwt_stego.py` | Wavelet per frame + frame selection | Skips every Nth frame |

**Video-Specific Innovations:**
- **Optical Flow** (Farneback): Identifies low-motion regions (safe to embed)
- **Frame Selection**: Skip frames (embed_every_n=2) to maintain temporal coherence
- **Temporal Consistency**: Ensures no visible flicker across frames
- **Codec Robustness**: Handles H.264/H.265 compression during encoding/decoding

---

## 5. FILES STRUCTURE & ROLES

```
steganography-project/
│
├── config/
│   └── settings.py               # Global config (encryption params, paths, training hyperparams)
│
├── core/                         # Core algorithms (no ML, pure classical methods)
│   ├── encryption/
│   │   ├── aes_cipher.py        # AES-256-GCM encryption + PBKDF2 key derivation
│   │   └── integrity.py         # SHA-256 hashing (integrity verification)
│   ├── image/
│   │   ├── lsb.py               # Image LSB (1-4 bits/pixel)
│   │   ├── dct_stego.py         # Image DCT + QIM
│   │   └── dwt_stego.py         # Image DWT (multi-level Haar)
│   ├── audio/
│   │   ├── lsb.py               # Audio LSB on PCM
│   │   └── dwt_stego.py         # Audio DWT (Daubechies-4)
│   ├── video/
│   │   ├── frame_utils.py       # PyAV frame extraction, optical flow, codec handling
│   │   ├── lsb.py               # Video LSB + motion compensation
│   │   ├── dct_stego.py         # Video DCT + temporal tracking
│   │   └── dwt_stego.py         # Video DWT + frame selection
│   └── metrics/
│       └── evaluate.py          # PSNR, SSIM, MS-SSIM, LPIPS, BER computation
│
├── models/                       # Deep Learning Models (PyTorch)
│   ├── layers.py                # CBAM attention, ConvNeXt blocks, noise layers
│   ├── losses.py                # SteganoLoss (MSE + MS-SSIM + LPIPS + Freq), WGAN-GP
│   ├── train.py                 # Training pipeline + datasets
│   ├── unet/
│   │   └── encoder_decoder.py   # Attention U-Net++ (encoder-decoder)
│   ├── hidden/
│   │   └── hidden_model.py      # HiDDeN + WGAN-GP discriminator
│   └── invertible/
│       └── inn_model.py         # INN with Haar wavelet + 3D temporal
│
├── api/
│   └── main.py                  # FastAPI backend with 6 endpoint pairs
│
├── frontend/                     # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/          # UI components (Navbar, FileUpload, MetricsDisplay)
│   │   ├── pages/               # Image, Audio, Video encode/decode pages
│   │   └── utils/               # API client
│   └── package.json
│
├── scripts/
│   ├── run_demo.py              # CLI demo script
│   └── download_test_data.py    # Dataset downloader
│
├── data/                        # Test media files
│   ├── images/
│   ├── audio/
│   └── videos/
│
└── PLANNING.md                  # Project planning document
```

---

## 6. DATASET STRUCTURE

**Where Data Lives:** `/data/` directory
- **Images**: `data/images/` — PNG, JPG, BMP test images
- **Audio**: `data/audio/` — WAV files (16-bit PCM, 44.1 kHz)
- **Videos**: `data/videos/` — MP4, AVI, MKV files

**How Datasets Are Loaded:**

```python
from models.train import StegoImageDataset, StegoVideoDataset

# Image Dataset
image_dataset = StegoImageDataset(
    "data/images",
    image_size=256,
    transform=torchvision.transforms.Compose([...])  # Random crop, flip, color jitter
)

# Video Dataset  
video_dataset = StegoVideoDataset(
    "data/videos",
    frame_size=256,
    temporal_window=5  # 5 consecutive frames per sample
)

loader = DataLoader(image_dataset, batch_size=8, shuffle=True)
```

---

## 7. TRAINING DEEP LEARNING MODELS

### Three DL Models Available:

#### a) Attention U-Net++ (Image Focus)

```python
from models.unet import AttentionUNet++

model = AttentionUNet++(
    msg_length=128,  # bits of secret message
    in_channels=3,   # RGB image
    out_channels=3   # RGB output
)
```

**Features:**
- **Encoder-Decoder** with dense skip connections
- **CBAM attention** for spatial feature selection
- **ConvNeXt blocks** for modern architecture
- **Multi-scale message injection** at different depths

#### b) HiDDeN Adversarial (Robustness)

```python
from models.hidden import HiDDeNSteganography, HiDDeNDiscriminator

encoder = HiDDeNSteganography(msg_length=128)
decoder = HiDDeNSteganography(msg_length=128, decoder=True)
discriminator = HiDDeNDiscriminator()
```

**Features:**
- **Encoder + Decoder + WGAN-GP Discriminator**
- **Spectral normalization** in discriminator
- **Noise layer** simulates: JPEG compression, Gaussian noise, crop, blur, codec
- **Frequency loss** to minimize spectral anomalies (avoid steganalysis detection)

#### c) Invertible Neural Network (INN) (Mathematically Reversible)

```python
from models.invertible import InvertibleSteganography

model = InvertibleSteganography(
    msg_length=128,
    use_temporal_attention=True  # For video
)
```

**Features:**
- **Normalizing flow** with affine coupling layers
- **Haar wavelet lifting** for invertibility guarantee
- **3D temporal attention** for video (tracks motion across frames)
- **Mathematically reversible**: Can perfectly recover all cover pixels

---

## 8. TRAINING PIPELINE

### Run Training:

```python
from models.train import train_hidden, train_unet, train_inn
from torch.utils.data import DataLoader

# 1. Create dataset & dataloader
dataset = StegoImageDataset("data/images", image_size=256)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Train HiDDeN model
model = HiDDeNSteganography(msg_length=128)
train_hidden(
    model, 
    train_loader=loader,
    val_loader=val_loader,
    epochs=100,
    device="cuda"
)
```

### Training Features:

- **Mixed Precision (AMP)**: FP16 + FP32 for speed
- **Scheduler**: Cosine annealing with warm restarts
- **WandB Integration**: Real-time experiment tracking (loss curves, metrics dashboards)
- **Multi-GPU**: Data parallel training support
- **Checkpointing**: Save/resume best models

### Loss Functions Combined:

```
Total Loss = λ₁·MSE + λ₂·MS-SSIM + λ₃·LPIPS + λ₄·BCE(msg) + λ₅·Frequency + λ₆·WGAN-GP
```

- **MSE**: Pixel-level fidelity
- **MS-SSIM**: Multi-scale structural similarity (perceptual)
- **LPIPS**: Deep perceptual loss (AlexNet/VGG features)
- **BCE**: Message recovery accuracy
- **Frequency**: Spectral flatness (avoid steganalysis detection)
- **WGAN-GP**: Adversarial training (fool discriminator)

---

## 9. TESTING & EVALUATION

### Metrics Computed (`/core/metrics/evaluate.py`):

| Metric | Target | Meaning |
|--------|--------|---------|
| **PSNR** | > 35 dB | Peak Signal-to-Noise Ratio (higher = less distortion) |
| **SSIM** | > 0.97 | Structural Similarity (1.0 = identical) |
| **MS-SSIM** | > 0.95 | Multi-scale SSIM (perceptual quality) |
| **MSE** | < 10 | Mean Squared Error (lower = better) |
| **SNR** | > 30 dB | Signal-to-Noise Ratio |
| **LPIPS** | < 0.05 | Learned Perceptual Image Patch Similarity |
| **BER** | 0% | Bit Error Rate (message corruption) |

### Test Flow:

```python
from core.metrics import compute_all_metrics

# Encode → Decode → Measure
stego_image = encoder.encode(cover_image, secret_data)
metrics = compute_all_metrics(cover_image, stego_image)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
```

---

## 10. API ENDPOINTS (FastAPI)

**Backend running at:** `http://localhost:8000`

### Endpoints:

```
POST /api/image/encode       # Hide message in image
POST /api/image/decode       # Extract message from image
POST /api/audio/encode       # Hide message in audio
POST /api/audio/decode       # Extract message from audio
POST /api/video/encode       # Hide message in video  
POST /api/video/decode       # Extract message from video
GET  /api/methods            # List available methods
GET  /api/metrics            # Get evaluation results
```

### Example Encode Request:

```bash
curl -X POST http://localhost:8000/api/image/encode \
  -F "cover=@input.jpg" \
  -F "message=Secret" \
  -F "password=mypassword" \
  -F "method=dwt"
```

---

## 11. FRONTEND (React)

**Running at:** `http://localhost:5173`

### Pages:

- **Image**: Encode/Decode images with LSB/DCT/DWT/U-Net methods
- **Audio**: Encode/Decode audio with LSB/DWT
- **Video**: Encode/Decode video with LSB/DCT/DWT/INN/HiDDeN
- **Metrics Dashboard**: Visualize PSNR, SSIM, BER graphs

### User Flow:

1. Select modality (Image/Audio/Video)
2. Upload cover media
3. Enter secret message
4. Enter password
5. Choose method
6. Click "Encode" → Download stego media + metrics
7. To decode: Upload stego media + enter password

---

## 12. COMPLETE END-TO-END EXAMPLE

### CLI Demo:

```python
# Run: python scripts/run_demo.py

# 1. Read cover image
cover = cv2.imread("data/images/test.jpg")

# 2. Encrypt message
cipher = AESCipher("mypassword")
encrypted = cipher.encrypt_message("Secret message")

# 3. Embed in image using DWT
from core.image import ImageDWT
encoder = ImageDWT(wavelet="haar", level=2, alpha=5.0)
stego = encoder.encode(cover, encrypted)

# 4. Compute metrics
from core.metrics import compute_all_metrics
metrics = compute_all_metrics(cover, stego)
print(f"PSNR: {metrics['psnr']:.2f} dB")

# 5. Save stego
cv2.imwrite("output.jpg", stego)

# 6. Extract & Decrypt
extracted = encoder.decode(stego)
decrypted_msg = cipher.decrypt_message(extracted)
print(f"Recovered: {decrypted_msg}")
```

---

## 13. SECURITY PROPERTIES

| Property | Implementation |
|----------|-----------------|
| **Confidentiality** | AES-256-GCM (256-bit keys, authenticated) |
| **Integrity** | GCM authentication tag + SHA-256 hash |
| **Key Derivation** | PBKDF2-HMAC-SHA256 (600k iterations, resistant to brute force) |
| **Imperceptibility** | Classical methods use adaptive embedding; DL models trained with perceptual losses |
| **Anti-Steganalysis** | HiDDeN adversarial training + frequency domain optimization |

---

## 14. RUN COMMANDS SUMMARY

### Backend Setup

```bash
cd steganography-project
pip install -r requirements.txt
python scripts/download_test_data.py
uvicorn api.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Access at http://localhost:5173
```

### CLI Demo

```bash
python scripts/run_demo.py
```

### Train Deep Learning Models

```bash
python -c "from models.train import train_hidden; ..."
```

---

## CONCLUSION

This is a sophisticated, production-ready steganography system! The classical methods (LSB/DCT/DWT) provide a baseline, while deep learning models push robustness to state-of-the-art levels with adversarial training and perceptual losses. The system seamlessly integrates encryption, steganography, and evaluation into a complete web-based platform with both CLI and GUI interfaces.

### Key Achievements:

✅ Multi-modal support (Image, Audio, Video)  
✅ 6 steganography methods (3 classical + 3 DL)  
✅ Military-grade encryption (AES-256-GCM)  
✅ PBKDF2 key derivation (OWASP compliant)  
✅ Comprehensive evaluation metrics  
✅ Production-ready FastAPI backend  
✅ Interactive React frontend  
✅ Adversarial robustness training  
✅ Temporal consistency for video  
✅ Codec-robust noise simulation  

---

**Date Generated:** 2026-04-21  
**Project Status:** B.Tech Major Project (Project-II), AY 2025-26
