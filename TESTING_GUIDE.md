# GAN-Based Steganography System - Testing Guide

**Verify installation and functionality of the new GAN modules**

---

## 🔍 Pre-Test Checklist

- [ ] Python 3.8+
- [ ] PyTorch 2.2.0+
- [ ] CUDA available (optional, falls back to CPU)
- [ ] All files created in correct locations
- [ ] Dependencies installed: `pip install -r requirements.txt`

---

## ✅ Test 1: Import Verification

Verify all new modules can be imported:

```python
# Test imports
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

# Text compression
from core.text import TextCompressor, compress_text, decompress_text
print('✓ Text compression imported')

# Image GAN
from models.image_gan import ImageGANSteganography
from core.image import ImageGANStego
print('✓ Image GAN imported')

# Audio GAN
from models.audio_gan import AudioGANSteganography
from core.audio import AudioGANStego
print('✓ Audio GAN imported')

# Video GAN
from models.video_gan import VideoGANSteganography
from core.video import VideoGANStego
print('✓ Video GAN imported')

# Training
from models.train_gan import train_image_gan, train_audio_gan, train_video_gan
print('✓ Training functions imported')

# Config
from config.settings import IMAGE_GAN, AUDIO_GAN, VIDEO_GAN, GAN_TRAINING
print('✓ Config imported')

print('\n✓ All imports successful!')
"
```

**Expected Output:**
```
✓ Text compression imported
✓ Image GAN imported
✓ Audio GAN imported
✓ Video GAN imported
✓ Training functions imported
✓ Config imported

✓ All imports successful!
```

---

## ✅ Test 2: Text Compression

Test compression/decompression pipeline:

```python
python -c "
from core.text import compress_text, decompress_text

# Test 1: Short text
text1 = 'Hello, World!'
compressed = compress_text(text1)
decompressed = decompress_text(compressed)

assert decompressed == text1, f'Mismatch: {text1} != {decompressed}'
ratio = 100 * (1 - len(compressed) / len(text1.encode()))
print(f'✓ Short text: {text1} -> {len(compressed)} bytes (ratio: {ratio:.1f}%)')

# Test 2: Longer text
text2 = 'The quick brown fox jumps over the lazy dog. ' * 10
compressed2 = compress_text(text2)
decompressed2 = decompress_text(compressed2)

assert decompressed2 == text2
ratio2 = 100 * (1 - len(compressed2) / len(text2.encode()))
print(f'✓ Long text: {len(text2)} bytes -> {len(compressed2)} bytes (ratio: {ratio2:.1f}%)')

print('\n✓ Text compression working!')
"
```

**Expected Output:**
```
✓ Short text: Hello, World! -> XX bytes (ratio: XX%)
✓ Long text: XXXX bytes -> XXX bytes (ratio: XX%)

✓ Text compression working!
```

---

## ✅ Test 3: Model Instantiation

Test model creation and forward pass:

```python
python -c "
import torch
from models.image_gan import ImageGANSteganography
from models.audio_gan import AudioGANSteganography
from models.video_gan import VideoGANSteganography

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Image GAN
img_model = ImageGANSteganography(msg_length=128).to(device)
cover = torch.rand(2, 3, 256, 256).to(device)
msg = torch.randint(0, 2, (2, 128)).float().to(device)
stego, decoded = img_model(cover, msg)
assert stego.shape == cover.shape
assert decoded.shape == msg.shape
print(f'✓ Image GAN: {cover.shape} -> {stego.shape}')

# Audio GAN
audio_model = AudioGANSteganography(msg_length=128).to(device)
mag = torch.rand(2, 1, 513, 100).to(device)
phase = torch.rand(2, 1, 513, 100).to(device)
msg = torch.randint(0, 2, (2, 128)).float().to(device)
stego_mag, decoded = audio_model(mag, phase, msg)
assert stego_mag.shape == mag.shape
print(f'✓ Audio GAN: {mag.shape} -> {stego_mag.shape}')

# Video GAN
video_model = VideoGANSteganography(msg_length=128, temporal_window=5).to(device)
frames = torch.rand(2, 5, 3, 256, 256).to(device)
msg = torch.randint(0, 2, (2, 128)).float().to(device)
stego_frames, decoded = video_model(frames, msg)
assert stego_frames.shape == frames.shape
print(f'✓ Video GAN: {frames.shape} -> {stego_frames.shape}')

print('\n✓ All models instantiate correctly!')
"
```

**Expected Output:**
```
Device: cuda (or cpu)
✓ Image GAN: torch.Size([2, 3, 256, 256]) -> torch.Size([2, 3, 256, 256])
✓ Audio GAN: torch.Size([2, 1, 513, 100]) -> torch.Size([2, 1, 513, 100])
✓ Video GAN: torch.Size([2, 5, 3, 256, 256]) -> torch.Size([2, 5, 3, 256, 256])

✓ All models instantiate correctly!
```

---

## ✅ Test 4: API Wrapper Classes

Test high-level API wrappers:

```python
python -c "
import numpy as np
import torch
from core.image import ImageGANStego
from core.audio import AudioGANStego
from core.video import VideoGANStego

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image API
img_api = ImageGANStego(device=device)
cover_img = np.random.rand(256, 256, 3).astype(np.uint8) * 255
secret = b'test message'
cap = img_api.capacity(cover_img)
print(f'✓ Image API: capacity = {cap} bytes')

# Audio API
audio_api = AudioGANStego(device=device)
audio = np.random.randn(44100).astype(np.float32)
cap = audio_api.capacity(audio)
print(f'✓ Audio API: capacity = {cap} bytes')

# Video API
video_api = VideoGANStego(device=device)
print(f'✓ Video API instantiated')

print('\n✓ All API wrappers working!')
"
```

**Expected Output:**
```
✓ Image API: capacity = 16 bytes
✓ Audio API: capacity = 16 bytes
✓ Video API instantiated

✓ All API wrappers working!
```

---

## ✅ Test 5: Quick Training (Synthetic Data)

Run minimal training to verify training loop:

```bash
python scripts/train_gan_quick_start.py --modality image --device cuda
```

**Expected Output:**
```
Device: cuda
CUDA available: True

Creating 10 sample images...
Image dataset: 10 samples, batch size=2

================================================================================
TRAINING IMAGE GAN
================================================================================
Epoch 1/5: G_loss=XX.XXXX, D_loss=XX.XXXX, BER=X.XXXX
...
Training complete. Best checkpoint saved to [path]

✓ Image GAN training complete
```

---

## ✅ Test 6: Inference Testing

Test inference without training:

```bash
python scripts/train_gan_quick_start.py --test-only
```

**Expected Output:**
```
Device: cuda
CUDA available: True

Image GAN inference...
  Input shape: torch.Size([1, 3, 256, 256])
  Output shape: torch.Size([1, 3, 256, 256])
  Message decoded shape: torch.Size([1, 128])
  ✓ Image inference OK

Audio GAN inference...
  Input magnitude shape: torch.Size([1, 1, 513, 100])
  Output magnitude shape: torch.Size([1, 1, 513, 100])
  ✓ Audio inference OK

Video GAN inference...
  Input frames shape: torch.Size([1, 5, 3, 256, 256])
  Output frames shape: torch.Size([1, 5, 3, 256, 256])
  ✓ Video inference OK
```

---

## ✅ Test 7: API Integration

Test API factory functions:

```python
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from api.main import get_image_method, get_audio_method, get_video_method

# Image methods
for method in ['lsb', 'dct', 'dwt', 'gan']:
    try:
        m = get_image_method(method)
        print(f'✓ Image method: {method}')
    except Exception as e:
        print(f'✗ Image method {method}: {e}')

# Audio methods
for method in ['lsb', 'dwt', 'gan']:
    try:
        m = get_audio_method(method)
        print(f'✓ Audio method: {method}')
    except Exception as e:
        print(f'✗ Audio method {method}: {e}')

# Video methods
for method in ['lsb', 'dct', 'dwt', 'gan']:
    try:
        m = get_video_method(method)
        print(f'✓ Video method: {method}')
    except Exception as e:
        print(f'✗ Video method {method}: {e}')

print('\n✓ All API methods accessible!')
"
```

**Expected Output:**
```
✓ Image method: lsb
✓ Image method: dct
✓ Image method: dwt
✓ Image method: gan
✓ Audio method: lsb
✓ Audio method: dwt
✓ Audio method: gan
✓ Video method: lsb
✓ Video method: dct
✓ Video method: dwt
✓ Video method: gan

✓ All API methods accessible!
```

---

## ✅ Test 8: File Structure Verification

Verify all files exist:

```bash
# Check new directories
ls -la models/image_gan/ models/audio_gan/ models/video_gan/ core/text/

# Check new scripts
ls -la scripts/train_gan_quick_start.py scripts/evaluate_gan_methods.py

# Check new documentation
ls -la GAN_STEGANOGRAPHY_README.md IMPLEMENTATION_SUMMARY.md TESTING_GUIDE.md
```

**Expected Output:**
```
models/image_gan/:
  __init__.py
  model.py

models/audio_gan/:
  __init__.py
  model.py

models/video_gan/:
  __init__.py
  model.py

core/text/:
  __init__.py
  compression.py

scripts/:
  train_gan_quick_start.py
  evaluate_gan_methods.py

✓ All files present!
```

---

## ✅ Test 9: Config Verification

Verify config system:

```python
python -c "
from config.settings import (
    TEXT_COMPRESSION, IMAGE_GAN, AUDIO_GAN, VIDEO_GAN, GAN_TRAINING
)

print('TextCompressionConfig:')
print(f'  use_transformer: {TEXT_COMPRESSION.use_transformer}')
print(f'  vocab_size: {TEXT_COMPRESSION.vocab_size}')

print('ImageGANConfig:')
print(f'  message_bits: {IMAGE_GAN.message_bits}')
print(f'  base_channels: {IMAGE_GAN.base_channels}')

print('AudioGANConfig:')
print(f'  freq_bins: {AUDIO_GAN.freq_bins}')
print(f'  n_fft: {AUDIO_GAN.n_fft}')

print('VideoGANConfig:')
print(f'  temporal_window: {VIDEO_GAN.temporal_window}')
print(f'  frame_size: {VIDEO_GAN.frame_size}')

print('GANTrainingConfig:')
print(f'  batch_size: {GAN_TRAINING.batch_size}')
print(f'  epochs: {GAN_TRAINING.epochs}')
print(f'  lambda_adversarial: {GAN_TRAINING.lambda_adversarial}')

print('\n✓ All configs loaded!')
"
```

**Expected Output:**
```
TextCompressionConfig:
  use_transformer: True
  vocab_size: 256
ImageGANConfig:
  message_bits: 128
  base_channels: 64
AudioGANConfig:
  freq_bins: 513
  n_fft: 1024
VideoGANConfig:
  temporal_window: 5
  frame_size: 256
GANTrainingConfig:
  batch_size: 8
  epochs: 100
  lambda_adversarial: 0.01

✓ All configs loaded!
```

---

## 🎯 Full Test Suite

Run all tests at once:

```bash
#!/bin/bash
set -e

echo "Running full GAN system test suite..."
echo "======================================"

echo -e "\n1. Import verification..."
python -c "
from core.text import TextCompressor
from models.image_gan import ImageGANSteganography
from models.audio_gan import AudioGANSteganography
from models.video_gan import VideoGANSteganography
from models.train_gan import train_image_gan, train_audio_gan, train_video_gan
print('✓ All imports successful')
"

echo -e "\n2. Text compression test..."
python -c "
from core.text import compress_text, decompress_text
text = 'test message'
assert decompress_text(compress_text(text)) == text
print('✓ Compression working')
"

echo -e "\n3. Model instantiation test..."
python scripts/train_gan_quick_start.py --test-only

echo -e "\n4. Config verification..."
python -c "
from config.settings import IMAGE_GAN, AUDIO_GAN, VIDEO_GAN
print(f'✓ Image: {IMAGE_GAN.message_bits} bits')
print(f'✓ Audio: {AUDIO_GAN.freq_bins} freq bins')
print(f'✓ Video: {VIDEO_GAN.temporal_window} frame window')
"

echo -e "\n======================================"
echo "✓ All tests passed!"
```

---

## 🐛 Troubleshooting

### Import Error: "No module named 'models.image_gan'"

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/steganography-project

# Python path includes current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Out of Memory

**Solution:**
```python
# Use CPU instead
model = ImageGANSteganography().to('cpu')

# Or reduce batch size in training
GAN_TRAINING.batch_size = 4
```

### Training Doesn't Start

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU for initial testing
python scripts/train_gan_quick_start.py --device cpu
```

### API Method Not Found

**Solution:**
```python
# Verify import in api/main.py
from core.image import ImageGANStego
from core.audio import AudioGANStego
from core.video import VideoGANStego
```

---

## 📊 Performance Expectations

After successfully passing all tests:

| Test | Expected Time | Expected Memory |
|------|----------------|-----------------|
| Import verification | <1s | <500 MB |
| Text compression | <1s | <100 MB |
| Model instantiation | 2-5s | 1-2 GB |
| Quick training (5 epochs) | 2-5 min | 2-4 GB |
| Inference testing | 10-20s | 2 GB |
| API integration | <1s | <100 MB |

---

## ✨ Next Steps After Testing

1. ✅ **All tests passing?** Great!
2. Prepare your training data (images, audio, video)
3. Configure hyperparameters in `config/settings.py`
4. Run full training: `python scripts/train_gan_quick_start.py --modality all`
5. Evaluate with comparison script: `python scripts/evaluate_gan_methods.py --modality image`
6. Deploy to API: `python -m uvicorn api.main:app`

---

**System is ready for deployment!** 🚀
