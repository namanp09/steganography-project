# 🚀 Running GAN Steganography System Locally

**Complete guide to train, test, and deploy the system on your machine**

---

## ✅ Prerequisites

- Python 3.8+
- PyTorch 2.2.0+
- ~2GB RAM minimum (4GB recommended)
- Optional: CUDA for GPU acceleration

---

## 📦 Installation

```bash
# Navigate to project
cd /Users/i_naman.padiyar/Desktop/steganography-project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## 🧪 Phase 1: Testing (5 minutes)

### Test Inference (No Training)
```bash
# Verify all models can perform forward passes
python scripts/train_gan_quick_start.py --test-only
```

**Expected Output:**
```
✓ Image GAN inference OK
✓ Audio GAN inference OK
✓ Video GAN inference OK
```

### Test Text Compression
```bash
python -c "
from core.text import compress_text, decompress_text

text = 'Secret message to hide'
compressed = compress_text(text)
decompressed = decompress_text(compressed)

ratio = 100 * (1 - len(compressed) / len(text.encode()))
print(f'✓ Text: {text}')
print(f'✓ Compressed: {len(compressed)} bytes (ratio: {ratio:.1f}%)')
print(f'✓ Decompressed: {decompressed}')
"
```

---

## 🏋️ Phase 2: Quick Training (15 minutes)

### Train Image GAN
```bash
python scripts/train_gan_quick_start.py --modality image
```

**Expected:** 5 epochs, ~3s/batch, checkpoint saved to `models/checkpoints/image_gan_quickstart/`

### Train Audio GAN
```bash
python scripts/train_gan_quick_start.py --modality audio
```

### Train Video GAN
```bash
python scripts/train_gan_quick_start.py --modality video
```

### Train All Three
```bash
python scripts/train_gan_quick_start.py --modality all
```

**Checkpoints Location:**
```
models/checkpoints/
├── image_gan_quickstart/best_model.pth
├── audio_gan_quickstart/best_model.pth
└── video_gan_quickstart/best_model.pth
```

---

## 🔍 Phase 3: Evaluation

### Compare Methods
```bash
# Image baselines vs GAN
python scripts/evaluate_gan_methods.py \
  --modality image \
  --data-dir data \
  --image-gan-model models/checkpoints/image_gan_quickstart/best_model.pth

# Audio
python scripts/evaluate_gan_methods.py \
  --modality audio \
  --audio-gan-model models/checkpoints/audio_gan_quickstart/best_model.pth

# Video
python scripts/evaluate_gan_methods.py \
  --modality video \
  --video-gan-model models/checkpoints/video_gan_quickstart/best_model.pth
```

---

## 🌐 Phase 4: API Deployment

### Start FastAPI Server
```bash
# Default: localhost:8000
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints Available:**
```
POST /api/image/encode     - Hide data in image with GAN
POST /api/image/decode     - Extract data from stego image
POST /api/audio/encode     - Hide data in audio with GAN
POST /api/audio/decode     - Extract data from audio
POST /api/video/encode     - Hide data in video with GAN
POST /api/video/decode     - Extract data from video
GET  /api/methods          - List all available methods
```

### Test Image Encoding (with GAN)
```bash
# Create test image (512x512 random)
python -c "
import cv2
import numpy as np
img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
cv2.imwrite('/tmp/test_image.png', img)
print('✓ Test image created: /tmp/test_image.png')
"

# Encode
curl -X POST http://localhost:8000/api/image/encode \
  -F "cover=@/tmp/test_image.png" \
  -F "message=SecretGANMessage" \
  -F "method=gan" \
  -F "password=mypassword" \
  --output /tmp/stego_image.png

echo "✓ Stego image saved to /tmp/stego_image.png"

# Decode
curl -X POST http://localhost:8000/api/image/decode \
  -F "stego=@/tmp/stego_image.png" \
  -F "method=gan" \
  -F "password=mypassword"
```

### Switch Methods in API
```bash
# Using classical methods
curl -X POST http://localhost:8000/api/image/encode \
  -F "cover=@image.png" \
  -F "message=hello" \
  -F "method=lsb" \
  -F "password=pass"

# Using GAN
curl -X POST http://localhost:8000/api/image/encode \
  -F "cover=@image.png" \
  -F "message=hello" \
  -F "method=gan" \
  -F "password=pass"

# Available image methods: lsb, dct, dwt, gan
# Available audio methods: lsb, dwt, gan
# Available video methods: lsb, dct, dwt, gan
```

---

## 📊 Full Training (Production)

For real-world deployment, train on larger datasets:

```bash
# 1. Prepare dataset
mkdir -p data/{images,audio,videos}
# Copy your images to data/images/
# Copy your audio files to data/audio/
# Copy your video files to data/videos/

# 2. Create training script
cat > train_production.py << 'EOF'
from torch.utils.data import DataLoader
from models.image_gan import ImageGANSteganography
from models.train_gan import train_image_gan
from scripts.train_gan_quick_start import create_sample_image_dataset

model = ImageGANSteganography()
train_loader = create_sample_image_dataset(num_samples=1000)  # 1000 images
train_image_gan(model, train_loader, epochs=100, device="cuda")
EOF

# 3. Run training
python train_production.py
```

---

## 💾 Using Trained Models

### Load & Use Trained GAN for Inference
```python
from core.image import ImageGANStego
from core.audio import AudioGANStego
from core.video import VideoGANStego
import numpy as np

# Image
img_gan = ImageGANStego(
    model_path="models/checkpoints/image_gan_quickstart/best_model.pth",
    device="cuda"
)
cover_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
secret = b"Hide me!"
stego = img_gan.encode(cover_img, secret)
revealed = img_gan.decode(stego)
print(f"Original: {secret}")
print(f"Revealed: {revealed}")

# Audio
audio_gan = AudioGANStego(
    model_path="models/checkpoints/audio_gan_quickstart/best_model.pth",
    device="cuda"
)
audio = np.random.randn(44100).astype(np.float32)
stego_audio, sr = audio_gan.encode(audio, 44100, secret)
revealed = audio_gan.decode(stego_audio)
print(f"Audio - Revealed: {revealed}")

# Video
video_gan = VideoGANStego(
    model_path="models/checkpoints/video_gan_quickstart/best_model.pth",
    device="cuda"
)
meta = video_gan.encode("input.mp4", secret, "output.mp4")
revealed = video_gan.decode("output.mp4")
print(f"Video - Revealed: {revealed}")
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
edit config/settings.py
# Change: GAN_TRAINING.batch_size = 8 -> 2 or 4

# Or run on CPU (slower)
python scripts/train_gan_quick_start.py --device cpu
```

### Model Not Loading
```bash
# Verify checkpoint exists
ls -la models/checkpoints/image_gan_quickstart/best_model.pth

# If missing, retrain
python scripts/train_gan_quick_start.py --modality image
```

### CUDA Not Available
```python
# Works on CPU (slower)
model = ImageGANSteganography().to("cpu")

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📈 Performance Expectations

| Task | Time | GPU | CPU |
|------|------|-----|-----|
| Inference (image) | 1-2s | <1s | 1-2s |
| Inference (audio) | 0.5-1s | <0.5s | 0.5-1s |
| Inference (video) | 2-5s | 1-2s | 2-5s |
| Quick training (5 epochs) | ~2min | 30s | 2min |
| Full training (100 epochs) | ~40min | 6min | 40min |

---

## 🎓 Verification Checklist

- [ ] Installation successful: `python -c "import torch; print(torch.__version__)"`
- [ ] Inference tests pass: `python scripts/train_gan_quick_start.py --test-only`
- [ ] Text compression works: `python -c "from core.text import compress_text, decompress_text"`
- [ ] Training completes: `python scripts/train_gan_quick_start.py --modality image`
- [ ] API starts: `python -m uvicorn api.main:app --reload`
- [ ] API endpoint responds: `curl http://localhost:8000/api/methods`

---

## 📚 Next Steps

1. **Train on real data**: Prepare your own image/audio/video datasets
2. **Fine-tune hyperparameters**: Adjust learning rate, batch size in `config/settings.py`
3. **Deploy to server**: Use Gunicorn + Nginx for production
4. **Monitor training**: Use WandB for experiment tracking

```bash
# Install wandb (optional)
pip install wandb

# Login
wandb login

# Training will automatically log to your dashboard
python scripts/train_gan_quick_start.py --modality all
```

---

## 📖 Documentation

- **GAN_STEGANOGRAPHY_README.md** - Architecture & theory
- **IMPLEMENTATION_SUMMARY.md** - Implementation details
- **TESTING_GUIDE.md** - Comprehensive test scenarios
- **RUN_LOCALLY.md** - This file

---

**Ready to hide secrets with GAN-based steganography!** 🎓

