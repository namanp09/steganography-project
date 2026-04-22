# CLAUDE.md — AI-Enhanced Steganography Project

## Project Overview
Multi-modal steganography system (image, audio, video) with GAN-based hiding and AES-256-GCM encryption. **Video steganography is the primary focus.**

## How to Run

```bash
# Backend (from project root)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
/usr/local/bin/node /usr/local/bin/npm --prefix frontend run dev
# Runs on localhost:5173
```

## Architecture

```
api/main.py              # FastAPI — all endpoints
config/settings.py       # All config (model sizes, paths, hyperparams)
core/
  image/gan_stego.py     # Image GAN encode/decode wrapper
  audio/gan_stego.py     # Audio GAN encode/decode wrapper
  video/gan_stego.py     # Video GAN encode/decode wrapper  ← most recently edited
  encryption.py          # AES-256-GCM cipher
models/
  image_gan/             # ImageGANSteganography architecture
  audio_gan/             # AudioGANSteganography architecture
  video_gan/model.py     # VideoGANSteganography — expects T=temporal_window frames
  checkpoints/
    image_gan_improved/best_model.pth
    audio_gan_improved/best_model.pth
    video_gan_improved/best_model.pth
scripts/                 # Training scripts (train_*_improved.py)
frontend/src/            # React + Vite + Tailwind
```

## Key Config (config/settings.py)

| Config | Value |
|--------|-------|
| `IMAGE_GAN.image_size` | 64 |
| `IMAGE_GAN.message_bits` | 128 |
| `IMAGE_GAN.base_channels` | 32 |
| `AUDIO_GAN.freq_bins` | 128 |
| `AUDIO_GAN.message_bits` | 128 |
| `AUDIO_GAN.base_channels` | 32 |
| `VIDEO_GAN.frame_size` | 64 |
| `VIDEO_GAN.message_bits` | 128 |
| `VIDEO_GAN.base_channels` | 16 |
| `VIDEO_GAN.temporal_window` | 5 |
| `PATHS.models_dir` | `models/checkpoints/` |

## Trained Model Accuracy (CPU, synthetic data)

| Model | Train Acc | Test Acc | Notes |
|-------|-----------|----------|-------|
| Image GAN | 63.4% | ~51% | Checkpoint: image_gan_improved |
| Video GAN | 61.6% | ~51% | Checkpoint: video_gan_improved |
| Audio GAN | 59.3% | ~51% | Checkpoint: audio_gan_improved |

**~51% test accuracy = near-random.** GAN encode produces imperceptible stego output (good PSNR), but message decode is unreliable. GPU training needed to reach 90%+ accuracy for reliable decode.

## Important Design Decisions

### GAN method skips AES encryption
In `api/main.py`, GAN encode/decode bypasses AES because the model can't guarantee exact bit recovery (needed for AES). GAN embeds raw UTF-8 bytes; decode does best-effort UTF-8 conversion.

### Video GAN processes in temporal windows
`core/video/gan_stego.py` slices video into 5-frame windows (`VIDEO_GAN.temporal_window`) before passing to the model. The model architecture (`models/video_gan/model.py:142`) hardcodes `T=5` — passing more frames causes a shape error.

### extract_frames returns a tuple
`core/video/frame_utils.py` — `extract_frames()` returns `(frames_list, metadata_dict)`. Always unpack both.

### Audio GAN freq_bins = 128 (not 513)
Was reduced from 513 → 128 in `config/settings.py` for training speed. Any new audio GAN training must use `freq_bins=128`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/image/encode` | Hide message in image |
| POST | `/api/image/decode` | Extract from image |
| POST | `/api/audio/encode` | Hide message in audio |
| POST | `/api/audio/decode` | Extract from audio |
| POST | `/api/video/encode` | Hide message in video |
| POST | `/api/video/decode` | Extract from video |
| GET | `/api/methods` | List all methods |
| GET | `/api/models/status` | Model training status |
| GET | `/health` | Health check |

All encode/decode endpoints accept `multipart/form-data` with fields: `cover`/`stego` (file), `message` (str), `method` (lsb/dct/dwt/gan), `password` (str), `seed` (optional int).

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `shape '[1,16,N,16,16]' is invalid` | Passing all N frames to VideoGAN (expects T=5) | Process in temporal windows — already fixed in `core/video/gan_stego.py` |
| `size mismatch for discriminator` | Wrong `base_ch` when loading checkpoint | Use `VIDEO_GAN.base_channels` (16), not hardcoded 32 |
| `Can't parse 'dsize'` | `resize=int` instead of tuple | Pass `resize=(VIDEO_GAN.frame_size, VIDEO_GAN.frame_size)` |
| `src is not numpy array` | Forgot to unpack `extract_frames` tuple | `frames, metadata = extract_frames(...)` |
| `expected 2 channels, got 1` | Optical flow tensor shape wrong | Pass `flow_tensor = None` (optical flow disabled) |
| AES decryption fails on GAN decode | GAN bit accuracy too low for AES | GAN method bypasses AES — already fixed in `api/main.py` |

## GPU Training (Next Step)
Current models trained on CPU with synthetic data → low accuracy. For reliable decode (90%+ accuracy):
- Use Google Colab or any CUDA GPU
- Train with real image/video/audio data
- `base_channels=64`, `image_size=128`, `epochs=200`, `batch_size=32`
- Script template: `colab_train_gpu.py` (at project root)

## Testing

```bash
# Test all 3 GAN models (loads checkpoints, runs 5 inference passes each)
python test_all_gans.py

# Full encode/decode test script
python scripts/test_gan_complete.py
```

## npm path on this machine
```bash
/usr/local/bin/node /usr/local/bin/npm ...
```
`npm` is not in PATH by default — use full path.
