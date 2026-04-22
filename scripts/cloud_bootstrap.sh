#!/usr/bin/env bash
# One-shot setup on a Linux cloud GPU (RunPod, Lambda, Azure VM, Kaggle, etc.)
# Prereq: NVIDIA driver + CUDA runtime on the image; then:
#
#   curl -sL https://raw.githubusercontent.com/.../cloud_bootstrap.sh | bash
#   # or: clone repo, chmod +x scripts/cloud_bootstrap.sh, ./scripts/cloud_bootstrap.sh
#
# Set REPO_DIR to the folder that contains this `scripts/` directory (default: current dir).

set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
echo "REPO_DIR=$REPO_DIR"

if ! command -v python3 >/dev/null; then
  echo "python3 not found" >&2
  exit 1
fi

# PyTorch with CUDA (pick wheel matching your image; see https://pytorch.org/get-started/locally/)
if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  echo "Installing PyTorch CUDA cu124 (override INSTALL_TORCH=0 to skip)..."
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi
pip install -q cryptography pillow numpy scipy

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible — use a GPU image or check driver'; print('OK:', torch.cuda.get_device_name(0))"

# Training (tune EPOCHS / SAMPLES / MODALITY)
EPOCHS="${EPOCHS:-200}"
SAMPLES="${SAMPLES:-8000}"
BATCH="${BATCH:-32}"
MODALITY="${MODALITY:-image}"
exec python3 scripts/train_production_gan_gpu.py \
  --modality "$MODALITY" \
  --device cuda \
  --epochs "$EPOCHS" \
  --train-samples "$SAMPLES" \
  --batch "$BATCH"
