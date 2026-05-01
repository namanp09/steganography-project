# Stage 1: Build frontend
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend + built frontend
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first (separate index URL), then remaining deps
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY config/ config/
COPY core/ core/
COPY api/ api/
COPY models/ models/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Create necessary directories
RUN mkdir -p uploads outputs data/images data/audio data/video models/checkpoints

EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
