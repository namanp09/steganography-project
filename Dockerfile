FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app

# Create necessary directories
RUN mkdir -p uploads outputs data/images data/audio data/video models/checkpoints

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
