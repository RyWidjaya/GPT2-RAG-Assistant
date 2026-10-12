# Step 1: Base image with Python 3.10 Debian slim
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_MODEL_REPO=XAEOZ/gpt2-squad-rag

# Install essential system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set application working directory inside container
WORKDIR /app

# Step 2: Install dependencies
# Layer caching optimization: copy requirements first so rebuilds are fast
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller download ~200MB vs ~2.5GB CUDA wheels), then requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Step 3: Pre-download model weights at build time for sub-2s cold starts on Cloud Run
RUN mkdir -p checkpoints && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='XAEOZ/gpt2-squad-rag', filename='gpt2_124M_squad_ft.pt', local_dir='checkpoints')"

# Step 4: Copy application source code and web assets
COPY . .

# Cloud Run injects $PORT (default 8080)
EXPOSE 8080

# Step 5: Launch FastAPI server using Uvicorn
CMD ["sh", "-c", "uvicorn rag_server:app --host 0.0.0.0 --port ${PORT:-8080}"]

