FROM python:3.11-slim

# ffmpeg is required by torchaudio and faster-whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caches before code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY *.py ./
COPY .env.example .env.example

# Pre-download Whisper small model during build so cold starts are fast
# (the model is ~244MB and gets cached in /root/.cache/huggingface)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Render passes $PORT at runtime; default 8000 for local docker run
ENV PORT=8000

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
