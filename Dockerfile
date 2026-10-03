FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_MODEL=tiny

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download whisper model so first job is fast
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY *.py ./

EXPOSE 7860
# Use $PORT when set by the host (Render), else 7860
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"
