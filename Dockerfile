# UDAAN ASR server (asr-server.py): offline GPU image.
#
# faster-whisper large-v3 transcription plus speaker diarization:
#   NVIDIA Nemotron-3-Diarization for meetings of up to 8 speakers,
#   pyannote speaker-diarization-community-1 above that.
# No model is baked in. All three are mounted read-only at /app/models;
# README.md next to this file has the folder layout and the one-time downloads.
#
# Build from this folder, which holds asr-server.py and is the build context:
#   docker build -t udaan-asr .
# .dockerignore keeps the context to the two files copied below, so a models/
# folder in here is never sent to the builder.

# The plain CUDA runtime image, without the system cuDNN: torch's pip wheels
# bring cuDNN 9.10 and cuBLAS, and LD_LIBRARY_PATH below makes CTranslate2
# (Whisper) load those same copies. Two cuDNN versions in one process can fail
# with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH once pyannote or the NVIDIA
# diarizer runs after Whisper.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg turns webm/opus/m4a uploads into 16 kHz WAV; libsndfile backs soundfile.
# libpython3.10 is needed by torchcodec's FFmpeg 4 build, which pyannote uses
# to decode audio files (without it: "Could not load libtorchcodec").
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libpython3.10 ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# CUDA 12.6 builds of torch; they run on Turing (T4, RTX 20xx) and newer.
# Installed first so nothing in requirements.txt pulls another torch.
RUN python3 -m pip install torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt \
    && python3 -m pip check

# One cuDNN and one cuBLAS for everything: torch's pip copies (see FROM above).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}
RUN test -f /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
    && test -f /usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib/libcublas.so.12

WORKDIR /app

# whisper_online.py from ufal/whisper_streaming, which asr-server.py imports
# for the live /transcribe-stream WebSocket. Pinned to a commit and checked.
ARG WHISPER_STREAMING_COMMIT=4fde005708f98620fd2c7700b675c9a3e1b3f080
ARG WHISPER_ONLINE_SHA256=f0a7bd26bff879cf1e2ee4589bd88c38275c678d8773a47563ceb0e7d970b850
RUN curl -fsSL -o /app/whisper_online.py \
        "https://raw.githubusercontent.com/ufal/whisper_streaming/${WHISPER_STREAMING_COMMIT}/whisper_online.py" \
    && echo "${WHISPER_ONLINE_SHA256}  /app/whisper_online.py" | sha256sum -c -

COPY asr-server.py /app/asr-server.py

# Offline at runtime: nothing is downloaded; every model loads from /app/models.
# NUM_WORKERS and BEAM_SIZE match the earlier production image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYANNOTE_METRICS_ENABLED=0 \
    WHISPER_MODEL_PATH=/app/models/faster-whisper-large-v3 \
    NUM_WORKERS=1 \
    BEAM_SIZE=5 \
    DIARIZATION_ENABLED=1 \
    PYANNOTE_MODEL_PATH=/app/models/speaker-diarization-community-1/config.yaml \
    NEMOTRON_DIAR_MODEL_PATH=/app/models/Nemotron-3-Diarization \
    NEMOTRON_DIAR_DTYPE=float32 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
CMD ["python3", "asr-server.py"]
