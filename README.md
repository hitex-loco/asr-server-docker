# UDAAN ASR server: Docker image and install guide

`asr-server.py` (in this folder) is the transcription server behind the
extension's voice typing, audio uploads and Meeting Mode. It runs offline on a
GPU server:

- **Transcription:** faster-whisper large-v3.
- **Speaker diarization** ("With speaker ID" uploads and Meeting Mode with
  Speaker ID = Yes):
  - **NVIDIA Nemotron-3-Diarization** when the request's max speakers is
    **8 or less**, or when it sends no speaker count;
  - **pyannote speaker-diarization-community-1** when max speakers is
    **above 8**.

The API is unchanged: same endpoints, same form fields, same responses and
progress events, so the browser extensions need no update.

## Files in this folder

| File | What it is |
|---|---|
| `asr-server.py` | The server itself |
| `requirements.txt` | Pinned Python packages (torch is installed separately, first) |
| `Dockerfile` | The GPU image; this folder is its build context |
| `.dockerignore` | Sends only `asr-server.py` and `requirements.txt` to the build |
| `docker-compose.yml` | Builds and runs the image, models mounted from `./models` or `ASR_MODELS_DIR` |
| `.gitignore` | Keeps a `models/` folder here out of git |

Every command below runs from this folder.

## What is new (Nemotron-3-Diarization)

- One new model folder, `models/Nemotron-3-Diarization/`. The existing
  `faster-whisper-large-v3/` and `speaker-diarization-community-1/` folders stay
  exactly as they are.
- One new Python package: `transformers==5.20.2`, which needs
  `huggingface_hub` 1.31 or newer. The image pins `huggingface_hub==1.33.0`.
- NVIDIA's model has no speaker-count setting. Min/max speakers are still
  accepted from the extension. They pick the diarizer, and pyannote still
  uses them, but they are not passed to the NVIDIA model.
- With the NVIDIA model, each transcribed word goes to the speaker the model
  rates most likely during that word's own audio. The model gives a 0–1
  score per speaker every 10 ms. This places words at speaker changes better
  than matching words to cut-off speaker segments, which pyannote still uses.
  - If the model hears nobody during a word, the segment rule decides.
  - A word stays with the previous speaker only while that speaker is still
    clearly audible in it and nearly as likely as the best. This stops single
    words from flipping speaker in overlapping speech.
- The NVIDIA model runs in **float32** by default. NVIDIA lists Ampere and
  newer GPUs; float32 also works on Turing (T4, RTX 20xx) and costs little for
  a 100M-parameter model.
- If the NVIDIA model is missing or fails to load, the server still starts
  and ordinary transcription works. Diarize requests that would use it fail
  with a 503 "NVIDIA diarization model not loaded …" error; there is **no**
  silent fallback to pyannote. Requests above 8 speakers keep working on
  pyannote.

## Which diarizer a request uses

| `max_speakers` sent | `min_speakers` sent | Diarizer |
|---|---|---|
| 1 to 8 | anything | NVIDIA Nemotron |
| 9 or more | anything | pyannote (with the min/max hints) |
| none | none, or 1 to 8 | NVIDIA Nemotron |
| none | 9 or more | pyannote (NVIDIA can't find 9 or more speakers) |

The extension always sends both numbers when Speaker ID is on (defaults 2 and
4), so in practice the **Max speakers** box decides. A count of 0 or less
counts as "none". The server log line `Diarized … | diarizer=nemotron` (or
`pyannote`) shows which one ran.

## Host folder layout

Put all three models in one folder on the GPU host. It is mounted read-only
at `/app/models` in the container:

```
<host models folder>/                     e.g. /data/asr-models  →  /app/models
├── faster-whisper-large-v3/             existing, unchanged (model.bin, config.json, tokenizer.json, …)
├── speaker-diarization-community-1/     existing, unchanged (config.yaml + the subfolders it came with)
└── Nemotron-3-Diarization/              NEW
    ├── config.json                      1.5 kB
    ├── processor_config.json            0.6 kB
    └── model.safetensors                397 MB
```

The image expects exactly these paths:

- `/app/models/faster-whisper-large-v3`
- `/app/models/speaker-diarization-community-1/config.yaml`
- `/app/models/Nemotron-3-Diarization`

If your folders are named differently, set the matching variable in the
environment variables table below instead of renaming them.

## One-time model downloads (on a machine with internet)

Use the Hugging Face CLI (`pip install -U huggingface_hub`, command `hf`).

**New: NVIDIA Nemotron-3-Diarization** (about 400 MB, no account or token
needed). Only these three files are used. The repo also holds a `.nemo`, a
`.gguf` and demo media, which this server doesn't need.

```bash
hf download nvidia/Nemotron-3-Diarization \
    config.json processor_config.json model.safetensors \
    --local-dir models/Nemotron-3-Diarization
```

Copy `models/Nemotron-3-Diarization/` into the host models folder, next to the
two existing folders. Any leftover `.cache/` folder that `hf` creates inside it
is harmless.

**Existing models:** only for a fresh server. Skip this if they are already
in place.

```bash
# faster-whisper large-v3 (about 3 GB)
hf download Systran/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3

# pyannote community-1 is gated: accept its terms on
# https://huggingface.co/pyannote/speaker-diarization-community-1 first,
# then download with a token from that account.
hf download pyannote/speaker-diarization-community-1 \
    --local-dir models/speaker-diarization-community-1 --token <your HF token>
```

## Build the image (on a machine with internet)

The build downloads Python packages, so it needs network access. The running
container needs none. Run it from **this folder**:

```bash
cd asr-server-docker
docker build -t udaan-asr .
```

`.dockerignore` limits the build context to `asr-server.py` and
`requirements.txt`, so a `models/` folder here is never sent to the builder.
For the compose file, use Docker Compose 2.30 or newer (its `gpus: all` needs
2.30).

The base image is `nvidia/cuda:12.6.3-runtime-ubuntu22.04`, with Python 3.10.
It deliberately has no system cuDNN: torch's pip packages bring cuDNN 9.10 and
cuBLAS, and `LD_LIBRARY_PATH` makes Whisper (CTranslate2) use those same copies.
Two cuDNN versions in one process can break diarization after Whisper has run.

The image includes:

- torch 2.8.0 (CUDA 12.6);
- the pinned packages in `requirements.txt`, checked with `pip check`;
- `ffmpeg`;
- `whisper_online.py` from `ufal/whisper_streaming`, at a pinned commit with
  a checksum check.

To move the image to an offline host:

```bash
docker save udaan-asr | gzip > udaan-asr.tar.gz
# on the GPU host:
gunzip -c udaan-asr.tar.gz | docker load
```

## Run

```bash
docker run -d --name udaan-asr --gpus all --restart unless-stopped \
    -p 8000:8000 \
    -v /data/asr-models:/app/models:ro \
    udaan-asr
```

Or, from this folder, with compose (it builds too):

```bash
ASR_MODELS_DIR=/data/asr-models docker compose up --build -d
```

The extension's **ASR Server URL** setting stays the same: `http://<host>:8000`.

Startup log lines to look for (`docker logs udaan-asr`):

```
Loading diarization pipeline from local path: /app/models/speaker-diarization-community-1/config.yaml
Diarization pipeline on CUDA.
Loading NVIDIA diarization model from local path: /app/models/Nemotron-3-Diarization
NVIDIA diarization model on CUDA (float32), warm-up 0.xxs.
```

## Environment variables

All of them have working defaults in the image. Override one with `-e NAME=value`
on `docker run`, or in the compose file's `environment:` list (which sets
`NUM_WORKERS=2`); `docker compose up -d` applies a change without a rebuild.

| Variable | Image default | Meaning |
|---|---|---|
| `WHISPER_MODEL_PATH` | `/app/models/faster-whisper-large-v3` | faster-whisper model folder |
| `DEVICE` | `cuda` | `cuda`, `cpu` or `auto`; also used by both diarizers |
| `COMPUTE_TYPE` | `float16` | faster-whisper precision |
| `BEAM_SIZE` / `BATCH_SIZE` / `NUM_WORKERS` | `5` / `8` / `1` | faster-whisper tuning (beam and workers match the earlier production image). `NUM_WORKERS` = transcriptions run on the GPU at the same time; it doesn't speed up one file, and each extra worker needs more GPU memory |
| `DIARIZATION_ENABLED` | `1` | `0` turns both diarizers off (diarize endpoints then return 503) |
| `PYANNOTE_MODEL_PATH` | `/app/models/speaker-diarization-community-1/config.yaml` | pyannote pipeline `config.yaml` |
| `HF_TOKEN` | not set | only used when `PYANNOTE_MODEL_PATH` is empty, and needs `HF_HUB_OFFLINE=0` |
| `NEMOTRON_DIAR_MODEL_PATH` | `/app/models/Nemotron-3-Diarization` | folder with the three NVIDIA files |
| `NEMOTRON_DIAR_DTYPE` | `float32` | `float32` (Turing and older); `float16` or `bfloat16` on Ampere and newer |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` / `1` | never download at runtime |
| `PYANNOTE_METRICS_ENABLED` | `0` | pyannote's optional usage telemetry stays off |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | listen address |

## Smoke tests

Run these after every build. Tests 2 and 3 are required: each must end with
`"phase": "done"`, which proves the GPU libraries work for both diarizers after
Whisper has run. Use a real recording with two or more people talking.

```bash
# 1. Health: "diarization":true when at least one diarizer loaded.
curl -s http://localhost:8000/health

# 2. NVIDIA path (max_speakers 4). Streams progress events, ends with "phase": "done".
curl -sN -F "audio=@meeting.wav" -F language=en -F min_speakers=2 -F max_speakers=4 \
    http://localhost:8000/transcribe-diarize-streaming

# 3. pyannote path (max_speakers 10).
curl -sN -F "audio=@meeting.wav" -F language=en -F min_speakers=2 -F max_speakers=10 \
    http://localhost:8000/transcribe-diarize-streaming

# 4. Which diarizer ran:
docker logs udaan-asr 2>&1 | grep "diarizer="
#   … | diarizer=nemotron | hints=min=2,max=4 (ignored)
#   … | diarizer=pyannote | hints=min=2,max=10
```

Then, in the browser:

- Run a Meeting Mode recording with Speaker ID = Yes. The defaults (2 to 4)
  use the NVIDIA model.
- Upload an audio file with "With speaker ID" and Max speakers 10. That uses
  pyannote.

Both transcripts should read `Speaker_1: …`, `Speaker_2: …`, and the
👥 Rename speakers chip should work as before.

Before relying on it for real meetings, also run one long recording (1 to 2
hours) through test 2. That confirms the time and the memory on your server.

## Updating an image you already run

If you'd rather keep your current image than build this one, the changes are:

1. Copy the new `asr-server.py` (it now lives in `asr-server-docker/`) over
   the old one.
2. `pip install transformers==5.20.2 "huggingface_hub>=1.31,<2"`. Your torch
   must be 2.6 or newer; pyannote 4 already needs 2.8.
3. Add `models/Nemotron-3-Diarization/` next to the other model folders, or
   set `NEMOTRON_DIAR_MODEL_PATH`. This step is required. Without it, Meeting
   Mode's default Speaker ID (2 to 4 speakers) returns the "NVIDIA diarization
   model not loaded" error.
4. Keep `DIARIZATION_ENABLED=1`. Setting `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1` is recommended.

## Troubleshooting

- **503 "NVIDIA diarization model not loaded …".** The startup log says why:
  - a missing `model.safetensors` in `NEMOTRON_DIAR_MODEL_PATH`;
  - `transformers` older than 5.18 ("Unrecognized model" / unknown
    `nemotron3_diarization`).

  Requests above 8 speakers still work. If no GPU is visible (`--gpus all`
  missing), the container doesn't get this far: Whisper can't load on CUDA and
  the server exits at startup.
- **503 "Diarization pipeline not loaded …".** That is pyannote. Check that
  `PYANNOTE_MODEL_PATH` points at the `config.yaml`. Only requests above 8
  speakers are affected.
- **Out of GPU memory.** Whisper large-v3, pyannote and the NVIDIA model
  share one GPU. The NVIDIA model needs about 0.4 GB in float32, plus about
  1 GB while a 2-hour recording is diarized. NVIDIA requests are diarized one
  at a time, and that memory is handed back afterwards, also when a run fails.
- **Out of host RAM.** Preparing a 2-hour recording for the NVIDIA model
  briefly needs about 5 to 6 GB of RAM, on top of Whisper. Give the container
  that much headroom. Only one recording is prepared at a time.
- **`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, or other cuDNN errors** in an
  image of your own: two cuDNN versions are loaded. Point `LD_LIBRARY_PATH` at
  torch's `nvidia/cudnn/lib` and `nvidia/cublas/lib`, as this Dockerfile does.
- **A modified `whisper_online.py`.** If your current image carries its own
  copy, replace the `curl` step in the Dockerfile with
  `COPY whisper_online.py /app/`, put the file in this folder next to
  `asr-server.py`, and add `!whisper_online.py` to `.dockerignore`.
- **NeMo.** NVIDIA's model card also shows NeMo. Don't use it here: released
  `nemo-toolkit` 3.0.0 can't load this checkpoint, which needs NeMo's
  unreleased main branch. The transformers route above doesn't need NeMo.
