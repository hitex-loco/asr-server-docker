"""
Voice Typing ASR Backend (Offline)
===================================
FastAPI server using faster-whisper with a locally stored
Systran/faster-whisper-large-v3 model. No internet required at runtime.

Speaker diarization (optional, DIARIZATION_ENABLED=1) runs on NVIDIA
Nemotron-3-Diarization for meetings of up to 8 speakers and on pyannote for
larger ones (see _pick_diarizer). Both load from local files only.

Install, model downloads, the /app/models folder layout and the Docker image:
    README.md in this folder (asr-server-docker/)

Run without Docker (models under ./models and whisper_online.py next to this file):
    python asr-server.py
"""

import os
import sys
import time
import json
import asyncio
import tempfile
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0",
    category=UserWarning,
)
from faster_whisper import WhisperModel
try:
    # faster-whisper >= 1.0 ships BatchedInferencePipeline for parallel
    # decoding of VAD-detected speech chunks. Same model, same flags,
    # same beam search — only the scheduling changes. Much faster on
    # files >30s with no accuracy loss.
    from faster_whisper import BatchedInferencePipeline
    _HAS_BATCHED = True
except ImportError:
    BatchedInferencePipeline = None
    _HAS_BATCHED = False

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from whisper_online import FasterWhisperASR, OnlineASRProcessor

# ─── Configuration ───────────────────────────────────────────────────────────

# Use abspath so the path resolves correctly regardless of cwd
MODEL_PATH = os.getenv(
    "WHISPER_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "faster-whisper-large-v3"),
)

DEVICE       = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "2"))
BEAM_SIZE    = int(os.getenv("BEAM_SIZE", "4"))

# Diarization config. Disabled by default so the server still boots without
# a pyannote model on disk or an HF token. Set DIARIZATION_ENABLED=1 to turn on.
#   PYANNOTE_MODEL_PATH → local pipeline config.yaml (offline, preferred)
#                        e.g. /app/models/speaker-diarization-community-1/config.yaml
#   HF_TOKEN            → HuggingFace token; only used when no local path is set
DIARIZATION_ENABLED  = os.getenv("DIARIZATION_ENABLED", "0") == "1"
PYANNOTE_MODEL_PATH  = os.getenv("PYANNOTE_MODEL_PATH", "").strip()
HF_TOKEN             = os.getenv("HF_TOKEN", "").strip() or None

# NVIDIA Nemotron-3-Diarization, loaded under the same DIARIZATION_ENABLED
# switch. It has 8 speaker channels and takes no speaker-count hints, so
# requests whose max_speakers is above 8 go to pyannote instead (see
# _pick_diarizer).
#   NEMOTRON_DIAR_MODEL_PATH → local folder holding config.json,
#                              processor_config.json and model.safetensors
#   NEMOTRON_DIAR_DTYPE      → float32 (default; the right choice on Turing
#                              and older GPUs), float16 or bfloat16
NEMOTRON_DIAR_MODEL_PATH = os.getenv(
    "NEMOTRON_DIAR_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Nemotron-3-Diarization"),
).strip()
NEMOTRON_DIAR_DTYPE   = os.getenv("NEMOTRON_DIAR_DTYPE", "float32").strip().lower()
NEMOTRON_MAX_SPEAKERS = 8

MAX_FILE_SIZE_MB = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-typing-asr")

# ─── Validate Model Path ────────────────────────────────────────────────────

model_dir = Path(MODEL_PATH)

if not model_dir.exists():
    logger.error(
        "\n" + "=" * 60 + "\n"
        "  MODEL NOT FOUND at: %s\n\n"
        "  Download it first (requires internet, one-time only):\n\n"
        "    pip install huggingface_hub\n"
        "    huggingface-cli download Systran/faster-whisper-large-v3 \\\n"
        "        --local-dir %s\n\n"
        "  Then re-run this server.\n"
        + "=" * 60,
        MODEL_PATH, model_dir.resolve()
    )
    raise SystemExit(1)

if not (model_dir / "model.bin").exists():
    logger.error(
        "\n" + "=" * 60 + "\n"
        "  model.bin NOT FOUND in: %s\n\n"
        "  The directory exists but appears incomplete.\n"
        "  Re-download with:\n"
        "    huggingface-cli download Systran/faster-whisper-large-v3 \\\n"
        "        --local-dir %s\n"
        + "=" * 60,
        MODEL_PATH, model_dir.resolve()
    )
    raise SystemExit(1)

found_files = sorted(f.name for f in model_dir.iterdir() if f.is_file())
logger.info("Model dir contents: %s", ", ".join(found_files))

# ─── Load Model ──────────────────────────────────────────────────────────────

logger.info("Loading model from: %s", MODEL_PATH)
logger.info("Device: %s | Compute type: %s | Workers: %d", DEVICE, COMPUTE_TYPE, NUM_WORKERS)

model = WhisperModel(
    MODEL_PATH,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=os.cpu_count() or 4,
    num_workers=NUM_WORKERS,
    local_files_only=True,  # Never attempt any network download
)

logger.info("Model loaded — fully offline, ready to serve.")


# ─── Batched Inference Pipeline (faster-whisper ≥ 1.0) ──────────────────────
# Same model, same beam search, same VAD — chunks are scheduled in parallel
# on the GPU instead of serially. Much faster on long files with no accuracy
# loss. Falls back to the serial path if the installed faster-whisper is too
# old to provide BatchedInferencePipeline.
batched_model = None
if _HAS_BATCHED:
    try:
        batched_model = BatchedInferencePipeline(model=model)
        logger.info("Batched inference pipeline ready.")
    except Exception as e:
        logger.warning("Could not build BatchedInferencePipeline (%s); using serial.", e)
        batched_model = None
else:
    logger.info("faster-whisper has no BatchedInferencePipeline; using serial inference.")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))


# ─── Warm-up ────────────────────────────────────────────────────────────────
# First request after boot pays GPU kernel-compilation cost (~5-15s).
# Run a tiny silent transcribe at startup so the first real user doesn't see it.
try:
    _warm_audio = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
    _wm_t0 = time.time()
    _segs, _info = model.transcribe(_warm_audio, language="en", beam_size=1, vad_filter=False)
    for _ in _segs:
        pass
    if batched_model is not None:
        _segs2, _ = batched_model.transcribe(_warm_audio, language="en", beam_size=1, vad_filter=False, batch_size=2)
        for _ in _segs2:
            pass
    logger.info("Warm-up complete in %.2fs.", time.time() - _wm_t0)
except Exception as e:
    logger.warning("Warm-up skipped: %s", e)


# ─── Language-specific decoding hints (#7) ──────────────────────────────────
# A short prompt in the target language biases the decoder toward correct
# tokenization, including code-switched (e.g. Hinglish) speech. No effect on
# accuracy when the user says nothing in that language — but a meaningful
# boost when they do.
LANGUAGE_PROMPTS = {
    "hi": "नमस्ते। मैं कैसे आपकी मदद कर सकता हूँ?",
    "en": "Hello. How can I help you today?",
}


# ─── Diarization Pipeline (optional) ────────────────────────────────────────

diarization_pipeline = None

if DIARIZATION_ENABLED:
    try:
        import torch
        from pyannote.audio import Pipeline as PyannotePipeline

        if PYANNOTE_MODEL_PATH:
            logger.info("Loading diarization pipeline from local path: %s", PYANNOTE_MODEL_PATH)
            diarization_pipeline = PyannotePipeline.from_pretrained(PYANNOTE_MODEL_PATH)
        else:
            # Fallback: pull a default pipeline from HuggingFace. pyannote.audio
            # 4.x renamed `use_auth_token` → `token` to match the rest of the HF
            # ecosystem. We try the new kwarg first, then fall back for older
            # pyannote installs.
            default_pipeline_id = "pyannote/speaker-diarization-community-1"
            logger.info("Loading diarization pipeline: %s", default_pipeline_id)
            try:
                diarization_pipeline = PyannotePipeline.from_pretrained(
                    default_pipeline_id,
                    token=HF_TOKEN,
                )
            except TypeError:
                # Older pyannote.audio (<4.0) used use_auth_token
                diarization_pipeline = PyannotePipeline.from_pretrained(
                    default_pipeline_id,
                    use_auth_token=HF_TOKEN,
                )

        # Match Whisper's device choice. "auto" → cuda if available.
        if DEVICE == "cuda" or (DEVICE == "auto" and torch.cuda.is_available()):
            diarization_pipeline.to(torch.device("cuda"))
            logger.info("Diarization pipeline on CUDA.")
        else:
            logger.info("Diarization pipeline on CPU.")

    except Exception as e:
        logger.error("Failed to load diarization pipeline: %s", e, exc_info=True)
        logger.error("Diarize requests routed to pyannote (above %d speakers) will return 503. "
                     "Server will still serve /transcribe.", NEMOTRON_MAX_SPEAKERS)
        diarization_pipeline = None


def _diarization_to_turns(diarization):
    """
    Normalise pyannote diarization output across versions.

    pyannote 3.x: pipeline returns an Annotation with .itertracks().
    pyannote 4.x: pipeline returns a DiarizeOutput wrapper; the actual
    Annotation lives at .speaker_diarization (or .diarization on some
    versions). Falls back to attribute search before giving up.

    Returns a list of (start, end, raw_speaker_label) tuples in
    chronological order.
    """
    # If it already has itertracks (pyannote 3.x Annotation), use directly.
    annotation = diarization
    if not hasattr(annotation, "itertracks"):
        # pyannote 4.x DiarizeOutput wrapper — try known attribute names
        for attr in ("speaker_diarization", "diarization", "annotation"):
            inner = getattr(annotation, attr, None)
            if inner is not None and hasattr(inner, "itertracks"):
                annotation = inner
                break
        else:
            # Last resort: scan all attributes for something that quacks
            for name in dir(annotation):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(annotation, name)
                except Exception:
                    continue
                if hasattr(val, "itertracks"):
                    annotation = val
                    break
            else:
                raise AttributeError(
                    "pyannote diarization output has no .itertracks() and no "
                    "recognisable Annotation attribute. Got type: %s, attrs: %s"
                    % (type(diarization).__name__, [a for a in dir(diarization) if not a.startswith("_")])
                )

    return [
        (float(seg.start), float(seg.end), str(spk))
        for seg, _, spk in annotation.itertracks(yield_label=True)
    ]


# ─── NVIDIA Nemotron-3-Diarization (optional) ───────────────────────────────
#
# Runs through transformers (>= 5.18 supports it natively; the released NeMo
# 3.0.0 can't restore this checkpoint). Its offline forward splits a whole
# recording into chunks with a speaker cache, so long meetings work.

nemotron_processor = None
nemotron_model = None
nemotron_device = None
nemotron_dtype = None
# One request at a time: even a 2 h meeting takes only seconds, and its feature
# extraction briefly needs several GB of host RAM (plus about 1 GB on the GPU),
# so parallel uploads must not stack those peaks.
nemotron_lock = threading.Lock()


def _nemotron_turns(waveform: np.ndarray):
    """
    Diarize a 16 kHz mono float32 waveform with Nemotron-3-Diarization.

    Returns (turns, frame_probs):
      turns       — (start, end, raw_speaker_label) tuples sorted by start, the
                    same shape _diarization_to_turns gives for pyannote.
                    Speakers are numbered by first arrival; overlapping speech
                    gives overlapping turns.
      frame_probs — float32 array (frames, 8): each speaker channel's
                    probability for every NEMOTRON_FRAME_SEC (10 ms) frame,
                    used to assign Whisper words (_speakers_from_frame_probs).
    """
    import torch  # type: ignore

    error = None
    with nemotron_lock, torch.inference_mode():
        try:
            inputs = nemotron_processor(waveform, sampling_rate=16000)
            attention_mask = inputs.get("attention_mask")  # stays on the CPU
            inputs = inputs.to(nemotron_device, dtype=nemotron_dtype)
            logits = nemotron_model(**inputs).logits.float().cpu()
        except Exception as e:
            # Keep only the message: the traceback would hold the forward's GPU
            # tensors and keep them out of the cache release below.
            logger.error("NVIDIA diarization failed: %s", e, exc_info=True)
            error = "%s: %s" % (type(e).__name__, e)
        inputs = None
        if nemotron_device.type == "cuda":
            # Give the cached activations back so Whisper (CTranslate2) can use
            # them, after a failure (e.g. out of memory) as well.
            torch.cuda.empty_cache()
    if error:
        raise RuntimeError(error)
    segments = nemotron_processor.extract_speaker_dict(logits, attention_mask)[0]
    turns = sorted(
        ((float(seg["Start"]), float(seg["End"]), "NEMOTRON_%d" % int(seg["Speaker"])) for seg in segments),
        key=lambda turn: turn[0],
    )
    frame_probs = logits[0].sigmoid().numpy().astype(np.float32, copy=False)
    if attention_mask is not None:
        frame_probs = frame_probs[: int(attention_mask[0].sum())]
    return turns, frame_probs


# Word-to-speaker assignment from the frame probabilities (Nemotron path only).
NEMOTRON_FRAME_SEC = 0.01        # one probability row per 10 ms of audio
NEMOTRON_ACTIVE_PROB = 0.5       # extract_speaker_dict's default threshold
NEMOTRON_WORD_MIN_SCORE = 0.05   # below this the model hears nobody in the word
NEMOTRON_WORD_KEEP_MARGIN = 0.1  # keep the previous word's speaker unless another is clearly ahead
NEMOTRON_WORD_KEEP_RATIO = 0.5   # ...and only while it scores at least half the best (it is really there)


def _speakers_from_frame_probs(words: list, frame_probs: np.ndarray, fallback: Callable) -> list:
    """
    Pick each word's raw speaker label ("NEMOTRON_<n>") from Nemotron's
    per-frame probabilities: the speaker with the highest mean probability over
    the word's own frames. This keeps the model's confidence instead of the
    0.5-thresholded turns, so words near a speaker change or in a short gap go
    to the right person more often.

    - Candidates are only the channels that cross NEMOTRON_ACTIVE_PROB
      somewhere, i.e. the speakers that have turns, so the set of speakers is
      the same as with turn-based alignment.
    - A word whose best score is below NEMOTRON_WORD_MIN_SCORE (a Whisper
      timing slip into silence) uses fallback(start, end), the turn rule.
    - If the previous word's speaker is still audible in this word (at least
      NEMOTRON_WORD_MIN_SCORE and NEMOTRON_WORD_KEEP_RATIO of the best) and
      within NEMOTRON_WORD_KEEP_MARGIN of the best, it is kept, so single
      words don't flip speaker inside overlapping speech. A previous speaker
      who is silent in the word is never kept.
    """
    n_frames = frame_probs.shape[0]
    candidates = [int(c) for c in np.flatnonzero((frame_probs > NEMOTRON_ACTIVE_PROB).any(axis=0))] if n_frames else []
    if not candidates:
        return [fallback(w["start"], w["end"]) for w in words]
    position = {"NEMOTRON_%d" % c: j for j, c in enumerate(candidates)}
    cand_probs = frame_probs[:, candidates]

    out = []
    prev = None  # index into candidates of the previous word's speaker
    for w in words:
        i0 = min(max(int(round(w["start"] / NEMOTRON_FRAME_SEC)), 0), n_frames - 1)
        i1 = min(max(int(np.ceil(w["end"] / NEMOTRON_FRAME_SEC)), i0 + 1), n_frames)
        scores = cand_probs[i0:i1].mean(axis=0)
        best = int(scores.argmax())
        if scores[best] < NEMOTRON_WORD_MIN_SCORE:
            label = fallback(w["start"], w["end"])
            prev = position.get(label)
            out.append(label)
            continue
        if (
            prev is not None
            and scores[prev] >= max(NEMOTRON_WORD_MIN_SCORE, NEMOTRON_WORD_KEEP_RATIO * scores[best])
            and scores[best] - scores[prev] < NEMOTRON_WORD_KEEP_MARGIN
        ):
            best = prev
        prev = best
        out.append("NEMOTRON_%d" % candidates[best])
    return out


if DIARIZATION_ENABLED:
    try:
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        _nemotron_dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if NEMOTRON_DIAR_DTYPE not in _nemotron_dtypes:
            raise ValueError(
                "NEMOTRON_DIAR_DTYPE must be float32, float16 or bfloat16, got %r" % NEMOTRON_DIAR_DTYPE
            )
        if not os.path.isfile(os.path.join(NEMOTRON_DIAR_MODEL_PATH, "model.safetensors")):
            raise FileNotFoundError("model.safetensors not found in %s" % NEMOTRON_DIAR_MODEL_PATH)

        logger.info("Loading NVIDIA diarization model from local path: %s", NEMOTRON_DIAR_MODEL_PATH)
        nemotron_dtype = _nemotron_dtypes[NEMOTRON_DIAR_DTYPE]
        # Match Whisper's device choice. "auto" → cuda if available.
        use_cuda = DEVICE == "cuda" or (DEVICE == "auto" and torch.cuda.is_available())
        nemotron_device = torch.device("cuda" if use_cuda else "cpu")
        nemotron_processor = AutoProcessor.from_pretrained(NEMOTRON_DIAR_MODEL_PATH, local_files_only=True)
        nemotron_model = AutoModelForAudioFrameClassification.from_pretrained(
            NEMOTRON_DIAR_MODEL_PATH, local_files_only=True, dtype=nemotron_dtype,
        )
        nemotron_model.to(nemotron_device).eval()

        # Warm-up on 1 s of silence so the first request doesn't pay CUDA
        # start-up, and so a broken install fails here instead of per request.
        t0 = time.time()
        _nemotron_turns(np.zeros(16000, dtype=np.float32))
        logger.info(
            "NVIDIA diarization model on %s (%s), warm-up %.2fs.",
            nemotron_device.type.upper(), NEMOTRON_DIAR_DTYPE, time.time() - t0,
        )

    except Exception as e:
        logger.error(
            "Failed to load the NVIDIA diarization model: %s", e,
            exc_info=not isinstance(e, FileNotFoundError),
        )
        logger.error(
            "Diarize requests routed to it (max_speakers of %d or less, or no speaker "
            "count) will return 503.", NEMOTRON_MAX_SPEAKERS,
        )
        nemotron_processor = None
        nemotron_model = None


# ─── Diarizer Routing ───────────────────────────────────────────────────────

def _pick_diarizer(min_speakers: Optional[int], max_speakers: Optional[int]) -> str:
    """
    Choose the diarizer for one request:
      max_speakers above 8        → "pyannote" (Nemotron has 8 speaker channels)
      max_speakers 1..8           → "nemotron"
      no max, min_speakers above 8 → "pyannote" (Nemotron can't reach 9)
      no speaker count            → "nemotron"
    Counts of 0 or less count as absent, as they do for the pyannote hints.
    """
    max_n = max_speakers if max_speakers is not None and max_speakers > 0 else None
    min_n = min_speakers if min_speakers is not None and min_speakers > 0 else None
    if max_n is not None:
        return "pyannote" if max_n > NEMOTRON_MAX_SPEAKERS else "nemotron"
    if min_n is not None and min_n > NEMOTRON_MAX_SPEAKERS:
        return "pyannote"
    return "nemotron"


def _diarizer_unavailable_detail(backend: str) -> Optional[str]:
    """The 503 detail when the chosen diarizer isn't loaded, else None."""
    if backend == "pyannote":
        if diarization_pipeline is None:
            return ("Diarization pipeline not loaded. Set DIARIZATION_ENABLED=1 and "
                    "provide PYANNOTE_MODEL_PATH or HF_TOKEN, then restart the server.")
        return None
    if nemotron_model is None:
        return ("NVIDIA diarization model not loaded. Set DIARIZATION_ENABLED=1, put the "
                "model in NEMOTRON_DIAR_MODEL_PATH and check the server log, then restart the server.")
    return None


def _run_diarization(
    waveform: Optional[np.ndarray],
    file_path: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
):
    """
    Diarize one request with the backend _pick_diarizer chooses.

    Returns (turns, backend, hints, frame_probs): turns as (start, end,
    raw_speaker_label) in chronological order, backend "nemotron" or
    "pyannote", the hints text for the summary log line, and Nemotron's
    per-frame speaker probabilities (None for pyannote). Never falls back to
    the other backend.
    """
    # Optional speaker-count hints from the client. pyannote constrains its
    # auto-detected speaker count to this range, which materially improves
    # attribution on meetings with a known headcount. Nemotron has no such
    # option, so they only choose the backend there.
    dia_kwargs = {}
    if min_speakers is not None and min_speakers > 0:
        dia_kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None and max_speakers > 0:
        dia_kwargs["max_speakers"] = int(max_speakers)
    hints = ("min=%s,max=%s" % (min_speakers, max_speakers)) if dia_kwargs else "none"

    backend = _pick_diarizer(min_speakers, max_speakers)
    unavailable = _diarizer_unavailable_detail(backend)
    if unavailable:
        raise RuntimeError(unavailable)

    if backend == "nemotron":
        if waveform is None:
            raise RuntimeError("Could not decode the audio for NVIDIA diarization.")
        turns, frame_probs = _nemotron_turns(waveform)
        return turns, backend, (hints + " (ignored)" if dia_kwargs else hints), frame_probs

    if waveform is not None:
        try:
            import torch  # type: ignore
            audio_in = {
                "waveform": torch.from_numpy(waveform).unsqueeze(0),
                "sample_rate": 16000,
            }
            diarization = diarization_pipeline(audio_in, **dia_kwargs)
        except Exception as e:
            logger.warning("Diarize from waveform failed (%s); using file path.", e)
            diarization = diarization_pipeline(file_path, **dia_kwargs)
    else:
        diarization = diarization_pipeline(file_path, **dia_kwargs)
    # Handles both pyannote 3.x (Annotation) and 4.x (DiarizeOutput wrapper)
    return _diarization_to_turns(diarization), backend, hints, None


# ─── Shared Model Wrapper for Streaming ─────────────────────────────────────

class SharedModelASR(FasterWhisperASR):
    """FasterWhisperASR that reuses the already-loaded WhisperModel instance."""
    def __init__(self, lan, shared_model, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan in (None, "auto") else lan
        self.model = shared_model


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice Typing ASR Server (Offline)",
    description="Offline speech-to-text powered by Systran/faster-whisper-large-v3",
    version="2.0.0",
)

# CORS — allow all origins WITHOUT credentials.
# (allow_origins=["*"] + allow_credentials=True is spec-invalid and
# causes some browsers to reject preflight responses.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Blocking transcription (runs in thread pool) ───────────────────────────

def _transcribe_sync(file_path: str, language: Optional[str], task: str) -> dict:
    """
    Runs the actual Whisper inference. This is CPU/GPU-bound and MUST NOT
    run on the async event loop — it would block /health and all other
    requests for the entire duration.
    """
    kwargs = {
        "beam_size": BEAM_SIZE,
        "task": task if task in ("transcribe", "translate") else "transcribe",
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,  # #5: avoid clipping word boundaries
        },
        # #3: prevent cascade errors and loop hallucinations on long-form audio
        "condition_on_previous_text": False,
        # #6: discourage "the the the…" / "thanks for watching" loops
        "repetition_penalty": 1.1,
    }

    lang_norm = language.strip() if language and language.strip() else None
    if lang_norm:
        kwargs["language"] = lang_norm
        # #7: language-biasing prompt when we know the language
        if lang_norm in LANGUAGE_PROMPTS:
            kwargs["initial_prompt"] = LANGUAGE_PROMPTS[lang_norm]

    t0 = time.time()
    # #1: batched model when available; identical output, parallelised chunks
    if batched_model is not None:
        segments_iter, info = batched_model.transcribe(file_path, batch_size=BATCH_SIZE, **kwargs)
    else:
        segments_iter, info = model.transcribe(file_path, **kwargs)

    segments = []
    full_text_parts = []

    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  text,
            })
            full_text_parts.append(text)

    elapsed = time.time() - t0
    full_text = " ".join(full_text_parts)

    logger.info(
        "Done: lang=%s (prob=%.2f) | audio=%.1fs | processed in %.2fs | chars=%d",
        info.language, info.language_probability, info.duration, elapsed, len(full_text)
    )

    return {
        "text":                 full_text,
        "language":             info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration":       round(info.duration, 2),
        "processing_time":      round(elapsed, 3),
        "segments":             segments,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_extension(content_type: str, filename: Optional[str]) -> str:
    """Pick a file extension from the content type or filename."""
    ext_map = {
        "audio/webm":  ".webm",
        "audio/wav":   ".wav",
        "audio/wave":  ".wav",
        "audio/x-wav": ".wav",
        "audio/mp3":   ".mp3",
        "audio/mpeg":  ".mp3",
        "audio/ogg":   ".ogg",
        "audio/flac":  ".flac",
        "audio/mp4":   ".m4a",
        "audio/x-m4a": ".m4a",
    }

    # content_type may be "audio/webm;codecs=opus" — strip the parameters
    base_type = (content_type or "").split(";")[0].strip().lower()
    ext = ext_map.get(base_type, "")

    if not ext and filename:
        ext = Path(filename).suffix.lower()

    return ext or ".webm"


def _decode_audio_16k_mono(src_path: str):
    """
    Decode an audio file to a mono 16kHz float32 numpy array in one pass,
    so both pyannote and whisper can consume it without each re-decoding.
    Returns (waveform_np, sample_rate). Raises if neither librosa nor
    soundfile is available.
    """
    # Prefer librosa (handles webm/m4a/etc via audioread/ffmpeg fallback).
    try:
        import librosa  # type: ignore
        wav, sr = librosa.load(src_path, sr=16000, mono=True)
        return wav.astype(np.float32, copy=False), 16000
    except Exception as e:
        # Fallback: transcode via ffmpeg to wav, then read with soundfile.
        import soundfile as sf  # type: ignore
        wav_path = _ensure_wav(src_path)
        data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            # Only librosa-failed path lacks resampler; transcode forces 16k.
            pass
        # Cleanup transcoded wav if we created one
        if wav_path != src_path:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        return data.astype(np.float32, copy=False), int(sr)


def _ensure_wav(src_path: str) -> str:
    """
    Pyannote loads audio via torchaudio→soundfile, which can't decode
    webm/opus/m4a. Whisper uses ffmpeg directly so it doesn't have this
    problem. For the diarization path we transcode to mono 16 kHz wav
    on the fly. Returns the path the caller should use; if it differs
    from src_path, the caller is responsible for cleaning it up.
    """
    ext = Path(src_path).suffix.lower()
    if ext in (".wav", ".flac", ".ogg"):
        return src_path

    wav_path = src_path + ".wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", wav_path,
    ]
    subprocess.run(cmd, check=True)
    return wav_path


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":       "ok",
        "model":        "Systran/faster-whisper-large-v3",
        "model_path":   MODEL_PATH,
        "device":       DEVICE,
        "compute_type": COMPUTE_TYPE,
        "offline":      True,
        "diarization":  diarization_pipeline is not None or nemotron_model is not None,
    }


@app.get("/health")
async def health():
    return {
        "status":       "healthy",
        "model":        "Systran/faster-whisper-large-v3",
        "model_path":   MODEL_PATH,
        "device":       DEVICE,
        "compute_type": COMPUTE_TYPE,
        "beam_size":    BEAM_SIZE,
        "offline":      True,
        "diarization":  diarization_pipeline is not None or nemotron_model is not None,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile    = File(None, description="Audio file (webm, wav, mp3, ogg, flac, m4a)"),
    file: UploadFile    = File(None, description="Audio file (webm, wav, mp3, ogg, flac, m4a)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code. Omit for auto-detect."),
    task: Optional[str]     = Form(None, description="'transcribe' (default) or 'translate' (to English)"),
):
    """
    Transcribe an audio file. Fully offline — no network calls.
    """
    audio = audio or file
    # ── Read & validate ──
    contents = await audio.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="File too large (%.1fMB). Max: %dMB" % (file_size_mb, MAX_FILE_SIZE_MB),
        )

    ext = _resolve_extension(audio.content_type, audio.filename)

    logger.info(
        "Received: %s | type=%s | size=%.2fMB | lang=%s | task=%s",
        audio.filename, audio.content_type, file_size_mb,
        language or "auto", task or "transcribe"
    )

    # ── Write temp file & transcribe in thread ──
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Offload to thread pool so the event loop stays free
        result = await asyncio.to_thread(
            _transcribe_sync,
            tmp_path,
            language,
            task or "transcribe",
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed: %s" % str(e))

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─── Diarization + Transcription (batch) ────────────────────────────────────

def _diarize_and_transcribe_sync(
    file_path: str,
    language: Optional[str],
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """
    Run speaker diarization (Nemotron or pyannote, see _pick_diarizer) and
    whisper transcription, then give each whisper word a speaker: from
    Nemotron's per-frame speaker probabilities, or for pyannote the turn with
    the largest time overlap.

    Segments from the same speaker adjacent in time are merged so that the
    final output is one line per speaker-turn, not one line per whisper cut.
    """
    # #2: Decode the audio once and feed both pipelines from the same array.
    # Falls back to per-pipeline decoding if librosa/soundfile aren't installed.
    waveform = None
    try:
        waveform, _sr = _decode_audio_16k_mono(file_path)
    except Exception as e:
        logger.warning("Audio pre-decode failed (%s); falling back to per-pipeline decode.", e)

    # ── 1. Diarize ──
    # Nemotron or pyannote, chosen from the speaker-count hints (see
    # _pick_diarizer). turns: list of (start, end, raw_speaker_label) in
    # chronological order.
    t_dia = time.time()
    turns, diarizer, hints, frame_probs = _run_diarization(waveform, file_path, min_speakers, max_speakers)
    dia_elapsed = time.time() - t_dia

    # ── 2. Transcribe ──
    t_asr = time.time()
    kwargs = {
        "beam_size": BEAM_SIZE,
        "task": "transcribe",
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,  # #5
        },
        "word_timestamps": True,  # finer-grained alignment for short speaker turns
        "condition_on_previous_text": False,  # #3
        "repetition_penalty": 1.1,  # #6
    }
    lang_norm = language.strip() if language and language.strip() else None
    if lang_norm:
        kwargs["language"] = lang_norm
        if lang_norm in LANGUAGE_PROMPTS:
            kwargs["initial_prompt"] = LANGUAGE_PROMPTS[lang_norm]  # #7

    # #1 + #2: prefer batched + pre-decoded waveform
    audio_input = waveform if waveform is not None else file_path
    if batched_model is not None:
        segments_iter, info = batched_model.transcribe(audio_input, batch_size=BATCH_SIZE, **kwargs)
    else:
        segments_iter, info = model.transcribe(audio_input, **kwargs)

    # Collect per-word timestamps for word-level speaker alignment. We don't
    # need the per-segment list anymore (the old segment-level alignment is
    # replaced below), but we keep gathering individual words so each word
    # can be tagged with its own best-overlap speaker.
    all_words = []  # list of {"start", "end", "text"}
    for seg in segments_iter:
        text = seg.text.strip()
        seg_words = getattr(seg, "words", None) or []
        any_word_added = False
        for w in seg_words:
            wtext = (w.word or "").strip()
            if not wtext:
                continue
            all_words.append({
                "start": float(w.start),
                "end":   float(w.end),
                "text":  wtext,
            })
            any_word_added = True
        # Fallback: if a segment has no usable word timestamps, treat the
        # whole segment text as one pseudo-word so it doesn't disappear.
        if not any_word_added and text:
            all_words.append({
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  text,
            })
    asr_elapsed = time.time() - t_asr

    # ── 3. Word-level alignment to speakers ──
    # Tag every word with a speaker (Nemotron: from its per-frame speaker
    # probabilities; pyannote: the turn it overlaps most), then group
    # consecutive same-speaker words into lines. This cleanly splits a
    # whisper segment when a speaker change happens mid-segment.
    # _best_speaker is the turn rule (also Nemotron's fallback for words in
    # which the model hears nobody).
    def _best_speaker(s_start: float, s_end: float) -> Optional[str]:
        best_spk, best_overlap = None, 0.0
        for t_start, t_end, spk in turns:
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, spk
        # Tiny words ("uh", "the") sometimes land entirely in a silence gap
        # between turns. Fall back to the turn whose midpoint is closest to
        # the word — keeps short interjections attached to the surrounding
        # speaker instead of becoming "Speaker_?".
        if best_spk is None and turns:
            mid = 0.5 * (s_start + s_end)
            best_dist = float("inf")
            for t_start, t_end, spk in turns:
                t_mid = 0.5 * (t_start + t_end)
                d = abs(t_mid - mid)
                if d < best_dist:
                    best_dist, best_spk = d, spk
        return best_spk

    # ── 4. Normalize speaker labels to "Speaker_1", "Speaker_2", ... in first-appearance order ──
    label_map: dict = {}
    def _friendly(raw: Optional[str]) -> str:
        if raw is None:
            return "Speaker_?"
        if raw not in label_map:
            label_map[raw] = "Speaker_%d" % (len(label_map) + 1)
        return label_map[raw]

    # ── 5. Tag each word, then group consecutive same-speaker words into lines ──
    # Nemotron: from its per-frame speaker probabilities; pyannote: by turn overlap.
    if frame_probs is not None:
        raw_speakers = _speakers_from_frame_probs(all_words, frame_probs, _best_speaker)
    else:
        raw_speakers = [_best_speaker(w["start"], w["end"]) for w in all_words]
    tagged_words = []
    for w, raw in zip(all_words, raw_speakers):
        tagged_words.append((_friendly(raw), w["start"], w["end"], w["text"]))

    lines = []   # list of {speaker, start, end, text}
    for spk, w_start, w_end, w_text in tagged_words:
        if lines and lines[-1]["speaker"] == spk:
            lines[-1]["end"] = w_end
            lines[-1]["text"] += w_text if w_text.startswith(" ") else (" " + w_text)
        else:
            lines.append({
                "speaker": spk,
                "start":   w_start,
                "end":     w_end,
                "text":    w_text.lstrip(),
            })
    # Normalise whitespace inside each line
    for ln in lines:
        ln["text"] = " ".join(ln["text"].split())

    # ── 6. Formatted plain-text output for direct pasting ──
    # Blank line between speakers for readability when pasted into a document.
    formatted = "\n\n".join("%s: %s" % (ln["speaker"], ln["text"]) for ln in lines)

    logger.info(
        "Diarized: lang=%s | audio=%.1fs | diarize=%.2fs | asr=%.2fs | speakers=%d | lines=%d | diarizer=%s | hints=%s",
        info.language, info.duration, dia_elapsed, asr_elapsed, len(label_map), len(lines),
        diarizer, hints,
    )

    return {
        "text":                 formatted,
        "language":             info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration":       round(info.duration, 2),
        "num_speakers":         len(label_map),
        "diarization_time":     round(dia_elapsed, 3),
        "transcription_time":   round(asr_elapsed, 3),
        "segments": [
            {
                "speaker": ln["speaker"],
                "start":   round(ln["start"], 2),
                "end":     round(ln["end"], 2),
                "text":    ln["text"],
            }
            for ln in lines
        ],
    }


@app.post("/transcribe-diarize")
async def transcribe_diarize(
    audio: UploadFile    = File(..., description="Audio file (webm, wav, mp3, ogg, flac, m4a)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code. Omit for auto-detect."),
    min_speakers: Optional[int] = Form(None, description="Minimum speaker count hint (pyannote only; the NVIDIA model ignores it)."),
    max_speakers: Optional[int] = Form(None, description="Maximum speaker count hint. Above 8 uses pyannote; 8 or less, or none, uses the NVIDIA model, which ignores it."),
):
    """
    Transcribe an audio file AND identify speakers. Returns diarized text in
    the form "Speaker_1: ...\\n\\nSpeaker_2: ...".

    Requires DIARIZATION_ENABLED=1 and a loaded diarizer: the NVIDIA model
    when max_speakers is 8 or less (or not sent), pyannote above that.
    """
    unavailable = _diarizer_unavailable_detail(_pick_diarizer(min_speakers, max_speakers))
    if unavailable:
        raise HTTPException(status_code=503, detail=unavailable)

    contents = await audio.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="File too large (%.1fMB). Max: %dMB" % (file_size_mb, MAX_FILE_SIZE_MB),
        )

    ext = _resolve_extension(audio.content_type, audio.filename)

    logger.info(
        "Diarize request: %s | type=%s | size=%.2fMB | lang=%s",
        audio.filename, audio.content_type, file_size_mb, language or "auto",
    )

    tmp_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Pyannote can't decode webm/opus/m4a — transcode to wav if needed.
        wav_path = await asyncio.to_thread(_ensure_wav, tmp_path)

        result = await asyncio.to_thread(
            _diarize_and_transcribe_sync,
            wav_path,
            language,
            min_speakers,
            max_speakers,
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Diarization failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Diarization failed: %s" % str(e))

    finally:
        # Cleanup both files; wav_path may equal tmp_path when no transcode was needed.
        seen = set()
        for p in (tmp_path, wav_path):
            if p and p not in seen:
                seen.add(p)
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ─── SSE Streaming Endpoints (progress events) ──────────────────────────────
#
# These endpoints mirror /transcribe and /transcribe-diarize but stream
# progress events as Server-Sent Events while inference runs. The client
# receives a series of `data: {json}\n\n` chunks describing the current
# phase, percent complete, and (where available) an ETA derived from real
# throughput. The final event always has phase="done" and carries the same
# JSON payload the non-streaming endpoints return today.
# ─────────────────────────────────────────────────────────────────────────────


def _sse(payload: dict) -> str:
    """Format a Server-Sent Events frame."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


# Sentinel pushed onto the worker→generator queue when the worker exits
_STREAM_DONE = object()


async def _stream_worker_events(worker: Callable[[Callable[[dict], None]], None]):
    """
    Run a blocking worker in a thread; bridge events it emits onto an
    asyncio queue and yield them as SSE frames.

    `worker(emit)` is called once on a background thread. `emit(payload)`
    is a thread-safe callback the worker uses to push events.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def emit(payload: dict) -> None:
        # Called from the worker thread — schedule the put on the loop.
        loop.call_soon_threadsafe(q.put_nowait, payload)

    def run() -> None:
        try:
            worker(emit)
        except Exception as e:
            logger.error("Streaming worker failed: %s", e, exc_info=True)
            try:
                loop.call_soon_threadsafe(
                    q.put_nowait,
                    {"phase": "error", "detail": str(e)},
                )
            except RuntimeError:
                pass
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _STREAM_DONE)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    try:
        while True:
            event = await q.get()
            if event is _STREAM_DONE:
                break
            yield _sse(event)
    finally:
        # Best-effort: if the client disconnects mid-stream the thread will
        # finish on its own; we just stop reading from the queue.
        pass


def _transcribe_sync_streaming(
    file_path: str,
    language: Optional[str],
    task: str,
    emit: Callable[[dict], None],
) -> None:
    """
    Streaming variant of _transcribe_sync. Emits:
      - {"phase":"transcribing","audio_duration":...,"language":...,"percent":0}
      - per-segment {"phase":"transcribing","percent":...,"audio_done":...,"eta_seconds":...,"partial_text":...}
      - final {"phase":"done", <same payload as /transcribe>}
    """
    kwargs = {
        "beam_size": BEAM_SIZE,
        "task": task if task in ("transcribe", "translate") else "transcribe",
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,  # #5
        },
        "condition_on_previous_text": False,  # #3
        "repetition_penalty": 1.1,  # #6
    }
    lang_norm = language.strip() if language and language.strip() else None
    if lang_norm:
        kwargs["language"] = lang_norm
        if lang_norm in LANGUAGE_PROMPTS:
            kwargs["initial_prompt"] = LANGUAGE_PROMPTS[lang_norm]  # #7

    t0 = time.time()
    # #1: batched pipeline when available
    if batched_model is not None:
        segments_iter, info = batched_model.transcribe(file_path, batch_size=BATCH_SIZE, **kwargs)
    else:
        segments_iter, info = model.transcribe(file_path, **kwargs)

    # info.duration is populated after the initial decode/VAD pass returns.
    audio_duration = float(info.duration) if info.duration else 0.0
    emit({
        "phase": "transcribing",
        "audio_duration": round(audio_duration, 2),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "percent": 0,
    })

    segments = []
    full_text_parts = []

    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  text,
            })
            full_text_parts.append(text)

        elapsed = time.time() - t0
        audio_done = float(seg.end)
        if audio_duration > 0:
            percent = min(99.0, (audio_done / audio_duration) * 100.0)
        else:
            percent = 0.0
        throughput = audio_done / elapsed if elapsed > 0 else 0.0
        eta = ((audio_duration - audio_done) / throughput) if throughput > 0 else None

        emit({
            "phase": "transcribing",
            "percent": round(percent, 1),
            "audio_done": round(audio_done, 2),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta, 1) if eta is not None and eta >= 0 else None,
            "partial_text": text,
        })

    elapsed = time.time() - t0
    full_text = " ".join(full_text_parts)

    logger.info(
        "Done (stream): lang=%s (prob=%.2f) | audio=%.1fs | processed in %.2fs | chars=%d",
        info.language, info.language_probability, audio_duration, elapsed, len(full_text)
    )

    emit({
        "phase": "done",
        "text":                 full_text,
        "language":             info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration":       round(audio_duration, 2),
        "processing_time":      round(elapsed, 3),
        "segments":             segments,
    })


def _diarize_and_transcribe_sync_streaming(
    file_path: str,
    language: Optional[str],
    emit: Callable[[dict], None],
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> None:
    """
    Streaming variant of _diarize_and_transcribe_sync. Emits phase events
    for diarize → transcribe → align, with per-segment progress during
    transcription. Final event mirrors /transcribe-diarize's JSON.
    """
    # #2: Decode the audio once and feed both pipelines from the same array.
    waveform = None
    try:
        waveform, _sr = _decode_audio_16k_mono(file_path)
    except Exception as e:
        logger.warning("Audio pre-decode failed (%s); falling back to per-pipeline decode.", e)

    # ── 1. Diarize ──
    # Nemotron or pyannote, chosen from the speaker-count hints (see
    # _pick_diarizer). turns: list of (start, end, raw_speaker_label) in
    # chronological order.
    emit({"phase": "diarizing"})
    t_dia = time.time()
    turns, diarizer, hints, frame_probs = _run_diarization(waveform, file_path, min_speakers, max_speakers)
    dia_elapsed = time.time() - t_dia
    emit({
        "phase": "diarized",
        "diarization_time": round(dia_elapsed, 3),
        "num_raw_turns": len(turns),
    })

    # ── 2. Transcribe (streamed) ──
    t_asr = time.time()
    kwargs = {
        "beam_size": BEAM_SIZE,
        "task": "transcribe",
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,  # #5
        },
        "word_timestamps": True,
        "condition_on_previous_text": False,  # #3
        "repetition_penalty": 1.1,  # #6
    }
    lang_norm = language.strip() if language and language.strip() else None
    if lang_norm:
        kwargs["language"] = lang_norm
        if lang_norm in LANGUAGE_PROMPTS:
            kwargs["initial_prompt"] = LANGUAGE_PROMPTS[lang_norm]  # #7

    # #1 + #2: prefer batched + pre-decoded waveform
    audio_input = waveform if waveform is not None else file_path
    if batched_model is not None:
        segments_iter, info = batched_model.transcribe(audio_input, batch_size=BATCH_SIZE, **kwargs)
    else:
        segments_iter, info = model.transcribe(audio_input, **kwargs)
    audio_duration = float(info.duration) if info.duration else 0.0
    emit({
        "phase": "transcribing",
        "audio_duration": round(audio_duration, 2),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "percent": 0,
    })

    # Collect both segments (for the streamed progress UI) and words
    # (for word-level speaker alignment in step 3). word_timestamps=True
    # in kwargs above ensures Whisper produces per-word timing.
    whisper_segs = []
    all_words = []  # list of {"start", "end", "text"} for every word
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            whisper_segs.append({
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  text,
            })
        # Pull per-word timestamps. Whisper occasionally returns seg.words=None
        # (rare; happens when word-alignment failed on a chunk). Fall back to
        # treating the whole segment as one pseudo-word so we never lose text.
        seg_words = getattr(seg, "words", None) or []
        any_word_added = False
        for w in seg_words:
            wtext = (w.word or "").strip()
            if not wtext:
                continue
            all_words.append({
                "start": float(w.start),
                "end":   float(w.end),
                "text":  wtext,
            })
            any_word_added = True
        if not any_word_added and text:
            # Fallback: no usable word timestamps for this segment
            all_words.append({
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  text,
            })

        elapsed = time.time() - t_asr
        audio_done = float(seg.end)
        if audio_duration > 0:
            percent = min(99.0, (audio_done / audio_duration) * 100.0)
        else:
            percent = 0.0
        throughput = audio_done / elapsed if elapsed > 0 else 0.0
        eta = ((audio_duration - audio_done) / throughput) if throughput > 0 else None

        emit({
            "phase": "transcribing",
            "percent": round(percent, 1),
            "audio_done": round(audio_done, 2),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta, 1) if eta is not None and eta >= 0 else None,
            "partial_text": text,
        })

    asr_elapsed = time.time() - t_asr

    # ── 3. Word-level alignment to speakers ──
    # Previously we tagged whole whisper segments with a single speaker, which
    # collapsed two-speaker exchanges within one segment to a single label.
    # Now we tag each *word* with a speaker (Nemotron: from its per-frame
    # speaker probabilities; pyannote: its best-overlap turn), then group
    # consecutive same-speaker words into lines so a speaker change inside a
    # segment cleanly splits into two lines. _best_speaker is the turn rule
    # (also Nemotron's fallback for words in which the model hears nobody).
    emit({"phase": "aligning"})

    def _best_speaker(s_start: float, s_end: float) -> Optional[str]:
        best_spk, best_overlap = None, 0.0
        for t_start, t_end, spk in turns:
            overlap = max(0.0, min(s_end, t_end) - max(s_start, t_start))
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, spk
        # Tiny words (e.g. "uh", "the") sometimes fall entirely in a silence
        # gap between turns and get no overlap. In that case use the midpoint
        # of the word to find the nearest turn — keeps short interjections
        # attached to the surrounding speaker instead of becoming "Speaker_?".
        if best_spk is None and turns:
            mid = 0.5 * (s_start + s_end)
            # Pick the turn whose midpoint is closest to this word's midpoint
            best_dist = float("inf")
            for t_start, t_end, spk in turns:
                t_mid = 0.5 * (t_start + t_end)
                d = abs(t_mid - mid)
                if d < best_dist:
                    best_dist, best_spk = d, spk
        return best_spk

    label_map: dict = {}
    def _friendly(raw: Optional[str]) -> str:
        if raw is None:
            return "Speaker_?"
        if raw not in label_map:
            label_map[raw] = "Speaker_%d" % (len(label_map) + 1)
        return label_map[raw]

    # Tag every word: Nemotron from its per-frame speaker probabilities,
    # pyannote with its best-overlap speaker.
    if frame_probs is not None:
        raw_speakers = _speakers_from_frame_probs(all_words, frame_probs, _best_speaker)
    else:
        raw_speakers = [_best_speaker(w["start"], w["end"]) for w in all_words]
    tagged_words = []
    for w, raw in zip(all_words, raw_speakers):
        tagged_words.append((_friendly(raw), w["start"], w["end"], w["text"]))

    # Group consecutive same-speaker words into lines. Lines now break on
    # speaker change rather than on whisper-segment boundaries.
    lines = []
    for spk, w_start, w_end, w_text in tagged_words:
        if lines and lines[-1]["speaker"] == spk:
            lines[-1]["end"] = w_end
            # Whisper word .text already includes a leading space when needed,
            # so just concatenate. Strip the line at the end to clean any
            # stray leading space.
            lines[-1]["text"] += w_text if w_text.startswith(" ") else (" " + w_text)
        else:
            lines.append({
                "speaker": spk,
                "start":   w_start,
                "end":     w_end,
                "text":    w_text.lstrip(),
            })
    # Normalise whitespace in each line
    for ln in lines:
        ln["text"] = " ".join(ln["text"].split())

    formatted = "\n\n".join("%s: %s" % (ln["speaker"], ln["text"]) for ln in lines)

    logger.info(
        "Diarized (stream): lang=%s | audio=%.1fs | diarize=%.2fs | asr=%.2fs | speakers=%d | lines=%d | diarizer=%s | hints=%s",
        info.language, audio_duration, dia_elapsed, asr_elapsed, len(label_map), len(lines),
        diarizer, hints,
    )

    emit({
        "phase": "done",
        "text":                 formatted,
        "language":             info.language,
        "language_probability": round(info.language_probability, 3),
        "audio_duration":       round(audio_duration, 2),
        "num_speakers":         len(label_map),
        "diarization_time":     round(dia_elapsed, 3),
        "transcription_time":   round(asr_elapsed, 3),
        "segments": [
            {
                "speaker": ln["speaker"],
                "start":   round(ln["start"], 2),
                "end":     round(ln["end"], 2),
                "text":    ln["text"],
            }
            for ln in lines
        ],
    })


@app.post("/transcribe-streaming")
async def transcribe_streaming(
    audio: UploadFile    = File(..., description="Audio file (webm, wav, mp3, ogg, flac, m4a)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code. Omit for auto-detect."),
    task: Optional[str]     = Form(None, description="'transcribe' (default) or 'translate' (to English)"),
):
    """
    Streaming version of /transcribe. Returns an SSE stream of progress
    events. The final event has phase="done" and carries the same JSON
    payload /transcribe returns.
    """
    contents = await audio.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="File too large (%.1fMB). Max: %dMB" % (file_size_mb, MAX_FILE_SIZE_MB),
        )

    ext = _resolve_extension(audio.content_type, audio.filename)

    logger.info(
        "Stream request: %s | type=%s | size=%.2fMB | lang=%s | task=%s",
        audio.filename, audio.content_type, file_size_mb,
        language or "auto", task or "transcribe"
    )

    # Write the temp file synchronously here so pre-stream validation errors
    # surface as HTTP errors. Cleanup happens after the stream finishes.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    async def stream():
        try:
            yield _sse({
                "phase": "received",
                "file_size_mb": round(file_size_mb, 3),
                "filename": audio.filename or "",
            })

            def worker(emit):
                _transcribe_sync_streaming(tmp_path, language, task or "transcribe", emit)

            async for chunk in _stream_worker_events(worker):
                yield chunk
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
        },
    )


@app.post("/transcribe-diarize-streaming")
async def transcribe_diarize_streaming(
    audio: UploadFile    = File(..., description="Audio file (webm, wav, mp3, ogg, flac, m4a)"),
    language: Optional[str] = Form(None, description="ISO 639-1 language code. Omit for auto-detect."),
    min_speakers: Optional[int] = Form(None, description="Minimum speaker count hint (pyannote only; the NVIDIA model ignores it)."),
    max_speakers: Optional[int] = Form(None, description="Maximum speaker count hint. Above 8 uses pyannote; 8 or less, or none, uses the NVIDIA model, which ignores it."),
):
    """
    Streaming version of /transcribe-diarize. Returns an SSE stream of
    progress events (diarizing → transcribing → aligning → done). The
    final event carries the same JSON payload /transcribe-diarize returns.
    """
    unavailable = _diarizer_unavailable_detail(_pick_diarizer(min_speakers, max_speakers))
    if unavailable:
        raise HTTPException(status_code=503, detail=unavailable)

    contents = await audio.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="File too large (%.1fMB). Max: %dMB" % (file_size_mb, MAX_FILE_SIZE_MB),
        )

    ext = _resolve_extension(audio.content_type, audio.filename)

    logger.info(
        "Diarize-stream request: %s | type=%s | size=%.2fMB | lang=%s",
        audio.filename, audio.content_type, file_size_mb, language or "auto",
    )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Pyannote can't decode webm/opus/m4a — transcode to wav up-front.
    # We do this before opening the stream so the user sees a transcoding
    # phase event only when it actually happens.
    needs_transcode = Path(tmp_path).suffix.lower() not in (".wav", ".flac", ".ogg")

    async def stream():
        wav_path = tmp_path
        try:
            yield _sse({
                "phase": "received",
                "file_size_mb": round(file_size_mb, 3),
                "filename": audio.filename or "",
            })

            if needs_transcode:
                yield _sse({"phase": "transcoding"})
                t_tx = time.time()
                wav_path = await asyncio.to_thread(_ensure_wav, tmp_path)
                yield _sse({
                    "phase": "transcoded",
                    "transcode_time": round(time.time() - t_tx, 3),
                })

            def worker(emit):
                _diarize_and_transcribe_sync_streaming(
                    wav_path, language, emit, min_speakers, max_speakers
                )

            async for chunk in _stream_worker_events(worker):
                yield chunk
        finally:
            seen = set()
            for p in (tmp_path, wav_path):
                if p and p not in seen:
                    seen.add(p)
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── WebSocket Streaming Endpoint ────────────────────────────────────────

@app.websocket("/transcribe-stream")
async def transcribe_stream(websocket: WebSocket):
    await websocket.accept()

    language = None
    processor = None
    pcm_buffer = np.array([], dtype=np.float32)
    cumulative_confirmed = ""
    processing_task = None
    last_process_time = 0.0
    MIN_PROCESS_INTERVAL = 1.0   # seconds between process_iter calls
    MIN_SAMPLES = 16000          # 1 second of audio at 16kHz

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])

                if data["type"] == "config":
                    language = data.get("language")
                    if language == "auto":
                        language = None
                    asr = SharedModelASR(language, model)
                    processor = OnlineASRProcessor(asr)
                    logger.info("Streaming session started: lang=%s", language or "auto")

                elif data["type"] == "stop":
                    # Wait for any in-flight processing
                    if processing_task and not processing_task.done():
                        await processing_task

                    # Feed any remaining buffered audio
                    if len(pcm_buffer) > 0 and processor:
                        await asyncio.to_thread(processor.insert_audio_chunk, pcm_buffer)
                        pcm_buffer = np.array([], dtype=np.float32)

                    # Final transcription of complete audio buffer
                    if processor and len(processor.audio_buffer) > 0:
                        t0 = time.time()
                        def _final_transcribe():
                            prompt, _ = processor.prompt()
                            res = processor.asr.transcribe(processor.audio_buffer, init_prompt=prompt)
                            parts = []
                            for seg in res:
                                text = seg.text.strip()
                                if not text:
                                    continue
                                if seg.no_speech_prob > 0.8 and text.lower().rstrip(".!,") in HALLUCINATIONS:
                                    continue
                                parts.append(text)
                            return " ".join(parts).strip()
                        total = await asyncio.to_thread(_final_transcribe)
                        elapsed = time.time() - t0

                        logger.info("Streaming session done: %d chars in %.2fs", len(total), elapsed)
                        await websocket.send_json({
                            "type": "final",
                            "text": total,
                            "language": language or "auto",
                            "processing_time": round(elapsed, 3),
                        })
                    else:
                        await websocket.send_json({
                            "type": "final",
                            "text": "",
                            "language": language or "auto",
                            "processing_time": 0,
                        })
                    await websocket.close()
                    break

            elif "bytes" in message:
                audio = np.frombuffer(message["bytes"], dtype=np.float32)
                pcm_buffer = np.append(pcm_buffer, audio)

                # Trigger processing when enough audio and no in-flight task
                now = time.time()
                ready = (
                    processor is not None
                    and len(pcm_buffer) >= MIN_SAMPLES
                    and (processing_task is None or processing_task.done())
                    and now - last_process_time >= MIN_PROCESS_INTERVAL
                )
                if ready:
                    last_process_time = now
                    chunk = pcm_buffer.copy()
                    pcm_buffer = np.array([], dtype=np.float32)

                    # Common Whisper hallucination phrases on silence
                    HALLUCINATIONS = {
                        "thank you", "thank you.", "thanks for watching",
                        "thank you for watching", "thank you for watching.",
                        "thanks for watching.", "please subscribe",
                        "please subscribe.", "like and subscribe",
                        "subtitles by the amara.org community",
                        "you", "the end", "the end.",
                    }

                    async def run_process(audio_chunk, ws=websocket):
                        nonlocal cumulative_confirmed
                        def _sync():
                            processor.insert_audio_chunk(audio_chunk)

                            # Transcribe the full audio buffer
                            prompt, _ = processor.prompt()
                            res = processor.asr.transcribe(processor.audio_buffer, init_prompt=prompt)

                            # Extract text from all segments
                            raw_parts = []
                            tsw = []
                            for seg in res:
                                text = seg.text.strip()
                                if not text:
                                    continue
                                # Filter hallucinations: short text with high no_speech_prob
                                if seg.no_speech_prob > 0.8 and text.lower().rstrip(".!,") in HALLUCINATIONS:
                                    continue
                                raw_parts.append(text)
                                if seg.words:
                                    for word in seg.words:
                                        tsw.append((word.start, word.end, word.word))

                            raw_text = " ".join(raw_parts).strip()

                            # Run hypothesis buffer for tracking
                            processor.transcript_buffer.insert(tsw, processor.buffer_time_offset)
                            o = processor.transcript_buffer.flush()
                            processor.commited.extend(o)

                            # Trim buffer if needed
                            if processor.buffer_trimming_way == "segment":
                                s = processor.buffer_trimming_sec
                            else:
                                s = 30
                            if len(processor.audio_buffer) / processor.SAMPLING_RATE > s:
                                processor.chunk_completed_segment(res)

                            return raw_text

                        raw_text = await asyncio.to_thread(_sync)

                        # Send the full raw transcription as interim —
                        # most responsive for the user
                        if raw_text:
                            await ws.send_json({
                                "type": "interim",
                                "text": raw_text,
                            })

                    processing_task = asyncio.create_task(run_process(chunk))

    except WebSocketDisconnect:
        logger.info("Streaming client disconnected")
        if processing_task and not processing_task.done():
            processing_task.cancel()
    except Exception as e:
        logger.error("Streaming error: %s", e, exc_info=True)
        if processing_task and not processing_task.done():
            processing_task.cancel()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting offline ASR server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)