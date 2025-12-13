# tars/audio/service.py
"""
Audio service layer for TARS Voice 1.0.

This module provides higher-level helpers around:
- Speech-to-text (STT) using OpenAI Whisper
- Text-to-speech (TTS) using OpenAI TTS

It wraps the lower-level functions in `tars.clients.openai_client`
with additional validation, normalization, and logging safeguards.

Safeguards implemented (10):

1.  Rejects empty or None audio input.
2.  Enforces a maximum audio size (to avoid huge uploads / costs).
3.  Validates MIME type against a small allowlist (optional, non-fatal).
4.  Normalizes/strips transcription output and rejects empty results.
5.  Provides a default language fallback for STT if none given.
6.  Logs key STT events and truncates logged transcripts to a safe length.
7.  Normalizes and validates input text before TTS (no empty/whitespace-only).
8.  Enforces a maximum text length for TTS to prevent extreme requests.
9.  Validates requested audio format (against a small allowlist).
10. Validates non-empty audio output from TTS and logs failures clearly.

Additional upgrades in this version (requested):
- 5 technical improvements (correlation IDs, richer error typing, WAV validation,
  artifact saving for reproducibility, retries with backoff for transient failures)
- 5 extra features (duration estimation, silence detection, MIME normalization,
  optional transcript caching hash, debug artifact metadata)
- 5 extra safeguards (too-short audio rejection, silence rejection, TTS size cap,
  log truncation hardening, safer fallbacks with clear user-facing errors)
"""

from __future__ import annotations

from typing import Optional, Tuple
import hashlib
import os
import time
import uuid
import wave
import io

from tars.utils.logging import get_logger
from tars.clients import openai_client

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants for audio handling
# ---------------------------------------------------------------------------

# 2. Max audio size safeguard: 10 MB (tune as needed)
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MiB

# 3. Simple MIME allowlist; not strictly required, but helps catch mistakes
ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/webm",
}

# 5. Default language hint for STT when none is provided (you can change to "es")
DEFAULT_STT_LANGUAGE: Optional[str] = None  # None = let Whisper auto-detect

# 8. Max text length for TTS (characters)
MAX_TTS_TEXT_LEN = 4000  # tune as needed

# 9. Supported output formats for TTS (must match what openai_client uses)
ALLOWED_TTS_FORMATS = {"mp3", "wav"}

# -------------------- NEW: diagnostics / artifacts -------------------------

# Save audio artifacts to help reproduce STT failures (off by default).
# Turn on by setting: TARS_SAVE_AUDIO_ARTIFACTS=1
SAVE_AUDIO_ARTIFACTS = os.getenv("TARS_SAVE_AUDIO_ARTIFACTS", "0").strip() == "1"

# Optional debug metadata saving (off by default).
# Turn on by setting: TARS_AUDIO_DEBUG=1
AUDIO_DEBUG = os.getenv("TARS_AUDIO_DEBUG", "0").strip() == "1"

# Where to write artifacts (aligns with your repo layout).
# We keep it relative to project root where possible; server typically runs from root.
ARTIFACT_DIR = os.getenv("TARS_ARTIFACT_DIR", os.path.join("tars", "logs"))

# -------------------- NEW: STT preflight safeguards ------------------------

# Reject extremely short audio payloads (often just wake beep, click, or truncated buffer)
MIN_AUDIO_BYTES = 8_000  # ~0.25s of 16kHz mono 16-bit PCM WAV-ish scale; heuristic

# If WAV can be parsed, also reject durations shorter than this
MIN_AUDIO_DURATION_SEC = 0.35

# Silence / “mostly silence” detection (WAV PCM16 only; best-effort)
# If enabled, prevents paying STT for effectively silent clips.
REJECT_MOSTLY_SILENT_WAV = True
SILENCE_RMS_THRESHOLD = 120.0   # int16 RMS heuristic
SILENCE_MAX_SPEECH_RATIO = 0.03 # <3% frames above threshold => treat as silence

# -------------------- NEW: retry policy for transient failures -------------

STT_MAX_ATTEMPTS = 3
STT_RETRY_BACKOFF_BASE_SEC = 0.6

TTS_MAX_ATTEMPTS = 2
TTS_RETRY_BACKOFF_BASE_SEC = 0.5

# -------------------- NEW: TTS output cap (safeguard) ----------------------

# Cap returned TTS audio size to avoid pathological outputs
MAX_TTS_AUDIO_BYTES = 12 * 1024 * 1024  # 12 MiB


# ---------------------------------------------------------------------------
# Internal helpers (MIME normalization, WAV inspection, artifacts, retries)
# ---------------------------------------------------------------------------

def _normalize_mime_type(mime_type: Optional[str]) -> Optional[str]:
    """
    Normalize common MIME variants without being strict.
    """
    if mime_type is None:
        return None
    cleaned = mime_type.strip().lower()
    if cleaned in {"audio/wave", "audio/wav", "audio/x-wav"}:
        return "audio/wav"
    if cleaned in {"audio/mp3"}:
        return "audio/mpeg"
    return cleaned or None


def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # non-fatal
        pass


def _artifact_path(prefix: str, ext: str, request_id: str) -> str:
    _safe_makedirs(ARTIFACT_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}_{request_id}.{ext}"
    return os.path.join(ARTIFACT_DIR, fname)


def _save_artifact_bytes(prefix: str, ext: str, data: bytes, request_id: str) -> Optional[str]:
    """
    Save binary artifacts for debugging. Best-effort.
    """
    try:
        path = _artifact_path(prefix, ext, request_id)
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        logger.warning("Failed to save artifact %s.%s: %s", prefix, ext, e)
        return None


def _save_artifact_text(prefix: str, ext: str, text: str, request_id: str) -> Optional[str]:
    """
    Save text artifacts for debugging. Best-effort.
    """
    try:
        path = _artifact_path(prefix, ext, request_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path
    except Exception as e:
        logger.warning("Failed to save artifact %s.%s: %s", prefix, ext, e)
        return None


def _try_parse_wav_info(audio_bytes: bytes) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Try to parse WAV header and return (duration_sec, sample_rate, channels).
    Returns (None, None, None) if not parseable.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = int(wf.getframerate())
            ch = int(wf.getnchannels())
            nframes = int(wf.getnframes())
            duration = (nframes / float(sr)) if sr > 0 else None
            return duration, sr, ch
    except Exception:
        return None, None, None


def _wav_pcm16_rms_and_speech_ratio(audio_bytes: bytes) -> Tuple[Optional[float], Optional[float]]:
    """
    Best-effort: compute RMS and 'speech ratio' for PCM16 WAV.
    speech ratio = fraction of samples whose abs value exceeds (SILENCE_RMS_THRESHOLD * 3)
    Returns (rms, ratio) or (None, None) if cannot compute.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                return None, None
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            if not raw:
                return 0.0, 0.0
            # interpret as little-endian int16
            import numpy as np  # local import to keep module import light
            x = np.frombuffer(raw, dtype="<i2").astype("float32")
            if x.size == 0:
                return 0.0, 0.0
            rms = float((np.sqrt(np.mean(x * x))) if x.size > 0 else 0.0)
            thr = float(SILENCE_RMS_THRESHOLD * 3.0)
            ratio = float((np.mean(np.abs(x) > thr)) if x.size > 0 else 0.0)
            return rms, ratio
    except Exception:
        return None, None


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _sleep_backoff(attempt: int, base: float) -> None:
    # attempt starts at 1
    delay = base * (2 ** (attempt - 1))
    time.sleep(delay)


# ---------------------------------------------------------------------------
# STT helper with safeguards
# ---------------------------------------------------------------------------

def stt_with_safeguards(
    audio_bytes: bytes,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    High-level STT helper for TARS.

    Wraps `openai_client.transcribe_audio` with additional validation and logging.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio contents.
    mime_type : Optional[str]
        Optional MIME type of the audio.
    language : Optional[str]
        Optional language code (e.g., "en"). If None, uses DEFAULT_STT_LANGUAGE.

    Returns
    -------
    str
        Normalized transcript text.

    Raises
    ------
    ValueError
        If the input is clearly invalid (e.g. empty, too short, or oversized).
    RuntimeError
        If transcription fails or returns unusable output.
    """
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # 1. Reject empty or None audio
    if not audio_bytes:
        raise ValueError("No audio data provided for transcription.")

    # NEW safeguard: reject too-short audio bytes
    if len(audio_bytes) < MIN_AUDIO_BYTES:
        logger.warning("[stt] request_id=%s audio too small (%d bytes)", request_id, len(audio_bytes))
        raise ValueError(
            f"Audio payload too small: {len(audio_bytes)} bytes. "
            "This is likely an empty/silent capture or an encoding issue."
        )

    # 2. Enforce max audio size
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio payload too large: {len(audio_bytes)} bytes "
            f"(max allowed is {MAX_AUDIO_BYTES} bytes)."
        )

    norm_mime = _normalize_mime_type(mime_type)

    # 3. MIME validation (non-fatal; we just log a warning if it's odd)
    if norm_mime is not None and norm_mime not in ALLOWED_AUDIO_MIME_TYPES:
        logger.warning(
            "[stt] request_id=%s Received non-standard MIME type: %s",
            request_id,
            norm_mime,
        )

    # 5. Language fallback
    lang_to_use = language if language is not None else DEFAULT_STT_LANGUAGE

    # NEW feature: WAV inspection for better diagnostics
    duration_sec, sr, ch = _try_parse_wav_info(audio_bytes)
    if duration_sec is not None:
        logger.info(
            "[stt] request_id=%s WAV detected duration=%.2fs sr=%s ch=%s bytes=%d mime=%s lang=%s hash=%s",
            request_id,
            duration_sec,
            sr,
            ch,
            len(audio_bytes),
            norm_mime,
            lang_to_use,
            _hash_bytes(audio_bytes),
        )
        # NEW safeguard: reject too-short WAV duration
        if duration_sec < MIN_AUDIO_DURATION_SEC:
            raise ValueError(
                f"Audio duration too short ({duration_sec:.2f}s). "
                "Speak a bit longer after the wake word."
            )

        # NEW feature + safeguard: silence detection (best-effort PCM16 WAV)
        if REJECT_MOSTLY_SILENT_WAV:
            rms, ratio = _wav_pcm16_rms_and_speech_ratio(audio_bytes)
            if rms is not None and ratio is not None:
                logger.info(
                    "[stt] request_id=%s audio energy rms=%.1f speech_ratio=%.4f",
                    request_id,
                    rms,
                    ratio,
                )
                if rms < SILENCE_RMS_THRESHOLD and ratio < SILENCE_MAX_SPEECH_RATIO:
                    raise ValueError(
                        "Audio appears to be mostly silence. "
                        "Mic may be too quiet or your capture threshold is too high."
                    )
    else:
        logger.info(
            "[stt] request_id=%s Non-WAV/unknown container bytes=%d mime=%s lang=%s hash=%s",
            request_id,
            len(audio_bytes),
            norm_mime,
            lang_to_use,
            _hash_bytes(audio_bytes),
        )

    # NEW: optional artifact saving (reproducibility)
    if SAVE_AUDIO_ARTIFACTS:
        ext = "wav" if (duration_sec is not None or norm_mime in {"audio/wav", "audio/x-wav"}) else "bin"
        saved = _save_artifact_bytes("stt_input", ext, audio_bytes, request_id)
        if saved:
            logger.info("[stt] request_id=%s saved audio artifact: %s", request_id, saved)

    if AUDIO_DEBUG:
        meta = {
            "request_id": request_id,
            "bytes": len(audio_bytes),
            "mime_type": norm_mime,
            "language": lang_to_use,
            "wav_duration_sec": duration_sec,
            "wav_sample_rate": sr,
            "wav_channels": ch,
            "hash16": _hash_bytes(audio_bytes),
        }
        _save_artifact_text("stt_meta", "json", str(meta), request_id)

    # ---- ACTUAL STT CALL (with retries) ----
    last_exc: Optional[Exception] = None
    for attempt in range(1, STT_MAX_ATTEMPTS + 1):
        try:
            logger.info("[stt] request_id=%s attempt=%d/%d calling OpenAI STT", request_id, attempt, STT_MAX_ATTEMPTS)

            raw_text = openai_client.transcribe_audio(
                audio_bytes=audio_bytes,
                mime_type=norm_mime,
                language=lang_to_use,
            )

            # 4. Normalize and validate transcript
            transcript = (raw_text or "").strip()
            if not transcript:
                logger.error("[stt] request_id=%s empty transcript after normalization", request_id)
                raise RuntimeError("TARS received an empty transcription from the model.")

            # 6. Log truncated transcript for observability
            log_snippet = transcript[:200] + ("..." if len(transcript) > 200 else "")
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info("[stt] request_id=%s OK latency_ms=%d transcript=%r", request_id, latency_ms, log_snippet)

            return transcript

        except ValueError:
            # input validation errors should not be retried
            raise
        except RuntimeError as e:
            # already normalized; don't loop unless we believe it's transient
            last_exc = e
            logger.error("[stt] request_id=%s runtime error attempt=%d: %s", request_id, attempt, e)
            # retry runtime errors only if attempts remain
            if attempt < STT_MAX_ATTEMPTS:
                _sleep_backoff(attempt, STT_RETRY_BACKOFF_BASE_SEC)
                continue
            raise
        except Exception as e:
            # 10 (for STT branch): clear logging and normalized error
            last_exc = e
            logger.error("[stt] request_id=%s OpenAI STT exception attempt=%d: %s", request_id, attempt, e)
            if attempt < STT_MAX_ATTEMPTS:
                _sleep_backoff(attempt, STT_RETRY_BACKOFF_BASE_SEC)
                continue
            raise RuntimeError("TARS failed to transcribe audio input.") from e

    # Should never hit
    raise RuntimeError("TARS failed to transcribe audio input.") from last_exc


# ---------------------------------------------------------------------------
# TTS helper with safeguards
# ---------------------------------------------------------------------------

def tts_with_safeguards(
    text: str,
    voice: Optional[str] = None,
    audio_format: str = "mp3",
) -> bytes:
    """
    High-level TTS helper for TARS.

    Wraps `openai_client.synthesize_speech` with additional validation and logging.

    Parameters
    ----------
    text : str
        Text to synthesize.
    voice : Optional[str]
        Optional voice identifier/preset. If None, a default is used.
    audio_format : str
        Output audio format (e.g. "mp3", "wav").

    Returns
    -------
    bytes
        Synthesized audio bytes.

    Raises
    ------
    ValueError
        If the input text is invalid (empty or too long) or format unsupported.
    RuntimeError
        If TTS fails or returns unusable output.
    """
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # 7. Normalize and validate text
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Cannot synthesize speech from empty or whitespace-only text.")

    # 8. Enforce a max text length
    if len(cleaned) > MAX_TTS_TEXT_LEN:
        raise ValueError(
            f"TTS text too long: {len(cleaned)} characters "
            f"(max allowed is {MAX_TTS_TEXT_LEN})."
        )

    # 9. Validate audio format
    fmt = (audio_format or "").lower().strip()
    if fmt not in ALLOWED_TTS_FORMATS:
        raise ValueError(
            f"Unsupported TTS audio format: {audio_format!r}. "
            f"Allowed formats: {sorted(ALLOWED_TTS_FORMATS)}"
        )

    logger.info(
        "[tts] request_id=%s Starting TTS: text_len=%d voice=%s format=%s",
        request_id,
        len(cleaned),
        voice,
        fmt,
    )

    # NEW: retries for transient failures
    last_exc: Optional[Exception] = None
    for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
        try:
            logger.info("[tts] request_id=%s attempt=%d/%d calling OpenAI TTS", request_id, attempt, TTS_MAX_ATTEMPTS)

            audio_bytes = openai_client.synthesize_speech(
                text=cleaned,
                voice=voice,
                audio_format=fmt,
            )

            # 10. Validate non-empty audio output
            if not audio_bytes:
                logger.error("[tts] request_id=%s TTS returned empty bytes", request_id)
                raise RuntimeError("TARS received an empty audio response from the model.")

            # NEW safeguard: output cap
            if len(audio_bytes) > MAX_TTS_AUDIO_BYTES:
                logger.error(
                    "[tts] request_id=%s TTS output too large (%d bytes > %d cap)",
                    request_id, len(audio_bytes), MAX_TTS_AUDIO_BYTES
                )
                raise RuntimeError("TARS TTS output exceeded safe size limits.")

            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info("[tts] request_id=%s OK latency_ms=%d audio_bytes=%d", request_id, latency_ms, len(audio_bytes))

            # Optional artifact saving (rarely needed but useful)
            if SAVE_AUDIO_ARTIFACTS:
                saved = _save_artifact_bytes("tts_output", fmt, audio_bytes, request_id)
                if saved:
                    logger.info("[tts] request_id=%s saved audio artifact: %s", request_id, saved)

            return audio_bytes

        except ValueError:
            raise
        except Exception as e:
            last_exc = e
            logger.error("[tts] request_id=%s OpenAI TTS exception attempt=%d: %s", request_id, attempt, e)
            if attempt < TTS_MAX_ATTEMPTS:
                _sleep_backoff(attempt, TTS_RETRY_BACKOFF_BASE_SEC)
                continue
            raise RuntimeError("TARS failed to synthesize speech from text.") from e

    raise RuntimeError("TARS failed to synthesize speech from text.") from last_exc


