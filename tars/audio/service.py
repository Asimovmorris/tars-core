# tars/audio/service.py
"""
Audio service layer for TARS Voice 1.0.

High-level helpers around:
- Speech-to-text (STT) using OpenAI Whisper
- Text-to-speech (TTS) using OpenAI TTS

This module wraps lower-level functions in `tars.clients.openai_client`
with additional validation, normalization, retries, logging, and guardrails.

Key additions in this revision (keeps API compatible):
- Optional STT transcript caching (hash-based) to reduce repeated costs on retries/tests.
- Optional WAV post-processing hook for "TARS FX" (off by default).
- Automatic TTS fallback routing: if a chosen voice fails, retry once with default voice.
- Default TTS voice/format can be controlled via environment variables.
- More structured error types while preserving ValueError/RuntimeError outward semantics.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
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
# Error typing (internal). We still raise ValueError/RuntimeError to callers
# unless they choose to catch these explicitly.
# ---------------------------------------------------------------------------

class AudioServiceError(RuntimeError):
    """Base class for audio service failures."""


class AudioInputError(ValueError):
    """Input is invalid (empty/too short/unsupported)."""


class AudioUpstreamError(AudioServiceError):
    """Upstream service call failed (STT/TTS provider errors, network, etc.)."""


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

# -------------------- Diagnostics / artifacts ------------------------------

SAVE_AUDIO_ARTIFACTS = os.getenv("TARS_SAVE_AUDIO_ARTIFACTS", "0").strip() == "1"
AUDIO_DEBUG = os.getenv("TARS_AUDIO_DEBUG", "0").strip() == "1"
ARTIFACT_DIR = os.getenv("TARS_ARTIFACT_DIR", os.path.join("tars", "logs"))

# -------------------- STT preflight safeguards -----------------------------

MIN_AUDIO_BYTES = 8_000          # heuristic
MIN_AUDIO_DURATION_SEC = 0.35

REJECT_MOSTLY_SILENT_WAV = True
SILENCE_RMS_THRESHOLD = 120.0
SILENCE_MAX_SPEECH_RATIO = 0.03

# -------------------- Retry policy -----------------------------------------

STT_MAX_ATTEMPTS = 3
STT_RETRY_BACKOFF_BASE_SEC = 0.6

TTS_MAX_ATTEMPTS = 2
TTS_RETRY_BACKOFF_BASE_SEC = 0.5

# -------------------- TTS output cap ---------------------------------------

MAX_TTS_AUDIO_BYTES = 12 * 1024 * 1024  # 12 MiB

# -------------------- NEW: Defaults (voice/format) -------------------------

# If caller passes audio_format="mp3"/"wav", that wins.
# If caller passes audio_format=None/"", use env default, else "mp3".
DEFAULT_TTS_FORMAT = os.getenv("TARS_TTS_DEFAULT_FORMAT", "mp3").strip().lower() or "mp3"

# If caller passes voice="...", that wins.
# If caller passes voice=None, use env default (optional); else None (provider default).
DEFAULT_TTS_VOICE = os.getenv("TARS_TTS_DEFAULT_VOICE", "").strip() or None

# -------------------- NEW: Optional STT caching ----------------------------

STT_CACHE_ENABLED = os.getenv("TARS_STT_CACHE", "0").strip() == "1"
STT_CACHE_DIR = os.getenv("TARS_STT_CACHE_DIR", os.path.join("tars", "logs", "stt_cache"))
STT_CACHE_MAX_AGE_SEC = int(os.getenv("TARS_STT_CACHE_MAX_AGE_SEC", "604800"))  # 7 days

# -------------------- NEW: Optional "TARS FX" hook for WAV -----------------

TTS_ENABLE_FX = os.getenv("TARS_TTS_ENABLE_FX", "0").strip() == "1"
TTS_FX_PRESET = os.getenv("TARS_TTS_FX_PRESET", "medium").strip().lower()  # low|medium|high


# ---------------------------------------------------------------------------
# Internal helpers (MIME normalization, WAV inspection, artifacts, retries)
# ---------------------------------------------------------------------------

def _normalize_mime_type(mime_type: Optional[str]) -> Optional[str]:
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
        pass


def _artifact_path(prefix: str, ext: str, request_id: str) -> str:
    _safe_makedirs(ARTIFACT_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}_{request_id}.{ext}"
    return os.path.join(ARTIFACT_DIR, fname)


def _save_artifact_bytes(prefix: str, ext: str, data: bytes, request_id: str) -> Optional[str]:
    try:
        path = _artifact_path(prefix, ext, data=prefix and ext and request_id)  # type: ignore
    except TypeError:
        # compatibility if a user has an older version of this file imported elsewhere
        path = _artifact_path(prefix, ext, request_id)

    try:
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        logger.warning("Failed to save artifact %s.%s: %s", prefix, ext, e)
        return None


def _save_artifact_text(prefix: str, ext: str, text: str, request_id: str) -> Optional[str]:
    try:
        path = _artifact_path(prefix, ext, request_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path
    except Exception as e:
        logger.warning("Failed to save artifact %s.%s: %s", prefix, ext, e)
        return None


def _try_parse_wav_info(audio_bytes: bytes) -> Tuple[Optional[float], Optional[int], Optional[int]]:
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
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                return None, None
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
            if not raw:
                return 0.0, 0.0

            import numpy as np  # local import
            x = np.frombuffer(raw, dtype="<i2").astype("float32")
            if x.size == 0:
                return 0.0, 0.0

            rms = float(np.sqrt(np.mean(x * x)))
            thr = float(SILENCE_RMS_THRESHOLD * 3.0)
            ratio = float(np.mean(np.abs(x) > thr))
            return rms, ratio
    except Exception:
        return None, None


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _sleep_backoff(attempt: int, base: float) -> None:
    delay = base * (2 ** (attempt - 1))
    time.sleep(delay)


# -------------------- NEW: STT cache helpers -------------------------------

def _stt_cache_path(key16: str) -> str:
    _safe_makedirs(STT_CACHE_DIR)
    return os.path.join(STT_CACHE_DIR, f"{key16}.txt")


def _stt_cache_get(key16: str) -> Optional[str]:
    if not STT_CACHE_ENABLED:
        return None
    path = _stt_cache_path(key16)
    try:
        st = os.stat(path)
        age = time.time() - st.st_mtime
        if age > STT_CACHE_MAX_AGE_SEC:
            return None
        txt = open(path, "r", encoding="utf-8").read()
        txt = (txt or "").strip()
        return txt or None
    except Exception:
        return None


def _stt_cache_put(key16: str, transcript: str) -> None:
    if not STT_CACHE_ENABLED:
        return
    path = _stt_cache_path(key16)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(transcript)
    except Exception:
        pass


# -------------------- NEW: optional WAV "TARS FX" hook ---------------------

def _apply_tars_fx_wav(audio_bytes: bytes, preset: str) -> bytes:
    """
    Best-effort: apply a lightweight "radio/robot" style effect to PCM16 WAV.
    Off by default. If anything fails, return original audio_bytes.

    This is intentionally conservative (no extra heavy deps).
    """
    if not TTS_ENABLE_FX:
        return audio_bytes
    if not audio_bytes or audio_bytes[:4] != b"RIFF":
        return audio_bytes

    try:
        import numpy as np  # local import

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sw != 2 or sr <= 0 or not raw:
            return audio_bytes

        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)

        # preset mapping (subtle; we don't want distortion artifacts)
        preset = (preset or "medium").lower()
        if preset == "low":
            drive = 1.15
            noise = 0.002
            hp = 120.0
        elif preset == "high":
            drive = 1.35
            noise = 0.006
            hp = 180.0
        else:  # medium
            drive = 1.25
            noise = 0.004
            hp = 150.0

        # 1) High-pass (one-pole) to reduce boom
        rc = 1.0 / (2.0 * 3.14159 * hp)
        dt = 1.0 / float(sr)
        alpha = rc / (rc + dt)
        y = np.zeros_like(x)
        y_prev = 0.0
        x_prev = 0.0
        for i in range(x.shape[0]):
            y_i = alpha * (y_prev + x[i] - x_prev)
            y[i] = y_i
            y_prev = y_i
            x_prev = x[i]

        # 2) Mild saturation (tanh) + tiny noise (robot texture)
        y = np.tanh(y * drive)
        y = y + (np.random.randn(y.shape[0]).astype(np.float32) * noise)

        # 3) Final limit
        y = np.clip(y, -0.95, 0.95)

        out_i16 = (y * 32767.0).astype("<i2").tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(out_i16)

        return buf.getvalue()

    except Exception as e:
        logger.warning("[tts_fx] Failed to apply FX preset=%s: %s", preset, e)
        return audio_bytes


# ---------------------------------------------------------------------------
# STT helper with safeguards
# ---------------------------------------------------------------------------

def stt_with_safeguards(
    audio_bytes: bytes,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    if not audio_bytes:
        raise AudioInputError("No audio data provided for transcription.")

    if len(audio_bytes) < MIN_AUDIO_BYTES:
        logger.warning("[stt] request_id=%s audio too small (%d bytes)", request_id, len(audio_bytes))
        raise AudioInputError(
            f"Audio payload too small: {len(audio_bytes)} bytes. "
            "This is likely an empty/silent capture or an encoding issue."
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise AudioInputError(
            f"Audio payload too large: {len(audio_bytes)} bytes "
            f"(max allowed is {MAX_AUDIO_BYTES} bytes)."
        )

    norm_mime = _normalize_mime_type(mime_type)

    if norm_mime is not None and norm_mime not in ALLOWED_AUDIO_MIME_TYPES:
        logger.warning("[stt] request_id=%s Received non-standard MIME type: %s", request_id, norm_mime)

    lang_to_use = language if language is not None else DEFAULT_STT_LANGUAGE

    # Hash key for caching and correlation (privacy-safe)
    key16 = _hash_bytes(audio_bytes)

    # NEW: cache short-circuit
    cached = _stt_cache_get(key16)
    if cached:
        logger.info("[stt] request_id=%s cache_hit hash=%s", request_id, key16)
        return cached

    duration_sec, sr, ch = _try_parse_wav_info(audio_bytes)
    if duration_sec is not None:
        logger.info(
            "[stt] request_id=%s WAV duration=%.2fs sr=%s ch=%s bytes=%d mime=%s lang=%s hash=%s",
            request_id, duration_sec, sr, ch, len(audio_bytes), norm_mime, lang_to_use, key16
        )

        if duration_sec < MIN_AUDIO_DURATION_SEC:
            raise AudioInputError(
                f"Audio duration too short ({duration_sec:.2f}s). "
                "Speak a bit longer after the wake word."
            )

        if REJECT_MOSTLY_SILENT_WAV:
            rms, ratio = _wav_pcm16_rms_and_speech_ratio(audio_bytes)
            if rms is not None and ratio is not None:
                logger.info("[stt] request_id=%s rms=%.1f speech_ratio=%.4f", request_id, rms, ratio)
                if rms < SILENCE_RMS_THRESHOLD and ratio < SILENCE_MAX_SPEECH_RATIO:
                    raise AudioInputError(
                        "Audio appears to be mostly silence. "
                        "Mic may be too quiet or your capture threshold is too high."
                    )
    else:
        logger.info(
            "[stt] request_id=%s Non-WAV bytes=%d mime=%s lang=%s hash=%s",
            request_id, len(audio_bytes), norm_mime, lang_to_use, key16
        )

    if SAVE_AUDIO_ARTIFACTS:
        ext = "wav" if (duration_sec is not None or norm_mime in {"audio/wav", "audio/x-wav"}) else "bin"
        saved = _save_artifact_bytes("stt_input", ext, audio_bytes, request_id)
        if saved:
            logger.info("[stt] request_id=%s saved artifact=%s", request_id, saved)

    if AUDIO_DEBUG:
        meta: Dict[str, Any] = {
            "request_id": request_id,
            "bytes": len(audio_bytes),
            "mime_type": norm_mime,
            "language": lang_to_use,
            "wav_duration_sec": duration_sec,
            "wav_sample_rate": sr,
            "wav_channels": ch,
            "hash16": key16,
        }
        _save_artifact_text("stt_meta", "json", str(meta), request_id)

    last_exc: Optional[Exception] = None
    for attempt in range(1, STT_MAX_ATTEMPTS + 1):
        try:
            logger.info("[stt] request_id=%s attempt=%d/%d calling STT", request_id, attempt, STT_MAX_ATTEMPTS)

            raw_text = openai_client.transcribe_audio(
                audio_bytes=audio_bytes,
                mime_type=norm_mime,
                language=lang_to_use,
            )

            transcript = (raw_text or "").strip()
            if not transcript:
                logger.error("[stt] request_id=%s empty transcript after normalization", request_id)
                raise AudioUpstreamError("TARS received an empty transcription from the model.")

            # log snippet (hardened)
            snippet = transcript[:200]
            if len(transcript) > 200:
                snippet += "..."
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info("[stt] request_id=%s OK latency_ms=%d transcript=%r", request_id, latency_ms, snippet)

            # NEW: cache store
            _stt_cache_put(key16, transcript)

            return transcript

        except AudioInputError:
            raise
        except AudioUpstreamError as e:
            last_exc = e
            logger.error("[stt] request_id=%s upstream error attempt=%d: %s", request_id, attempt, e)
            if attempt < STT_MAX_ATTEMPTS:
                _sleep_backoff(attempt, STT_RETRY_BACKOFF_BASE_SEC)
                continue
            raise RuntimeError(str(e)) from e
        except Exception as e:
            last_exc = e
            logger.error("[stt] request_id=%s exception attempt=%d: %s", request_id, attempt, e)
            if attempt < STT_MAX_ATTEMPTS:
                _sleep_backoff(attempt, STT_RETRY_BACKOFF_BASE_SEC)
                continue
            raise RuntimeError("TARS failed to transcribe audio input.") from e

    raise RuntimeError("TARS failed to transcribe audio input.") from last_exc


# ---------------------------------------------------------------------------
# TTS helper with safeguards
# ---------------------------------------------------------------------------

def tts_with_safeguards(
    text: str,
    voice: Optional[str] = None,
    audio_format: str = "mp3",
) -> bytes:
    request_id = str(uuid.uuid4())
    t0 = time.monotonic()

    cleaned = (text or "").strip()
    if not cleaned:
        raise AudioInputError("Cannot synthesize speech from empty or whitespace-only text.")

    if len(cleaned) > MAX_TTS_TEXT_LEN:
        raise AudioInputError(
            f"TTS text too long: {len(cleaned)} characters (max allowed is {MAX_TTS_TEXT_LEN})."
        )

    # NEW: resolve defaults safely
    fmt_raw = (audio_format or "").strip().lower()
    fmt = fmt_raw if fmt_raw else DEFAULT_TTS_FORMAT
    if fmt not in ALLOWED_TTS_FORMATS:
        raise AudioInputError(
            f"Unsupported TTS audio format: {audio_format!r}. Allowed: {sorted(ALLOWED_TTS_FORMATS)}"
        )

    chosen_voice = voice if voice is not None else DEFAULT_TTS_VOICE

    logger.info(
        "[tts] request_id=%s start text_len=%d voice=%s format=%s fx=%s preset=%s",
        request_id, len(cleaned), chosen_voice, fmt, TTS_ENABLE_FX, TTS_FX_PRESET
    )

    last_exc: Optional[Exception] = None

    def _do_tts(v: Optional[str]) -> bytes:
        return openai_client.synthesize_speech(text=cleaned, voice=v, audio_format=fmt)

    # We allow a “voice fallback” pass without increasing global attempts too much.
    # Pass 1: chosen_voice
    # Pass 2 (optional): default voice (None) if chosen_voice fails
    voice_passes = [chosen_voice]
    if chosen_voice is not None:
        voice_passes.append(None)

    for vpass_idx, vpass_voice in enumerate(voice_passes, start=1):
        for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
            try:
                logger.info(
                    "[tts] request_id=%s voice_pass=%d/%d attempt=%d/%d calling TTS voice=%s",
                    request_id, vpass_idx, len(voice_passes), attempt, TTS_MAX_ATTEMPTS, vpass_voice
                )

                audio_bytes = _do_tts(vpass_voice)

                if not audio_bytes:
                    logger.error("[tts] request_id=%s empty audio bytes", request_id)
                    raise AudioUpstreamError("TARS received an empty audio response from the model.")

                if len(audio_bytes) > MAX_TTS_AUDIO_BYTES:
                    logger.error(
                        "[tts] request_id=%s output too large (%d > %d)",
                        request_id, len(audio_bytes), MAX_TTS_AUDIO_BYTES
                    )
                    raise AudioUpstreamError("TARS TTS output exceeded safe size limits.")

                # NEW: optional FX hook (WAV only)
                if fmt == "wav" and TTS_ENABLE_FX:
                    audio_bytes = _apply_tars_fx_wav(audio_bytes, TTS_FX_PRESET)

                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.info("[tts] request_id=%s OK latency_ms=%d bytes=%d fmt=%s", request_id, latency_ms, len(audio_bytes), fmt)

                if SAVE_AUDIO_ARTIFACTS:
                    saved = _save_artifact_bytes("tts_output", fmt, audio_bytes, request_id)
                    if saved:
                        logger.info("[tts] request_id=%s saved artifact=%s", request_id, saved)

                return audio_bytes

            except AudioInputError:
                raise
            except AudioUpstreamError as e:
                last_exc = e
                logger.error("[tts] request_id=%s upstream error attempt=%d: %s", request_id, attempt, e)
                if attempt < TTS_MAX_ATTEMPTS:
                    _sleep_backoff(attempt, TTS_RETRY_BACKOFF_BASE_SEC)
                    continue
                break  # move to next voice pass (fallback) if available
            except Exception as e:
                last_exc = e
                logger.error("[tts] request_id=%s exception attempt=%d: %s", request_id, attempt, e)
                if attempt < TTS_MAX_ATTEMPTS:
                    _sleep_backoff(attempt, TTS_RETRY_BACKOFF_BASE_SEC)
                    continue
                break  # move to next voice pass (fallback) if available

        # If we get here, attempts exhausted for this voice pass; try next pass (if any)
        if vpass_voice is not None:
            logger.warning("[tts] request_id=%s voice=%s failed; retrying with default voice", request_id, vpass_voice)

    raise RuntimeError("TARS failed to synthesize speech from text.") from last_exc



