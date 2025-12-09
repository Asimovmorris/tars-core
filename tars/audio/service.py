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

These helpers are designed to be called from the FastAPI `/chat_audio`
endpoint or any other audio pipeline code.
"""

from __future__ import annotations

from typing import Optional

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
        If the input is clearly invalid (e.g. empty or oversized).
    RuntimeError
        If transcription fails or returns unusable output.
    """
    # 1. Reject empty or None audio
    if not audio_bytes:
        raise ValueError("No audio data provided for transcription.")

    # 2. Enforce max audio size
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio payload too large: {len(audio_bytes)} bytes "
            f"(max allowed is {MAX_AUDIO_BYTES} bytes)."
        )

    # 3. MIME validation (non-fatal; we just log a warning if it's odd)
    if mime_type is not None and mime_type not in ALLOWED_AUDIO_MIME_TYPES:
        logger.warning(
            "Received non-standard or unsupported MIME type for audio: %s",
            mime_type,
        )

    # 5. Language fallback
    lang_to_use = language if language is not None else DEFAULT_STT_LANGUAGE

    logger.info(
        "Starting STT: size=%d bytes, mime=%s, lang=%s",
        len(audio_bytes),
        mime_type,
        lang_to_use,
    )

    try:
        raw_text = openai_client.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language=lang_to_use,
        )
    except Exception as e:
        # 10 (for STT branch): clear logging and normalized error
        logger.error("STT request to OpenAI failed: %s", e)
        raise RuntimeError("TARS failed to transcribe audio input.") from e

    # 4. Normalize and validate transcript
    transcript = (raw_text or "").strip()
    if not transcript:
        logger.error("STT produced empty transcript after normalization.")
        raise RuntimeError("TARS received an empty transcription from the model.")

    # 6. Log truncated transcript for observability
    log_snippet = transcript[:200] + ("..." if len(transcript) > 200 else "")
    logger.info("STT completed. Transcript (truncated): %r", log_snippet)

    return transcript


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
    fmt = (audio_format or "").lower()
    if fmt not in ALLOWED_TTS_FORMATS:
        raise ValueError(
            f"Unsupported TTS audio format: {audio_format!r}. "
            f"Allowed formats: {sorted(ALLOWED_TTS_FORMATS)}"
        )

    logger.info(
        "Starting TTS: text_len=%d, voice=%s, format=%s",
        len(cleaned),
        voice,
        fmt,
    )

    try:
        audio_bytes = openai_client.synthesize_speech(
            text=cleaned,
            voice=voice,
            audio_format=fmt,
        )
    except Exception as e:
        logger.error("TTS request to OpenAI failed: %s", e)
        raise RuntimeError("TARS failed to synthesize speech from text.") from e

    # 10. Validate non-empty audio output
    if not audio_bytes:
        logger.error("TTS returned empty audio bytes from OpenAI.")
        raise RuntimeError("TARS received an empty audio response from the model.")

    logger.info("TTS completed successfully. Audio size: %d bytes", len(audio_bytes))

    return audio_bytes

