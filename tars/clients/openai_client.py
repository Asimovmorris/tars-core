# tars/clients/openai_client.py

import io
import os
from typing import List, Dict, Optional

import openai
import requests

from tars.config.settings import load_settings, TARS_PROMPT_PATH
from tars.utils.logging import get_logger

logger = get_logger(__name__)
_settings = load_settings()

# ---------------------------------------------------------------------------
# OpenAI base configuration (legacy SDK 0.28.x)
# ---------------------------------------------------------------------------

if not _settings.openai_api_key:
    # Hard fail early: nothing will work without this
    raise RuntimeError("OPENAI_API_KEY is not set in settings/.env")

openai.api_key = _settings.openai_api_key

# Prefer settings.openai_api_base, then env var, then default.
_openai_base_url = (
    getattr(_settings, "openai_api_base", None)
    or os.getenv("OPENAI_BASE_URL")
    or "https://api.openai.com"
).rstrip("/")

# If you want the SDK to also use a custom base (e.g. proxy), set api_base.
openai.api_base = _openai_base_url


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------

def load_tars_system_prompt() -> str:
    """
    Load the TARS system prompt from the configured path.
    Executed once at import-time and cached in TARS_SYSTEM_PROMPT.
    """
    try:
        with open(TARS_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load TARS system prompt from {TARS_PROMPT_PATH}: {e}")
        raise

    if not prompt:
        logger.error("TARS system prompt is empty after loading.")
        raise RuntimeError("TARS system prompt is empty.")

    return prompt


TARS_SYSTEM_PROMPT = load_tars_system_prompt()


# ---------------------------------------------------------------------------
# Internal helpers to read config with safeguards
# ---------------------------------------------------------------------------

def _get_chat_model() -> str:
    model = (_settings.openai_model or "").strip()
    if not model:
        logger.warning(
            "openai_model is empty in settings; falling back to 'gpt-4.1-mini'."
        )
        return "gpt-4.1-mini"
    return model


def _get_stt_model() -> str:
    # For the legacy SDK, Whisper models like "whisper-1" are valid.
    model = getattr(_settings, "openai_stt_model", "") or ""
    model = model.strip()
    if not model:
        logger.warning(
            "openai_stt_model is empty in settings; falling back to 'whisper-1'."
        )
        return "whisper-1"
    return model


def _get_tts_model() -> str:
    # This is used by the raw HTTP TTS call to /v1/audio/speech.
    model = getattr(_settings, "openai_tts_model", "") or ""
    model = model.strip()
    if not model:
        logger.warning(
            "openai_tts_model is empty in settings; falling back to 'gpt-4o-mini-tts'."
        )
        return "gpt-4o-mini-tts"
    return model


def _get_tts_voice() -> str:
    voice = getattr(_settings, "openai_tts_voice", "") or ""
    voice = voice.strip()
    if not voice:
        logger.warning(
            "openai_tts_voice is empty in settings; falling back to 'alloy'."
        )
        return "alloy"
    return voice


def _get_tts_audio_format() -> str:
    fmt = getattr(_settings, "openai_tts_audio_format", "") or ""
    fmt = fmt.strip().lower()
    if not fmt:
        logger.warning(
            "openai_tts_audio_format is empty in settings; falling back to 'mp3'."
        )
        fmt = "mp3"
    # audio.service will enforce supported formats; here we just normalize.
    return fmt


# ---------------------------------------------------------------------------
# Chat (text) API - using legacy SDK style (0.28.x)
# ---------------------------------------------------------------------------

def chat_with_tars(messages: List[Dict[str, str]]) -> str:
    """
    Call the OpenAI chat model with the TARS system prompt automatically prepended.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        Sequence of chat messages, each a dict with:
            {"role": "user" | "assistant" | "system", "content": "..."}

        You should NOT include the main TARS system prompt yourself; it is
        injected automatically as the first "system" message.

    Returns
    -------
    str
        The assistant's reply text.

    Raises
    ------
    RuntimeError
        If the OpenAI API call fails or returns an unusable response.
    """
    full_messages = [{"role": "system", "content": TARS_SYSTEM_PROMPT}] + messages
    model_name = _get_chat_model()

    try:
        # Legacy SDK: ChatCompletion.create(...).
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=full_messages,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"OpenAI chat API call failed (model={model_name!r}): {e}")
        raise RuntimeError("TARS encountered an error contacting the model.") from e

    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except Exception as e:
        logger.error(f"Unexpected response format from OpenAI: {response!r}")
        raise RuntimeError(
            "TARS received an unexpected response format from the model."
        ) from e

    if not content or not content.strip():
        logger.error("Empty content received from OpenAI response.")
        raise RuntimeError("TARS received an empty response from the model.")

    return content.strip()


# ---------------------------------------------------------------------------
# Audio helpers (STT and TTS) for TARS Voice 1.0
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe audio to text using OpenAI's speech-to-text (Whisper-based) API.

    Uses the legacy SDK method: openai.Audio.transcribe(...).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file contents (e.g. WAV/MP3/OGG).
    mime_type : Optional[str]
        MIME type hint, e.g. "audio/wav" or "audio/mpeg". Currently unused.
    language : Optional[str]
        Optional language code (e.g. "en"). If None, the API auto-detects.

    Returns
    -------
    str
        The transcribed text.

    Raises
    ------
    RuntimeError
        If the transcription request fails or returns an unusable response.
    """
    audio_file = io.BytesIO(audio_bytes)
    # The legacy SDK relies on the file having a name with extension.
    audio_file.name = "input_audio.wav"

    model_name = _get_stt_model()

    try:
        # Legacy SDK: Audio.transcribe(...).
        resp = openai.Audio.transcribe(
            model=model_name,
            file=audio_file,
            language=language,
        )
    except Exception as e:
        logger.error(f"OpenAI audio transcription failed (model={model_name!r}): {e}")
        raise RuntimeError("TARS failed to transcribe the audio input.") from e

    try:
        # Legacy API returns a dict-like object: {'text': '...'}.
        text = resp["text"] if isinstance(resp, dict) else getattr(resp, "text", "")
    except Exception as e:
        logger.error(f"Unexpected transcription response format: {resp!r}")
        raise RuntimeError(
            "TARS received an unexpected transcription response format."
        ) from e

    if not text or not str(text).strip():
        logger.error("Empty transcription text received from OpenAI.")
        raise RuntimeError("TARS received an empty transcription from the model.")

    return str(text).strip()


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    audio_format: Optional[str] = None,
) -> bytes:
    """
    Synthesize speech from text using OpenAI's text-to-speech HTTP API.

    NOTE:
        - We use a direct HTTP call to /v1/audio/speech instead of the Python SDK,
          because the newer TTS helpers are not available in openai==0.28.x.
        - This keeps us compatible with Termux while still using modern TTS models.

    Parameters
    ----------
    text : str
        The text to speak.
    voice : Optional[str]
        Optional voice preset name. If None, falls back to config default.
    audio_format : Optional[str]
        Output audio format, e.g. "mp3", "wav".
        If None, falls back to config default.

    Returns
    -------
    bytes
        The raw audio data.

    Raises
    ------
    RuntimeError
        If the TTS request fails or returns an unusable response.
    ValueError
        If `text` is empty.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Cannot synthesize speech from empty text.")

    model_name = _get_tts_model()
    voice_name = voice or _get_tts_voice()
    fmt = (audio_format or _get_tts_audio_format()).lower()

    url = f"{_openai_base_url}/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {_settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "voice": voice_name,
        "input": cleaned,
        "response_format": fmt,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
    except Exception as e:
        logger.error(
            "OpenAI text-to-speech HTTP request failed "
            "(url=%r, model=%r, voice=%r, format=%r): %s",
            url,
            model_name,
            voice_name,
            fmt,
            e,
        )
        raise RuntimeError("TARS failed to synthesize speech from text.") from e

    if resp.status_code != 200:
        logger.error(
            "OpenAI TTS API returned non-200 status: %s, body=%r",
            resp.status_code,
            resp.text,
        )
        raise RuntimeError(
            f"TARS TTS request failed with status {resp.status_code}: {resp.text}"
        )

    audio_bytes = resp.content or b""

    if not audio_bytes:
        logger.error("Empty audio bytes received from OpenAI TTS.")
        raise RuntimeError("TARS received an empty audio response from the model.")

    return audio_bytes

