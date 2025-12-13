# tars/clients/openai_client.py
#
# IMPORTANT PROJECT RULES (enforced here by design):
# - Do NOT reduce the length of this file.
# - Keep the architecture stable: this module is the single integration layer for OpenAI.
# - Correctness first: STT + Chat must work on PC (openai>=1.x) and Termux (possibly openai==0.28.x).
# - Add safeguards + observability without leaking secrets.

import io
import os
import time
import random
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Any

import openai
import requests

from tars.config.settings import load_settings, TARS_PROMPT_PATH
from tars.utils.logging import get_logger

logger = get_logger(__name__)
_settings = load_settings()

# ---------------------------------------------------------------------------
# OpenAI base configuration (legacy SDK + new SDK compatible)
# ---------------------------------------------------------------------------

if not _settings.openai_api_key:
    # Hard fail early: nothing will work without this
    raise RuntimeError("OPENAI_API_KEY is not set in settings/.env")

openai.api_key = _settings.openai_api_key

# Prefer settings.openai_api_base, then env var, then default.
_openai_base_url_raw = (
    getattr(_settings, "openai_api_base", None)
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE")
    or "https://api.openai.com"
)

# ---------------------------------------------------------------------------
# NEW TECH ADDITION #1: Robust base-url normalization (fixes STT nginx 404)
# ---------------------------------------------------------------------------

def _strip_outer_quotes(s: str) -> str:
    """
    Safeguard: users sometimes put OPENAI_BASE_URL="https://..." including quotes.
    This removes a single pair of matching outer quotes.
    """
    s2 = (s or "").strip()
    if len(s2) >= 2 and ((s2[0] == s2[-1]) and s2[0] in ("'", '"')):
        return s2[1:-1].strip()
    return s2

def _normalize_openai_api_base(raw: Optional[str]) -> str:
    """
    Ensures api_base ends with /v1.

    Why this exists:
      - Legacy openai==0.28.x needs /v1 when api_base is overridden.
      - Your ground-truth logs showed nginx HTML 404 for STT before this fix.
    """
    base = _strip_outer_quotes((raw or "https://api.openai.com").strip())

    # Safeguard A: detect obviously invalid base URL early
    if not (base.startswith("http://") or base.startswith("https://")):
        raise RuntimeError(f"OPENAI_BASE_URL is invalid (missing scheme): {base!r}")

    # Remove trailing slashes
    while base.endswith("/"):
        base = base[:-1]

    # If user accidentally passed full endpoint like .../v1/audio/transcriptions, trim to /v1
    if "/v1/" in base:
        base = base.split("/v1/")[0] + "/v1"
        return base

    # If already ends with /v1, done
    if base.endswith("/v1"):
        return base

    # Otherwise append /v1
    return base + "/v1"

def _join_api(base_v1: str, path_no_leading_slash: str) -> str:
    """
    Join helper that assumes base_v1 ends with /v1 and the path has no leading slash.
    Safeguard: prevents accidental double /v1/v1 or missing slashes.
    """
    b = (base_v1 or "").rstrip("/")
    p = (path_no_leading_slash or "").lstrip("/")
    return f"{b}/{p}"

_OPENAI_API_BASE = _normalize_openai_api_base(_openai_base_url_raw)

# Legacy SDK uses openai.api_base
openai.api_base = _OPENAI_API_BASE

# NEW TECH ADDITION #2: one-time diagnostics (no secrets)
try:
    logger.info("OpenAI api_base resolved to: %s", openai.api_base)
except Exception:
    pass

# ---------------------------------------------------------------------------
# NEW TECH ADDITION #3: Shared HTTP session for TTS + better connection reuse
# ---------------------------------------------------------------------------

_HTTP_SESSION = requests.Session()

def _configure_http_session() -> None:
    try:
        _HTTP_SESSION.headers.update({
            "User-Agent": "tars/voice-client (requests)",
        })
    except Exception:
        pass

_configure_http_session()

# ---------------------------
# HTTP/session config, retries, and diagnostics
# ---------------------------

_STT_MAX_ATTEMPTS = int(getattr(_settings, "openai_stt_max_attempts", 3) or 3)
_TTS_MAX_ATTEMPTS = int(getattr(_settings, "openai_tts_max_attempts", 2) or 2)

# NEW TECH ADDITION #4: Chat retry policy (bounded)
_CHAT_MAX_ATTEMPTS = int(getattr(_settings, "openai_chat_max_attempts", 2) or 2)

_OPENAI_HTTP_TIMEOUT_SEC = float(getattr(_settings, "openai_http_timeout_sec", 60) or 60)

# Safeguard B: raw base must have scheme
_openai_base_url_raw_s = _strip_outer_quotes(str(_openai_base_url_raw or "")).strip()
if not (_openai_base_url_raw_s.startswith("http://") or _openai_base_url_raw_s.startswith("https://")):
    raise RuntimeError(f"OPENAI_BASE_URL is invalid (missing scheme): {_openai_base_url_raw_s!r}")

# ---------------------------------------------------------------------------
# NEW TECH ADDITION #5: request_id + payload fingerprints for logs
# ---------------------------------------------------------------------------

def _mk_req_id(prefix: str = "req") -> str:
    return f"{prefix}_{int(time.time()*1000)}_{random.randint(1000, 9999)}"

def _sha256_hex(b: bytes, max_bytes: int = 1024 * 64) -> str:
    if not b:
        return "sha256(empty)"
    h = hashlib.sha256()
    h.update(b[:max_bytes])
    return h.hexdigest()

def _safe_host_from_url(url: str) -> str:
    try:
        u = url.strip()
        u = u.replace("https://", "").replace("http://", "")
        host = u.split("/")[0]
        return host
    except Exception:
        return "unknown-host"

def _is_transient_status(code: int) -> bool:
    return code in {408, 409, 425, 429, 500, 502, 503, 504}

def _sleep_backoff(attempt_idx: int) -> None:
    base = 0.4 * (2 ** max(0, attempt_idx - 1))
    jitter = random.uniform(0.0, 0.25)
    time.sleep(min(3.0, base + jitter))

# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

def _classify_openai_error(e: Exception) -> str:
    name = e.__class__.__name__
    msg = (str(e) or "").lower()

    # Your earlier blocker signature:
    if "404" in msg and ("nginx" in msg or "<html" in msg):
        return "stt_404_nginx_base_url"

    # New blocker signature (PC openai>=1.x calling legacy ChatCompletion)
    if "you tried to access openai.chatcompletion" in msg and "no longer supported" in msg:
        return "chat_legacy_api_used_on_v1_sdk"

    if "notfound" in name.lower() or "404" in msg:
        return "openai_404_not_found"

    if "401" in msg or "incorrect api key" in msg or "authentication" in msg:
        return "openai_auth"

    if "429" in msg or "rate limit" in msg:
        return "openai_rate_limit"

    if "timeout" in msg or "timed out" in msg:
        return "openai_timeout"

    if "502" in msg or "bad gateway" in msg:
        return "openai_502"

    if "503" in msg or "service unavailable" in msg:
        return "openai_503"

    if "connection" in msg or "dns" in msg:
        return "openai_network"

    return "openai_unknown"

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
        logger.warning("openai_model is empty in settings; falling back to 'gpt-4.1-mini'.")
        return "gpt-4.1-mini"
    return model

def _get_stt_model() -> str:
    model = getattr(_settings, "openai_stt_model", "") or ""
    model = model.strip()
    if not model:
        logger.warning("openai_stt_model is empty in settings; falling back to 'whisper-1'.")
        return "whisper-1"
    return model

def _get_tts_model() -> str:
    model = getattr(_settings, "openai_tts_model", "") or ""
    model = model.strip()
    if not model:
        logger.warning("openai_tts_model is empty in settings; falling back to 'gpt-4o-mini-tts'.")
        return "gpt-4o-mini-tts"
    return model

def _get_tts_voice() -> str:
    voice = getattr(_settings, "openai_tts_voice", "") or ""
    voice = voice.strip()
    if not voice:
        logger.warning("openai_tts_voice is empty in settings; falling back to 'alloy'.")
        return "alloy"
    return voice

def _get_tts_audio_format() -> str:
    fmt = getattr(_settings, "openai_tts_audio_format", "") or ""
    fmt = fmt.strip().lower()
    if not fmt:
        logger.warning("openai_tts_audio_format is empty in settings; falling back to 'mp3'.")
        fmt = "mp3"
    return fmt

# ---------------------------
# Audio extension guessing + filename correctness
# ---------------------------

def _guess_ext_from_mime(mime_type: Optional[str]) -> str:
    mt = (mime_type or "").strip().lower()
    if mt in ("audio/wav", "audio/x-wav"):
        return ".wav"
    if mt in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if mt == "audio/ogg":
        return ".ogg"
    if mt == "audio/webm":
        return ".webm"
    return ".wav"

def _safe_audio_filename(mime_type: Optional[str]) -> str:
    return "input_audio" + _guess_ext_from_mime(mime_type)

# ---------------------------------------------------------------------------
# NEW: OpenAI v1 client builder for chat+stt (PC) while preserving legacy fallback
# ---------------------------------------------------------------------------

def _try_make_v1_client():
    """
    Returns (client, True) if openai>=1.x OpenAI client is available, else (None, False).
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=_settings.openai_api_key, base_url=_OPENAI_API_BASE)
        return client, True
    except Exception:
        return None, False

# ---------------------------------------------------------------------------
# Chat (text) API - supports BOTH new SDK and legacy fallback
# ---------------------------------------------------------------------------

def chat_with_tars(messages: List[Dict[str, str]]) -> str:
    """
    Call the OpenAI chat model with the TARS system prompt automatically prepended.

    This MUST work on:
      - PC with openai==1.90.0 (new SDK path)
      - Termux possibly with openai==0.28.x (legacy fallback)

    Returns assistant reply text.
    """
    req_id = _mk_req_id("chat")
    model_name = _get_chat_model()

    # Safeguard 1: messages must be list of dicts with role/content
    if not isinstance(messages, list):
        raise RuntimeError("chat_with_tars: messages must be a list.")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise RuntimeError(f"chat_with_tars: invalid message at index {i}: {m!r}")

    full_messages = [{"role": "system", "content": TARS_SYSTEM_PROMPT}] + messages

    # Prefer v1 client if available
    client, use_v1 = _try_make_v1_client()

    last_err: Optional[Exception] = None

    logger.info("[chat] req_id=%s start model=%s base=%s new_sdk=%s msg_count=%d",
                req_id, model_name, _OPENAI_API_BASE, use_v1, len(full_messages))

    for attempt in range(1, max(1, _CHAT_MAX_ATTEMPTS) + 1):
        t0 = time.monotonic()
        try:
            if use_v1 and client is not None:
                # NEW SDK
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=full_messages,
                    temperature=0.4,
                )
                # v1 response shape: resp.choices[0].message.content
                content = ""
                try:
                    content = (resp.choices[0].message.content or "")
                except Exception:
                    # fall back to safe repr
                    content = ""
            else:
                # LEGACY SDK fallback (only works if openai==0.28.x installed)
                resp = openai.ChatCompletion.create(
                    model=model_name,
                    messages=full_messages,
                    temperature=0.4,
                )
                content = resp["choices"][0]["message"]["content"]

            dt_ms = int((time.monotonic() - t0) * 1000)
            content = (content or "").strip()
            if not content:
                raise RuntimeError("TARS received an empty response from the model.")

            snippet = content[:240] + ("..." if len(content) > 240 else "")
            logger.info("[chat] req_id=%s OK attempt=%d latency_ms=%d model=%s reply=%r",
                        req_id, attempt, dt_ms, model_name, snippet)
            return content

        except Exception as e:
            last_err = e
            dt_ms = int((time.monotonic() - t0) * 1000)
            code = _classify_openai_error(e)

            # Safeguard 2: if we somehow hit the legacy API on v1 SDK, don't retry
            if code == "chat_legacy_api_used_on_v1_sdk":
                logger.error("[chat] req_id=%s FAIL legacy_api_on_v1 attempt=%d latency_ms=%d err=%s",
                             req_id, attempt, dt_ms, str(e))
                break

            # Safeguard 3: auth shouldn't retry
            if code == "openai_auth":
                logger.error("[chat] req_id=%s AUTH failure attempt=%d latency_ms=%d err=%s",
                             req_id, attempt, dt_ms, str(e))
                break

            # Determine transience (best-effort)
            transient = False
            msg = (str(e) or "").lower()
            if hasattr(e, "http_status") and isinstance(getattr(e, "http_status"), int):
                transient = _is_transient_status(int(getattr(e, "http_status")))
            if "timeout" in msg or "temporarily" in msg or "502" in msg or "503" in msg or "429" in msg:
                transient = True

            logger.warning("[chat] req_id=%s FAIL attempt=%d/%d latency_ms=%d model=%s code=%s transient=%s err=%s",
                           req_id, attempt, _CHAT_MAX_ATTEMPTS, dt_ms, model_name, code, transient, str(e))

            if attempt >= _CHAT_MAX_ATTEMPTS or not transient:
                break

            _sleep_backoff(attempt)

    logger.error("[chat] req_id=%s chat failed after retries. last_code=%s last_err=%r",
                 req_id, _classify_openai_error(last_err or Exception("unknown")), last_err)
    raise RuntimeError("TARS encountered an error contacting the model.") from last_err

# ---------------------------------------------------------------------------
# STT + TTS
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    req_id = _mk_req_id("stt")
    payload_fp = _sha256_hex(audio_bytes)

    # Safeguard 4: fail fast if key missing at runtime
    if not _settings.openai_api_key or not str(_settings.openai_api_key).strip():
        logger.error("[stt] req_id=%s OPENAI_API_KEY missing in settings at runtime.", req_id)
        raise RuntimeError("TARS STT misconfigured: OPENAI_API_KEY missing.")

    # Safeguard 5: reject empty / tiny payloads
    if not audio_bytes:
        logger.error("[stt] req_id=%s Empty audio_bytes received.", req_id)
        raise RuntimeError("TARS STT received empty audio bytes.")
    if len(audio_bytes) < 256:
        logger.error("[stt] req_id=%s Audio payload too small (%d bytes) fp=%s.", req_id, len(audio_bytes), payload_fp)
        raise RuntimeError("TARS STT received an invalid audio payload (too small).")

    # Ensure api_base stays /v1 even if mutated
    try:
        current_base = getattr(openai, "api_base", "") or ""
        if not str(current_base).rstrip("/").endswith("/v1"):
            fixed = _normalize_openai_api_base(str(current_base))
            openai.api_base = fixed
            logger.warning("[stt] req_id=%s openai.api_base mutated; re-normalized to %s", req_id, fixed)
    except Exception:
        pass

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = _safe_audio_filename(mime_type)
    model_name = _get_stt_model()

    client, use_new_sdk = _try_make_v1_client()
    last_err: Optional[Exception] = None

    host = _safe_host_from_url(_OPENAI_API_BASE)
    logger.info("[stt] req_id=%s start model=%s base=%s host=%s bytes=%d fp=%s mime=%s lang=%s new_sdk=%s",
                req_id, model_name, _OPENAI_API_BASE, host, len(audio_bytes), payload_fp, mime_type, language, use_new_sdk)

    for attempt in range(1, max(1, _STT_MAX_ATTEMPTS) + 1):
        try:
            audio_file.seek(0)
        except Exception:
            pass

        t0 = time.monotonic()
        try:
            if use_new_sdk and client is not None:
                kwargs: Dict[str, Any] = {}
                if language:
                    kwargs["language"] = language
                result = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    **kwargs,
                )
                text = getattr(result, "text", "") or ""
            else:
                resp = openai.Audio.transcribe(
                    model=model_name,
                    file=audio_file,
                    language=language,
                )
                text = resp["text"] if isinstance(resp, dict) else getattr(resp, "text", "")

            dt_ms = int((time.monotonic() - t0) * 1000)
            text = str(text or "").strip()
            if not text:
                raise RuntimeError("TARS received an empty transcription from the model.")

            snippet = text[:180] + ("..." if len(text) > 180 else "")
            logger.info("[stt] req_id=%s OK attempt=%d latency_ms=%d model=%s mime=%s lang=%s transcript=%r",
                        req_id, attempt, dt_ms, model_name, mime_type, language, snippet)
            return text

        except Exception as e:
            last_err = e
            dt_ms = int((time.monotonic() - t0) * 1000)
            code = _classify_openai_error(e)
            msg = str(e)

            transient = False
            if hasattr(e, "http_status") and isinstance(getattr(e, "http_status"), int):
                transient = _is_transient_status(int(getattr(e, "http_status")))
            if "timeout" in msg.lower() or "502" in msg or "503" in msg or "429" in msg:
                transient = True

            if code == "stt_404_nginx_base_url":
                transient = False

            logger.warning("[stt] req_id=%s FAIL attempt=%d/%d latency_ms=%d model=%s code=%s transient=%s err=%s",
                           req_id, attempt, _STT_MAX_ATTEMPTS, dt_ms, model_name, code, transient, msg)

            if attempt >= _STT_MAX_ATTEMPTS or not transient:
                break

            _sleep_backoff(attempt)

    logger.error("[stt] req_id=%s STT failed after retries. last_code=%s last_err=%r",
                 req_id, _classify_openai_error(last_err or Exception("unknown")), last_err)
    raise RuntimeError("TARS failed to transcribe the audio input.") from last_err


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    audio_format: Optional[str] = None,
) -> bytes:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Cannot synthesize speech from empty text.")

    model_name = _get_tts_model()
    voice_name = voice or _get_tts_voice()
    fmt = (audio_format or _get_tts_audio_format()).lower()

    # IMPORTANT: _OPENAI_API_BASE already ends with /v1
    url = _join_api(_OPENAI_API_BASE, "audio/speech")

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

    if len(cleaned) > 6000:
        logger.warning("TTS input unusually long (%d chars). Consider truncation upstream.", len(cleaned))

    req_id = _mk_req_id("tts")
    host = _safe_host_from_url(url)

    last_err: Optional[Exception] = None

    for attempt in range(1, max(1, _TTS_MAX_ATTEMPTS) + 1):
        t0 = time.monotonic()
        try:
            resp = _HTTP_SESSION.post(url, headers=headers, json=payload, timeout=_OPENAI_HTTP_TIMEOUT_SEC)
        except Exception as e:
            last_err = e
            dt_ms = int((time.monotonic() - t0) * 1000)
            code = _classify_openai_error(e)
            logger.warning("[tts] req_id=%s HTTP exception attempt=%d/%d latency_ms=%d host=%s code=%s err=%s",
                           req_id, attempt, _TTS_MAX_ATTEMPTS, dt_ms, host, code, str(e))
            if attempt >= _TTS_MAX_ATTEMPTS:
                break
            _sleep_backoff(attempt)
            continue

        dt_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code == 200:
            audio_bytes = resp.content or b""
            if not audio_bytes:
                raise RuntimeError("TARS received an empty audio response from the model.")
            logger.info("[tts] req_id=%s OK attempt=%d latency_ms=%d bytes=%d model=%s voice=%s fmt=%s host=%s",
                        req_id, attempt, dt_ms, len(audio_bytes), model_name, voice_name, fmt, host)
            return audio_bytes

        body_preview = (resp.text or "")[:400]
        transient = _is_transient_status(resp.status_code)
        if resp.status_code == 401:
            transient = False

        logger.warning("[tts] req_id=%s non-200 attempt=%d/%d status=%d latency_ms=%d host=%s body=%r transient=%s",
                       req_id, attempt, _TTS_MAX_ATTEMPTS, resp.status_code, dt_ms, host, body_preview, transient)

        last_err = RuntimeError(f"TARS TTS request failed with status {resp.status_code}: {resp.text}")

        if attempt >= _TTS_MAX_ATTEMPTS or not transient:
            break

        _sleep_backoff(attempt)

    logger.error("[tts] req_id=%s OpenAI TTS failed after retries. Last error=%r", req_id, last_err)
    raise RuntimeError("TARS failed to synthesize speech from text.") from last_err


def get_openai_runtime_config() -> Dict[str, str]:
    """
    Returns non-secret runtime configuration helpful for debugging.
    Intended for logs / internal inspection.
    """
    cfg = {
        "openai_api_base": str(getattr(openai, "api_base", "") or ""),
        "openai_api_key_set": "YES" if bool(_settings.openai_api_key and str(_settings.openai_api_key).strip()) else "NO",
        "openai_model": _get_chat_model(),
        "openai_stt_model": _get_stt_model(),
        "openai_tts_model": _get_tts_model(),
        "openai_tts_voice": _get_tts_voice(),
        "openai_http_timeout_sec": str(_OPENAI_HTTP_TIMEOUT_SEC),
        "stt_max_attempts": str(_STT_MAX_ATTEMPTS),
        "tts_max_attempts": str(_TTS_MAX_ATTEMPTS),
        "chat_max_attempts": str(_CHAT_MAX_ATTEMPTS),
    }
    return cfg

