# tars/api/server.py
# -*- coding: utf-8 -*-
"""
TARS Core API (FastAPI).

Endpoints:
- /health
- /chat_audio : audio (STT -> TARSCore -> TTS)
- /memory/*   : memory helpers

This revision focuses on CONSISTENT AUDIO:
- WAV-first TTS by default (unless client asks for mp3)
- Response audio_mime_type always matches the returned bytes
- Voice fallback routing (OpenAI voice A -> fallback voice A -> no audio)
- System prompt file loading (server-side) for consistent persona across clients

Environment variables (recommended):
- TARS_DEFAULT_REPLY_AUDIO_FORMAT=wav
- TARS_DEFAULT_TTS_VOICE=onyx            (or any OpenAI voice you want)
- TARS_FALLBACK_TTS_VOICE=alloy
- TARS_SYSTEM_PROMPT_PATH=tars/config/tars_system_prompt.txt
- TARS_SESSION_DIR=tars/memory/sessions
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr

from tars.audio.service import stt_with_safeguards, tts_with_safeguards
from tars.utils.logging import get_logger

# Keep these imports aligned with your repo structure.
from tars.core.chat import TARSCore
from tars.core.state import ReasoningMode
from tars.core.channels import ResponseChannel

from tars.memory.repository import (
    save_memory_item,
    get_recent_memory_items,
    get_memory_items_by_type,
    search_memory_by_keyword,
)

logger = get_logger(__name__)

app = FastAPI(
    title="TARS Core API",
    description="Local API for TARS core (modes, memory, logging, audio).",
    version="1.6.1",
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Session storage for JSONL turns
_SESS_DIR = os.getenv("TARS_SESSION_DIR", os.path.join("tars", "memory", "sessions"))
os.makedirs(_SESS_DIR, exist_ok=True)

# Soft per-session rate limiter
_SESS_LAST_CALL: Dict[str, float] = {}
_SESS_RATE_LIMIT_WINDOW_S = float(os.getenv("TARS_RATE_LIMIT_WINDOW_S", "0.75"))

# Reset phrases that rotate session id
_RESET_SESSION_PHRASES = {
    "reset session",
    "reset the session",
    "new session",
    "start new session",
    "forget this conversation",
}

# WAV-first behavior
_DEFAULT_REPLY_AUDIO_FORMAT = os.getenv("TARS_DEFAULT_REPLY_AUDIO_FORMAT", "wav").strip().lower() or "wav"
_ALLOWED_REPLY_FORMATS = {"mp3", "wav"}

# Voice routing (OpenAI voice A)
_DEFAULT_REPLY_VOICE = os.getenv("TARS_DEFAULT_TTS_VOICE", "").strip() or None
_FALLBACK_REPLY_VOICE = os.getenv("TARS_FALLBACK_TTS_VOICE", "alloy").strip() or "alloy"

# System prompt file (server-side)
_SYSTEM_PROMPT_PATH = os.getenv(
    "TARS_SYSTEM_PROMPT_PATH",
    os.path.join("tars", "config", "tars_system_prompt.txt"),
)

# -----------------------------------------------------------------------------
# Boot: TARSCore + optional system prompt load
# -----------------------------------------------------------------------------

tars_core = TARSCore()


def _try_load_system_prompt_text(path: str) -> Optional[str]:
    try:
        if not os.path.exists(path):
            return None
        txt = open(path, "r", encoding="utf-8").read()
        txt = (txt or "").strip()
        return txt or None
    except Exception as e:
        logger.warning("Failed to read system prompt file %r: %s", path, e)
        return None


def _apply_system_prompt_if_supported(prompt_text: str) -> bool:
    """
    Applies system prompt to TARSCore if a compatible method exists.
    This is deliberately defensive: we don't crash if the core doesn't support it.
    """
    if not prompt_text:
        return False

    for method_name in ("set_system_prompt", "set_system_prompt_text", "load_system_prompt"):
        try:
            if hasattr(tars_core, method_name):
                fn = getattr(tars_core, method_name)
                fn(prompt_text)
                logger.info("Applied system prompt via tars_core.%s()", method_name)
                return True
        except Exception as e:
            logger.warning("System prompt apply failed via %s: %s", method_name, e)

    logger.info("No system prompt setter found on TARSCore; continuing without applying file.")
    return False


_system_prompt_text = _try_load_system_prompt_text(_SYSTEM_PROMPT_PATH)
_system_prompt_loaded = False
if _system_prompt_text:
    _system_prompt_loaded = _apply_system_prompt_if_supported(_system_prompt_text)
logger.info(
    "System prompt path=%s exists=%s loaded=%s",
    _SYSTEM_PROMPT_PATH,
    bool(_system_prompt_text),
    _system_prompt_loaded,
)

# -----------------------------------------------------------------------------
# Models (Memory)
# -----------------------------------------------------------------------------

class MemoryAddRequest(BaseModel):
    type: str
    label: Optional[str] = None
    content: str
    source_conversation_id: Optional[int] = None


class MemoryItemResponse(BaseModel):
    id: int
    created_at: str
    type: str
    label: Optional[str]
    content: str
    source_conversation_id: Optional[int]


# -----------------------------------------------------------------------------
# Models (Audio Chat)
# -----------------------------------------------------------------------------

class AudioChatRequest(BaseModel):
    """
    Input contract for /chat_audio.

    audio_base64 : base64-encoded audio bytes (WAV/MP3/OGG, etc.)
    mime_type    : optional MIME hint for incoming audio
    sample_rate  : optional metadata
    language     : optional STT language hint
    mode         : optional reasoning mode hint
    voice_style  : optional voice style hint for reply shaping
    session_id   : client session identifier
    channel      : channel hint (default VOICE)

    New (non-breaking):
    reply_audio_format : "wav" or "mp3" (default wav)
    reply_voice        : optional voice name/preset (OpenAI voices)
    """
    audio_base64: constr(min_length=1) = Field(..., description="Base64-encoded audio payload.")
    mime_type: Optional[str] = Field(default=None, description="Optional incoming MIME type.")
    sample_rate: Optional[int] = Field(default=None, description="Optional sample rate metadata.")
    language: Optional[str] = Field(default=None, description="Optional STT language code, e.g. 'en', 'es'.")
    mode: Optional[str] = Field(default=None, description="Optional reasoning mode hint.")
    voice_style: Optional[str] = Field(default=None, description="Optional voice style hint for spoken reply.")
    session_id: Optional[str] = Field(default=None, description="Client session identifier.")
    channel: Optional[str] = Field(default="VOICE", description="Optional channel hint (VOICE/TEXT).")

    reply_audio_format: Optional[str] = Field(
        default=None,
        description="Reply audio format: 'wav' or 'mp3'. If omitted, server default is used (wav).",
    )
    reply_voice: Optional[str] = Field(
        default=None,
        description="Preferred reply voice (OpenAI voice name). If omitted, server default is used.",
    )


class AudioChatResponse(BaseModel):
    """
    Output contract for /chat_audio.
    """
    reply: str
    mode: str
    transcript: str
    audio_base64: str
    audio_mime_type: str
    latency_ms: int
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_reset_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _apply_mode_hint(mode_hint: Optional[str]) -> None:
    if not mode_hint:
        return

    normalized = mode_hint.strip().lower()
    try:
        if normalized.startswith("analyst") or "analysis" in normalized:
            tars_core.set_mode(ReasoningMode.ANALYST)
        elif normalized.startswith("critic") or "devil" in normalized:
            tars_core.set_mode(ReasoningMode.CRITIC)
        elif "synth" in normalized or "synthetic" in normalized:
            tars_core.set_mode(ReasoningMode.SYNTHESIZER)
        else:
            logger.info("Unrecognized mode hint: %r; leaving mode unchanged.", mode_hint)
    except Exception as e:
        logger.error("Failed to apply mode hint %r: %s", mode_hint, e)


def _normalize_voice_style(style: Optional[str]) -> str:
    if not style:
        return "default"
    cleaned = style.strip().lower()
    if cleaned in ("brief", "short", "concise"):
        return "brief"
    if cleaned in ("story", "narrative"):
        return "story"
    if cleaned in ("technical", "tech", "dense"):
        return "technical"
    return "default"


def _truncate_for_voice_reply(reply: str, style: str) -> Tuple[str, bool]:
    reply = (reply or "").strip()
    if not reply:
        return "", False

    if style == "brief":
        cap = 360
    elif style == "technical":
        cap = 820
    elif style == "story":
        cap = 1200
    else:
        cap = 520

    if len(reply) <= cap:
        return reply, False
    return reply[:cap].rstrip() + "…", True


def _reply_format_and_mime(req_fmt: Optional[str]) -> Tuple[str, str]:
    """
    Resolve requested format against allowlist + defaults, and return (fmt, mime).
    """
    fmt = (req_fmt or _DEFAULT_REPLY_AUDIO_FORMAT or "wav").strip().lower()
    if fmt not in _ALLOWED_REPLY_FORMATS:
        logger.warning("Invalid reply_audio_format=%r; forcing default=%s", fmt, _DEFAULT_REPLY_AUDIO_FORMAT)
        fmt = _DEFAULT_REPLY_AUDIO_FORMAT if _DEFAULT_REPLY_AUDIO_FORMAT in _ALLOWED_REPLY_FORMATS else "wav"

    if fmt == "wav":
        return "wav", "audio/wav"
    return "mp3", "audio/mpeg"


# -----------------------------------------------------------------------------
# Session persistence (JSONL)
# -----------------------------------------------------------------------------

@dataclass
class _Turn:
    role: str
    content: str
    ts: float


def _session_path(session_id: str) -> str:
    safe = "".join(ch for ch in (session_id or "") if ch.isalnum() or ch in ("-", "_")).strip()
    safe = safe or "default"
    return os.path.join(_SESS_DIR, f"{safe}.jsonl")


def _append_session_turn(session_id: str, role: str, content: str) -> None:
    rec = {"ts": time.time(), "role": role, "content": content}
    try:
        with open(_session_path(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_session_turns(session_id: str, limit: int = 12) -> List[_Turn]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return []
    turns: List[_Turn] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    turns.append(_Turn(
                        role=obj.get("role", ""),
                        content=obj.get("content", ""),
                        ts=float(obj.get("ts", 0.0)),
                    ))
                except Exception:
                    continue
    except Exception:
        return []

    if limit and len(turns) > limit:
        turns = turns[-limit:]
    return turns


def _build_context_block(turns: List[_Turn], max_chars: int = 1800) -> str:
    lines: List[str] = []
    for t in turns:
        role = "User" if t.role == "user" else "Assistant"
        msg = (t.content or "").strip()
        if not msg:
            continue
        lines.append(f"{role}: {msg}")

    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "system_prompt_path": _SYSTEM_PROMPT_PATH,
        "system_prompt_loaded": bool(_system_prompt_text) and _system_prompt_loaded,
        "default_reply_audio_format": _DEFAULT_REPLY_AUDIO_FORMAT,
        "default_tts_voice": _DEFAULT_REPLY_VOICE,
        "fallback_tts_voice": _FALLBACK_REPLY_VOICE,
    }


# -----------------------------------------------------------------------------
# Chat endpoint (Audio)
# -----------------------------------------------------------------------------

@app.post("/chat_audio", response_model=AudioChatResponse)
def chat_audio(req: AudioChatRequest) -> AudioChatResponse:
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    session_id = (req.session_id or "").strip() or "default"

    # Rate limit
    now = time.time()
    last = _SESS_LAST_CALL.get(session_id)
    if last is not None and (now - last) < _SESS_RATE_LIMIT_WINDOW_S:
        logger.warning("[chat_audio] request_id=%s session_id=%s rate_limited delta=%.2fs", request_id, session_id, now - last)
        raise HTTPException(status_code=429, detail="Too many requests (rate limited).")
    _SESS_LAST_CALL[session_id] = now

    reply_fmt, reply_mime = _reply_format_and_mime(req.reply_audio_format)
    primary_voice = (req.reply_voice or _DEFAULT_REPLY_VOICE)
    fallback_voice = _FALLBACK_REPLY_VOICE  # always defined (string)

    logger.info(
        "[chat_audio] request_id=%s session_id=%s audio_b64_len=%d in_mime=%s lang=%s mode_hint=%s voice_style=%s out_fmt=%s out_mime=%s voice=%r",
        request_id, session_id, len(req.audio_base64), req.mime_type, req.language, req.mode, req.voice_style, reply_fmt, reply_mime, primary_voice
    )

    # Decode base64 audio
    if not req.audio_base64 or not req.audio_base64.strip():
        raise HTTPException(status_code=400, detail="Field 'audio_base64' must not be empty.")
    try:
        audio_bytes = base64.b64decode(req.audio_base64, validate=True)
    except Exception as exc:
        logger.warning("[chat_audio] request_id=%s invalid base64: %s", request_id, exc)
        raise HTTPException(status_code=400, detail="Invalid 'audio_base64' payload: could not decode base64.")

    # STT
    try:
        transcript = stt_with_safeguards(
            audio_bytes=audio_bytes,
            mime_type=req.mime_type,
            language=req.language,
        )
    except ValueError as exc:
        logger.warning("[chat_audio] request_id=%s STT validation error: %s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("[chat_audio] request_id=%s STT runtime error: %s", request_id, exc)
        raise HTTPException(status_code=502, detail="Speech-to-text service failed.")

    transcript = (transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript (no speech detected).")

    # Reset phrase -> new session id
    if _normalize_reset_phrase(transcript) in _RESET_SESSION_PHRASES:
        new_session_id = str(uuid.uuid4())
        logger.info("[chat_audio] request_id=%s session_id=%s reset_phrase=%r -> new_session_id=%s", request_id, session_id, transcript, new_session_id)
        session_id = new_session_id

    # Mode hint
    _apply_mode_hint(req.mode)

    normalized_voice_style = _normalize_voice_style(req.voice_style)

    # Load turns
    prior_turns = _load_session_turns(session_id=session_id, limit=12)
    context_block = _build_context_block(prior_turns, max_chars=1800)

    # Core
    try:
        if context_block:
            user_text_for_core = (
                "Conversation so far (most recent turns):\n"
                f"{context_block}\n\n"
                "Current user utterance:\n"
                f"{transcript}"
            )
        else:
            user_text_for_core = transcript

        reply = tars_core.process_user_text(
            user_text_for_core,
            channel=ResponseChannel.VOICE,
            voice_style=normalized_voice_style,
        )
        mode = tars_core.state.mode.value
    except RuntimeError as exc:
        logger.error("[chat_audio] request_id=%s session_id=%s TARSCore error: %s", request_id, session_id, exc)
        raise HTTPException(status_code=500, detail="TARS core text engine failed to process the transcript.")
    except Exception as exc:
        logger.error("[chat_audio] request_id=%s session_id=%s Unexpected core error: %s", request_id, session_id, exc)
        raise HTTPException(status_code=500, detail="Unexpected error in TARS core while processing transcript.")

    reply = (reply or "").strip()
    if not reply:
        logger.error("[chat_audio] request_id=%s session_id=%s Empty reply for transcript: %r", request_id, session_id, transcript)
        raise HTTPException(status_code=500, detail="TARS generated an empty reply.")

    # Truncate for voice before TTS
    reply, was_truncated = _truncate_for_voice_reply(reply, normalized_voice_style)

    # TTS (non-fatal) with fallback voice (A)
    audio_reply_base64 = ""
    error: Optional[str] = None
    tried: List[str] = []

    try:
        tried.append(f"primary={primary_voice!r}:{reply_fmt}")
        audio_reply_bytes = tts_with_safeguards(text=reply, voice=primary_voice, audio_format=reply_fmt)
        audio_reply_base64 = base64.b64encode(audio_reply_bytes).decode("ascii")
        error = None
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "[chat_audio] request_id=%s session_id=%s TTS primary failed: %s (voice=%r fmt=%s)",
            request_id, session_id, exc, primary_voice, reply_fmt
        )
        # Fallback voice attempt
        try:
            tried.append(f"fallback={fallback_voice!r}:{reply_fmt}")
            audio_reply_bytes = tts_with_safeguards(text=reply, voice=fallback_voice, audio_format=reply_fmt)
            audio_reply_base64 = base64.b64encode(audio_reply_bytes).decode("ascii")
            error = "tts_primary_failed_fallback_succeeded"
        except Exception as exc2:
            logger.error(
                "[chat_audio] request_id=%s session_id=%s TTS fallback failed: %s (voice=%r fmt=%s tried=%s)",
                request_id, session_id, exc2, fallback_voice, reply_fmt, tried
            )
            audio_reply_base64 = ""
            error = "tts_failed"

    # Persist turns
    try:
        _append_session_turn(session_id, "user", transcript)
        _append_session_turn(session_id, "assistant", reply)
    except Exception:
        pass

    latency_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "[chat_audio] request_id=%s session_id=%s OK latency_ms=%d mode=%s error=%s truncated=%s out_mime=%s tried=%s",
        request_id, session_id, latency_ms, mode, error, was_truncated, reply_mime, tried
    )

    return AudioChatResponse(
        reply=reply,
        mode=mode,
        transcript=transcript,
        audio_base64=audio_reply_base64,
        audio_mime_type=reply_mime,
        latency_ms=latency_ms,
        error=error,
    )


# -----------------------------------------------------------------------------
# Memory endpoints
# -----------------------------------------------------------------------------

@app.post("/memory/add", response_model=MemoryItemResponse)
def add_memory(req: MemoryAddRequest) -> MemoryItemResponse:
    mem_id = save_memory_item(
        type=req.type,
        label=req.label,
        content=req.content,
        source_conversation_id=req.source_conversation_id,
    )

    items = get_recent_memory_items(limit=1)
    if not items or items[0].id != mem_id:
        logger.warning("New memory id %s not found in recent list; returning fallback response.", mem_id)
        return MemoryItemResponse(
            id=mem_id,
            created_at="",
            type=req.type,
            label=req.label,
            content=req.content,
            source_conversation_id=req.source_conversation_id,
        )

    item = items[0]
    return MemoryItemResponse(
        id=item.id,
        created_at=item.created_at,
        type=item.type,
        label=item.label,
        content=item.content,
        source_conversation_id=item.source_conversation_id,
    )


@app.get("/memory/recent", response_model=List[MemoryItemResponse])
def recent_memory(limit: int = 20) -> List[MemoryItemResponse]:
    items = get_recent_memory_items(limit=limit)
    return [
        MemoryItemResponse(
            id=i.id,
            created_at=i.created_at,
            type=i.type,
            label=i.label,
            content=i.content,
            source_conversation_id=i.source_conversation_id,
        )
        for i in items
    ]


@app.get("/memory/type/{type_name}", response_model=List[MemoryItemResponse])
def memory_by_type(type_name: str, limit: int = 50) -> List[MemoryItemResponse]:
    items = get_memory_items_by_type(type_name=type_name, limit=limit)
    return [
        MemoryItemResponse(
            id=i.id,
            created_at=i.created_at,
            type=i.type,
            label=i.label,
            content=i.content,
            source_conversation_id=i.source_conversation_id,
        )
        for i in items
    ]


@app.get("/memory/search", response_model=List[MemoryItemResponse])
def memory_search(q: str, limit: int = 50) -> List[MemoryItemResponse]:
    items = search_memory_by_keyword(query=q, limit=limit)
    return [
        MemoryItemResponse(
            id=i.id,
            created_at=i.created_at,
            type=i.type,
            label=i.label,
            content=i.content,
            source_conversation_id=i.source_conversation_id,
        )
        for i in items
    ]

