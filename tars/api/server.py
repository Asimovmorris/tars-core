# tars/api/server.py
"""
FastAPI server for TARS core:

- /chat_text   : text-only interaction with TARSCore
- /chat_audio  : audio (STT -> TARSCore -> TTS)
- /memory/*    : manual memory management endpoints
- /health      : basic health check

Design practices applied:
1. Clear request/response contracts (Pydantic models).
2. Separation of concerns (TARSCore, audio service, memory repo).
3. Explicit logging with per-request correlation IDs.
4. Input validation + explicit, meaningful HTTP error codes.
5. Bounded resource usage (history/length handled in TARSCore).
6. Graceful degradation (e.g. TTS failures still return text).
7. No binary data in model prompts (audio never touches TARSCore directly).
8. Stable API semantics (POST for side-effects, GET for reads).
9. Centralized error handling boundaries around model calls.
10. Extensibility for mode, language, and now voice_style + reply-length control.
11. NEW: Session memory persisted in JSONL keyed by session_id for /chat_audio.
"""

import base64
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, constr

from tars.core.chat import TARSCore, ResponseChannel
from tars.core.modes import ReasoningMode
from tars.memory.repository import (
    save_memory_item,
    get_recent_memory_items,
    get_memory_items_by_type,
    search_memory_by_keyword,
)
from tars.audio.service import stt_with_safeguards, tts_with_safeguards
from tars.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="TARS Core API",
    description="Local API for TARS text-based core (modes, memory, logging, audio).",
    version="1.5.0",
)

# Single TARSCore instance shared by all requests
tars_core = TARSCore()

# ---------------------------------------------------------------------------
# Voice reply length configuration (Step 2: output-side control)
# ---------------------------------------------------------------------------

# Hard safety ceiling: never send more than this many characters to TTS
VOICE_REPLY_MAX_CHARS_HARD = 1400  # chars

# Default / style-specific soft limits:
VOICE_REPLY_LIMITS = {
    # style: (max_sentences, max_chars_soft)
    "brief": (4, 600),
    "default": (5, 700),
    "story": (7, 900),
    "technical": (6, 900),
}

# ---------------------------------------------------------------------------
# Models (Text Chat)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message in plain text.")


class ChatResponse(BaseModel):
    reply: str
    mode: str


# ---------------------------------------------------------------------------
# Models (Memory)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Models (Audio Chat)
# ---------------------------------------------------------------------------

class AudioChatRequest(BaseModel):
    """
    Input contract for /chat_audio.

    audio_base64 : base64-encoded audio bytes (WAV/MP3/OGG, etc.)
    mime_type    : optional MIME hint, e.g. "audio/wav"
    sample_rate  : optional metadata (not required by the API itself)
    language     : optional STT language hint, e.g. "en" or "es" (None = auto)
    mode         : optional reasoning mode hint, e.g. "analyst", "critic", "synthetic"
    voice_style  : optional style hint for spoken replies, e.g. "brief", "story", "technical"
    session_id   : client-provided session identifier (Android already sends this)
    channel      : optional channel hint (default "VOICE")
    """
    audio_base64: constr(min_length=1) = Field(
        ...,
        description="Base64-encoded audio payload (WAV/MP3/OGG).",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="Optional MIME type, e.g. 'audio/wav', 'audio/mpeg'.",
    )
    sample_rate: Optional[int] = Field(
        default=None,
        description="Optional sample rate in Hz (metadata / client-side only).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional STT language code, e.g. 'en' or 'es'.",
    )
    mode: Optional[str] = Field(
        default=None,
        description="Optional reasoning mode hint: 'analyst', 'critic', 'synthetic'.",
    )
    voice_style: Optional[str] = Field(
        default=None,
        description=(
            "Optional voice style hint for spoken reply, e.g. 'brief', 'story', 'technical'. "
            "If omitted, TARS uses a default concise voice hint and reply-length limits."
        ),
    )

    # ✅ session-aware continuity
    session_id: Optional[str] = Field(
        default=None,
        description="Client session identifier for persistent conversation across turns.",
    )

    channel: Optional[str] = Field(
        default="VOICE",
        description="Optional channel hint (VOICE/TEXT); VOICE default.",
    )


class AudioChatResponse(BaseModel):
    """
    Output contract for /chat_audio.

    reply           : TARS's text reply (possibly truncated for voice)
    mode            : TARS's current reasoning mode
    transcript      : STT transcription of user's speech
    audio_base64    : base64-encoded TTS audio (may be empty if TTS fails)
    audio_mime_type : MIME type of returned audio
    latency_ms      : total processing time for the request
    error           : null if OK, else a short error code/description
    """
    reply: str
    mode: str
    transcript: str
    audio_base64: str
    audio_mime_type: str = "audio/mpeg"
    latency_ms: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: map mode hint string -> ReasoningMode
# ---------------------------------------------------------------------------

def _apply_mode_hint(mode_hint: Optional[str]) -> None:
    """
    Interpret a user-provided mode hint string and apply it to TARSCore.

    This is an OPTIONAL override; if mode_hint is None or unrecognized,
    we leave the current mode unchanged.
    """
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
            # Unknown hint: ignore gracefully
            logger.info("Received unrecognized mode hint: %r; leaving mode unchanged.", mode_hint)
    except Exception as e:
        # Mode changes should never break the request
        logger.error("Failed to apply mode hint %r: %s", mode_hint, e)


# ---------------------------------------------------------------------------
# Helper: voice style normalization + truncation
# ---------------------------------------------------------------------------

def _normalize_voice_style(style: Optional[str]) -> str:
    """
    Normalize the voice_style field into one of:
        'brief', 'story', 'technical', 'default'

    Used both for system hints (in TARSCore) and for reply truncation logic.
    """
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


def _split_sentences_basic(text: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.

    We keep it intentionally naive but robust enough for truncation:
    - Split on '.', '?', '!' followed by whitespace.
    - Preserve punctuation at end of each sentence.
    """
    text = text.strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _truncate_for_voice_reply(
    text: str,
    voice_style: str,
) -> Tuple[str, bool]:
    """
    Apply length limits for voice replies.

    - Enforce style-specific sentence and soft char limits.
    - Enforce a hard char ceiling (VOICE_REPLY_MAX_CHARS_HARD).
    - Returns (truncated_text, was_truncated_flag).
    """
    original = (text or "").strip()
    if not original:
        return original, False

    style = _normalize_voice_style(voice_style)
    max_sentences, max_chars_soft = VOICE_REPLY_LIMITS.get(style, VOICE_REPLY_LIMITS["default"])

    # 1) Hard cap first as an absolute safety measure
    if len(original) > VOICE_REPLY_MAX_CHARS_HARD:
        logger.info(
            "Voice reply length %d > hard cap %d; applying hard truncation.",
            len(original),
            VOICE_REPLY_MAX_CHARS_HARD,
        )
        original = original[:VOICE_REPLY_MAX_CHARS_HARD].rstrip()

    # 2) Sentence-based truncation (soft)
    sentences = _split_sentences_basic(original)
    if not sentences:
        truncated = original[:max_chars_soft].rstrip()
        was_truncated = len(truncated) < len(original)
        return truncated, was_truncated

    if len(sentences) <= max_sentences and len(original) <= max_chars_soft:
        return original, False

    selected = sentences[:max_sentences]
    truncated_text = " ".join(selected).strip()

    # 3) Apply soft char limit
    if len(truncated_text) > max_chars_soft:
        truncated_text = truncated_text[:max_chars_soft].rstrip()

    was_truncated = len(truncated_text) < len(original)
    return truncated_text, was_truncated


# ---------------------------------------------------------------------------
# Chat endpoints (Text)
# ---------------------------------------------------------------------------

@app.post("/chat_text", response_model=ChatResponse)
def chat_text(req: ChatRequest) -> ChatResponse:
    """
    Send a text message to TARS and get a reply + current reasoning mode.
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()
    logger.info("[chat_text] request_id=%s message_len=%d", request_id, len(req.message))

    try:
        reply = tars_core.process_user_text(req.message, channel=ResponseChannel.TEXT)
        mode = tars_core.state.mode.value
        if not reply or not reply.strip():
            logger.error("[chat_text] Empty reply for request_id=%s message=%r", request_id, req.message)
            raise HTTPException(status_code=500, detail="TARS generated an empty reply; this should not happen.")
        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("[chat_text] request_id=%s OK latency_ms=%d mode=%s", request_id, latency_ms, mode)
        return ChatResponse(reply=reply, mode=mode)
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("[chat_text] RuntimeError request_id=%s error=%s", request_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("[chat_text] Unexpected error request_id=%s error=%s", request_id, e)
        raise HTTPException(status_code=500, detail="Unexpected error in chat_text.")


@app.get("/health")
def health_check() -> dict:
    """
    Very simple health check endpoint.
    """
    try:
        mode = tars_core.state.mode.value
    except Exception:
        mode = "unknown"
    return {"status": "ok", "mode": mode}


# ---------------------------------------------------------------------------
# Session conversation memory (persisted locally as JSONL)
# ---------------------------------------------------------------------------

_SESS_DIR = Path(__file__).resolve().parents[1] / "memory" / "sessions"
_SESS_DIR.mkdir(parents=True, exist_ok=True)

# Addition #4: per-session soft rate limiter (prevents accidental double-sends)
_SESS_RATE_LIMIT_WINDOW_S = 3.0
_SESS_LAST_CALL: Dict[str, float] = {}

def _session_file(session_id: str) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        safe = "default"
    return _SESS_DIR / f"{safe}.jsonl"

def _load_session_turns(session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    fp = _session_file(session_id)
    if not fp.exists():
        return []
    turns: List[Dict[str, Any]] = []
    try:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turns.append(json.loads(line))
                except Exception:
                    continue
        return turns[-limit:]
    except Exception:
        return []

def _append_session_turn(session_id: str, role: str, text: str) -> None:
    fp = _session_file(session_id)
    rec = {"role": role, "text": text, "ts": time.time()}
    try:
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _build_context_block(turns: List[Dict[str, Any]], max_chars: int = 1800) -> str:
    if not turns:
        return ""
    chunks: List[str] = []
    for t in turns:
        role = (t.get("role") or "").strip().lower()
        text = (t.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            chunks.append(f"User: {text}")
        elif role == "assistant":
            chunks.append(f"TARS: {text}")
        else:
            chunks.append(text)

    ctx = "\n".join(chunks).strip()
    if len(ctx) > max_chars:
        ctx = ctx[-max_chars:]
    return ctx

# Addition #3: voice-only session reset triggers
_RESET_SESSION_PHRASES = {
    "reset memory",
    "new session",
    "forget that",
    "clear memory",
}

def _normalize_reset_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Chat endpoint (Audio)
# ---------------------------------------------------------------------------

@app.post("/chat_audio", response_model=AudioChatResponse)
def chat_audio(req: AudioChatRequest) -> AudioChatResponse:
    """
    End-to-end audio chat:
      1) Decode base64 audio
      2) STT via stt_with_safeguards
      3) Optional mode override via 'mode'
      4) Session memory: inject recent turns (JSONL) into TARSCore input
      5) TARSCore: transcript -> reply (VOICE)
      6) Apply voice reply length control
      7) TTS via tts_with_safeguards
      8) Persist session turns
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    # session id (Android sends this)
    session_id = (req.session_id or "").strip() or "default"

    # Addition #4: soft rate limiter per session_id
    now = time.time()
    last = _SESS_LAST_CALL.get(session_id)
    if last is not None and (now - last) < _SESS_RATE_LIMIT_WINDOW_S:
        logger.warning(
            "[chat_audio] request_id=%s session_id=%s rate_limited delta=%.2fs",
            request_id, session_id, now - last,
        )
        raise HTTPException(status_code=429, detail="Too many requests (rate limited).")
    _SESS_LAST_CALL[session_id] = now

    logger.info(
        "[chat_audio] request_id=%s session_id=%s audio_b64_len=%d mime=%s lang=%s mode_hint=%s voice_style=%s",
        request_id,
        session_id,
        len(req.audio_base64),
        req.mime_type,
        req.language,
        req.mode,
        req.voice_style,
    )

    # 1. Validate and decode base64 audio
    if not req.audio_base64 or not req.audio_base64.strip():
        raise HTTPException(status_code=400, detail="Field 'audio_base64' must not be empty.")

    try:
        audio_bytes = base64.b64decode(req.audio_base64, validate=True)
    except Exception as exc:
        logger.warning("[chat_audio] request_id=%s invalid base64: %s", request_id, exc)
        raise HTTPException(status_code=400, detail="Invalid 'audio_base64' payload: could not decode base64.")

    # 2. STT: audio -> transcript
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

    # Addition #3: reset phrases rotate the session id (server-side safe reset)
    if _normalize_reset_phrase(transcript) in _RESET_SESSION_PHRASES:
        new_session_id = str(uuid.uuid4())
        logger.info(
            "[chat_audio] request_id=%s session_id=%s reset_phrase=%r -> new_session_id=%s",
            request_id, session_id, transcript, new_session_id
        )
        session_id = new_session_id

    # 3. Optional mode override
    _apply_mode_hint(req.mode)

    # Normalize voice_style for both TARSCore and truncation
    normalized_voice_style = _normalize_voice_style(req.voice_style)

    # Addition #1/#2: load recent turns and inject context
    prior_turns = _load_session_turns(session_id=session_id, limit=12)
    context_block = _build_context_block(prior_turns, max_chars=1800)
    turn_count = len(prior_turns)

    # 4. TARSCore text pipeline
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
        logger.error("[chat_audio] request_id=%s session_id=%s Unexpected TARSCore error: %s", request_id, session_id, exc)
        raise HTTPException(status_code=500, detail="Unexpected error in TARS core while processing transcript.")

    reply = (reply or "").strip()
    if not reply:
        logger.error("[chat_audio] request_id=%s session_id=%s Empty reply for transcript: %r", request_id, session_id, transcript)
        raise HTTPException(status_code=500, detail="TARS generated an empty reply; this should not happen.")

    # Addition #5: observability – log turn_count + context chars
    logger.info(
        "[chat_audio] request_id=%s session_id=%s turn_count=%d context_chars=%d transcript_len=%d reply_len=%d",
        request_id,
        session_id,
        turn_count,
        len(context_block or ""),
        len(transcript),
        len(reply),
    )

    # 5. Apply voice reply length control BEFORE TTS
    truncated_reply, was_truncated = _truncate_for_voice_reply(reply, normalized_voice_style)
    if was_truncated:
        logger.info(
            "[chat_audio] request_id=%s session_id=%s Voice reply truncated (style=%s, orig_len=%d, new_len=%d).",
            request_id,
            session_id,
            normalized_voice_style,
            len(reply),
            len(truncated_reply),
        )
    reply = truncated_reply

    # 6. TTS: reply -> audio (non-fatal)
    try:
        audio_reply_bytes = tts_with_safeguards(
            text=reply,
            voice=None,          # later: 'tars_default' from config
            audio_format="mp3",
        )
        audio_reply_base64 = base64.b64encode(audio_reply_bytes).decode("ascii")
        error = None
    except ValueError as exc:
        logger.warning("[chat_audio] request_id=%s session_id=%s TTS validation error: %s", request_id, session_id, exc)
        audio_reply_base64 = ""
        error = f"tts_validation_error: {exc}"
    except RuntimeError as exc:
        logger.error("[chat_audio] request_id=%s session_id=%s TTS runtime error: %s", request_id, session_id, exc)
        audio_reply_base64 = ""
        error = "tts_failed"

    # Persist session turns (so memory survives across requests/restarts)
    try:
        _append_session_turn(session_id, "user", transcript)
        _append_session_turn(session_id, "assistant", reply)
    except Exception:
        pass

    latency_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "[chat_audio] request_id=%s session_id=%s OK latency_ms=%d mode=%s error=%s truncated=%s",
        request_id,
        session_id,
        latency_ms,
        mode,
        error,
        was_truncated,
    )

    return AudioChatResponse(
        reply=reply,
        mode=mode,
        transcript=transcript,
        audio_base64=audio_reply_base64,
        audio_mime_type="audio/mpeg",
        latency_ms=latency_ms,
        error=error,
    )


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

@app.post("/memory/add", response_model=MemoryItemResponse)
def add_memory(req: MemoryAddRequest) -> MemoryItemResponse:
    """
    Manually add a memory item (e.g., original idea, project, position shift).
    """
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
def list_recent_memory(limit: int = Query(20, ge=1, le=200)) -> List[MemoryItemResponse]:
    """
    Return the most recent memory items, any type.
    """
    items = get_recent_memory_items(limit=limit)
    return [
        MemoryItemResponse(
            id=item.id,
            created_at=item.created_at,
            type=item.type,
            label=item.label,
            content=item.content,
            source_conversation_id=item.source_conversation_id,
        )
        for item in items
    ]


@app.get("/memory/by_type", response_model=List[MemoryItemResponse])
def list_memory_by_type(
    type: str = Query(..., description="Memory type, e.g., 'original_idea', 'project', 'position_shift'"),
    limit: int = Query(50, ge=1, le=500),
) -> List[MemoryItemResponse]:
    """
    Return recent memory items of a given type.
    """
    items = get_memory_items_by_type(type=type, limit=limit)
    return [
        MemoryItemResponse(
            id=item.id,
            created_at=item.created_at,
            type=item.type,
            label=item.label,
            content=item.content,
            source_conversation_id=item.source_conversation_id,
        )
        for item in items
    ]


@app.get("/memory/search", response_model=List[MemoryItemResponse])
def search_memory(
    q: str = Query(..., description="Keyword to search in memory label/content"),
    limit: int = Query(20, ge=1, le=200),
) -> List[MemoryItemResponse]:
    """
    Keyword search in memory items (label + content).
    """
    items = search_memory_by_keyword(keyword=q, limit=limit)
    return [
        MemoryItemResponse(
            id=item.id,
            created_at=item.created_at,
            type=item.type,
            label=item.label,
            content=item.content,
            source_conversation_id=item.source_conversation_id,
        )
        for item in items
    ]



