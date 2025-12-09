# tars/core/chat.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from enum import Enum

from tars.core.modes import ReasoningMode, random_mode
from tars.clients.openai_client import chat_with_tars
from tars.memory.repository import (
    initialize as init_memory,
    start_conversation,
    end_conversation,
    save_message,
    save_memory_item,
)
from tars.utils.logging import get_logger

logger = get_logger(__name__)

MEMORY_BLOCK_START = "<<MEMORY_BLOCK>>"
MEMORY_BLOCK_END = "<<END_MEMORY_BLOCK>>"

ALLOWED_MEMORY_TYPES = {
    "original_idea",
    "project",
    "position_shift",
    "todo",
    "other",
}

# SAFEGUARD 1: Bounds on user text + history size
MAX_USER_TEXT_CHARS = 8000          # hard cap on a single user message
MAX_HISTORY_MESSAGES = 40           # rolling window to avoid unbounded growth


class ResponseChannel(str, Enum):
    """
    Indicates where the reply will be consumed.

    TEXT  : normal chat (detailed, essay answers allowed).
    VOICE : spoken response (should be concise and conversational).
    """
    TEXT = "text"
    VOICE = "voice"


@dataclass
class ConversationState:
    mode: ReasoningMode
    history: List[Dict[str, str]] = field(default_factory=list)
    conversation_id: Optional[int] = None
    last_channel: ResponseChannel = ResponseChannel.TEXT  # for logging / introspection


class TARSCore:
    def __init__(self) -> None:
        # Ensure DB schema is ready
        init_memory()

        # Start a new conversation with a random mode
        mode = random_mode()
        conv_id = start_conversation(mode.value)

        self.state = ConversationState(
            mode=mode,
            history=[],
            conversation_id=conv_id,
            last_channel=ResponseChannel.TEXT,
        )

    def reset_conversation(self) -> None:
        """
        End current conversation (without a summary for now) and start a new one.
        """
        if self.state.conversation_id is not None:
            end_conversation(self.state.conversation_id)

        mode = random_mode()
        conv_id = start_conversation(mode.value)
        self.state = ConversationState(
            mode=mode,
            history=[],
            conversation_id=conv_id,
            last_channel=ResponseChannel.TEXT,
        )

    def set_mode(self, mode: ReasoningMode) -> None:
        self.state.mode = mode
        # Note: we could also update the conversations.mode column if desired.

    def _update_mode_from_text(self, text: str) -> None:
        """
        Interpret user commands that request a mode change.
        """
        lowered = text.lower()
        if "switch style" in lowered:
            self.state.mode = random_mode()
        elif "analyst mode" in lowered:
            self.state.mode = ReasoningMode.ANALYST
        elif "critic mode" in lowered:
            self.state.mode = ReasoningMode.CRITIC
        elif "synthesizer mode" in lowered or "synthetic mode" in lowered:
            self.state.mode = ReasoningMode.SYNTHESIZER

    # ---------- MEMORY BLOCK PARSING ----------

    def _extract_memory_from_reply(
        self, reply: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Look for a MEMORY_BLOCK at the end of the reply, parse it (if present),
        and return (visible_reply, memory_items).

        visible_reply: the reply with the MEMORY_BLOCK removed.
        memory_items: list of dicts with keys type/label/content (if any).
        """
        start_idx = reply.rfind(MEMORY_BLOCK_START)
        end_idx = reply.rfind(MEMORY_BLOCK_END)

        # No well-formed markers -> nothing to do
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return reply.strip(), []

        # Extract JSON text
        json_text = reply[start_idx + len(MEMORY_BLOCK_START) : end_idx].strip()

        # Remove the block from the visible reply
        visible = (reply[:start_idx] + reply[end_idx + len(MEMORY_BLOCK_END) :]).strip()

        if not json_text:
            logger.warning("Empty MEMORY_BLOCK JSON text.")
            return visible, []

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MEMORY_BLOCK JSON: {e}")
            return visible, []

        if not isinstance(data, list):
            logger.error("MEMORY_BLOCK JSON is not a list.")
            return visible, []

        items: List[Dict[str, str]] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content")
            if not content:
                continue
            mem_type = raw.get("type") or "other"
            if mem_type not in ALLOWED_MEMORY_TYPES:
                mem_type = "other"
            label = raw.get("label") or None

            items.append(
                {
                    "type": mem_type,
                    "label": label,
                    "content": content,
                }
            )

        return visible, items

    def _generate_summary(self) -> Optional[str]:
        """
        Ask the model for a concise summary of the conversation.

        We use the existing system prompt and current mode. We do NOT
        store the summary as a message; it is only saved in the DB.
        """
        if not self.state.history:
            return None

        # Use only the last N messages to keep context bounded
        N = min(len(self.state.history), MAX_HISTORY_MESSAGES)
        history_tail = self.state.history[-N:]

        summary_instruction = (
            "Please summarize this entire conversation for my future recall in 3–5 bullet points. "
            "Focus on: (1) core questions, (2) key ideas or decisions, (3) any explicit next steps. "
            "Do not introduce new ideas or speculation; just summarize what was actually discussed."
        )

        # Append the summary request as the last user message
        messages = history_tail + [{"role": "user", "content": summary_instruction}]

        try:
            summary = chat_with_tars(messages)
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            return None

    # ---------- VOICE STYLE HINTING ----------

    def _build_voice_style_hint(self, voice_style: Optional[str]) -> str:
        """
        Build a system message that instructs the model how to respond
        when the reply is going to be spoken (VOICE channel).

        This is a *hint*, not a hard truncation. Actual length control
        may be enforced separately in the audio pipeline.

        voice_style examples (case-insensitive):
            - None / "brief": short, 2–5 sentences, direct.
            - "story": slightly more narrative but still concise.
            - "technical": concise but with higher density of terms.
        """
        style = (voice_style or "").strip().lower()

        if style in ("", "brief", "default"):
            return (
                "You are responding in VOICE mode for spoken output. "
                "Answer concisely and conversationally: typically 2–5 sentences, "
                "no long bullet lists, and avoid overly dense academic phrasing. "
                "Prioritize clarity and pacing suitable for listening."
            )
        elif style == "story":
            return (
                "You are responding in VOICE mode for spoken output in story style. "
                "Use vivid but concise storytelling: typically 4–7 sentences, "
                "with clear narrative flow. Avoid long enumerations or dense jargon."
            )
        elif style == "technical":
            return (
                "You are responding in VOICE mode for spoken output in technical style. "
                "Be concise but information-dense: typically 3–6 sentences, "
                "using precise terminology but avoiding very long paragraphs or lists."
            )
        else:
            # Unknown style: fall back but log for later tuning.
            logger.info("Unrecognized voice_style=%r; using default brief voice hint.", voice_style)
            return (
                "You are responding in VOICE mode for spoken output. "
                "Answer concisely (2–5 sentences), focusing on the key point first."
            )

    # ---------- MAIN USER ENTRY POINT ----------

    def _trim_history_if_needed(self) -> None:
        """
        SAFEGUARD 2:
        Keep only a rolling window of the most recent messages in memory
        to avoid unbounded growth and overly large context for the model.
        """
        if len(self.state.history) > MAX_HISTORY_MESSAGES:
            # Drop the oldest messages, keep the most recent
            excess = len(self.state.history) - MAX_HISTORY_MESSAGES
            logger.info(
                "Trimming conversation history: dropping %d oldest messages (kept %d).",
                excess,
                MAX_HISTORY_MESSAGES,
            )
            self.state.history = self.state.history[-MAX_HISTORY_MESSAGES :]

    def process_user_text(
        self,
        text: str,
        channel: ResponseChannel = ResponseChannel.TEXT,
        voice_style: Optional[str] = None,
    ) -> str:
        """
        Core entry point: given user text, update history, persist to DB,
        handle memory extraction, and get TARS reply text (visible part).

        This is used both by the text API and by the audio pipeline
        (where 'text' is the STT transcript).

        Parameters
        ----------
        text : str
            User text input (from keyboard or STT).
        channel : ResponseChannel
            Where the reply will be consumed: TEXT (default) or VOICE.
            VOICE will cause a 'voice style' hint to be added as a system message.
        voice_style : Optional[str]
            Optional style hint for VOICE channel, e.g. "brief", "story", "technical".
        """
        self.state.last_channel = channel

        # SAFEGUARD 1 (part 2): sanitize and bound user text
        cleaned = (text or "").strip()
        if not cleaned:
            # For audio, an empty transcript is possible; better to respond cleanly.
            logger.warning("Received empty or whitespace-only user text; ignoring.")
            raise RuntimeError("No meaningful input text was provided to TARS.")

        if len(cleaned) > MAX_USER_TEXT_CHARS:
            logger.warning(
                "User text length %d exceeds MAX_USER_TEXT_CHARS=%d; truncating.",
                len(cleaned),
                MAX_USER_TEXT_CHARS,
            )
            cleaned = cleaned[:MAX_USER_TEXT_CHARS]

        # Adjust mode based on user directive, if any
        self._update_mode_from_text(cleaned)

        # Persist user message
        if self.state.conversation_id is not None:
            save_message(self.state.conversation_id, role="user", content=cleaned)

        # Build messages for the model
        mode_hint = {
            "role": "system",
            "content": f"Current reasoning mode: {self.state.mode.value}. Maintain this stance in your reply.",
        }

        system_messages: List[Dict[str, str]] = [mode_hint]

        # NEW: Voice style hint when channel is VOICE
        if channel == ResponseChannel.VOICE:
            voice_hint = self._build_voice_style_hint(voice_style)
            system_messages.append({"role": "system", "content": voice_hint})
            logger.info(
                "Applying voice style hint (channel=VOICE, style=%r) for user text length %d.",
                voice_style,
                len(cleaned),
            )

        # Apply history trimming before adding new messages
        self._trim_history_if_needed()

        messages = self.state.history + system_messages + [
            {"role": "user", "content": cleaned}
        ]

        # SAFEGUARD 3: defensive model call with explicit logging
        try:
            raw_reply = chat_with_tars(messages)
        except Exception as e:
            logger.error("chat_with_tars failed in process_user_text: %s", e)
            # Re-raise as RuntimeError so API layer can convert to HTTP error cleanly
            raise RuntimeError("TARS failed to generate a reply from the model.") from e

        # Extract memory items and visible reply
        visible_reply, mem_items = self._extract_memory_from_reply(raw_reply)
        visible_reply = (visible_reply or "").strip()

        if not visible_reply:
            logger.error(
                "Model returned empty visible_reply after MEMORY_BLOCK extraction. "
                "Raw reply was: %r",
                raw_reply,
            )
            raise RuntimeError(
                "TARS generated an empty reply after processing; this should not happen."
            )

        # Update in-memory history (without the mode/voice hints, and without the memory block)
        self.state.history.append({"role": "user", "content": cleaned})
        self.state.history.append({"role": "assistant", "content": visible_reply})

        # Persist TARS reply (visible part)
        if self.state.conversation_id is not None:
            save_message(
                self.state.conversation_id,
                role="assistant",
                content=visible_reply,
            )

        # Persist any memory items suggested by the model
        if self.state.conversation_id is not None and mem_items:
            for mem in mem_items:
                try:
                    save_memory_item(
                        type=mem["type"],
                        label=mem["label"],
                        content=mem["content"],
                        source_conversation_id=self.state.conversation_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to save memory item: {e}")

        return visible_reply

    def close(self, summary: Optional[str] = None) -> None:
        """
        Gracefully end current conversation with an optional summary.
        If no summary is provided, we attempt to generate one automatically.
        """
        if self.state.conversation_id is not None:
            if summary is None:
                summary = self._generate_summary()

            try:
                end_conversation(self.state.conversation_id, summary=summary)
            except Exception as e:
                logger.error(f"Failed to end conversation with summary: {e}")

            self.state.conversation_id = None

