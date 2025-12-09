# tars/memory/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def now_iso() -> str:
    return datetime.utcnow().strftime(ISO_FMT)


@dataclass
class Conversation:
    id: Optional[int]
    started_at: str
    ended_at: Optional[str]
    mode: str
    summary: Optional[str] = None


@dataclass
class Message:
    id: Optional[int]
    conversation_id: int
    role: str            # 'user' or 'assistant'
    timestamp: str
    content: str


@dataclass
class MemoryItem:
    id: Optional[int]
    created_at: str
    type: str            # 'original_idea', 'project', 'position_shift', etc.
    label: Optional[str]
    content: str
    source_conversation_id: Optional[int]

