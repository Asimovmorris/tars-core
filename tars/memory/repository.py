# tars/memory/repository.py

from typing import Optional, List

from tars.memory.db import get_connection, init_db
from tars.memory.models import Conversation, Message, MemoryItem, now_iso


def initialize() -> None:
    """
    Initialize DB schema. Call once at startup.
    """
    init_db()


def start_conversation(mode: str) -> int:
    """
    Create a new conversation row and return its id.
    """
    conn = get_connection()
    cur = conn.cursor()

    started_at = now_iso()
    cur.execute(
        """
        INSERT INTO conversations (started_at, mode)
        VALUES (?, ?)
        """,
        (started_at, mode),
    )
    conv_id = cur.lastrowid
    conn.commit()
    conn.close()
    return conv_id


def end_conversation(conversation_id: int, summary: Optional[str] = None) -> None:
    """
    Mark a conversation as ended; optionally store a summary.
    """
    conn = get_connection()
    cur = conn.cursor()

    ended_at = now_iso()
    cur.execute(
        """
        UPDATE conversations
        SET ended_at = ?, summary = ?
        WHERE id = ?
        """,
        (ended_at, summary, conversation_id),
    )
    conn.commit()
    conn.close()


def save_message(conversation_id: int, role: str, content: str) -> int:
    """
    Insert a message row and return its id.
    """
    conn = get_connection()
    cur = conn.cursor()

    timestamp = now_iso()
    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, timestamp, content)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, role, timestamp, content),
    )
    msg_id = cur.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def save_memory_item(
    type: str,
    label: Optional[str],
    content: str,
    source_conversation_id: Optional[int] = None,
) -> int:
    """
    Insert a memory item row and return its id.
    """
    conn = get_connection()
    cur = conn.cursor()

    created_at = now_iso()
    cur.execute(
        """
        INSERT INTO memory_items (created_at, type, label, content, source_conversation_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (created_at, type, label, content, source_conversation_id),
    )
    mem_id = cur.lastrowid
    conn.commit()
    conn.close()
    return mem_id


def get_recent_conversations(limit: int = 10) -> List[Conversation]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, started_at, ended_at, mode, summary
        FROM conversations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        Conversation(
            id=row["id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            mode=row["mode"],
            summary=row["summary"],
        )
        for row in rows
    ]


def get_recent_memory_items(limit: int = 20) -> List[MemoryItem]:
    """
    Return the most recent memory items (any type), newest first.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, type, label, content, source_conversation_id
        FROM memory_items
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        MemoryItem(
            id=row["id"],
            created_at=row["created_at"],
            type=row["type"],
            label=row["label"],
            content=row["content"],
            source_conversation_id=row["source_conversation_id"],
        )
        for row in rows
    ]


def get_memory_items_by_type(type: str, limit: int = 50) -> List[MemoryItem]:
    """
    Return recent memory items of a given type, newest first.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, type, label, content, source_conversation_id
        FROM memory_items
        WHERE type = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (type, limit),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        MemoryItem(
            id=row["id"],
            created_at=row["created_at"],
            type=row["type"],
            label=row["label"],
            content=row["content"],
            source_conversation_id=row["source_conversation_id"],
        )
        for row in rows
    ]


def search_memory_by_keyword(keyword: str, limit: int = 20) -> List[MemoryItem]:
    """
    Simple LIKE-based search for now. Later we can switch to full-text search.
    """
    conn = get_connection()
    cur = conn.cursor()

    pattern = f"%{keyword}%"
    cur.execute(
        """
        SELECT id, created_at, type, label, content, source_conversation_id
        FROM memory_items
        WHERE content LIKE ? OR label LIKE ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (pattern, pattern, limit),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        MemoryItem(
            id=row["id"],
            created_at=row["created_at"],
            type=row["type"],
            label=row["label"],
            content=row["content"],
            source_conversation_id=row["source_conversation_id"],
        )
        for row in rows
    ]

