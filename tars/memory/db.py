# tars/memory/db.py

import sqlite3
from pathlib import Path

from tars.config.settings import load_settings

_settings = load_settings()


def get_db_path() -> Path:
    return Path(_settings.db_path)


def get_connection() -> sqlite3.Connection:
    """
    Return a SQLite connection.
    Uses Row factory to allow dict-like access later if needed.
    Caller is responsible for closing.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the database schema if it does not exist.
    Safe to call multiple times.
    """
    conn = get_connection()
    cur = conn.cursor()

    # conversations: one per TARS session
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            mode TEXT NOT NULL,
            summary TEXT
        )
        """
    )

    # messages: logs of user and TARS messages
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,            -- 'user' or 'assistant'
            timestamp TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        """
    )

    # memory_items: distilled long-term items (future use)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            type TEXT NOT NULL,            -- 'original_idea', 'project', 'position_shift', etc.
            label TEXT,
            content TEXT NOT NULL,
            source_conversation_id INTEGER,
            FOREIGN KEY (source_conversation_id) REFERENCES conversations (id)
        )
        """
    )

    conn.commit()
    conn.close()

