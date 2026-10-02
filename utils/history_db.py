"""
utils/history_db.py — Async SQLite database layer for conversation history and settings.

Uses aiosqlite for native async non-blocking access.
WAL mode + 32MB cache are set at startup for maximum throughput.
"""
from __future__ import annotations

import sqlite3
import aiosqlite

from config import settings

DB_PATH = settings.DB_PATH
_MAX_CHAR_BUDGET = 32_000  # ~8,000 tokens


# ── One-time synchronous init (called before the event loop starts) ───────────

def init_db() -> None:
    """Create tables and run migrations synchronously at startup."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA cache_size=-32768;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now')),
                model      TEXT
            );
        """)

        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_history)").fetchall()]
        if "timestamp" in cols and "created_at" not in cols:
            conn.execute("ALTER TABLE conversation_history RENAME COLUMN timestamp TO created_at")

        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_history)").fetchall()]
        if "model" not in cols:
            conn.execute("ALTER TABLE conversation_history ADD COLUMN model TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON conversation_history(chat_id);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id  INTEGER NOT NULL,
                key      TEXT    NOT NULL,
                value    TEXT    NOT NULL,
                PRIMARY KEY (chat_id, key)
            );
        """)

        conn.commit()
    finally:
        conn.close()


# ── History ───────────────────────────────────────────────────────────────────

async def get_history(chat_id: int, limit: int) -> list[dict[str, str]]:
    """Return up to *limit* recent messages for *chat_id*, oldest-first."""
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            "SELECT role, content FROM conversation_history WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

    result: list[dict[str, str]] = []
    char_count = 0
    for role, content in rows:
        char_count += len(content)
        if char_count > _MAX_CHAR_BUDGET:
            break
        result.append({"role": role, "content": content})

    result.reverse()
    return result


async def append_history(chat_id: int, role: str, content: str, model: str | None = None) -> None:
    """Append a message turn to the conversation history."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO conversation_history (chat_id, role, content, model) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, model),
        )
        await conn.commit()


async def clear_history(chat_id: int) -> None:
    """Delete all history for *chat_id*."""
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("DELETE FROM conversation_history WHERE chat_id = ?", (chat_id,))
        await conn.commit()


async def delete_last_assistant_message(chat_id: int) -> bool:
    """Delete the most recent assistant message if it exists. Returns True if deleted."""
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            "SELECT id, role FROM conversation_history WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row and row[1] == "assistant":
            await conn.execute("DELETE FROM conversation_history WHERE id = ?", (row[0],))
            await conn.commit()
            return True
    return False


async def get_last_model(chat_id: int) -> str | None:
    """Return the model name used for the most recent assistant turn."""
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            "SELECT model FROM conversation_history WHERE chat_id = ? AND role = 'assistant' AND model IS NOT NULL ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


# ── Per-chat settings ─────────────────────────────────────────────────────────

async def get_setting(chat_id: int, key: str) -> str | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            "SELECT value FROM chat_settings WHERE chat_id = ? AND key = ?",
            (chat_id, key),
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


async def set_setting(chat_id: int, key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO chat_settings (chat_id, key, value) VALUES (?, ?, ?) ON CONFLICT(chat_id, key) DO UPDATE SET value = excluded.value",
            (chat_id, key, value),
        )
        await conn.commit()


# ── Global settings (chat_id = 0) ─────────────────────────────────────────────

async def get_global_setting(key: str) -> str | None:
    async with aiosqlite.connect(DB_PATH) as conn:
        async with conn.execute(
            "SELECT value FROM chat_settings WHERE chat_id = 0 AND key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


async def set_global_setting(key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO chat_settings (chat_id, key, value) VALUES (0, ?, ?) ON CONFLICT(chat_id, key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await conn.commit()
