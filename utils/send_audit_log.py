"""
utils/send_audit_log.py — Lightweight append-only audit trail for all UserBot sent messages.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

AUDIT_LOG_PATH = Path("/tmp/aiagent/send_audit.log")


def log_send(
    user_id: int,
    recipient: Any,
    text: str,
    source: str,
    confidence: int | None = None,
    success: bool = True,
    error: Any = None,
) -> None:
    """Log every confirmed send attempt (both success and failure) with full metadata."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "user_id": user_id,
            "recipient": str(recipient),
            "text": text,
            "source": source,
            "confidence": confidence,
            "success": success,
            "error": str(error) if error else None,
        }
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Never crash the main application if writing audit log fails
        pass
