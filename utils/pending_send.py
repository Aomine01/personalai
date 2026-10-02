"""
utils/pending_send.py — In-memory registry for pending outbound Telegram UserBot messages.

Enforces the safety confirmation gate: no message is sent to a contact without explicit
user confirmation displaying the exact final recipient and message text.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class PendingSend:
    user_id: int                      # the bot-chat user (owner), not the recipient
    recipient_entity_id: int | str    # Telegram entity id/username resolved by find_contact
    recipient_display_name: str       # human-readable name shown in the confirmation card
    match_confidence: int | None      # the fuzzy-match score, shown to user if fuzzy-matched
    final_text: str                   # the exact text that will be sent if confirmed
    source: str                       # "quick_suggestion" | "custom_text" | "custom_voice" | "voice_auto_route" | "ai_draft"
    created_at: float = field(default_factory=time.time)


class PendingSendRegistry:
    """One pending send per user at a time. A new pending send from the same user
    overwrites any prior unconfirmed one (prevents stale confirmations firing late)."""
    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[int, PendingSend] = {}
        self.ttl = ttl_seconds

    def set(self, pending: PendingSend) -> str:
        ref_id = uuid.uuid4().hex[:12]
        self._store[pending.user_id] = pending
        return ref_id

    def get(self, user_id: int) -> PendingSend | None:
        p = self._store.get(user_id)
        if p and (time.time() - p.created_at > self.ttl):
            del self._store[user_id]
            return None
        return p

    def clear(self, user_id: int) -> None:
        self._store.pop(user_id, None)


pending_send_registry = PendingSendRegistry(ttl_seconds=300)  # 5 min TTL
