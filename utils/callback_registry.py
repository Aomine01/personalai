import uuid
import time
from dataclasses import dataclass, field

@dataclass
class CallbackContext:
    action: str                  # e.g. "regen_image", "restyle_image", "regen_excel", "edit_excel", "news_more", "news_report_button", "file_followup"
    payload: dict                # arbitrary context needed to redo the action
    created_at: float = field(default_factory=time.time)

class CallbackRegistry:
    """In-memory store mapping short IDs -> full context, since callback_data is 64 bytes max.
    TTL-based eviction prevents unbounded growth."""
    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, CallbackContext] = {}
        self.ttl = ttl_seconds

    def register(self, action: str, payload: dict) -> str:
        self._evict_expired()
        ref_id = uuid.uuid4().hex[:12]   # short enough to fit in callback_data with room for the action prefix
        self._store[ref_id] = CallbackContext(action=action, payload=payload)
        return ref_id

    def resolve(self, ref_id: str) -> CallbackContext | None:
        self._evict_expired()
        return self._store.get(ref_id)

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v.created_at > self.ttl]
        for k in expired:
            del self._store[k]

callback_registry = CallbackRegistry()
