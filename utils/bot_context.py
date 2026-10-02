"""
utils/bot_context.py — Shared mutable state for all bot handlers.

All singletons that are defined here instead of in a service module
(to avoid circular imports) must be imported from here — never re-instantiated.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import MutableMapping

from config import settings

# Per-chat semaphore: prevents concurrent Ollama calls for the same user.
# Defined once here, imported everywhere — never re-instantiated in handlers.
_semaphores: MutableMapping[int, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(1))

# Max history turns to feed into LLM context
MAX_HISTORY: int = settings.CONTEXT_WINDOW * 2
