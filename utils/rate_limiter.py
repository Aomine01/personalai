"""
utils/rate_limiter.py — Per-user request rate limiter.

Implements a simple token-bucket rate limiter.  Each user gets a bucket of
`capacity` tokens that refills at `refill_rate` tokens per second.  Each heavy
handler call (image gen, web search + report, pdf gen, etc.) consumes one token.

Usage in any heavy handler::

    if not rate_limiter.allow(user_id):
        await update.message.reply_text(
            "⏳ You're sending requests too fast. Please wait a moment."
        )
        return

This is a best-effort in-process guard (not persistent across restarts) and is
intentionally lightweight — the per-chat asyncio.Semaphore in main.py provides
the strict concurrency guarantee; the rate limiter adds a sustained-rate cap.
"""
from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class _Bucket:
    """Single user's token bucket."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens: float = capacity
        self.last_refill: float = time.monotonic()


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Args:
        capacity:     Maximum tokens in the bucket (also the initial fill).
        refill_rate:  Tokens added per second (fractional allowed).

    Example — 5 requests/minute::

        rl = RateLimiter(capacity=5, refill_rate=5/60)
    """

    def __init__(self, capacity: float = 5.0, refill_rate: float = 5 / 60) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate  # tokens per second
        self._buckets: dict[int, _Bucket] = defaultdict(lambda: _Bucket(capacity))
        self._lock = Lock()

    def allow(self, user_id: int) -> bool:
        """
        Consume one token for *user_id*.

        Returns True if the request is allowed, False if the user is over-rate.
        Thread-safe; safe to call from asyncio tasks (no I/O, short critical section).
        """
        with self._lock:
            bucket = self._buckets[user_id]
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            # Refill tokens proportional to elapsed time
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False

    def reset(self, user_id: int) -> None:
        """Reset a specific user's bucket (useful in tests)."""
        with self._lock:
            self._buckets.pop(user_id, None)


# Module-level singleton — 5 heavy requests per user per minute
rate_limiter = RateLimiter(capacity=5.0, refill_rate=5 / 60)
