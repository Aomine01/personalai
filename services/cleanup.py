"""
services/cleanup.py — TTL-based cleanup of generated temp files.

Runs once on bot startup (sweep) and can be scheduled as a periodic task.
Default TTL: 24 hours. Files older than that in TMP_DIR are deleted.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# File extensions this bot generates (only delete our own files)
_MANAGED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"
})

DEFAULT_TTL_HOURS: int = 24


def sweep(ttl_hours: int = DEFAULT_TTL_HOURS) -> int:
    """
    Delete files in TMP_DIR older than *ttl_hours*.

    Returns:
        Number of files deleted.
    """
    tmp = Path(settings.TMP_DIR)
    if not tmp.exists():
        return 0

    cutoff = time.time() - (ttl_hours * 3600)
    deleted = 0

    for fpath in tmp.iterdir():
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in _MANAGED_EXTENSIONS:
            continue  # never touch unrecognised files
        try:
            if fpath.stat().st_mtime < cutoff:
                fpath.unlink()
                deleted += 1
                logger.debug("Deleted stale file: %s", fpath.name)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", fpath.name, exc)

    if deleted:
        logger.info("Startup cleanup: removed %d stale file(s) from %s", deleted, tmp)
    else:
        logger.info("Startup cleanup: no stale files in %s", tmp)

    return deleted
