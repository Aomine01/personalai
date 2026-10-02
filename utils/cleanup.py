"""
utils/cleanup.py — TTL-based cleanup of generated temp files.

sweep_tmp_dir() is called once on bot startup and can be scheduled
via python-telegram-bot's JobQueue for periodic sweeps.

Only deletes files with extensions produced by this bot — never touches
unrecognised files even if they appear in the same directory.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_MANAGED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg",
})

DEFAULT_TTL_HOURS: int = 24


def sweep_tmp_dir(max_age_hours: int = DEFAULT_TTL_HOURS) -> int:
    """
    Recursively delete files under TMP_DIR older than *max_age_hours*.

    Returns:
        Number of files deleted.
    """
    tmp = Path(settings.TMP_DIR)
    if not tmp.exists():
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0

    for fpath in tmp.rglob("*"):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in _MANAGED_EXTENSIONS:
            continue
        try:
            if fpath.stat().st_mtime < cutoff:
                fpath.unlink()
                deleted += 1
                logger.debug("Deleted stale file: %s", fpath)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", fpath, exc)

    if deleted:
        logger.info("Cleanup: removed %d stale file(s) (TTL=%dh)", deleted, max_age_hours)
    else:
        logger.info("Cleanup: no stale files found (TTL=%dh)", max_age_hours)

    return deleted


async def periodic_sweep(context) -> None:  # type: ignore[type-arg]
    """
    Async wrapper for use as a python-telegram-bot JobQueue callback.
    Schedule with: app.job_queue.run_repeating(periodic_sweep, interval=3600)
    """
    sweep_tmp_dir()
