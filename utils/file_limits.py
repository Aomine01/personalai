"""
utils/file_limits.py — Centralised file-upload validation.

Call `validate_upload()` BEFORE downloading any Telegram attachment.
Telegram provides `file_size` in the update metadata — use it for the
size check so we reject oversized files without ever touching the network.
"""
from __future__ import annotations

from pathlib import Path

from services.exceptions import FileTooLargeError

MAX_FILE_SIZE_MB: int = 20
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".xlsx", ".xls", ".csv"
})

MAX_PDF_PAGES_TO_READ: int = 50
MAX_CHARS_FOR_LLM: int = 40_000     # ~10 000 tokens — safe for 16k num_ctx


def validate_upload(file_size_bytes: int | None, filename: str) -> tuple[bool, str | None]:
    """
    Validate a Telegram upload before downloading.

    Args:
        file_size_bytes: Value from `document.file_size` (may be None if unknown).
        filename:        Original filename as reported by Telegram.

    Returns:
        (is_valid, error_message)
        error_message is None when is_valid is True.

    Note:
        Size is checked FIRST so we never block on extension for a 500MB file.
    """
    # Size check
    if file_size_bytes is not None and file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, (
            f"❌ File too large ({size_mb:.1f} MB). "
            f"The Telegram Bot API limits downloads to {MAX_FILE_SIZE_MB} MB. "
            "Please compress or split the file."
        )

    # Extension check
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        return False, (
            f"❌ Unsupported file type `{ext}`.\n"
            f"I can read: {supported}"
        )

    return True, None
