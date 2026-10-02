"""
services/exceptions.py — Typed exceptions for every failure domain.

Every handler must catch its relevant exception type and produce a
user-facing Telegram message. Never let a raw traceback reach the user,
and never swallow silently with bare `except Exception: pass`.
"""


class ImageGenerationError(Exception):
    """Raised when all image-generation backends (Pollinations / OpenAI / Stability) fail."""


class TableGenerationError(Exception):
    """Raised when the LLM cannot produce valid, schema-conformant JSON for a spreadsheet."""


class UnreadableFileError(Exception):
    """
    Raised when a file cannot be read due to:
      - password protection
      - corruption / invalid format
      - empty / image-only content (no extractable text)
    """


class StyleExtractionError(Exception):
    """Raised when a reference file's style profile cannot be extracted."""


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the Telegram Bot API 20MB download cap."""


class ReportGenerationError(Exception):
    """Raised when news/search report PDF generation fails (render or template error)."""
