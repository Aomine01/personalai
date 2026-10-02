"""
services/news_image_fetcher.py — Safe article image downloader for news reports.

Design decisions and guardrails (document here to prevent silent future regressions):

1. DEFAULT BEHAVIOUR — NO AI GENERATION:
   fetch_article_image() returns a placeholder FetchedImage (local_path=None,
   source="placeholder") when no real image is available.  This is the default.
   AI-generated images via Pollinations are opt-in ONLY (allow_ai_fallback=False
   everywhere unless a caller explicitly enables it).  This is intentional:
   displaying an AI-generated photo alongside a real news headline is a
   misinformation risk.  Do NOT change the default to True.

2. WHEN AI FALLBACK IS ENABLED (allow_ai_fallback=True):
   - The Pollinations prompt is always prefixed with "editorial illustration,
     not photorealistic" so the output is visually distinguishable from a
     real photograph.
   - is_ai_generated=True on the returned FetchedImage is the canonical flag
     the report/Telegram layer uses to show the mandatory "AI-generated
     illustration" caption.  Do not rely on source=="pollinations" alone —
     use is_ai_generated.

3. BATCH SAFETY:
   fetch_article_image() never raises — all failures return FetchedImage with
   source="placeholder".  fetch_all_images() uses return_exceptions=False so
   gather propagation is simple, but individual failures are swallowed per item.
   One broken image URL must never stall or fail the whole batch.

4. COPYRIGHT SCOPE:
   Downloaded images stay local to the user's own Telegram chat / personal
   report.  No shared cache, no multi-user serving.  This is a personal-use
   tool and must remain so.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from config import settings

logger = logging.getLogger(__name__)

_IMAGE_DIR = Path(settings.TMP_DIR) / "news_images"

# Safety caps for image downloads
_DOWNLOAD_TIMEOUT_S: float = 8.0          # per-image download timeout
_MAX_IMAGE_BYTES: int = 5 * 1024 * 1024  # 5 MB hard cap — reject larger responses
_MIN_IMAGE_BYTES: int = 2_000             # <2 KB is likely an error placeholder

_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif",
})


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class FetchedImage:
    """
    Result of a single article image fetch attempt.

    Attributes:
        local_path:     Absolute path to the downloaded image file, or None if
                        no image is available (placeholder case).
        is_ai_generated: True only if the image was produced by an AI generator
                        (Pollinations).  The rendering layer MUST show a visible
                        "AI-generated illustration" caption whenever this is True.
                        Use this flag — do not rely on source=="pollinations" alone.
        source:         "tavily" | "pollinations" | "placeholder"
    """
    local_path: str | None
    is_ai_generated: bool
    source: str  # "tavily" | "pollinations" | "placeholder"


_PLACEHOLDER = FetchedImage(local_path=None, is_ai_generated=False, source="placeholder")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _ext_from_content_type(ct: str) -> str:
    ct = ct.split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    return mapping.get(ct, "jpg")


async def _download_image(url: str, prefix: str = "article") -> str | None:
    """
    Download *url* with safety guardrails:
      - Timeout: _DOWNLOAD_TIMEOUT_S seconds.
      - Content-Type must be image/* (rejects HTML error pages, redirects, etc.).
      - Size cap: abort and return None if > _MAX_IMAGE_BYTES or < _MIN_IMAGE_BYTES.

    Returns the local file path on success, or None on any failure.
    Never raises.
    """
    _ensure_dir()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT_S),
            follow_redirects=True,
        ) as client:
            # Stream the response so we can abort early on size
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.debug("Image download non-200 (%d): %s", resp.status_code, url)
                    return None

                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type not in _ALLOWED_CONTENT_TYPES:
                    logger.debug(
                        "Rejected non-image content-type %r for URL: %s", content_type, url
                    )
                    return None

                # Buffer response, abort if too large
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes(chunk_size=65_536):
                    total += len(chunk)
                    if total > _MAX_IMAGE_BYTES:
                        logger.debug(
                            "Image exceeds %d B cap, aborting: %s", _MAX_IMAGE_BYTES, url
                        )
                        return None
                    chunks.append(chunk)

        data = b"".join(chunks)
        if len(data) < _MIN_IMAGE_BYTES:
            logger.debug("Image too small (%d B), treating as error placeholder: %s", len(data), url)
            return None

        ext = _ext_from_content_type(content_type)
        out_path = _IMAGE_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
        out_path.write_bytes(data)
        logger.info("Downloaded news image %s (%d B) from %s", out_path.name, len(data), url)
        return str(out_path)

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        logger.debug("Image download network error (%s): %s", type(exc).__name__, url)
        return None
    except Exception as exc:
        logger.warning("Unexpected image download error for %s: %s", url, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_article_image(
    item: "SearchItem",  # type: ignore[name-defined]  # forward ref, avoid circular import
    allow_ai_fallback: bool = False,
) -> FetchedImage:
    """
    Fetch the best available image for a single news article.

    Strategy:
      1. If item.image_url is present: attempt download with timeout, Content-Type
         check, and size cap.  On success → FetchedImage(path, is_ai_generated=False,
         source="tavily").
      2. If that fails or is absent:
         - allow_ai_fallback=True: generate an editorial illustration via Pollinations.
           Prompt always includes "editorial illustration, not photorealistic".
           Returns FetchedImage(path, is_ai_generated=True, source="pollinations").
           MANDATORY: caller must display "AI-generated illustration" caption.
         - allow_ai_fallback=False (DEFAULT): return placeholder immediately.
           The report/send layer renders a styled gradient card instead of an image.

    Never raises — always returns a FetchedImage (worst case: placeholder).
    This ensures a failed image never stalls or crashes a batch gather.

    NOTE: Do NOT change allow_ai_fallback default to True — see module docstring.
    """
    # 1. Try the real Tavily-supplied image URL
    if item.image_url:
        path = await _download_image(item.image_url, prefix="tavily")
        if path:
            return FetchedImage(local_path=path, is_ai_generated=False, source="tavily")

    # 2. AI fallback — opt-in only
    if allow_ai_fallback:
        try:
            from services.image_service import generate_image  # noqa: PLC0415

            # Editorial-illustration style qualifier is mandatory — never photorealistic
            ai_prompt = (
                f"editorial illustration, not photorealistic — {item.title[:120]}, "
                "digital art style, news magazine cover art, clean composition"
            )
            ai_path = await generate_image(ai_prompt)
            logger.info("AI fallback illustration generated for: %r", item.title[:60])
            return FetchedImage(local_path=ai_path, is_ai_generated=True, source="pollinations")
        except Exception as exc:
            logger.warning("AI image fallback failed for %r: %s", item.title[:60], exc)

    # 3. Placeholder — no image available
    return _PLACEHOLDER


async def fetch_all_images(
    items: list["SearchItem"],  # type: ignore[name-defined]
    allow_ai_fallback: bool = False,
) -> list[FetchedImage]:
    """
    Fetch images for all items concurrently.

    Uses asyncio.gather; individual failures are handled inside
    fetch_article_image() and never propagate (each item always returns a
    FetchedImage, defaulting to placeholder on failure).

    Args:
        items:            List of SearchItem from web_search_rich().
        allow_ai_fallback: Passed through to fetch_article_image(). Default OFF.
                           See module docstring for why this must stay False by default.

    Returns:
        List of FetchedImage, same length and order as items.
    """
    return list(
        await asyncio.gather(
            *(fetch_article_image(item, allow_ai_fallback=allow_ai_fallback) for item in items)
        )
    )
