"""
services/gemini_vision_service.py — Google Gemini Flash vision integration.

Uses the new google-genai SDK (google-generativeai is deprecated as of 2025).
Sends images to Gemini Flash, which excels at:
  - Recognising real people / celebrities
  - Reading text in images (OCR)
  - Understanding complex scenes

Free tier quota: 15 RPM, 1 500 req/day, 1M tokens/min.
Docs: https://ai.google.dev/gemini-api/docs
"""
from __future__ import annotations

import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)

# Model priority list — tried in order until one succeeds
_MODELS = [
    "models/gemini-3.5-flash",
    "models/gemini-3.5-flash-lite",
    "models/gemini-2.5-flash",
]


async def vision_chat(user_message: str, image_path: str) -> str:
    """
    Describe / analyse *image_path* in response to *user_message*.

    Falls back to a friendly error string (never raises) so callers
    don't need extra error handling.
    """
    if not settings.GEMINI_API_KEY:
        return (
            "⚠️ Gemini vision is not configured yet.\n\n"
            "To enable it:\n"
            "1. Go to https://aistudio.google.com/app/apikey (free)\n"
            "2. Create an API key\n"
            "3. Add `GEMINI_API_KEY=<your_key>` to your `.env` file and restart the bot."
        )

    try:
        from google import genai          # noqa: PLC0415
        import PIL.Image                  # noqa: PLC0415
    except ImportError:
        logger.error("google-genai or Pillow not installed")
        return (
            "⚠️ Vision dependency missing. Run:\n"
            "`pip install google-genai Pillow`"
        )

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        image = PIL.Image.open(image_path)

        # Give Gemini richer context so it answers the user's question directly
        prompt = (
            f"{user_message}\n\n"
            "If the image contains a recognisable person or celebrity, name them. "
            "If it contains text, read it. Be thorough and helpful."
        )

        last_exc: Exception | None = None
        for model_name in _MODELS:
            try:
                logger.info("Trying Gemini model: %s", model_name)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda m=model_name: client.models.generate_content(
                        model=m,
                        contents=[prompt, image],
                    ),
                )
                logger.info("Vision success with model: %s", model_name)
                return response.text.strip()
            except Exception as exc:
                logger.warning("Model %s failed: %s", model_name, exc)
                last_exc = exc
                # Small delay before trying next model
                await asyncio.sleep(1)

        # All models failed
        logger.exception("All Gemini models failed. Last error: %s", last_exc)
        return (
            "⚠️ Vision analysis is temporarily unavailable (Gemini servers are busy). "
            "Please try again in a moment."
        )

    except Exception as exc:
        logger.exception("Gemini vision error: %s", exc)
        return f"⚠️ Vision analysis failed: {exc}"
