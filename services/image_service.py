"""
services/image_service.py — Async image generation service.

Optimised for maximum speed:
  - Bypasses local Ollama prompt enhancement by default to eliminate local LLM inference lag (~10-30s).
  - Appends high-quality, lightweight render keywords (photorealistic, 8k, detailed) in Python instantly.
  - Uses the fast, unauthenticated Pollinations.ai endpoint.
  - Tries 'sana' model first ( MIT/NVIDIA, extremely fast, generates in ~2s).
  - Falls back to 'flux' on timeout or failure.
"""
from __future__ import annotations

import logging
import os
import random
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx

from config import settings
from services.exceptions import ImageGenerationError

logger = logging.getLogger(__name__)

_IMAGES_DIR = Path(settings.TMP_DIR) / "images"
_TIMEOUT = httpx.Timeout(20.0)      # Fast timeout since we want quick feedback
_MAX_RETRIES = 1

# Models targeted on the free image.pollinations.ai endpoint
_POLLINATIONS_MODELS = [
    ("sana", 1024, 1024),   # Sana (extremely fast, MIT/NVIDIA, high quality, ~2s)
    ("flux", 1024, 1024),   # Flux (high quality baseline, slower)
]

# Standard quality booster suffix to append to raw prompts
_QUALITY_SUFFIX = ", photorealistic, highly detailed, 8k resolution, cinematic lighting"


def _ensure_dirs() -> None:
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ── Pollinations.ai (multi-model tiered) ──────────────────────────────────────

async def _pollinations_generate(raw_prompt: str) -> str:
    """
    Download image from Pollinations.ai.
    Appends quality boosters, tries 'sana' first for speed, then 'flux'.
    """
    _ensure_dirs()
    
    # Append high-quality aesthetic keywords instantly (0ms latency)
    enhanced_prompt = raw_prompt.strip() + _QUALITY_SUFFIX
    encoded = quote(enhanced_prompt, safe="")
    
    last_exc: Exception | None = None

    for model_name, width, height in _POLLINATIONS_MODELS:
        for attempt in range(_MAX_RETRIES + 1):
            seed = random.randint(1, 99_999)
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?model={model_name}"
                f"&width={width}&height={height}"
                f"&seed={seed}&nologo=true"
            )
            logger.info(
                "Pollinations [%s] attempt %d/%d  seed=%d",
                model_name, attempt + 1, _MAX_RETRIES + 1, seed,
            )
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    if "image" not in content_type:
                        raise ImageGenerationError(
                            f"Unexpected content-type from {model_name}: {content_type!r}"
                        )

                    # Don't save tiny error placeholders (< 5KB)
                    if len(resp.content) < 5_000:
                        logger.warning(
                            "%s returned empty/error image (%d bytes)",
                            model_name, len(resp.content),
                        )
                        last_exc = ImageGenerationError(
                            f"{model_name} returned {len(resp.content)} byte image"
                        )
                        continue

                    ext = "jpg" if "jpeg" in content_type else "png"
                    out_path = _IMAGES_DIR / f"{model_name}_{uuid.uuid4().hex[:8]}.{ext}"
                    out_path.write_bytes(resp.content)

                    logger.info(
                        "Pollinations [%s]: saved %s (%d bytes)",
                        model_name, out_path.name, len(resp.content),
                    )
                    return str(out_path)

            except ImageGenerationError:
                raise
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exc = exc
                logger.warning(
                    "Pollinations [%s] attempt %d failed: %s",
                    model_name, attempt + 1, exc,
                )

        logger.warning("Model %s failed/timed out — falling back to next model", model_name)

    raise ImageGenerationError(
        f"All Pollinations models failed. Last error: {last_exc}"
    )


# ── OpenAI DALL-E 3 ───────────────────────────────────────────────────────────

async def _openai_generate(subject: str) -> str:
    _ensure_dirs()
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise ImageGenerationError("openai package not installed") from exc

    try:
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.images.generate(
            model="dall-e-3",
            prompt=subject,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        redirect_url: str = response.data[0].url  # type: ignore[index]
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as dl:
            resp = await dl.get(redirect_url)
            resp.raise_for_status()
        out_path = _IMAGES_DIR / f"dalle_{uuid.uuid4().hex[:8]}.png"
        out_path.write_bytes(resp.content)
        return str(out_path)
    except Exception as exc:
        raise ImageGenerationError(f"DALL-E 3 failed: {exc}") from exc


# ── Stability AI ──────────────────────────────────────────────────────────────

async def _stability_generate(subject: str) -> str:
    _ensure_dirs()
    import base64  # noqa: PLC0415
    try:
        body = {
            "text_prompts": [{"text": subject}],
            "cfg_scale": 7, "height": 1024, "width": 1024,
            "samples": 1, "steps": 30,
        }
        headers = {
            "Authorization": f"Bearer {settings.STABILITY_API_KEY}",
            "Accept": "application/json",
        }
        url = (
            "https://api.stability.ai/v1/generation/"
            "stable-diffusion-xl-1024-v1-0/text-to-image"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        image_bytes = base64.b64decode(data["artifacts"][0]["base64"])
        out_path = _IMAGES_DIR / f"stability_{uuid.uuid4().hex[:8]}.png"
        out_path.write_bytes(image_bytes)
        return str(out_path)
    except Exception as exc:
        raise ImageGenerationError(f"Stability AI failed: {exc}") from exc


# ── Local Stable Diffusion (Automatic1111) ───────────────────────────────────

async def _local_sd_generate(subject: str) -> str:
    """
    Generate an image using a local Stable Diffusion REST API (Automatic1111).
    """
    import base64
    _ensure_dirs()

    url = f"{settings.LOCAL_SD_URL.rstrip('/')}/sdapi/v1/txt2img"
    payload = {
        "prompt": subject.strip() + _QUALITY_SUFFIX,
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            image_bytes = base64.b64decode(data["images"][0])
            
            out_path = _IMAGES_DIR / f"local_sd_{uuid.uuid4().hex[:8]}.png"
            out_path.write_bytes(image_bytes)
            logger.info("Local SD: saved %s", out_path.name)
            return str(out_path)
    except Exception as exc:
        logger.error("Local Stable Diffusion request failed: %s", exc)
        raise ImageGenerationError(f"Local SD failed: {exc}") from exc


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_image(subject: str) -> str:
    """
    Generate an image and return the local file path.

    Selects backend by configuration presence: Local SD -> OpenAI → Stability → Pollinations.
    Always raises ImageGenerationError (never a raw exception) on failure.
    """
    if getattr(settings, "LOCAL_SD_URL", ""):
        logger.info("Image backend: Local Stable Diffusion (Automatic1111)")
        try:
            return await _local_sd_generate(subject)
        except Exception as exc:
            logger.warning("Local Stable Diffusion failed, falling back to other backends: %s", exc)

    if getattr(settings, "OPENAI_API_KEY", ""):
        logger.info("Image backend: OpenAI DALL-E 3")
        return await _openai_generate(subject)

    if getattr(settings, "STABILITY_API_KEY", ""):
        logger.info("Image backend: Stability AI")
        return await _stability_generate(subject)

    logger.info("Image backend: Pollinations.ai (free, fast path)")
    return await _pollinations_generate(subject)

