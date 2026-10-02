"""
services/ollama_service.py — Async wrapper around the local Ollama API.

Endpoint: POST http://localhost:11434/api/generate
Docs:     https://github.com/ollama/ollama/blob/main/docs/api.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import AsyncIterator

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ── Persistent HTTP client for Ollama (singleton, lives for process lifetime) ──
_ollama_client: httpx.AsyncClient | None = None


def get_ollama_client() -> httpx.AsyncClient:
    """Return the shared Ollama httpx client, creating it once on first call."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = httpx.AsyncClient(
            timeout=settings.OLLAMA_TIMEOUT,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            http2=False,
        )
    return _ollama_client


async def close_ollama_client() -> None:
    """Close the shared Ollama client on bot shutdown."""
    global _ollama_client
    if _ollama_client is not None:
        await _ollama_client.aclose()
        _ollama_client = None

_SYSTEM_PROMPT_BASE = (
    "You are a helpful, concise, and friendly personal assistant. "
    "You answer questions accurately. "
    "IMPORTANT: Always detect the language the user is writing in and reply in that exact same language. "
    "If the user writes in Uzbek, reply fully in Uzbek. "
    "If the user writes in Russian, reply in Russian. "
    "If the user writes in English, reply in English. "
    "Never say you cannot speak or understand a language — you are multilingual. "
    "When asked to produce files (PDF, Word document, image), "
    "simply acknowledge and let the system handle file creation."
)


def _system_prompt(persona_name: str | None = None) -> str:
    """Return system prompt with current date/time and optional persona name injected."""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M")

    if persona_name:
        identity = (
            f"Your name is {persona_name}. "
            f"Always refer to yourself as {persona_name}. "
            "Never say you are an AI language model or claim a different name. "
        )
    else:
        identity = ""

    return (
        f"{identity}"
        f"{_SYSTEM_PROMPT_BASE} "
        f"Today's date is {date_str} and the current time is {time_str}. "
        "Always use this information when the user asks about the date, time, day, or year."
    )


def _build_prompt(
    history: list[dict[str, str]],
    user_message: str,
    persona_name: str | None = None,
) -> str:
    """
    Format the conversation history into a single prompt string.
    Ollama's /api/generate endpoint expects a raw prompt; we simulate
    multi-turn by prepending recent history.
    """
    lines: list[str] = [f"System: {_system_prompt(persona_name)}\n"]
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


async def chat(
    user_message: str,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    persona_name: str | None = None,
    num_predict: int | None = None,
) -> str:
    """
    Send *user_message* to Ollama and return the complete assistant reply.

    Args:
        user_message:  The latest message from the user.
        history:       List of {"role": "user"|"assistant", "content": "..."} dicts.
        model:         Optional model name override.
        persona_name:  Optional bot persona name (e.g. "Friday") injected into system prompt.
        num_predict:   Optional maximum tokens to predict (default: 2048).

    Returns:
        The full text response from the model.
    """
    if history is None:
        history = []

    target_model = model or settings.DEFAULT_MODEL
    trimmed_history = history[-(settings.CONTEXT_WINDOW * 2):]
    prompt = _build_prompt(trimmed_history, user_message, persona_name)

    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": True,          # stream so we can concatenate chunks
        "keep_alive": settings.MODEL_SWITCH_KEEP_ALIVE,
        "options": {
            "temperature": 0.7,
            "num_predict": num_predict or 2048,
            # Apple Silicon M5 optimisations
            # Unified memory means ALL layers can stay on GPU (Metal)
            "num_gpu": 99,        # offload all layers to Metal GPU
            "num_thread": 4,      # efficiency cores for CPU-side work
            "num_ctx": 8192,      # context window (fits in 16GB easily)
            "f16_kv": True,       # half-precision KV cache → less memory
        },
    }

    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    full_response: list[str] = []

    try:
        client = get_ollama_client()
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON line from Ollama: %s", raw_line)
                    continue

                token = chunk.get("response", "")
                full_response.append(token)

                if chunk.get("done", False):
                    break
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"⚠️ The model `{target_model}` is not available locally. Please ensure it is pulled (run `ollama pull {target_model}`) and try again."
        try:
            err_json = exc.response.json()
            if "not found" in err_json.get("error", "").lower():
                return f"⚠️ The model `{target_model}` is not available locally. Please run `ollama pull {target_model}` and try again."
        except Exception:
            pass
        raise
    except httpx.ConnectError:
        return "⚠️ Couldn't reach the Ollama server. Make sure Ollama is running and try again."

    return "".join(full_response).strip()


async def stream_chat(
    user_message: str,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    persona_name: str | None = None,
) -> AsyncIterator[str]:
    """
    Async generator that yields individual tokens as they arrive from Ollama.
    Useful if you want to implement streaming replies in Telegram later.
    """
    if history is None:
        history = []

    target_model = model or settings.DEFAULT_MODEL
    trimmed_history = history[-(settings.CONTEXT_WINDOW * 2):]
    prompt = _build_prompt(trimmed_history, user_message, persona_name)

    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": settings.MODEL_SWITCH_KEEP_ALIVE,
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_gpu": 99,
            "num_thread": 4,
            "num_ctx": 8192,
            "f16_kv": True,
        },
    }

    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    try:
        client = get_ollama_client()
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")
                if token:
                    yield token

                if chunk.get("done", False):
                    break
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            yield f"⚠️ The model `{target_model}` is not available locally. Please run `ollama pull {target_model}` and try again."
            return
        try:
            err_json = exc.response.json()
            if "not found" in err_json.get("error", "").lower():
                yield f"⚠️ The model `{target_model}` is not available locally. Please run `ollama pull {target_model}` and try again."
                return
        except Exception:
            pass
        raise
    except httpx.ConnectError:
        yield "⚠️ Couldn't reach the Ollama server. Make sure Ollama is running and try again."
        return


# ---------------------------------------------------------------------------
# Structured (JSON-only) output — used for Excel data generation
# ---------------------------------------------------------------------------

_JSON_SYSTEM = (
    "You are a data-generation assistant. "
    "You MUST respond with ONLY valid JSON — no markdown fences, "
    "no explanation, no commentary, no trailing text. "
    "Your entire response must be parseable by json.loads()."
)


async def structured_chat(prompt: str, model: str | None = None) -> dict:  # type: ignore[type-arg]
    """
    Send *prompt* to Ollama with a strict JSON-only system instruction.

    Retries once on malformed JSON before raising.

    Returns:
        Parsed dict from the model's response.

    Raises:
        ValueError: if JSON cannot be parsed after 2 attempts or model is unavailable.
        httpx.HTTPError: on network failure.
    """
    full_prompt = f"{_JSON_SYSTEM}\n\nTask: {prompt}\n\nJSON response:"
    target_model = model or settings.DEFAULT_MODEL

    payload = {
        "model": target_model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",        # Ollama grammar-constrained JSON mode
        "keep_alive": settings.MODEL_SWITCH_KEEP_ALIVE,
        "options": {
            "temperature": 0.1,  # low temp = deterministic structured output
            "num_predict": 2048,
            "num_gpu": 99,
            "num_thread": 4,
            "num_ctx": 16384,
            "f16_kv": True,
        },
    }

    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    last_exc: Exception | None = None
    raw_text: str = ""

    for attempt in range(2):
        try:
            client = get_ollama_client()
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "").strip()

            # Strip accidental markdown fences if model ignores instructions
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                raw_text = parts[1] if len(parts) > 1 else raw_text
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            logger.info("structured_chat: valid JSON on attempt %d", attempt + 1)
            return parsed  # type: ignore[return-value]

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError(f"⚠️ Model `{target_model}` is not available locally. Please run `ollama pull {target_model}` and try again.")
            try:
                err_json = exc.response.json()
                if "not found" in err_json.get("error", "").lower():
                    raise ValueError(f"⚠️ Model `{target_model}` is not available locally. Please run `ollama pull {target_model}` and try again.")
            except Exception:
                pass
            raise
        except httpx.ConnectError as exc:
            raise ValueError("⚠️ Couldn't reach the Ollama server. Make sure Ollama is running and try again.") from exc
        except json.JSONDecodeError as exc:
            last_exc = exc
            logger.warning(
                "structured_chat attempt %d: JSONDecodeError — %s | raw=%r",
                attempt + 1, exc, raw_text[:200],
            )

    raise ValueError(
        f"Ollama returned invalid JSON after 2 attempts. Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Vision support — used for analyzing images via Llava
# ---------------------------------------------------------------------------

async def vision_chat(user_message: str, image_path: str) -> str:
    """
    Send *user_message* and the image at *image_path* to Ollama's vision model (llava).

    Args:
        user_message: The prompt or question about the image.
        image_path:   Absolute path to the downloaded image file.

    Returns:
        The text response from the vision model.
    """
    import base64  # noqa: PLC0415

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Could not read image file for base64 encoding: {exc}") from exc

    payload = {
        "model": "llava",
        "prompt": user_message,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.2,   # lower temp for descriptive accuracy
            "num_predict": 1024,
            "num_gpu": 99,
            "num_thread": 4,
            "num_ctx": 16384,
            "f16_kv": True,
        },
    }

    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    client = get_ollama_client()
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()
