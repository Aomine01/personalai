"""
services/search_service.py — Tavily web search integration.

Tavily provides an LLM-optimised search API (answer + source snippets).
Free tier: 1 000 requests/month.
Docs: https://docs.tavily.com
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


# ── Structured result types ───────────────────────────────────────────────────

@dataclass
class SearchItem:
    """
    A single search result.

    `snippet` is a short excerpt (a sentence or two) from the article — never
    a full-article reproduction.  This scope limit is intentional: including
    full article text in user-facing documents raises copyright concerns that
    are out of scope for this personal-use tool.
    """
    title: str
    url: str
    snippet: str          # short excerpt only — not a full article
    image_url: str | None
    published_date: str | None


@dataclass
class SearchResponse:
    """Structured response from a Tavily search, ready for report generation."""
    query: str
    answer: str            # synthesized summary from Tavily
    results: list[SearchItem]


# ── Snippet cleaner ──────────────────────────────────────────────────────────

def clean_snippet(text: str) -> str:
    """
    Strip raw markdown headers, HTML tags, and timestamp noise that Tavily
    sometimes includes in result snippets.
    E.g. '### Breaking News\n16:43 / 05.06 ### Something <br>else' → 'Something else'
    """
    # Remove markdown headers (##, ###, ####, etc.)
    text = re.sub(r'#{1,6}\s*', '', text)
    # Remove HTML tags (<br>, <br/>, <p>, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove timestamp patterns like "16:43 / 05.06.2026" or "17:35 |"
    text = re.sub(r'\b\d{1,2}:\d{2}\s*[/|]\s*\d{2}\.\d{2}(?:\.\d{4})?\b', '', text)
    # Collapse extra whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# ── Formatting helper (Markdown for Telegram) ─────────────────────────────────

def format_as_markdown(response: SearchResponse) -> str:
    """
    Convert a SearchResponse to a Markdown-formatted string for Telegram.
    This is a pure formatting function — no network I/O.
    """
    parts: list[str] = [f"🔎 *Web search: {response.query}*\n"]

    if response.answer:
        parts.append(f"💡 *Answer*: {response.answer}\n")

    if response.results:
        parts.append("📰 *Sources:*")
        for i, r in enumerate(response.results, 1):
            snippet = r.snippet[:200].strip()
            if snippet and not snippet.endswith("."):
                snippet += "…"
            parts.append(f"{i}. [{r.title}]({r.url})\n   _{snippet}_")

    if not response.answer and not response.results:
        return "🔎 No results found for that query."

    return "\n".join(parts)


# ── Core search function (single Tavily call site) ────────────────────────────

async def web_search_rich(query: str, max_results: int = 5) -> SearchResponse:
    """
    Run a Tavily search and return a structured SearchResponse.

    Includes image_url per result when Tavily provides one.
    This is the single code path that calls the Tavily API — both the
    plain-text function and the report generator use this, not duplicate calls.

    Raises:
        httpx.HTTPStatusError: on Tavily API errors (caller should handle).
        Exception:             on network or parsing failure.
    """
    if not settings.TAVILY_API_KEY:
        raise RuntimeError("Web search not configured (missing TAVILY_API_KEY).")

    payload: dict[str, Any] = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",       # 'basic' is free-tier; 'advanced' costs credits
        "include_answer": True,        # Tavily synthesizes a direct answer
        "include_raw_content": False,
        "include_images": True,        # request image URLs alongside results
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(_TAVILY_URL, json=payload)
        resp.raise_for_status()

    data: dict[str, Any] = resp.json()

    # Tavily returns a top-level "images" list and also sometimes image_url per result
    # Build a url->image_url mapping from the images list for quick lookup
    top_images: list[str] = data.get("images", [])
    raw_results: list[dict] = data.get("results", [])

    items: list[SearchItem] = []
    for i, r in enumerate(raw_results[:max_results]):
        # Prefer per-result image_url; fall back to the parallel images list
        img_url: str | None = r.get("image") or r.get("image_url") or None
        if not img_url and i < len(top_images):
            img_url = top_images[i] or None

        raw_snippet = r.get("content", "")[:600].strip()
        items.append(SearchItem(
            title=r.get("title", "Untitled"),
            url=r.get("url", ""),
            snippet=clean_snippet(raw_snippet)[:400],
            image_url=img_url if img_url and img_url.startswith("http") else None,
            published_date=r.get("published_date"),
        ))

    return SearchResponse(
        query=query,
        answer=data.get("answer", "").strip(),
        results=items,
    )


# ── Plain-text legacy wrapper (backward-compatible) ───────────────────────────

async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return a Markdown-formatted string for Telegram.

    Internally calls web_search_rich() + format_as_markdown() so there is
    exactly one Tavily-calling code path in this module.

    Returns a Markdown-formatted string ready to send to Telegram.
    """
    if not settings.TAVILY_API_KEY:
        return "❌ Web search is not configured (missing TAVILY_API_KEY)."

    try:
        response = await web_search_rich(query, max_results=max_results)
        return format_as_markdown(response)

    except httpx.HTTPStatusError as exc:
        logger.error("Tavily HTTP error: %s", exc)
        if exc.response.status_code == 401:
            return "❌ Web search authentication failed. Check your TAVILY_API_KEY."
        return f"⚠️ Web search failed (HTTP {exc.response.status_code}). Please try again."
    except Exception as exc:
        logger.exception("Tavily search error: %s", exc)
        return "⚠️ Web search is temporarily unavailable. Please try again."
