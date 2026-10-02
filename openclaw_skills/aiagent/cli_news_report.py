"""
openclaw_skills/aiagent/cli_news_report.py — Thin CLI wrapper for news report generation.

This file is intentionally minimal: all search, image-fetch, and report-generation
logic lives in the shared modules under services/ and generators/.  This wrapper
simply glues them together for command-line use by the Antigravity/openclaw skill.

Design rule: Do NOT add any styling or fetch logic here.
             Any change to report appearance must go into:
               - generators/news_report_gen.py   (layout, PDF rendering)
               - templates/news_report.html       (visual design)
               - services/news_image_fetcher.py   (image download)

Usage (from the aiagent project root):
    python openclaw_skills/aiagent/cli_news_report.py "latest AI news"

Output:
    Report generated: /tmp/aiagent/reports/news_report_<hash>.pdf (renderer: weasyprint)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure the aiagent project root is on the import path so shared modules resolve
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.search_service import web_search_rich          # noqa: E402
from services.news_image_fetcher import fetch_all_images     # noqa: E402
from generators.news_report_gen import (                     # noqa: E402
    generate_news_report,
    ArticleCard,
)


async def main(query: str) -> None:
    """
    Run a web search, fetch article images (real only — AI fallback OFF),
    generate a PDF report, and print the output path.

    allow_ai_fallback=False is intentional — see services/news_image_fetcher.py.
    AI-generated images are opt-in only and must never be the silent default.
    """
    print(f"🔎 Searching for: {query!r}")
    response = await web_search_rich(query)
    print(f"   Found {len(response.results)} results.")

    print("📥 Fetching article images (real only, no AI fallback)…")
    images = await fetch_all_images(response.results, allow_ai_fallback=False)

    articles = [
        ArticleCard(
            title=item.title,
            url=item.url,
            snippet=item.snippet,
            image=img,
            published_date=item.published_date,
        )
        for item, img in zip(response.results, images)
    ]

    real_images = sum(1 for a in articles if a.image.local_path)
    placeholders = len(articles) - real_images
    print(f"   Images: {real_images} real, {placeholders} placeholder(s).")

    print("📄 Generating PDF report…")
    path, renderer = generate_news_report(query, response.answer, articles)
    print(f"✅ Report generated: {path} (renderer: {renderer})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cli_news_report.py \"<search query>\"")
        sys.exit(1)

    search_query = " ".join(sys.argv[1:])
    asyncio.run(main(search_query))
