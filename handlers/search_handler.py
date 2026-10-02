"""
handlers/search_handler.py — Web search and news report handler.

Contains: _send_news_report, _add_pending_report, _build_sources_text, _handle_web_search
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path

import httpx
from telegram import Document, InputMediaPhoto, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from config import settings
from services import image_service, ollama_service
from services import gemini_vision_service
from services import weather_service, search_service
from services.news_image_fetcher import fetch_all_images
from services.exceptions import (
    FileTooLargeError,
    ImageGenerationError,
    ReportGenerationError,
    StyleExtractionError,
    TableGenerationError,
    UnreadableFileError,
)
from services.file_reader import ExtractedDocument, build_llm_prompt, read_file
from services.style_extractor import extract_pdf_style, validate_docx_reference
from generators.docx_gen import generate_docx, generate_docx_from_reference
from generators.pdf_gen import generate_pdf
from generators.xlsx_gen import generate_excel
from generators.news_report_gen import generate_news_report, ArticleCard
from utils import history_db
from utils.bot_context import _semaphores, MAX_HISTORY as _MAX_HISTORY
from utils.callback_registry import callback_registry
from utils.pending_send import PendingSend, pending_send_registry
from utils.rate_limiter import rate_limiter
from utils.send_audit_log import log_send
from utils.cleanup import sweep_tmp_dir
from utils.telegram_formatter import format_for_telegram, split_message
from utils.typing_refresher import TypingRefresher
from utils.model_router import should_use_deep_model
from utils.file_limits import MAX_FILE_SIZE_BYTES, validate_upload
from utils.intent_parser import (
    Intent,
    extract_image_subject,
    extract_search_query,
    parse_intent,
    parse_web_search_flags,
)
from utils.telegram_ui import build_result_keyboard, send_status, edit_to_result_text, replace_with_media
from services.chart_service import render_bar_chart, render_table_snapshot
from services.weather_service import extract_city

logger = logging.getLogger(__name__)
_UPLOAD_DIR = Path(settings.TMP_DIR) / "uploads"

from handlers._shared import _send_html_reply, _get_history, _append, _cleanup

async def _send_news_report(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    answer: str,
    articles: list[ArticleCard],
) -> None:
    """
    Generate and send the news PDF report to *chat_id*.
    Handles ReportGenerationError gracefully — user gets a friendly message.
    """
    status = await context.bot.send_message(
        chat_id=chat_id,
        text="⏳ Generating your news report PDF…",
    )
    report_path: str | None = None
    try:
        report_path, renderer = generate_news_report(query, answer, articles)
        caption = (
            f"📄 *News Report:* _{query}_\n"
            f"_Renderer: {renderer}_"
        )
        with open(report_path, "rb") as pdf_file:
            await context.bot.send_document(
                chat_id=chat_id,
                document=pdf_file,
                filename=f"news_report_{query[:30].replace(' ', '_')}.pdf",
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                read_timeout=60,
                write_timeout=60,
                connect_timeout=30,
            )
    except ReportGenerationError as exc:
        logger.error("News report generation failed chat_id=%d: %s", chat_id, exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Could not generate the news report. The search results were sent above.",
        )
    except Exception as exc:
        logger.exception("Unexpected error sending news report chat_id=%d: %s", chat_id, exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ An unexpected error occurred while generating the report.",
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass
        if report_path:
            _cleanup(report_path)


# In-memory cache: "chat_id:query" → (answer, articles) for callback re-use
# Capped at 100 entries; oldest entry is evicted when at capacity (LRU via OrderedDict).
_pending_reports: OrderedDict[str, tuple[str, list[ArticleCard]]] = OrderedDict()
_PENDING_REPORTS_MAX = 100


def _add_pending_report(key: str, value: tuple) -> None:
    """Insert into _pending_reports with LRU capping."""
    if key in _pending_reports:
        _pending_reports.move_to_end(key)
    _pending_reports[key] = value
    if len(_pending_reports) > _PENDING_REPORTS_MAX:
        _pending_reports.popitem(last=False)  # evict oldest


def _build_sources_text(response: "SearchResponse") -> str:  # type: ignore[name-defined]
    """
    Return a numbered sources-only list in Markdown format (no duplicate summary)
    for use as a follow-up message when the AI summary is already shown
    as the photo caption.
    """
    from services.search_service import SearchResponse  # noqa: PLC0415
    if not response.results:
        return ""
    lines = ["📰 **Sources:**"]
    for i, r in enumerate(response.results, 1):
        snippet = (r.snippet or "")[:200].strip()
        if snippet and not snippet.endswith("."):
            snippet += "…"
        lines.append(
            f"{i}. [{r.title}]({r.url})\n   _{snippet}_"
        )
    return "\n".join(lines)


async def _handle_web_search(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Search the web via Tavily, send enriched results with images, and optionally generate a PDF report."""
    assert update.message is not None
    assert update.effective_user is not None

    # ── Rate-limit check (per-user token bucket) ─────────────────────────────
    user_id: int = update.effective_user.id
    if not rate_limiter.allow(user_id):
        await update.message.reply_text(
            "⏳ You're sending requests too quickly. Please wait a moment and try again."
        )
        return

    query, wants_report = parse_web_search_flags(text)
    if not query:
        query = extract_search_query(text) or text

    status = await update.message.reply_text(
        f"🔎 Searching the web for: _{query}_…", parse_mode=ParseMode.MARKDOWN
    )

    articles: list[ArticleCard] = []
    answer: str = ""
    search_response = None

    try:
        async with _semaphores[chat_id]:
            async with TypingRefresher(context.bot, chat_id):
                # ── Step 1: Structured search ─────────────────────────────────
                try:
                    search_response = await search_service.web_search_rich(query)
                    answer = search_response.answer
                except Exception as search_exc:
                    logger.exception("web_search_rich failed chat_id=%d: %s", chat_id, search_exc)
                    try:
                        await status.delete()
                    except Exception:
                        pass
                    await update.message.reply_text("⚠️ Web search failed. Please try again.")
                    return

                # ── Step 2: Fetch article images concurrently ──────────────────
                # allow_ai_fallback=False (default) — see news_image_fetcher.py
                fetched_images = await fetch_all_images(
                    search_response.results,
                    allow_ai_fallback=False,
                )

                # Build ArticleCard list (for report generation + photo album)
                for item, img in zip(search_response.results, fetched_images):
                    articles.append(ArticleCard(
                        title=item.title,
                        url=item.url,
                        snippet=item.snippet,
                        image=img,
                        published_date=item.published_date,
                    ))

        # ── Step 3: Build the photo album from real images ────────────────────
        # Placeholder images (local_path=None) are PDF-only; never sent as Telegram photos.
        media_batch: list[InputMediaPhoto] = []
        for card in articles[:5]:  # scan up to first 5 cards
            if len(media_batch) >= 3:
                break
            if card.image and card.image.local_path:
                title_safe = html.escape(card.title[:95])
                caption_lines = [f"<b>{title_safe}</b>"]
                if card.image.is_ai_generated:
                    caption_lines.append("🤖 AI-generated illustration")  # mandatory
                try:
                    with open(card.image.local_path, "rb") as f:
                        media_batch.append(
                            InputMediaPhoto(
                                media=f.read(),
                                caption="\n".join(caption_lines),
                                parse_mode=ParseMode.HTML,
                            )
                        )
                except Exception as img_exc:
                    logger.warning("Could not read image for album: %s", img_exc)

        # ── Step 4: Compose the text summary + sources ────────────────────────
        markdown_result = search_service.format_as_markdown(search_response)
        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", markdown_result)

        # Register callback payloads
        ref_report = callback_registry.register("news_report", {"query": query, "chat_id": chat_id})
        ref_more = callback_registry.register("news_more", {"query": query, "chat_id": chat_id, "limit": 8})
        reply_markup = build_result_keyboard([
            ("📄 Generate Report", f"news_report:{ref_report}"),
            ("🔍 More results", f"news_more:{ref_more}"),
        ])

        # ── Step 5: Send — unified or text-only ──────────────────────────────
        if media_batch:
            # Put the AI answer summary as the caption on the first photo
            if answer:
                summary_safe = html.escape(answer[:700])
                summary_caption = f"💡 <b>Summary:</b> {summary_safe}"
                media_batch[0] = InputMediaPhoto(
                    media=media_batch[0].media,
                    caption=summary_caption,
                    parse_mode=ParseMode.HTML,
                )
            try:
                await context.bot.send_media_group(chat_id=chat_id, media=media_batch)
            except Exception as album_exc:
                logger.warning("Photo album send failed: %s", album_exc)
            # Follow up with the sources list (no duplicate summary)
            sources_only = _build_sources_text(search_response)
            await _send_html_reply(
                update, context, chat_id,
                sources_only if sources_only else markdown_result,
                reply_markup=reply_markup,
            )
        else:
            # No images — send full text reply
            await _send_html_reply(update, context, chat_id, markdown_result, reply_markup=reply_markup)

        # Cache results for possible callback-triggered report generation
        cache_key = f"{chat_id}:{query[:200]}"
        _add_pending_report(cache_key, (answer, articles))

        # ── Step 6: Auto-generate PDF if user asked for a report ──────────────
        if wants_report and articles:
            await _send_news_report(chat_id, context, query, answer, articles)

    except Exception as exc:
        logger.exception("_handle_web_search failed chat_id=%d: %s", chat_id, exc)
        await update.message.reply_text(
            "⚠️ Web search is temporarily unavailable. Please try again."
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass


