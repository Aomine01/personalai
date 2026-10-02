"""
main.py — Multi-modal AI Telegram Bot (Phase 2).

Handler routing
───────────────
/start /help /clear /status   → command handlers
<document upload>             → _handle_file_upload()   [routed by DocumentHandler,
                                                          bypasses intent parser entirely]
<text message>                → dispatch() → intent → sub-handler

Per-chat concurrency guard
──────────────────────────
asyncio.Semaphore(1) per chat_id, stored in _semaphores dict.
All Ollama-backed ops (chat, file analysis, excel gen) acquire it.
Prevents self-inflicted pile-ups on the local 16GB machine.

Style-matched generation flow
──────────────────────────────
User uploads a reference file → stored in context.user_data['pending_reference']
User then sends CREATE_STYLED_DOC intent → _handle_styled_doc() reads pending_reference,
calls style_extractor + appropriate generator, clears the pending state.

WeasyPrint / PDF rendering
───────────────────────────
If WeasyPrint system libs are missing (Cairo/Pango), pdf_gen falls back to fpdf2
and the user is explicitly told styling won't reflect the reference. See pdf_gen.py.
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
from telegram import BotCommand, Document, InputMediaPhoto, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import settings
from generators.docx_gen import generate_docx, generate_docx_from_reference
from generators.pdf_gen import generate_pdf
from generators.xlsx_gen import generate_excel
from generators.news_report_gen import generate_news_report, ArticleCard
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
from utils.cleanup import periodic_sweep, sweep_tmp_dir
from utils.telegram_formatter import format_for_telegram, split_message
from utils.typing_refresher import TypingRefresher
from utils import history_db
from utils.model_router import should_use_deep_model
from utils.file_limits import MAX_FILE_SIZE_BYTES, validate_upload
from utils.intent_parser import (
    Intent,
    extract_image_subject,
    extract_search_query,
    parse_intent,
    parse_web_search_flags,
)
from utils.rate_limiter import rate_limiter
from services.weather_service import extract_city
from utils.callback_registry import callback_registry
from utils.telegram_ui import build_result_keyboard, send_status, edit_to_result_text, replace_with_media
from services.chart_service import render_bar_chart, render_table_snapshot
from utils.pending_send import PendingSend, pending_send_registry
from utils.send_audit_log import log_send
from utils.bot_context import _semaphores, MAX_HISTORY as _MAX_HISTORY


def build_confirmation_card(pending: PendingSend) -> tuple[str, InlineKeyboardMarkup]:
    """
    Build universal confirmation card before sending any message to a Telegram contact.
    
    Format:
    📤 Ready to send
    
    To: {recipient_display_name}  (match confidence: {match_confidence}%)
    Message: "{final_text}"
    
    ⚠️ Not fully certain this is the right contact — please check. (if match_confidence < 85)
    
    [✅ Send]  [✏️ Edit]  [❌ Cancel]
    """
    to_line = f"<b>To:</b> {html.escape(pending.recipient_display_name)}"
    if pending.match_confidence is not None:
        to_line += f"  <i>(match confidence: {pending.match_confidence}%)</i>"

    lines = [
        "📤 <b>Ready to send</b>",
        "",
        to_line,
        f"<b>Message:</b> <i>\"{html.escape(pending.final_text)}\"</i>",
    ]

    if pending.match_confidence is not None and pending.match_confidence < 85:
        lines.append("")
        lines.append("⚠️ <i>Not fully certain this is the right contact — please check.</i>")

    text = "\n".join(lines)

    ref_confirm = callback_registry.register("confirm_send", {"user_id": pending.user_id})
    ref_edit = callback_registry.register("edit_send", {"user_id": pending.user_id})
    ref_cancel = callback_registry.register("cancel_send", {"user_id": pending.user_id})

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Send", callback_data=f"confirm_send:{ref_confirm}"),
            InlineKeyboardButton("✏️ Edit", callback_data=f"edit_send:{ref_edit}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_send:{ref_cancel}"),
        ]
    ])
    return text, keyboard

# ── Persona name detection ────────────────────────────────────────────────────
_PERSONA_PATTERNS = [
    # "call yourself X", "call you X", "I'll call you X", "I will call you X"
    re.compile(r"(?:i(?:'ll| will| want to)?\s+)?call\s+you(?:rself)?\s+([A-Za-z][\w\-]{0,29})", re.I),
    # "your name is X", "your name should be X"
    re.compile(r"your\s+name\s+(?:is|should\s+be|will\s+be)\s+([A-Za-z][\w\-]{0,29})", re.I),
    # "from now you are X", "from now on you are X"
    re.compile(r"from\s+now\s+(?:on\s+)?you\s+are\s+([A-Za-z][\w\-]{0,29})", re.I),
    # "you are X" / "you're X" (standalone — short message only to avoid false positives)
    re.compile(r"^you(?:'re|\s+are)\s+([A-Za-z][\w\-]{0,29})\s*[.!?]?$", re.I),
    # "so from now you are X"
    re.compile(r"so\s+from\s+now\s+(?:on\s+)?you\s+are\s+([A-Za-z][\w\-]{0,29})", re.I),
]

_STOP_WORDS = {"a", "an", "the", "my", "is", "are", "you", "me", "bot", "ai", "assistant"}


def _detect_persona_change(text: str) -> str | None:
    """Return new persona name if the message is a rename command, else None."""
    for pattern in _PERSONA_PATTERNS:
        m = pattern.search(text.strip())
        if m:
            name = m.group(1).strip().rstrip(".,!?")
            if name.lower() not in _STOP_WORDS and len(name) >= 2:
                return name.capitalize()
    return None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Conversation state ────────────────────────────────────────────────────────
# _MAX_HISTORY and _semaphores are imported from utils.bot_context (shared singletons)
# to allow handler modules to import them without re-instantiation.

_UPLOAD_DIR = Path(settings.TMP_DIR) / "uploads"


async def _get_history(chat_id: int) -> list[dict[str, str]]:
    return await history_db.get_history(chat_id, _MAX_HISTORY)


async def _append(chat_id: int, role: str, content: str, model: str | None = None) -> None:
    await history_db.append_history(chat_id, role, content, model)


def _get_chat_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("🔁 Regenerate", callback_data="regen"),
            InlineKeyboardButton("🗑 Clear History", callback_data="clear"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


async def _send_html_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None
) -> None:
    formatted = format_for_telegram(text)
    chunks = split_message(formatted)
    for idx, chunk in enumerate(chunks):
        markup = reply_markup if idx == len(chunks) - 1 else None
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=ParseMode.HTML,
                reply_markup=markup,
            )
        except Exception as exc:
            logger.warning("HTML send failed: %s, falling back to plain text", exc)
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                reply_markup=markup,
            )
        if len(chunks) > 1:
            await asyncio.sleep(0.5)


def clear_pending_states(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear temporary stateful prompts/inputs to prevent unexpected behavior."""
    if context.user_data:
        context.user_data.pop("pending_command_intent", None)
        context.user_data.pop("pending_excel_edit", None)
        context.user_data.pop("pending_file_ask", None)
        context.user_data.pop("pending_tg_message", None)
        context.user_data.pop("pending_tg_contacts", None)
        context.user_data.pop("pending_tg_custom_target", None)



def _cleanup(path: str | None) -> None:
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Helper: safe Telegram reply ───────────────────────────────────────────────

async def _reply(update: Update, text: str, parse_mode: str | None = None) -> None:
    assert update.message is not None
    try:
        await update.message.reply_text(text, parse_mode=parse_mode)
    except Exception:
        await update.message.reply_text(text)   # plain-text fallback


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_user is not None
    name = update.effective_user.first_name or "there"
    await _reply(
        update,
        f"👋 Hello *{name}*! I'm your AI personal assistant powered by `{settings.OLLAMA_MODEL}`.\n\n"
        "*What I can do:*\n"
        "• 💬 Chat — just send me any message\n"
        "• 🌤 Weather — _\"weather in London\"_ or _\"forecast for Tokyo\"_\n"
        "• 🔎 Web search — _\"search for latest AI news\"_ or _\"who won the World Cup\"_\n"
        "• 🖼 Generate image — _\"generate an image of a sunset\"_\n"
        "• 👁 Analyze photo — send any photo to identify people, text, or scenes\n"
        "• 📊 Excel spreadsheet — _\"create an excel of top programming languages\"_\n"
        "• 📄 PDF report — _\"create a PDF summary\"_\n"
        "• 📝 Word report — _\"write a report\"_\n"
        "• 🔍 Analyze file — upload any PDF, DOCX, XLSX, or CSV\n"
        "• 🎨 Style-matched generation — upload a reference file, then say "
        "_\"generate a doc like this\"_\n\n"
        "Use /help for details or /clear to reset our conversation.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    await _reply(
        update,
        "📖 *Commands*\n"
        "/start — Greeting & feature overview\n"
        "/clear — Wipe conversation history\n"
        "/status — Show pending reference file\n"
        "/help — This message\n\n"
        "📌 *Trigger phrases*\n"
        "• `generate an image of <subject>`\n"
        "• `create an excel spreadsheet of <topic>`\n"
        "• `create a PDF summary`\n"
        "• `write a report` / `create a Word document`\n"
        "• `match this style` / `generate a doc like this` (after uploading a reference)\n\n"
        "📎 *Upload formats (max 20 MB)*: .pdf, .docx, .xlsx, .csv\n"
        "_Upload any file without a caption to summarize it, or add a caption as your question._",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    await history_db.clear_history(update.effective_chat.id)
    if hasattr(context, "user_data") and context.user_data:
        context.user_data.pop("pending_reference", None)
    await _reply(update, "🗑 Conversation history cleared. Fresh start!")



async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    ref = context.user_data.get("pending_reference") if context.user_data else None
    if ref:
        await _reply(
            update,
            f"📎 Pending reference file: `{Path(ref).name}`\n"
            "Send _\"generate a doc like this\"_ or _\"match this style\"_ to use it.",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await _reply(update, "ℹ️ No reference file pending. Upload a .docx or .pdf to set one.")


async def cmd_deep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.message is not None
    assert update.effective_chat is not None

    text = " ".join(context.args).strip() if context.args else ""
    if not text:
        await _reply(
            update,
            "🧠 *Deep Routing Mode*\n\n"
            "Forces the bot to use the deep reasoning model (`gpt-oss:20b`) for this specific query.\n\n"
            "*Usage*: `/deep <your question>`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await dispatch(update, context, forced_model=settings.DEEP_MODEL, forced_text=text)


async def cmd_fast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.message is not None
    assert update.effective_chat is not None

    text = " ".join(context.args).strip() if context.args else ""
    if not text:
        await _reply(
            update,
            "⚡ *Fast Routing Mode*\n\n"
            "Forces the bot to use the fast conversational model (`qwen3:8b`) for this specific query.\n\n"
            "*Usage*: `/fast <your question>`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await dispatch(update, context, forced_model=settings.DEFAULT_MODEL, forced_text=text)



# ── Intent sub-handlers ───────────────────────────────────────────────────────


async def _handle_weather(update: Update, text: str, chat_id: int) -> None:
    """Fetch current weather or 5-day forecast via OpenWeatherMap."""
    assert update.message is not None
    await update.message.chat.send_action(ChatAction.TYPING)

    city = extract_city(text)
    if not city:
        # Ask Ollama to extract the city gracefully
        await _reply(
            update,
            "🌍 Which city would you like the weather for? (e.g. *London*, *New York*, *Tokyo*)",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    is_forecast = bool(
        re.search(r"\b(forecast|week|tomorrow|5.day|5 day|next few days)\b", text, re.IGNORECASE)
    )

    if is_forecast:
        reply = await weather_service.get_forecast(city)
    else:
        reply = await weather_service.get_current_weather(city)

    await _reply(update, reply, parse_mode=ParseMode.MARKDOWN)



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


async def _handle_chat(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    forced_model: str | None = None,
    persona_name: str | None = None,
) -> None:
    assert update.message is not None

    target_model = forced_model
    if not target_model:
        target_model = settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL

    last_model = await history_db.get_last_model(chat_id)
    prev_model = last_model or settings.DEFAULT_MODEL
    if prev_model != target_model:
        icon = "🧠" if target_model == settings.DEEP_MODEL else "⚡"
        model_name = "deep reasoning model" if target_model == settings.DEEP_MODEL else "fast conversational model"
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{icon} Switching to the {model_name} (`{target_model}`), this may take a few extra seconds to load..."
        )

    logger.debug("chat_id=%d  routing request to model: %s  persona=%s  text=%r", chat_id, target_model, persona_name, text[:50])

    async with _semaphores[chat_id]:
        logger.debug("chat_id=%d  acquiring semaphore for chat", chat_id)
        try:
            async with TypingRefresher(context.bot, chat_id):
                reply = await ollama_service.chat(
                    text,
                    await _get_history(chat_id),
                    model=target_model,
                    persona_name=persona_name,
                )
        except Exception as exc:
            logger.exception("Ollama chat error chat_id=%d: %s", chat_id, exc)
            await _reply(
                update,
                "⚠️ Couldn't reach the AI model. Make sure Ollama is running and try again.",
            )
            return

    await _append(chat_id, "user", text)
    await _append(chat_id, "assistant", reply, model=target_model)
    await _send_html_reply(update, context, chat_id, reply, reply_markup=_get_chat_keyboard())



async def _handle_image(update: Update, text: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_chat is not None
    subject = extract_image_subject(text)

    # Context lookback logic: if subject is generic/pronoun, search user history
    # for the last descriptive user message to use as the image prompt.
    if subject.lower() in ("", "it", "this", "image", "pic", "picture", "no", "yes"):
        history = await _get_history(chat_id)
        for turn in reversed(history):
            if turn["role"] == "user":
                candidate = extract_image_subject(turn["content"])
                if len(candidate) > 5 and not candidate.lower().startswith("/"):
                    subject = candidate
                    break

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    status_msg_id, _ = await send_status(
        update, context, f"🎨 Generating image of *{subject}*… this can take up to 60 seconds."
    )

    path: str | None = None
    try:
        path = await image_service.generate_image(subject)
        # Extract model name from filename prefix (e.g., 'gpt-image-2' from 'gpt-image-2_abc.png')
        filename = Path(path).name
        model_used = filename.split('_')[0] if '_' in filename else "AI Model"
        
        # Register callback payloads
        ref_regen = callback_registry.register("regen_image", {"prompt": subject, "chat_id": chat_id})
        ref_restyle = callback_registry.register("restyle_image", {"prompt": subject, "chat_id": chat_id})
        
        keyboard = build_result_keyboard([
            ("🔄 Regenerate", f"regen_image:{ref_regen}"),
            ("🎨 Restyle Image", f"restyle_image:{ref_restyle}"),
        ])
        
        caption = (
            f"🖼 *{subject}*\n"
            f"_Generated with Pollinations.ai · {model_used}_"
        )
        
        await replace_with_media(
            context=context,
            chat_id=chat_id,
            status_message_id=status_msg_id,
            media_path=path,
            media_type="photo",
            caption=caption,
            keyboard=keyboard
        )
    except ImageGenerationError as exc:
        logger.error("Image generation failed chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=(
                f"❌ Image generation failed: {exc}\n\n"
                "Pollinations.ai may be temporarily unavailable. Try again in a moment."
            )
        )
    except Exception as exc:
        logger.exception("Unexpected image error chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text="⚠️ An unexpected error occurred during image generation."
        )
    finally:
        _cleanup(path)


async def _handle_pdf(update: Update, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE, model: str | None = None) -> None:
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)

    target_model = model or (settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL)

    # Detect whether user wants a CUSTOM content PDF or a CONVERSATION SUMMARY.
    # IMPORTANT: 'history' alone must NOT match -- "history of Uzbekistan" is custom content.
    # Only match when the user clearly refers to the conversation itself.
    _SUMMARY_HINTS = re.compile(
        r"\b(summary|summarize|transcript|our talk|discussion|compile|export)\b"
        r"|\b(conversation|chat|discussion|text|slide|file|document)\s+(history|summary|log|transcript|content|list|compile|export)\b"
        r"|\b(our|this|these|the)\s+(conversation|chat|discussion|text|texts|slide|slides|content|information)\b"
        r"|\b(chat|conversation)\b(?!.{0,30}\bhistory of\b)"
        r"|\b(what i sent|what i gave|texts i sent|slides i sent)\b",
        re.IGNORECASE,
    )
    is_summary_request = _SUMMARY_HINTS.search(text) or not text

    if is_summary_request:
        # -- Conversation summary PDF (original behaviour) ---------------------
        history = await _get_history(chat_id)
        if not history:
            await _reply(
                update,
                "ℹ️ No conversation history yet to summarize. Chat with me first, then ask for a PDF!",
            )
            return

        status_msg_id, _ = await send_status(
            update, context, "📝 Compiling and structuring your conversation history into a PDF…"
        )

        def _strip_html(s: str) -> str:
            """Strip HTML from history content before feeding to Ollama."""
            s = html.unescape(s)
            s = re.sub(r"</?\s*\w+\s*/?>", " ", s)   # strip all tags
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        def _clean(raw: str) -> str:
            """Strip all HTML/markdown from Ollama output for clean PDF rendering."""
            out = html.unescape(raw)
            out = re.sub(r"\u00a0", " ", out)             # non-breaking spaces
            out = re.sub(r"</?\s*p\s*/?>", "\n", out)     # <p> / </p> → newline
            out = re.sub(r"<br\s*/?>", "\n", out)          # <br> → newline
            out = re.sub(r"<[^>]+>", "", out)              # remove remaining tags
            out = re.sub(r"\*\*(.+?)\*\*", r"\1", out, flags=re.DOTALL)
            out = re.sub(r"\*(.+?)\*",     r"\1", out, flags=re.DOTALL)
            out = re.sub(r"__(.+?)__",     r"\1", out, flags=re.DOTALL)
            out = re.sub(r"_(.+?)_",       r"\1", out, flags=re.DOTALL)
            out = re.sub(r"^\s*[-*]{3,}\s*$", "", out, flags=re.MULTILINE)
            out = re.sub(r"[ \t]+", " ", out)              # collapse inline spaces
            out = re.sub(r"\n{3,}", "\n\n", out)
            return out.strip()

        compile_prompt = (
            "You are a professional document compiler.\n"
            "Below is the conversation history between the User and the Assistant. "
            "Extract all the core information, slides, or texts provided by the user and organize them into a clean, professional, structured document.\n"
            "Rules:\n"
            "- Do not include conversational chatter (like 'Got it', 'How can I help', 'did you get it').\n"
            "- If the user specifically asks for a chat log or raw transcript, format it as a clean dialogue log. Otherwise, compile the core information and slides into structured sections.\n"
            "- Do not include 'User:' or 'Assistant:' prefixes unless formatting as a dialogue log.\n"
            "- Preserve the exact order of the content sent by the user.\n"
            "- Format headings with '## Heading Name'.\n"
            "- Format lists with '- item'.\n"
            "- Do not include HTML tags, markdown bold (**), or any XML/HTML markup. Plain text only.\n\n"
            "Conversation History:\n"
        )

        for turn in history:
            compile_prompt += f"{turn['role'].upper()}: {_strip_html(turn['content'])}\n"

        try:
            async with _semaphores[chat_id]:
                ai_content = await ollama_service.chat(
                    compile_prompt,
                    history=[],
                    model=target_model,
                )
        except Exception as exc:
            logger.exception("Ollama failed compiling history PDF chat_id=%d: %s", chat_id, exc)
            await _reply(update, "⚠️ The AI model is unavailable right now. Please try again.")
            try:
                await status.delete()
            except Exception:
                pass
            return
        finally:
            try:
                await status.delete()
            except Exception:
                pass

        # ── Pre-clean entire Ollama response BEFORE section parsing ─────────────
        # Model sometimes echoes HTML tags inline (</p><p>) even when instructed
        # not to. Cleaning here ensures the PDF builder never receives raw tags.
        ai_content = _clean(ai_content)

        # Parse AI response into sections (split on ## headings)
        raw_sections: list[dict] = []
        current_heading = "Summary"
        current_body: list[str] = []

        for line in ai_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                if current_body:
                    body_text = _clean("\n".join(current_body))
                    if body_text:
                        raw_sections.append({"heading": _clean(current_heading), "body": body_text})
                current_heading = stripped.lstrip("# ").strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            body_text = _clean("\n".join(current_body))
            if body_text:
                raw_sections.append({"heading": _clean(current_heading), "body": body_text})

        sections  = raw_sections or [{"heading": "Summary", "body": _clean(ai_content)}]
        pdf_title = "Conversation Summary"
        filename  = "conversation_summary.pdf"
        generated_image_path = None

    else:
        # -- Custom-content PDF ------------------------------------------------
        history = await _get_history(chat_id)

        # -- Step 1: Detect if user wants an image ---------------------------
        _VISUAL_HINTS = re.compile(
            r"\b(img|imgs|image|images|photo|photos|picture|pictures|pic|pics|poster|posters|"
            r"banner|banners|cover|covers|fla[gq]s?|flagg|logo|logos|map|maps|diagram|diagrams|"
            r"icon|icons|coat of arms|symbol|emblem|visual|visuals|illustration|illustrations|art)\b",
            re.IGNORECASE,
        )
        wants_image = bool(_VISUAL_HINTS.search(text))
        generated_image_path: str | None = None

        status_msg_id, _ = await send_status(
            update, context, "📝 Thinking about your request and writing the PDF…"
        )

        # ── Step 2: Ask Ollama for a clean short title ──────────────────────
        try:
            raw_title = await ollama_service.chat(
                f"The user asked: \"{text}\"\n\n"
                "Based on the request and conversation history, reply with ONLY a short, clean document title (4-7 words max). "
                "No quotes, no punctuation at the end, no explanation. Just the title.",
                history=history[-4:] if history else [],
                model=target_model,
            )
            # Clean up: strip quotes, newlines, trailing punctuation
            topic = re.sub(r'^[\s\'"]+|[\s\'".,!?]+$', "", raw_title.strip().splitlines()[0])
            topic = topic[:80]  # hard cap
        except Exception:
            # Fallback: strip obvious boilerplate with regex
            topic_pat = re.compile(
                r"^(?:please\s+)?(?:gen[a-z]*|create|make|produce|write|give me|send me)\s+"
                r"(?:me\s+)?(?:a\s+|an\s+)?(?:pdf|PDF|word|doc|document)\s*(?:file\s*)?"
                r"(?:in\s+which\s+there\s+(?:will\s+)?be\s+)?"
                r"(?:a\s+|an\s+|the\s+)?",
                re.IGNORECASE,
            )
            topic = topic_pat.sub("", text).strip()
            topic = topic[:1].upper() + topic[1:] if topic else text.strip()[:80]

        # ── Step 3: Fetch real image from web (NO AI fallback for documents) ──
        async def _fetch_real_image(query: str) -> str | None:
            """Try to download a real image via Tavily image search."""
            if not settings.TAVILY_API_KEY:
                return None
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                    resp = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": settings.TAVILY_API_KEY,
                            "query": query,
                            "search_depth": "basic",
                            "include_images": True,
                            "include_answer": False,
                            "max_results": 5,
                        },
                        timeout=15,
                    )
                    data = resp.json()
                images: list[str] = data.get("images", [])
                if not images:
                    logger.info("Tavily search returned no images for query: %s", query)
                    return None
                # Download the first usable real image
                for img_url in images[:6]:
                    if not img_url or not img_url.startswith("http"):
                        continue
                    ext = img_url.split("?")[0].rsplit(".", 1)[-1].lower()
                    if ext not in {"jpg", "jpeg", "png", "webp"}:
                        ext = "jpg"
                    try:
                        async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=headers) as dl:
                            logger.info("Attempting to download real image from web: %s", img_url)
                            r = await dl.get(img_url)
                            if r.status_code == 200 and len(r.content) > 3000:
                                from pathlib import Path as _P  # noqa: PLC0415
                                img_dir = _P(settings.TMP_DIR) / "images"
                                img_dir.mkdir(parents=True, exist_ok=True)
                                save_path = str(img_dir / f"web_{uuid.uuid4().hex[:8]}.{ext}")
                                with open(save_path, "wb") as f:
                                    f.write(r.content)
                                logger.info("Downloaded real image from web: %s", img_url)
                                return save_path
                            else:
                                logger.warning("Failed download for %s: HTTP %d (size: %d)", img_url, r.status_code, len(r.content))
                    except Exception as dl_exc:
                        logger.warning("Image download failed (%s): %s", img_url, dl_exc)
                        continue
            except Exception as exc:
                logger.warning("Tavily image search failed: %s", exc)
            return None

        if wants_image:
            await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)
            image_query = topic
            try:
                extract_prompt = (
                    f"The user wants to find a real image/poster/logo for this document.\n"
                    f"Document topic: \"{topic}\"\n"
                    f"User request: \"{text}\"\n\n"
                    "Extract the exact subject or entity they want an image of. "
                    "Reply with ONLY a clean search query (1-4 words max) to find this real image on the web (e.g. \"Marvel MCU Movies Poster\", \"Flag of Uzbekistan\", \"Avengers Doomsday\", etc.). "
                    "Do not include any quotes, markdown, or other text."
                )
                res = await ollama_service.chat(extract_prompt, history=history[-4:] if history else [], model=target_model)
                res_clean = re.sub(r'^[\s\'"]+|[\s\'".,!?]+$', "", res.strip().splitlines()[0])
                if res_clean:
                    image_query = res_clean
            except Exception as e:
                logger.warning("Failed to extract clean image query via LLM: %s", e)

            logger.info("Searching Tavily for real image query: %s", image_query)
            generated_image_path = await _fetch_real_image(image_query)
            if generated_image_path:
                logger.info("Using real web image for PDF: %s", generated_image_path)
            else:
                logger.info("No real web image found for %r; omitting image from document.", image_query)
                generated_image_path = None

        # ── Step 4: Generate document content with Ollama ───────────────────
        try:
            async with _semaphores[chat_id]:
                ai_content = await ollama_service.chat(
                    f"Write a comprehensive, professional, and visually structured document about: {topic}\n"
                    f"User's request: \"{text}\"\n\n"
                    "Rules you MUST follow:\n"
                    "- Start directly with the first section heading. Do NOT write conversational introductions, placeholders, or meta-commentary (such as 'posters will be inserted here' or 'here is the requested list').\n"
                    "- Use '## Section Heading' for main categories/headings.\n"
                    "- For items/movies/products/entries, format as clean bullet points with bold titles: '- **Item Name (Year)**: Director / Details / Key points...'\n"
                    "- Be thorough, detailed, and complete every entry fully without trailing off.\n"
                    "- Maintain a high-end executive tone.",
                    history=history[-6:] if history else [],
                    model=target_model,
                    num_predict=3500,
                )
        except Exception as exc:
            logger.exception("Ollama failed generating PDF content chat_id=%d: %s", chat_id, exc)
            await _reply(update, "⚠️ The AI model is unavailable right now. Please try again.")
            _cleanup(generated_image_path)
            try:
                await status.delete()
            except Exception:
                pass
            return
        finally:
            try:
                await status.delete()
            except Exception:
                pass

        def _clean(raw: str) -> str:
            """Strip HTML tags and excess separators while preserving markdown formatting for PDF rendering."""
            out = html.unescape(raw)
            out = re.sub(r"<[^>]+>", "", out)                             # strip raw HTML tags
            out = re.sub(r"^\s*[-*]{3,}\s*$", "", out, flags=re.MULTILINE)  # --- separators
            out = re.sub(r"\n{3,}", "\n\n", out)                          # excess blank lines
            return out.strip()

        # Parse AI response into sections (split on ## headings)
        raw_sections: list[dict] = []
        current_heading = topic
        current_body: list[str] = []

        for line in ai_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                if current_body:
                    body_text = _clean("\n".join(current_body))
                    if body_text:
                        raw_sections.append({"heading": _clean(current_heading), "body": body_text})
                current_heading = stripped.lstrip("# ").strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            body_text = _clean("\n".join(current_body))
            if body_text:
                raw_sections.append({"heading": _clean(current_heading), "body": body_text})

        sections  = raw_sections or [{"heading": topic, "body": _clean(ai_content)}]
        pdf_title = topic
        safe_stem = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()
        filename  = f"{safe_stem[:40]}.pdf"

        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", f"[Generated PDF about: {topic}]")

    path: str | None = None
    try:
        path, _ = generate_pdf(
            sections,
            title=pdf_title,
            image_path=generated_image_path,
        )
        img_note = " 🖼 Image included!" if generated_image_path else ""
        safe_caption_title = pdf_title
        if len(safe_caption_title) > 60:
            safe_caption_title = safe_caption_title[:57] + "..."
        caption = f"📄 Here's your PDF: *{safe_caption_title}*!{img_note}"
        
        # Register callback payload
        ref_regen = callback_registry.register("regen_pdf", {"prompt": text, "chat_id": chat_id})
        keyboard = build_result_keyboard([("🔄 Regenerate", f"regen_pdf:{ref_regen}")])
        
        await replace_with_media(
            context=context,
            chat_id=chat_id,
            status_message_id=status_msg_id,
            media_path=path,
            media_type="document",
            caption=caption,
            keyboard=keyboard
        )
    except Exception as exc:
        logger.exception("PDF generation failed chat_id=%d: %s", chat_id, exc)
        try:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ Sorry, couldn't create the PDF: {exc}"
            )
        except Exception:
            await _reply(update, f"❌ Sorry, couldn't create the PDF: {exc}")
    finally:
        _cleanup(path)
        _cleanup(generated_image_path)  # remove temp image after PDF is sent


async def _handle_docx(update: Update, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE, model: str | None = None) -> None:
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)

    target_model = model or (settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL)

    # Detect whether this is a CONVERSATION SUMMARY or CUSTOM-CONTENT doc request.
    # IMPORTANT: 'history' alone must NOT match — "history of Rome" is custom content.
    _SUMMARY_HINTS = re.compile(
        r"\b(summary|summarize|transcript|our talk|discussion|compile|export)\b"
        r"|\b(conversation|chat|discussion|text|slide|file|document)\s+(history|summary|log|transcript|content|list|compile|export)\b"
        r"|\b(our|this|these|the)\s+(conversation|chat|discussion|text|texts|slide|slides|content|information)\b"
        r"|\b(chat|conversation)\b(?!.{0,30}\bhistory of\b)"
        r"|\b(what i sent|what i gave|texts i sent|slides i sent)\b",
        re.IGNORECASE,
    )
    is_summary_request = _SUMMARY_HINTS.search(text) or not text

    if is_summary_request:
        # ── Conversation history export ───────────────────────────────────────
        history = await _get_history(chat_id)
        if not history:
            await _reply(
                update,
                "ℹ️ No conversation history yet. Chat with me first, then ask for a Word document!",
            )
            return

        status_msg_id, _ = await send_status(
            update, context, "📝 Compiling and structuring your conversation history into a Word document…"
        )

        compile_prompt = (
            "You are a professional document compiler.\n"
            "Below is the conversation history between the User and the Assistant. "
            "Extract all the core information, slides, or texts provided by the user and organize them into a clean, professional, structured Word document.\n"
            "Rules:\n"
            "- Do not include conversational chatter (like 'Got it', 'How can I help', 'did you get it').\n"
            "- If the user specifically asks for a chat log or raw transcript, format it as a clean dialogue log. Otherwise, compile the core information and slides into structured sections.\n"
            "- Do not include 'User:' or 'Assistant:' prefixes unless formatting as a dialogue log.\n"
            "- Preserve the exact order of the content sent by the user.\n"
            "- Format headings with '## Heading Name'.\n"
            "- Format lists with '- item'.\n"
            "- Do not include HTML tags or markdown bold (**).\n\n"
            "Conversation History:\n"
        )
        for turn in history:
            compile_prompt += f"{turn['role'].upper()}: {turn['content']}\n"

        try:
            async with _semaphores[chat_id]:
                ai_content = await ollama_service.chat(
                    compile_prompt,
                    history=[],
                    model=target_model,
                )
        except Exception as exc:
            logger.exception("Ollama failed for docx compile chat_id=%d: %s", chat_id, exc)
            await _reply(update, "⚠️ The AI model is unavailable right now. Please try again.")
            return

        def _clean(raw: str) -> str:
            out = html.unescape(raw)
            out = re.sub(r"<[^>]+>", "", out)
            out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
            out = re.sub(r"\*(.+?)\*",     r"\1", out)
            out = re.sub(r"__(.+?)__",     r"\1", out)
            out = re.sub(r"_(.+?)_",       r"\1", out)
            out = re.sub(r"^\s*[-*]{3,}\s*$", "", out, flags=re.MULTILINE)
            out = re.sub(r"\n{3,}", "\n\n", out)
            return out.strip()

        # Parse AI response into sections (split on ## headings)
        raw_sections: list[dict] = []
        current_heading = "Summary"
        current_body: list[str] = []

        for line in ai_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                if current_body:
                    body_text = _clean("\n".join(current_body))
                    if body_text:
                        raw_sections.append({"heading": _clean(current_heading), "body": body_text})
                current_heading = stripped.lstrip("# ").strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            body_text = _clean("\n".join(current_body))
            if body_text:
                raw_sections.append({"heading": _clean(current_heading), "body": body_text})

        sections = raw_sections or [{"heading": "Summary", "body": _clean(ai_content)}]
        doc_title = "Conversation Report"
        filename  = "conversation_report.docx"

    else:
        # ── Custom-content Word document ──────────────────────────────────────
        history = await _get_history(chat_id)
        topic_pat = re.compile(
            r"(?:please\s+)?(?:gen[a-z]*|create|make|write|produce|give me|send me)\s+"
            r"(?:me\s+)?(?:a\s+|an\s+)?(?:word\s+)?(?:doc(?:ument)?|file|report)?\s*(?:file\s+)?\b"
            r"|\b(?:want|need)\b.{0,20}\b(?:doc(?:ument)?|word)\s+(?:file\s+)?\b"
            r"|\b(?:i want you to\s+(?:gen[a-z]*|create|make|write))\b.{0,10}\b(?:a\s+)?(?:doc(?:ument)?)?\s*",
            re.IGNORECASE,
        )
        topic = topic_pat.sub("", text).strip()
        topic = topic[:1].upper() + topic[1:] if topic else text.strip()

        status_msg_id, _ = await send_status(
            update, context, f"📝 Writing a Word document about *{topic}*…"
        )

        try:
            async with _semaphores[chat_id]:
                ai_content = await ollama_service.chat(
                    f"Write a detailed, well-structured document about: {topic}\n"
                    f"User's request: \"{text}\"\n\n"
                    "Rules:\n"
                    "- Use '## Section Name' for headings\n"
                    "- Plain prose only — no HTML, no **bold** markers\n"
                    "- Use '- item' for bullet lists\n"
                    "- Be thorough and professional",
                    history=history[-6:] if history else [],
                    model=target_model,
                )
        except Exception as exc:
            logger.exception("Ollama failed for docx chat_id=%d: %s", chat_id, exc)
            await _reply(update, "⚠️ The AI model is unavailable right now. Please try again.")
            return

        # Parse into sections
        raw_sections: list[dict] = []
        current_heading = topic
        current_body: list[str] = []
        for line in ai_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("# "):
                if current_body:
                    body = "\n".join(current_body).strip()
                    if body:
                        raw_sections.append({"heading": current_heading, "body": body})
                current_heading = stripped.lstrip("# ").strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body:
            body = "\n".join(current_body).strip()
            if body:
                raw_sections.append({"heading": current_heading, "body": body})

        sections  = raw_sections or [{"heading": topic, "body": ai_content}]
        doc_title = topic
        safe_stem = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_").lower()
        filename  = f"{safe_stem[:40]}.docx"

        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", f"[Generated Word document about: {topic}]")

    path: str | None = None
    try:
        path = await asyncio.get_event_loop().run_in_executor(
            None, generate_docx, sections, doc_title, True
        )
        safe_caption_title = doc_title
        if len(safe_caption_title) > 60:
            safe_caption_title = safe_caption_title[:57] + "..."
        
        # Register callback payload
        ref_regen = callback_registry.register("regen_docx", {"prompt": text, "chat_id": chat_id})
        keyboard = build_result_keyboard([("🔄 Regenerate", f"regen_docx:{ref_regen}")])
        caption = (
            f"📝 Here's your Word document: *{safe_caption_title}*!\n\n"
            "_⚠️ To update the Table of Contents: open in Word → Ctrl+A → F9._"
        )
        
        await replace_with_media(
            context=context,
            chat_id=chat_id,
            status_message_id=status_msg_id,
            media_path=path,
            media_type="document",
            caption=caption,
            keyboard=keyboard
        )
    except Exception as exc:
        logger.exception("DOCX generation failed chat_id=%d: %s", chat_id, exc)
        try:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ Sorry, couldn't create the Word document: {exc}"
            )
        except Exception:
            await _reply(update, f"❌ Sorry, couldn't create the Word document: {exc}")
    finally:
        _cleanup(path)


async def _handle_excel(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    model: str | None = None
) -> None:
    assert update.message is not None
    import uuid  # noqa: PLC0415

    target_model = model or (settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL)

    status_msg_id, _ = await send_status(
        update, context, "📊 Generating your spreadsheet… asking the AI to structure the data."
    )

    path: str | None = None
    try:
        async with _semaphores[chat_id]:
            logger.info("chat_id=%d  acquiring semaphore for excel gen", chat_id)
            path, title, headers, rows = await generate_excel(text, model=target_model)

        # Decision heuristic: check if any non-first column is mostly numeric (>=70% values are float parseable)
        numeric_col_idx = -1
        n_rows = len(rows)
        if n_rows > 0:
            n_cols = len(headers)
            for col_idx in range(1, n_cols):
                numeric_count = 0
                for r in rows:
                    if col_idx < len(r):
                        val = r[col_idx].strip()
                        for char in ["$", "€", "£", "¥", "%", " "]:
                            val = val.replace(char, "")
                        try:
                            float(val)
                            numeric_count += 1
                        except ValueError:
                            pass
                if numeric_count / n_rows >= 0.7:
                    numeric_col_idx = col_idx
                    break

        # Render and send visualization if applicable
        if numeric_col_idx != -1 and n_rows > 0:
            labels = [r[0] if len(r) > 0 else f"Row {idx}" for idx, r in enumerate(rows)]
            values = []
            for r in rows:
                if numeric_col_idx < len(r):
                    val = r[numeric_col_idx].strip()
                    for char in ["$", "€", "£", "¥", "%", " "]:
                        val = val.replace(char, "")
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(0.0)
                else:
                    values.append(0.0)
            
            chart_path = str(Path(settings.TMP_DIR) / f"chart_{uuid.uuid4().hex[:8]}.png")
            render_bar_chart(labels, values, f"{title} ({headers[numeric_col_idx]})", chart_path)
            
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            with open(chart_path, "rb") as cf:
                await context.bot.send_photo(chat_id=chat_id, photo=cf, caption=f"📊 Visualized: {title}")
            _cleanup(chart_path)
        elif n_rows > 0:
            # Table snapshot fallback for non-numeric/pure text tables
            snapshot_path = str(Path(settings.TMP_DIR) / f"snapshot_{uuid.uuid4().hex[:8]}.png")
            render_table_snapshot(headers, rows, title, snapshot_path)
            
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            with open(snapshot_path, "rb") as sf:
                await context.bot.send_photo(chat_id=chat_id, photo=sf, caption=f"📋 Table Snapshot: {title}")
            _cleanup(snapshot_path)

        # Register callback payloads
        ref_regen = callback_registry.register("regen_excel", {"prompt": text, "chat_id": chat_id})
        ref_edit = callback_registry.register("edit_excel", {"prompt": text, "chat_id": chat_id})

        keyboard = build_result_keyboard([
            ("🔄 Regenerate", f"regen_excel:{ref_regen}"),
            ("✏️ Edit data", f"edit_excel:{ref_edit}"),
        ])

        # Sequentially trigger upload doc chat action
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        await replace_with_media(
            context=context,
            chat_id=chat_id,
            status_message_id=status_msg_id,
            media_path=path,
            media_type="document",
            caption=f"📊 *{title}*\n_Your Excel spreadsheet is ready!_",
            keyboard=keyboard
        )
    except TableGenerationError as exc:
        logger.warning("TableGenerationError chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=(
                "❌ The AI couldn't produce well-structured table data for that request.\n\n"
                "Try rephrasing with explicit columns, e.g.:\n"
                "_\"create an excel of the top 5 programming languages with columns: Name, Year Created, Paradigm, Popularity\"_"
            )
        )
    except Exception as exc:
        logger.exception("Excel generation failed chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"❌ Couldn't generate the spreadsheet: {exc}"
        )
    finally:
        _cleanup(path)


async def _handle_styled_doc(
    update: Update, text: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE, model: str | None = None
) -> None:
    """
    Generate a style-matched document using a previously uploaded reference file.
    If no reference file is pending, fall back to plain generation with an explanation.
    """
    assert update.message is not None

    target_model = model or (settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL)

    ref_path: str | None = context.user_data.get("pending_reference") if context.user_data else None

    if not ref_path or not Path(ref_path).exists():
        # No reference — fall back to plain and explain
        await _reply(
            update,
            "ℹ️ No reference file found. I'll generate a plain report.\n\n"
            "_Tip: upload a .docx or .pdf first, then say \"generate a doc like this\" "
            "to match its style._",
            parse_mode=ParseMode.MARKDOWN,
        )
        await _handle_docx(update, chat_id, text, context=context, model=target_model)
        return

    ref = Path(ref_path)
    ext = ref.suffix.lower()

    # Extract the TOPIC from the user's message (strip boilerplate)
    # e.g. "generate a doc like this but for August month" → "August month"
    topic_pat = re.compile(
        r"(?:gen[a-z]*|create|make|write|produce)\s+(?:a\s+|an\s+)?(?:doc(?:ument)?|file|report)?\s*"
        r"(?:like\s+(?:this|that|it)|similar to|based on|matching|in the same style|using this)\b.{0,60}"
        r"|\b(?:match(?:ing)?|clone|copy|replicate|use)\s+(?:this|that|the)?\s*(?:style|format|template)?\s*",
        re.IGNORECASE,
    )
    raw_topic = topic_pat.sub("", text).strip()
    # Also strip file-name references (e.g. "Ganiyev Sharofiddin oylik reja")
    ref_name_stem = ref.stem  # e.g. "Ganiyev Sharofiddin oylik reja"
    raw_topic = re.sub(re.escape(ref_name_stem), "", raw_topic, flags=re.IGNORECASE).strip()
    # Strip leading conjunctions left over
    raw_topic = re.sub(r"^(but|and|for|\s)+", "", raw_topic, flags=re.IGNORECASE).strip()
    topic = raw_topic[:1].upper() + raw_topic[1:] if raw_topic else ref_name_stem

    status_msg_id, _ = await send_status(
        update, context, f"🎨 Generating *{topic}* in the style of *{ref.name}*…"
    )
    await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)

    # Generate topic-specific content with Ollama (not just conversation history)
    try:
        async with _semaphores[chat_id]:
            ai_content = await ollama_service.chat(
                f"Write a detailed, professional document about: {topic}\n\n"
                f"Context: The user uploaded a reference file named '{ref.name}'. "
                f"Replicate the STRUCTURE and FORMAT of that document type, but fill it with content for: {topic}\n\n"
                "Rules:\n"
                "- Use '## Section' for headings\n"
                "- Plain prose — no **bold**, no HTML\n"
                "- Be thorough and professional",
                history=await _get_history(chat_id),
                model=target_model,
            )
    except Exception as exc:
        logger.exception("Ollama failed for styled doc chat_id=%d: %s", chat_id, exc)
        await _reply(update, "⚠️ AI model unavailable. Please try again.")
        try:
            await status.delete()
        except Exception:
            pass
        return

    # Parse AI response into sections
    raw_sections: list[dict] = []
    current_heading = topic
    current_body: list[str] = []
    for line in ai_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            if current_body:
                body = "\n".join(current_body).strip()
                if body:
                    raw_sections.append({"heading": current_heading, "body": body})
            current_heading = stripped.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            raw_sections.append({"heading": current_heading, "body": body})
    sections = raw_sections or [{"heading": topic, "body": ai_content}]
    path: str | None = None
    style_note = ""
    try:
        if ext == ".docx":
            validate_docx_reference(ref_path)
            path = await asyncio.get_event_loop().run_in_executor(
                None, generate_docx_from_reference, ref_path, sections, "Style-Matched Report"
            )
            style_note = (
                f"\n\n_Styles sourced from: {ref.name}_\n"
                "_Style matching covers fonts and paragraph styles; exact layout may vary._"
            )

        elif ext == ".pdf":
            try:
                style_profile = extract_pdf_style(ref_path)
                path_str, wp_used = generate_pdf(sections, style=style_profile, title="Style-Matched Report")
                path = path_str
                style_note = (
                    f"\n\n_Style extracted from: {ref.name}_\n"
                    f"_Heading: {style_profile.heading_font} {style_profile.heading_size_pt}pt "
                    f"{style_profile.heading_color_hex}_\n"
                    f"_Body: {style_profile.body_font}_\n"
                    f"_Columns: {style_profile.columns}_\n"
                    "_Note: matches look-and-feel (fonts/colors), not pixel-exact layout._"
                )
                if not wp_used:
                    style_note += (
                        "\n_⚠️ WeasyPrint unavailable — fpdf2 fallback used. "
                        "Install: `brew install pango cairo gdk-pixbuf gobject-introspection`_"
                    )
            except StyleExtractionError as exc:
                logger.warning("Style extraction failed: %s — using plain PDF", exc)
                path_str, _ = generate_pdf(sections, title="Report")
                path = path_str
                style_note = f"\n\n_⚠️ Could not extract styles from reference: {exc}. Plain formatting used._"

        else:
            await _reply(update, f"❌ Style matching requires a .docx or .pdf reference. Got: {ext}")
            return

        filename = "styled_report.docx" if ext == ".docx" else "styled_report.pdf"
        
        # Register callback payloads
        ref_regen_styled = callback_registry.register("regen_styled_doc", {
            "prompt": text, "chat_id": chat_id, "ref_path": ref_path, "ext": ext, "topic": topic, "use_style": True
        })
        ref_regen_plain = callback_registry.register("regen_styled_doc", {
            "prompt": text, "chat_id": chat_id, "ref_path": ref_path, "ext": ext, "topic": topic, "use_style": False
        })
        
        keyboard = build_result_keyboard([
            ("🔄 Regenerate with Style", f"regen_styled:{ref_regen_styled}"),
            ("🎨 Try without reference style", f"regen_styled:{ref_regen_plain}"),
        ])
        
        await replace_with_media(
            context=context,
            chat_id=chat_id,
            status_message_id=status_msg_id,
            media_path=path,
            media_type="document",
            caption=f"🎨 Style-matched document ready!{style_note}",
            keyboard=keyboard
        )

    except StyleExtractionError as exc:
        logger.error("Style extraction failed chat_id=%d: %s", chat_id, exc)
        try:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ Couldn't extract styles from reference: {exc}"
            )
        except Exception:
            await _reply(update, f"❌ Couldn't extract styles from reference: {exc}")
    except Exception as exc:
        logger.exception("Styled doc generation failed chat_id=%d: %s", chat_id, exc)
        try:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ Couldn't generate styled document: {exc}"
            )
        except Exception:
            await _reply(update, f"❌ Couldn't generate styled document: {exc}")
    finally:
        _cleanup(path)


# ── File upload handler (bypasses intent parser) ──────────────────────────────

async def _handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any document upload.

    Flow:
    1. validate_upload() pre-download (size + extension) — no network if invalid
    2. Download to /tmp/aiagent/uploads/
    3. read_file() → ExtractedDocument
    4. If file is a reference-style candidate (.docx/.pdf), store in user_data
    5. Build prompt + call Ollama; reply with analysis
    6. Catch UnreadableFileError specifically for clear user messages
    """
    assert update.message is not None
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    doc: Document | None = update.message.document
    if doc is None:
        return

    filename = doc.file_name or "upload"
    file_size = doc.file_size

    # ── 1. Pre-download validation ─────────────────────────────────────────────
    valid, err_msg = validate_upload(file_size, filename)
    if not valid:
        await _reply(update, err_msg or "❌ Invalid file.")
        return

    # ── 2. Download ────────────────────────────────────────────────────────────
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    import uuid  # noqa: PLC0415
    safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    tmp_path = str(_UPLOAD_DIR / safe_name)

    status = await update.message.reply_text(
        f"📂 Received *{filename}* — downloading and reading…",
        parse_mode=ParseMode.MARKDOWN,
    )
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(tmp_path)

    except Exception as exc:
        logger.exception("Download failed for %s: %s", filename, exc)
        await _reply(update, f"❌ Failed to download the file: {exc}")
        try:
            await status.delete()
        except Exception:
            pass
        return

    # ── 3. Extract text ────────────────────────────────────────────────────────
    extracted: ExtractedDocument | None = None
    try:
        extracted = read_file(tmp_path)

    except UnreadableFileError as exc:
        logger.warning("UnreadableFileError %s: %s", filename, exc)
        await _reply(
            update,
            f"❌ Couldn't read *{filename}*: {exc}\n\n"
            "The file may be:\n"
            "• Password-protected\n"
            "• Corrupted or invalid\n"
            "• Containing only images/scans without OCR text",
            parse_mode=ParseMode.MARKDOWN,
        )
        _cleanup(tmp_path)
        try:
            await status.delete()
        except Exception:
            pass
        return

    except Exception as exc:
        logger.exception("Unexpected read error %s: %s", filename, exc)
        await _reply(update, "⚠️ Something went wrong while reading the file. Please try again.")
        _cleanup(tmp_path)
        try:
            await status.delete()
        except Exception:
            pass
        return

    # ── 4. Store as pending reference (for style-clone flow) ──────────────────
    ext = Path(filename).suffix.lower()
    if ext in (".docx", ".pdf") and context.user_data is not None:
        context.user_data["pending_reference"] = tmp_path
        logger.info(
            "chat_id=%d: stored pending reference file: %s", chat_id, filename
        )
        ref_notice = (
            f"\n\n_📎 Tip: this file is now saved as a style reference. "
            f"Say \"generate a doc like this\" or \"match this style\" to clone its formatting._"
        )
    else:
        ref_notice = ""

    # ── 5. User's question / default ──────────────────────────────────────────
    user_question = (update.message.caption or "").strip()
    if not user_question:
        user_question = "Summarize this document and highlight the most important points."

    target_model = settings.DEEP_MODEL if should_use_deep_model(user_question) else settings.DEFAULT_MODEL

    last_model = await history_db.get_last_model(chat_id)
    prev_model = last_model or settings.DEFAULT_MODEL
    if prev_model != target_model:
        icon = "🧠" if target_model == settings.DEEP_MODEL else "⚡"
        model_name = "deep reasoning model" if target_model == settings.DEEP_MODEL else "fast conversational model"
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{icon} Switching to the {model_name} (`{target_model}`), this may take a few extra seconds to load..."
        )

    full_prompt = build_llm_prompt(extracted, filename, user_question)

    # ── 6. Call Ollama ────────────────────────────────────────────────────────
    try:
        async with _semaphores[chat_id]:
            logger.info("chat_id=%d  acquiring semaphore for file analysis", chat_id)
            async with TypingRefresher(context.bot, chat_id):
                reply = await ollama_service.chat(full_prompt, history=[], model=target_model)
    except Exception as exc:
        logger.exception("Ollama failed during file analysis chat_id=%d: %s", chat_id, exc)
        await _reply(update, "⚠️ The AI model is unavailable right now. Please try again.")
        _cleanup(tmp_path)
        try:
            await status.delete()
        except Exception:
            pass
        return

    await _append(chat_id, "user", f"[Uploaded: {filename}] {user_question}")
    await _append(chat_id, "assistant", reply, model=target_model)

    # Prepend any warnings from extraction
    warning_text = ""
    if extracted.warnings:
        warning_text = "⚠️ " + " | ".join(extracted.warnings) + "\n\n"

    # Register callback payload for follow-up questions
    ref_id = callback_registry.register("file_followup", {"file_path": tmp_path, "filename": filename})
    keyboard = build_result_keyboard([("❓ Ask another question", f"file_followup:{ref_id}")])

    await _send_html_reply(update, context, chat_id, warning_text + reply + ref_notice, reply_markup=keyboard)

    try:
        await status.delete()
    except Exception:
        pass

    # Keep the file in tmp so follow-up questions can read it. It will be swept later by periodic cleanup.
    logger.info("Preserving uploaded file at %s for follow-up questions", tmp_path)


# ── Photo handler stub ─────────────────────────────────────────────────────────
# Vision analysis is NOT in scope for Phase 2. This handler gives a clear
# "not supported yet" reply instead of silence or a crash.

async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.message is not None
    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    if not update.message.photo:
        return

    caption = (update.message.caption or "").strip()
    if caption:
        intent = parse_intent(caption)
        if intent == Intent.GENERATE_IMAGE:
            await _handle_image(update, caption, chat_id, context)
            return

    # The last photo in the list is the highest resolution
    photo_file = update.message.photo[-1]
    
    if photo_file.file_size and photo_file.file_size > 20 * 1024 * 1024:
        await _reply(update, "❌ The photo is too large (exceeds 20MB). Please compress it.")
        return

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    import uuid  # noqa: PLC0415
    safe_name = f"photo_{uuid.uuid4().hex[:8]}.jpg"
    tmp_path = str(_UPLOAD_DIR / safe_name)

    status = await update.message.reply_text(
        "👁️ Received image — downloading and analyzing..."
    )
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        tg_file = await photo_file.get_file(read_timeout=60, write_timeout=60, connect_timeout=30)
        await tg_file.download_to_drive(tmp_path, read_timeout=60, write_timeout=60)
    except Exception as exc:
        logger.exception("Photo download failed: %s", exc)
        await _reply(update, f"❌ Failed to download the photo: {exc}")
        try:
            await status.delete()
        except Exception:
            pass
        return

    user_question = (update.message.caption or "").strip()
    if not user_question:
        user_question = "Describe this image in detail and tell me what you see."

    try:
        # Use Gemini Flash for vision (free, better celebrity/scene recognition)
        # Falls back to a helpful message if GEMINI_API_KEY is not set
        async with TypingRefresher(context.bot, chat_id):
            reply = await gemini_vision_service.vision_chat(user_question, tmp_path)
    except Exception as exc:
        logger.exception("Gemini vision analysis failed chat_id=%d: %s", chat_id, exc)
        await _reply(
            update,
            "⚠️ The vision model is currently unavailable. Please try again in a moment."
        )
        _cleanup(tmp_path)
        try:
            await status.delete()
        except Exception:
            pass
        return

    await _append(chat_id, "user", f"[Sent Photo] {user_question}")
    await _append(chat_id, "assistant", reply)

    await _send_html_reply(update, context, chat_id, reply, reply_markup=_get_chat_keyboard())

    try:
        await status.delete()
    except Exception:
        pass

    _cleanup(tmp_path)


# ── Sticker handler ───────────────────────────────────────────────────────────

async def _handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any Telegram sticker (static, animated, video, custom emoji)."""
    assert update.message is not None
    assert update.message.sticker is not None
    assert update.effective_chat is not None

    sticker = update.message.sticker
    chat_id = update.effective_chat.id
    emoji   = sticker.emoji or "🎭"
    set_name = sticker.set_name or "unknown"

    # ── Static sticker (WEBP image) — send to Gemini Vision ───────────────────
    if not sticker.is_animated and not sticker.is_video:
        status = await update.message.reply_text("🖼️ Analyzing your sticker…")
        await update.message.chat.send_action(ChatAction.TYPING)

        safe_name = f"sticker_{uuid.uuid4().hex[:8]}.webp"
        tmp_path  = str(_UPLOAD_DIR / safe_name)

        try:
            tg_file = await sticker.get_file()
            await tg_file.download_to_drive(tmp_path)
        except Exception as exc:
            logger.exception("Sticker download failed: %s", exc)
            await _reply(update, f"❌ Couldn't download the sticker: {exc}")
            try:
                await status.delete()
            except Exception:
                pass
            return

        # Caption or generic question
        user_question = (update.message.caption or "").strip()
        if not user_question:
            user_question = (
                f"This is a Telegram sticker (emoji: {emoji}, pack: {set_name}). "
                "Describe what you see — the character, expression, mood, and art style."
            )

        try:
            async with TypingRefresher(context.bot, chat_id):
                reply = await gemini_vision_service.vision_chat(user_question, tmp_path)
        except Exception as exc:
            logger.exception("Vision analysis for sticker failed chat_id=%d: %s", chat_id, exc)
            reply = (
                f"{emoji} I received your sticker from the *{set_name}* pack, "
                "but the vision model is temporarily unavailable to analyze it."
            )
        finally:
            _cleanup(tmp_path)
            try:
                await status.delete()
            except Exception:
                pass

        await _append(chat_id, "user", f"[Sticker {emoji}] {user_question}")
        await _append(chat_id, "assistant", reply)
        await _send_html_reply(update, context, chat_id, reply, reply_markup=_get_chat_keyboard())
        return

    # ── Animated (TGS) or Video (WEBM) sticker — react with Ollama ────────────
    sticker_type = "animated" if sticker.is_animated else "video"

    target_model = settings.DEFAULT_MODEL

    try:
        async with _semaphores[chat_id]:
            async with TypingRefresher(context.bot, chat_id):
                reply = await ollama_service.chat(
                    f"The user sent a {sticker_type} Telegram sticker. "
                    f"Its associated emoji is: {emoji}. "
                    f"The sticker pack name is: {set_name}. "
                    "React naturally and creatively to this sticker — treat it like the user "
                    "sent you that emoji as an expression. Keep it short and fun (1-2 sentences).",
                    history=await _get_history(chat_id),
                    model=target_model,
                )
    except Exception as exc:
        logger.warning("Ollama reaction to sticker failed: %s", exc)
        reply = f"{emoji} Nice sticker!"

    await _append(chat_id, "user", f"[{sticker_type.capitalize()} sticker {emoji}]")
    await _append(chat_id, "assistant", reply, model=target_model)
    await _send_html_reply(update, context, chat_id, reply, reply_markup=_get_chat_keyboard())


# ── Voice Message Transcription & Handler ────────────────────────────────────

async def _handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Download voice note / audio, transcribe it to text, and route through dispatch().
    """
    from services.voice_service import transcribe_voice_message  # noqa: PLC0415
    assert update.message is not None
    assert update.effective_chat is not None

    chat_id = update.effective_chat.id
    voice = update.message.voice or update.message.audio
    if not voice:
        return

    status_msg_id, _ = await send_status(update, context, "🎙 Transcribing voice message…")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    tmp_path = None
    try:
        # Download voice file
        file_obj = await voice.get_file(read_timeout=30.0)
        ext = ".ogg" if update.message.voice else (Path(voice.file_name or "audio.mp3").suffix or ".mp3")
        tmp_path = os.path.join(settings.TMP_DIR, f"voice_{voice.file_unique_id}{ext}")
        await file_obj.download_to_drive(tmp_path)

        # Transcribe audio to text
        transcribed_text = await transcribe_voice_message(tmp_path)
        logger.info("chat_id=%d  voice transcribed: %r", chat_id, transcribed_text)

        if not transcribed_text:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text="🔇 Could not recognize speech in the audio. Please try speaking clearly."
            )
            return

        # Check if user explicitly entered reply mode for a Telegram contact
        pending_custom = context.user_data.pop("pending_tg_custom_target", None) if context.user_data else None
        if pending_custom:
            target_name = pending_custom.get("name", "Contact")
            target_id = pending_custom.get("user_id")
            confidence = pending_custom.get("match_confidence")

            pending = PendingSend(
                user_id=chat_id,
                recipient_entity_id=target_id,
                recipient_display_name=target_name,
                match_confidence=confidence,
                final_text=transcribed_text,
                source="custom_voice",
            )
            pending_send_registry.set(pending)
            card_text, card_keyboard = build_confirmation_card(pending)

            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg_id,
                text=card_text,
                parse_mode=ParseMode.HTML,
                reply_markup=card_keyboard,
            )
            return

        # Show transcription feedback to user
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"🎙 <i>\"{html.escape(transcribed_text)}\"</i>\n\n⚙️ <i>Processing request…</i>",
            parse_mode=ParseMode.HTML,
        )

        # Mark voice origin so if dispatch resolves a SEND_TELEGRAM_MESSAGE intent, source is tagged as voice_auto_route
        if context.user_data is not None:
            context.user_data["_is_voice_dispatch"] = True
        try:
            # Route the transcribed text directly through the agent dispatch
            await dispatch(update, context, forced_text=transcribed_text)
        finally:
            if context.user_data is not None:
                context.user_data.pop("_is_voice_dispatch", None)

        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
        except Exception:
            pass



    except Exception as exc:
        logger.exception("Voice processing failed chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"❌ Voice processing failed: {exc}"
        )
    finally:
        _cleanup(tmp_path)


# ── Personal Telegram UserBot Messenger ──────────────────────────────────────

def _clean_contact_name(raw_name: str) -> str:
    if not raw_name:
        return ""
    name = raw_name.strip().strip("\"'")
    name = re.sub(
        r"^(?:контакту|контакт|пользователю|пользователь|человеку|человек|другу|друг|коллеге|коллега|"
        r"nomli\s+kontaktni?|kontakt(?:ini|ni|ga)?|"
        r"to\s+the\s+contact|the\s+contact|to\s+contact|contact|to\s+user|the\s+user|user|to\s+person|person|"
        r"named|called|name|с\s+именем|по\s+имени|ismidagi)\s+",
        "", name, flags=re.IGNORECASE
    ).strip()
    name = re.sub(r"^(?:named|called|name|с\s+именем|по\s+имени|nomli)\s+", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"(?:ga|ka|qa)$", "", name, flags=re.IGNORECASE).strip()

    STOPWORDS_LOCAL = {
        "to", "the", "a", "an", "for", "in", "on", "at", "by", "from", "of",
        "contact", "contacts", "user", "users", "person", "people",
        "message", "messages", "text", "texts", "dm", "voice",
        "him", "her", "them", "me", "it",
        "kontakt", "kontaktga", "kontaktni", "odam", "kishi",
        "контакт", "контакту", "пользователь", "пользователю", "человек", "человеку",
        "сообщение", "текст", "смс",
    }
    if name.lower() in STOPWORDS_LOCAL or len(name) <= 1:
        return ""
    return name


def _extract_contact_and_message_regex(text: str) -> tuple[str, str]:
    """
    Extract (contact_name, message_text) from multilingual natural language requests (Uzbek, Russian, English).
    """

    # 0. Normalize unicode / smart quotes
    for q in ['“', '”', '«', '»', '„']:
        text = text.replace(q, '"')
    for q in ['‘', '’', '`']:
        text = text.replace(q, "'")

    # 1. Two quotes: "Sherdor" "salom"
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    if len(quotes) >= 2:
        c = _clean_contact_name(quotes[0])
        if c:
            return c, quotes[1].strip()

    # 2. English: "Send a message to the contact named TR number and say hi how are you"
    m_en_named = re.search(
        r'^(?:send\s+(?:a\s+)?(?:message|msg|text)\s+to\s+(?:the\s+)?(?:contact|person|user)?\s*(?:named|called|name)?\s*|text\s+|message\s+|dm\s+)'
        r'([A-Za-z0-9_\s]{2,30}?)'
        r'(?:\s+(?:and\s+say|saying|that\s+says?|to\s+say|and\s+tell\s+(?:him|her|them)?|say|that|:)\s+|:\s*)(.+)$',
        text, re.IGNORECASE
    )
    if m_en_named:
        c = _clean_contact_name(m_en_named.group(1))
        if c:
            return c, m_en_named.group(2).strip().strip("'\"")

    # 3. Colon separator: e.g. "напиши контакту Алишер Свифт: привет как дела" or "send to Alisher Swift: hi"
    m_col = re.search(
        r'^(?:напиши|отправь|скинь|пошли|напомни|скажи|сообщи|перешли|text|message|dm|send\s+(?:a\s+)?(?:message|msg|text)\s+to)\s+(?:контакту|контакт|пользователю|пользователь|человеку|человек|to\s+contact|contact)?\s*([A-Za-zА-Яа-я0-9_\s]+?)\s*:\s*(.+)$',
        text, re.IGNORECASE
    )
    if m_col:
        c = _clean_contact_name(m_col.group(1))
        if c and 1 <= len(c.split()) <= 4:
            return c, m_col.group(2).strip().strip("'\"")

    # 4. Greeting separator: "напиши Sherdor привет как дела" / "text Alisher Swift hello how are you"
    m_greet = re.search(
        r'^(?:напиши|отправь|скинь|пошли|напомни|скажи|сообщи|перешли|text|message|dm|send\s+(?:a\s+)?(?:message|msg|text)\s+to)\s+(?:контакту|контакт|пользователю|пользователь|человеку|человек|to\s+contact|contact)?\s*([A-Za-zА-Яа-я0-9_\s]+?)\s+(?:,\s*)?(привет|здравствуй|добрый|салом|salom|hello|hi|hey|assalomu)\b(.+)$',
        text, re.IGNORECASE
    )
    if m_greet:
        c = _clean_contact_name(m_greet.group(1))
        if c and 1 <= len(c.split()) <= 3:
            return c, (m_greet.group(2) + m_greet.group(3)).strip().strip("'\"")

    # 5. Uzbek patterns
    m_uz1 = re.search(
        r'["\']?([A-Za-zА-Яа-я0-9_\s]+?)["\']?\s*(?:kontakt(?:ini|ni)?|nomli\s+kontaktni?)?\s*(?:top|topgin|qidir)\s+va\s+["\']?(.+?)["\']?\s*(?:deb\s+)?(?:yoz|yozgin|yubor|yuborgin|jo\'?nat|jo\'?natgin)\b',
        text, re.IGNORECASE
    )
    if m_uz1:
        c = _clean_contact_name(m_uz1.group(1))
        if c:
            return c, m_uz1.group(2).strip().strip("'\"")

    m_uz2 = re.search(
        r'["\']?([A-Za-zА-Яа-я0-9_\s]+?)(?:ga|ka|qa)\s+["\']?(.+?)["\']?\s*(?:deb\s+)?(?:yoz|yozgin|yubor|yuborgin|jo\'?nat|jo\'?natgin)\b',
        text, re.IGNORECASE
    )
    if m_uz2:
        c = _clean_contact_name(m_uz2.group(1))
        if c:
            return c, m_uz2.group(2).strip().strip("'\"")

    # 6. Russian patterns
    m_ru = re.search(
        r'(?:найди|поищи)\s+(?:контакт|человека)?\s*["\']?([A-Za-zА-Яа-я0-9_\s]+?)["\']?\s+и\s+(?:напиши|отправь|скинь)\s+(?:ему|ей|им)?\s*["\']?(.+?)["\']?$',
        text, re.IGNORECASE
    )
    if m_ru:
        c = _clean_contact_name(m_ru.group(1))
        if c:
            return c, m_ru.group(2).strip().strip("'\"")

    m_ru_explicit = re.search(
        r'^(?:напиши|отправь|скинь|пошли|напомни|скажи|сообщи|перешли)\s+(?:контакту|контакт|пользователю|пользователь|человеку|человек)?\s*([A-Za-zА-Яа-я0-9_\s]+?)\s+(?:что|о\s+том\s+что|чтобы|о|об)\s+(.+)$',
        text, re.IGNORECASE
    )
    if m_ru_explicit:
        c = _clean_contact_name(m_ru_explicit.group(1))
        if c and 1 <= len(c.split()) <= 3:
            return c, m_ru_explicit.group(2).strip().strip("'\"")

    # 7. English general patterns
    m_en1 = re.search(
        r'(?:find\s+(?:a\s+)?(?:contact|person|user)\s+(?:called|named)\s+)(.+?)'
        r'\s+and\s+(?:text|message|send|write)\s+(?:him|her|them|to\s+him|to\s+her)?\s*["\']?(.+?)["\']?$',
        text, re.IGNORECASE
    )
    if m_en1:
        c = _clean_contact_name(m_en1.group(1))
        if c:
            return c, m_en1.group(2).strip().strip("'\"")

    m_en2 = re.search(
        r'send\s+(?:a\s+)?(?:message|msg|text)\s+to\s+(.+?)(?:\s+saying\s+|\s+that\s+says?\s+|\s+that\s+)([\"\'].+?[\"\']|.{3,}?)$',
        text, re.IGNORECASE
    )
    if m_en2:
        c = _clean_contact_name(m_en2.group(1))
        if c:
            return c, m_en2.group(2).strip().strip("'\"")

    return "", ""


async def _extract_contact_and_message(text: str) -> tuple[str, str]:
    """Extract contact name and message text using regex first, and robust LLM JSON fallback."""
    import json as _json  # noqa: PLC0415
    c, m = _extract_contact_and_message_regex(text)
    if c and m and len(c.split()) <= 4:
        return c, m

    # Fallback to high-accuracy structured few-shot LLM extraction
    prompt = (
        "Extract the recipient contact name and the message to send from the user's command.\n"
        "Output ONLY JSON: {\"contact\": \"<name>\", \"message\": \"<message>\"}\n\n"
        "Examples:\n"
        "Command: \"Send a message to the contact named TR number say hi how are you\"\n"
        "JSON: {\"contact\": \"TR number\", \"message\": \"hi how are you\"}\n\n"
        "Command: \"Send a message to the contact name Turkish number and say Hi, how are you? How it's going?\"\n"
        "JSON: {\"contact\": \"Turkish number\", \"message\": \"Hi, how are you? How it's going?\"}\n\n"
        "Command: \"Найди контакт с именем Ходжикабир Биз и отправь ему сообщение Привет как дела\"\n"
        "JSON: {\"contact\": \"Ходжикабир Биз\", \"message\": \"Привет как дела\"}\n\n"
        "Command: \"Ali ga salom deb yoz\"\n"
        "JSON: {\"contact\": \"Ali\", \"message\": \"salom\"}\n\n"
        f"Command: \"{text}\"\n"
        "JSON:"
    )
    try:
        resp = await ollama_service.chat(prompt, history=[], model=settings.DEFAULT_MODEL)
        resp = resp.strip()
        if "```" in resp:
            resp = resp.split("```")[1]
            if resp.startswith("json"):
                resp = resp[4:]
        data = _json.loads(resp.strip())
        extracted_c = _clean_contact_name(str(data.get("contact", "")))
        extracted_m = str(data.get("message", "")).strip()
        if extracted_c or extracted_m:
            return extracted_c or c, extracted_m or m
    except Exception as e:
        logger.debug("LLM extraction fallback exception: %s", e)
    return c, m



async def _reformulate_message(raw_instruction: str, recipient_name: str) -> str:
    """
    Use AI to turn a 3rd-person instruction into a proper direct message.
    """
    prompt = (
        f"You are a Telegram messaging assistant. The user wants to send a direct message to '{recipient_name}'.\n"
        f"User instruction: \"{raw_instruction}\"\n\n"
        "Rewrite this as a direct, polite Telegram message to the recipient.\n"
        "Rules:\n"
        "- Convert 3rd-person references ('he/she must', 'tell them') to 2nd-person ('You must', 'Please...').\n"
        "- If it is already a direct message, keep it as is.\n"
        "- Keep the same language (Uzbek, Russian, or English).\n"
        "- Do NOT add conversational replies like 'Here is your message' or 'Let me know if you need help'.\n"
        "- Output ONLY the message text."
    )
    try:
        result = await ollama_service.chat(prompt, history=[], model=settings.DEFAULT_MODEL)
        result = result.strip().strip('"\'')
        if result.startswith("```"):
            lines = result.splitlines()
            result = "\n".join(lines[1:-1] if len(lines) > 2 else lines)
        result = re.sub(r"^(?:Here is the message|Message):\s*", "", result, flags=re.IGNORECASE).strip().strip('"\'')
        if result and len(result) >= 2:
            return result
    except Exception as e:
        logger.warning("Message reformulation failed: %s", e)
    return raw_instruction


async def _handle_send_telegram_message(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Search personal Telegram contacts and send a direct message on behalf of the user.
    """
    from services import telethon_service  # noqa: PLC0415
    assert update.message is not None

    # Check session exists
    if not await telethon_service.is_connected():
        await update.message.reply_text(
            "⚠️ *UserBot not set up yet.*\n\n"
            "Run this command in your terminal to complete the one-time login:\n"
            "```\n.venv/bin/python setup_userbot.py\n```\n"
            "Then send your message again.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    contact_name, message_body = await _extract_contact_and_message(text)

    # If contact_name is a pronoun or empty while text contains pronouns → resolve to last contact
    PRONOUNS_SET = {
        "ей", "ему", "им", "её", "его", "нем", "ней",
        "him", "her", "them", "it",
        "uni", "unga", "ulardan", "ularga", "ularni", "o'shanga"
    }
    is_pronoun_contact = (contact_name.lower().strip() in PRONOUNS_SET) or (
        not contact_name and bool(re.search(r"\b(ей|ему|им|её|его|him|her|them|uni|unga|ularga|ularni)\b", text, re.IGNORECASE))
    )

    if is_pronoun_contact:
        last_contact = context.user_data.get("last_tg_contact_name", "") if context.user_data else ""
        if not last_contact:
            # Scan recent history entries for a previously sent-to name
            history = await _get_history(chat_id)
            for entry in reversed(history[-10:]):
                m = re.search(r"Sent message to ([^:\"]+)", entry.get("content", ""))
                if m:
                    last_contact = m.group(1).strip()
                    break
        if last_contact:
            contact_name = last_contact
        else:
            contact_name = ""

    # If still no message_body but we have contact_name (e.g. "напомни ей о встрече в 16:00")
    if not message_body and contact_name:
        stripped = re.sub(
            r"^(напомни|скажи|сообщи|перешли|text|message|dm|send|напиши|отправь|передай)\s+",
            "", text, flags=re.IGNORECASE
        ).strip()
        stripped = re.sub(r"\b(ей|ему|им|её|его|her|him|them|uni|unga)\b", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"^(что\s+|о\s+том\s+что\s+|о\s+|об\s+)", "", stripped, flags=re.IGNORECASE).strip()
        if len(stripped) > 3:
            message_body = stripped

    if not contact_name and not message_body:
        await update.message.reply_text(
            "❓ I couldn't figure out *who* to message or *what* to say.\n\n"
            "Try:\n"
            "• *\"find a contact called Ali and text him hi!\"*\n"
            "• *\"send a message to John: how are you?\"*\n"
            "• *\"text Sarah good morning!\"*",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Pending state: if no contact_name yet, ask who to text
    if not contact_name:
        context.user_data["pending_tg_message"] = message_body
        await update.message.reply_text(
            f"📨 Got the message: *\"{message_body}\"*\n\nWho should I send this to? (type the contact name)",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status = await update.message.reply_text(
        f"🔍 Searching for *{contact_name}* in your Telegram contacts…",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        matches = await telethon_service.find_contact(contact_name)

        if not matches:
            if context.user_data is not None and message_body:
                context.user_data["pending_tg_message"] = message_body
            await status.edit_text(
                f"❌ No contact found matching *{contact_name}* in your Telegram contacts.\n\n"
                f"💬 Message to send: _{message_body}_\n\n"
                "Please reply with their exact name, username (@username), or phone number to send.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        best = matches[0]

        # If multiple strong ambiguous matches exist, ask for confirmation
        needs_disambiguation = False
        if len(matches) > 1 and matches[1]["_score"] >= 40:
            top_score = matches[0]["_score"]
            second_score = matches[1]["_score"]
            if top_score < 75 or (top_score - second_score < 15):
                needs_disambiguation = True

        if needs_disambiguation:
            options = "\n".join(
                f"  {i+1}. {r['name']}" + (f" (@{r['username']})" if r["username"] else "")
                for i, r in enumerate(matches[:5])
            )
            # Escape markdown in names/usernames to prevent parse crashes
            clean_options = options.replace("_", "\\_").replace("*", "\\*")
            clean_c = contact_name.replace("_", "\\_").replace("*", "\\*")
            context.user_data["pending_tg_contacts"] = matches[:5]
            context.user_data["pending_tg_message_for_contact"] = message_body
            await status.edit_text(
                f"👥 Found multiple matches for *{clean_c}*:\n\n{clean_options}\n\n"
                "Reply with the number of the person you want to message.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return


        # Single clear match — reformulate instruction into proper outgoing message
        recipient_name = best["name"]
        recipient_label = recipient_name
        if best.get("username"):
            recipient_label += f" (@{best['username']})"

        final_message = await _reformulate_message(message_body, recipient_name)

        is_voice = bool(context.user_data and context.user_data.get("_is_voice_dispatch"))
        pending = PendingSend(
            user_id=chat_id,
            recipient_entity_id=best["entity"],
            recipient_display_name=recipient_label,
            match_confidence=best.get("_score"),
            final_text=final_message,
            source="voice_auto_route" if is_voice else "custom_text",
        )
        pending_send_registry.set(pending)

        # Save for pronoun-based follow-ups ("напомни ей...", "text him also...")
        if context.user_data is not None:
            context.user_data["last_tg_contact_name"] = recipient_name

        card_text, card_keyboard = build_confirmation_card(pending)
        await status.edit_text(
            card_text,
            parse_mode=ParseMode.HTML,
            reply_markup=card_keyboard,
        )

    except Exception as exc:
        logger.exception("_handle_send_telegram_message failed: %s", exc)
        await status.edit_text(
            f"❌ Failed to prepare message: {exc}",
        )



# ── Personal Telegram UserBot Reader ─────────────────────────────────────────

def _extract_contact_to_read(text: str) -> str:
    """Extract the target contact name from a read messages request."""
    for q in ['“', '”', '«', '»', '„']:
        text = text.replace(q, '"')
    for q in ['‘', '’', '`']:
        text = text.replace(q, "'")

    # 1. If in quotes
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    if quotes:
        return quotes[0].strip()

    # 2. Russian: что написал X / покажи сообщения от X
    m_ru = re.search(
        r'(?:что\s+(?:мне\s+)?(?:написал|написала|прислал|прислала|отправил|отправила|сказал|сказала)|покажи\s+(?:сообщения\s+)?(?:от|с)|чат\s+с)\s+([A-Za-zА-Яа-я0-9_\s]+)',
        text, re.IGNORECASE
    )
    if m_ru:
        return m_ru.group(1).strip()

    # 3. Uzbek: X nima deb yozdi / Xdan kelgan xabarlar
    m_uz = re.search(
        r'([A-Za-zА-Яа-я0-9_\s]+?)(?:dan|bilan|\s+nima|\s+nimani)',
        text, re.IGNORECASE
    )
    if m_uz:
        cand = m_uz.group(1).strip()
        if cand.lower() not in {"telegram", "kontakt", "xabar"}:
            return cand

    # 4. English: what did X send / read messages from X
    m_en = re.search(
        r'(?:what\s+(?:did\s+)?|messages?\s+(?:from|with)\s+|chat\s+with\s+)(.+?)(?:\s+send|\s+sent|\s+text|\s+write|\s+say|\?|$)',
        text, re.IGNORECASE
    )
    if m_en:
        return m_en.group(1).strip()

    return text.strip()


async def _handle_read_telegram_messages(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Search for contact, fetch recent messages from personal Telegram, and display them.
    """
    from services import telethon_service  # noqa: PLC0415
    assert update.message is not None

    if not await telethon_service.is_connected():
        await update.message.reply_text(
            "⚠️ UserBot is not logged in. Please run `setup_userbot.py`.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    contact_query = _extract_contact_to_read(text)
    if not contact_query:
        await update.message.reply_text(
            "❓ Please specify which contact's messages you want to read (e.g. `what did \"Sherdor\" send me?`).",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    status_msg_id, _ = await send_status(update, context, f"🔍 Fetching recent messages with *{contact_query}*…")

    try:
        contact_info, msgs = await telethon_service.get_contact_messages(contact_query, limit=6)
        if not contact_info:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"❌ No contact or chat found matching *{contact_query}* in your Telegram."
            )
            return

        if not msgs:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=f"ℹ️ No recent messages found in your chat with *{contact_info['name']}*."
            )
            return

        lines = [f"💬 **Recent messages with {contact_info['name']}:**\n"]
        for m in msgs:
            prefix = "🟢 **You**" if m["is_me"] else f"👤 **{contact_info['name']}**"
            time_tag = f" _({m['date']})_" if m["date"] else ""
            lines.append(f"{prefix}{time_tag}:\n\"{m['text']}\"\n")

        full_reply = "\n".join(lines)

        # Store in history so user can ask follow-ups
        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", full_reply)

        await _send_html_reply(update, context, chat_id, full_reply)
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
        except Exception:
            pass

    except Exception as exc:
        logger.exception("_handle_read_telegram_messages failed: %s", exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"❌ Failed to fetch messages: {exc}"
        )


# ── Local Project Analyzer ───────────────────────────────────────────────────

async def _handle_local_project(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    model: str | None = None
) -> None:
    """
    Scan, inspect, and analyze a local project on the user's Mac.
    """
    from services.local_project_service import resolve_local_project_path, analyze_local_project  # noqa: PLC0415
    assert update.message is not None

    status_msg_id, _ = await send_status(
        update, context, "🔍 Locating and inspecting local project on your Mac…"
    )
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        project_path = resolve_local_project_path(text)
        if not project_path:
            await edit_to_result_text(
                context=context,
                chat_id=chat_id,
                message_id=status_msg_id,
                text=(
                    "⚠️ Could not locate a project folder matching your query on your Mac.\n\n"
                    "_Tip: Specify the folder name or path, e.g.:_\n"
                    "`~/Desktop/jamgarma/yangiavlod` or `analyze project yangiavlod`"
                )
            )
            return

        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"📂 Found `{project_path.name}` at `{project_path}`.\n🧠 Analyzing codebase architecture & dependencies…",
            parse_mode=ParseMode.MARKDOWN
        )

        async with _semaphores[chat_id]:
            analysis, info = await analyze_local_project(project_path, user_query=text, model=model)

        # Store context in chat history for follow-up questions
        await _append(chat_id, "user", f"[Local Project: {info['project_name']}] {text}")
        await _append(chat_id, "assistant", analysis)

        # Register callback for PDF export
        ref_pdf = callback_registry.register(
            "regen_pdf",
            {"prompt": f"Write an executive technical specification and architecture report for project {info['project_name']}:\n{analysis}", "chat_id": chat_id}
        )
        keyboard = build_result_keyboard([("📄 Export as PDF Report", f"regen_pdf:{ref_pdf}")])

        header = f"🚀 **Project Analysis:** `{info['project_name']}`\n📍 *{info['absolute_path']}* ({info['total_files']} files)\n\n"
        full_text = header + analysis

        await _send_html_reply(update, context, chat_id, full_text, reply_markup=keyboard)
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
        except Exception:
            pass

    except Exception as exc:
        logger.exception("Local project analysis failed chat_id=%d: %s", chat_id, exc)
        await edit_to_result_text(
            context=context,
            chat_id=chat_id,
            message_id=status_msg_id,
            text=f"❌ Error analyzing local project: {exc}"
        )


# ── Text dispatcher ───────────────────────────────────────────────────────────

async def dispatch(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    forced_model: str | None = None,
    forced_text: str | None = None,
) -> None:
    text = forced_text if forced_text is not None else (update.message.text.strip() if update.message and update.message.text else "")
    if not text:
        return

    if text.lower() in ("cancel", "stop", "exit", "quit"):
        clear_pending_states(context)
        await update.message.reply_text("🛑 Cancelled pending action.")
        return

    assert update.effective_chat is not None
    chat_id = update.effective_chat.id

    # Track owner chat ID so Telethon background notifications know where to reach the user
    asyncio.create_task(history_db.set_global_setting("owner_chat_id", str(chat_id)))

    # Check for pending custom message to a personal Telegram contact
    pending_custom = context.user_data.pop("pending_tg_custom_target", None) if context.user_data else None
    if pending_custom:
        target_name = pending_custom.get("name", "Contact")
        target_id = pending_custom.get("user_id")
        confidence = pending_custom.get("match_confidence")

        pending = PendingSend(
            user_id=chat_id,
            recipient_entity_id=target_id,
            recipient_display_name=target_name,
            match_confidence=confidence,
            final_text=text,
            source="custom_text",
        )
        pending_send_registry.set(pending)

        card_text, card_keyboard = build_confirmation_card(pending)
        await update.message.reply_text(
            card_text,
            parse_mode=ParseMode.HTML,
            reply_markup=card_keyboard,
        )
        return


    # ── Persona / name change detection ──────────────────────────────────────
    new_name = _detect_persona_change(text)
    if new_name:
        await history_db.set_setting(chat_id, "persona_name", new_name)
        await _reply(
            update,
            f"🌟 Got it! I'll be **{new_name}** from now on. "
            f"Every reply in this chat will come from {new_name}. "
            "You can change my name again any time!",
        )
        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", f"Got it! I'll be {new_name} from now on.")
        return

    # Load saved persona for this chat
    persona_name = await history_db.get_setting(chat_id, "persona_name")


    intent = parse_intent(text)
    logger.debug("chat_id=%d  intent=%-20s  persona=%s  text=%r  forced_model=%s", chat_id, intent.name, persona_name, text[:60], forced_model)

    target_model = forced_model
    if not target_model:
        target_model = settings.DEEP_MODEL if should_use_deep_model(text) else settings.DEFAULT_MODEL

    # Check for pending command intents
    pending_intent = context.user_data.pop("pending_command_intent", None) if context.user_data else None
    if pending_intent:
        intent = pending_intent

    # Check for pending excel edit
    pending_excel = context.user_data.pop("pending_excel_edit", None) if context.user_data else None
    if pending_excel:
        original_prompt = pending_excel["original_prompt"]
        amended_prompt = f"{original_prompt} (Edits: {text})"
        await _handle_excel(update, amended_prompt, chat_id, context, model=target_model)
        return

    # Check for pending file followup questions
    pending_file = context.user_data.pop("pending_file_ask", None) if context.user_data else None
    if pending_file:
        file_path = pending_file["file_path"]
        filename = pending_file["filename"]
        try:
            extracted = read_file(file_path)
            full_prompt = build_llm_prompt(extracted, filename, text)
            status_msg_id, _ = await send_status(update, context, "📂 Processing follow-up question about the file...")
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            async with _semaphores[chat_id]:
                async with TypingRefresher(context.bot, chat_id):
                    reply = await ollama_service.chat(full_prompt, history=[], model=target_model)
            
            await _append(chat_id, "user", f"[File Question: {filename}] {text}")
            await _append(chat_id, "assistant", reply, model=target_model)
            
            ref_id = callback_registry.register("file_followup", {"file_path": file_path, "filename": filename})
            keyboard = build_result_keyboard([("❓ Ask another question", f"file_followup:{ref_id}")])
            await _send_html_reply(update, context, chat_id, reply, reply_markup=keyboard)
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass
        except Exception as e:
            logger.exception("Failed in file follow up: %s", e)
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed to answer question: {e}")
        return

    # 1. Check for pending Telegram contact selection FIRST (e.g. user typed "1" or contact name)
    pending_tg_contacts = context.user_data.pop("pending_tg_contacts", None) if context.user_data else None
    if pending_tg_contacts:
        msg_body = context.user_data.pop("pending_tg_message_for_contact", "") if context.user_data else ""
        if not msg_body:
            msg_body = context.user_data.pop("pending_tg_message", "") if context.user_data else ""
        selected = None
        if text.strip().isdigit():
            idx = int(text.strip()) - 1
            if 0 <= idx < len(pending_tg_contacts):
                selected = pending_tg_contacts[idx]
        if not selected:
            for cand in pending_tg_contacts:
                if text.strip().lower() in cand["name"].lower():
                    selected = cand
                    break
        if selected:
            recipient_label = selected["name"]
            if selected.get("username"):
                recipient_label += f" (@{selected['username']})"

            final_msg = await _reformulate_message(msg_body, selected["name"])

            pending = PendingSend(
                user_id=chat_id,
                recipient_entity_id=selected["entity"],
                recipient_display_name=recipient_label,
                match_confidence=selected.get("_score", 100),
                final_text=final_msg,
                source="custom_text",
            )
            pending_send_registry.set(pending)

            if context.user_data is not None:
                context.user_data["last_tg_contact_name"] = selected["name"]

            card_text, card_keyboard = build_confirmation_card(pending)
            await update.message.reply_text(
                card_text,
                parse_mode=ParseMode.HTML,
                reply_markup=card_keyboard,
            )
            return


    # 2. Check for pending Telegram UserBot message recipient
    pending_tg_msg = context.user_data.pop("pending_tg_message", None) if context.user_data else None
    if pending_tg_msg:
        cleaned = re.sub(r"^(?:it\s+is|it\'s|its|this\s+is|это|bu)\s+", "", text.strip(), flags=re.IGNORECASE).strip().strip(".'\"")
        await _handle_send_telegram_message(update, f'send to "{cleaned}": "{pending_tg_msg}"', chat_id, context)
        return

    if intent == Intent.READ_TELEGRAM_MESSAGES:
        await _handle_read_telegram_messages(update, text, chat_id, context)
    elif intent == Intent.SEND_TELEGRAM_MESSAGE:
        await _handle_send_telegram_message(update, text, chat_id, context)
    elif intent == Intent.LOCAL_PROJECT:
        await _handle_local_project(update, text, chat_id, context, model=target_model)
    elif intent == Intent.GENERATE_IMAGE:
        await _handle_image(update, text, chat_id, context)
    elif intent == Intent.WEATHER:
        await _handle_weather(update, text, chat_id)
    elif intent == Intent.WEB_SEARCH:
        await _handle_web_search(update, text, chat_id, context)
    elif intent == Intent.CREATE_PDF:
        await _handle_pdf(update, chat_id, text=text, context=context, model=target_model)
    elif intent == Intent.WRITE_REPORT:
        await _handle_docx(update, chat_id, text=text, context=context, model=target_model)
    elif intent == Intent.CREATE_EXCEL:
        await _handle_excel(update, text, chat_id, context, model=target_model)
    elif intent == Intent.CREATE_STYLED_DOC:
        await _handle_styled_doc(update, text, chat_id, context, model=target_model)
    else:
        await _handle_chat(update, text, chat_id, context, forced_model=target_model, persona_name=persona_name)


# ── Inline Command Handlers ───────────────────────────────────────────────────

async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_image(update, prompt, update.effective_chat.id, context)
    else:
        context.user_data["pending_command_intent"] = Intent.GENERATE_IMAGE
        await update.message.reply_text("🎨 What image would you like to generate? Please type your prompt.")

async def cmd_excel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_excel(update, prompt, update.effective_chat.id, context)
    else:
        context.user_data["pending_command_intent"] = Intent.CREATE_EXCEL
        await update.message.reply_text("📊 What spreadsheet would you like to create? Please type your prompt.")

async def cmd_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_pdf(update, update.effective_chat.id, text=prompt, context=context)
    else:
        context.user_data["pending_command_intent"] = Intent.CREATE_PDF
        await update.message.reply_text("📄 What would you like the PDF document to cover? Please type your prompt.")

async def cmd_doc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_docx(update, update.effective_chat.id, text=prompt, context=context)
    else:
        context.user_data["pending_command_intent"] = Intent.WRITE_REPORT
        await update.message.reply_text("📝 What would you like the Word document to cover? Please type your prompt.")

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_web_search(update, prompt, update.effective_chat.id, context)
    else:
        context.user_data["pending_command_intent"] = Intent.WEB_SEARCH
        await update.message.reply_text("📡 What news/web topic would you like to search for? Please type your query.")

async def cmd_project(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_local_project(update, prompt, update.effective_chat.id, context)
    else:
        context.user_data["pending_command_intent"] = Intent.LOCAL_PROJECT
        await update.message.reply_text("📂 What project or folder on your Mac would you like to inspect? (e.g. `~/Desktop/jamgarma/yangiavlod`)")

async def cmd_tg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a personal Telegram message. Usage: /tg <contact> <message>"""
    clear_pending_states(context)
    assert update.effective_chat is not None
    if context.args:
        prompt = " ".join(context.args)
        await _handle_send_telegram_message(update, prompt, update.effective_chat.id, context)
    else:
        await update.message.reply_text(
            "📨 *Personal Telegram Messenger*\n\n"
            "Usage:\n"
            "• `/tg Ali hi how are you?`\n"
            "• `/tg @username good morning!`\n\n"
            "Or just say naturally:\n"
            "• _\"find a contact called Tr number and text him hi how are you?\"_",
            parse_mode=ParseMode.MARKDOWN,
        )


# ── Global Callback Dispatcher ────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    # Immediacy constraint: query.answer() MUST be the very first line of execution
    await query.answer()

    chat_id = update.effective_chat.id
    data = query.data

    if not data:
        return

    # Handle built-in clear
    if data == "clear":
        await history_db.clear_history(chat_id)
        if hasattr(context, "user_data") and context.user_data:
            context.user_data.pop("pending_reference", None)
            context.user_data.pop("pending_excel_edit", None)
            context.user_data.pop("pending_file_ask", None)
            context.user_data.pop("pending_command_intent", None)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=chat_id,
            text="🗑 Conversation history cleared. Fresh start!",
        )
        return

    if data == "regen":
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        deleted = await history_db.delete_last_assistant_message(chat_id)
        if not deleted:
            await context.bot.send_message(
                chat_id=chat_id,
                text="ℹ️ No assistant message found to regenerate.",
            )
            return

        await query.edit_message_reply_markup(reply_markup=None)

        history = await history_db.get_history(chat_id, _MAX_HISTORY)
        user_message = None
        for turn in reversed(history):
            if turn["role"] == "user":
                user_message = turn["content"]
                break

        if not user_message:
            await context.bot.send_message(
                chat_id=chat_id,
                text="ℹ️ No user message found to regenerate.",
            )
            return

        await _run_chat_regeneration(update, context, chat_id, user_message)

    # Check for callback registry action format
    if ":" not in data:
        return

    action, ref_id = data.split(":", 1)
    ctx = callback_registry.resolve(ref_id)
    
    if ctx is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ This action has expired. Please try generating a new report or spreadsheet."
        )
        return

    payload = ctx.payload
    
    # Consolidate all callback actions that trigger heavy work: enforce rate limiter + semaphore
    HEAVY_ACTIONS = {
        "regen_image", "apply_style", "regen_pdf", "regen_docx", 
        "regen_styled_doc", "regen_excel", "news_report", "news_more"
    }
    
    if action in HEAVY_ACTIONS:
        user_id = update.effective_user.id
        if not rate_limiter.allow(user_id):
            await context.bot.send_message(
                chat_id=chat_id,
                text="⏳ You're sending requests too quickly. Please wait a moment and try again."
            )
            return

    # Callback action routing
    if action == "regen_image":
        prompt = payload["prompt"]
        await _handle_image(update, prompt, chat_id, context)
        
    elif action == "restyle_image":
        prompt = payload["prompt"]
        # Present choices
        ref_realistic = callback_registry.register("apply_style", {"prompt": prompt, "style": "photorealistic, cinematic lighting, 8k resolution, highly detailed", "chat_id": chat_id})
        ref_anime = callback_registry.register("apply_style", {"prompt": prompt, "style": "anime style, digital illustration, vibrant colors, expressive", "chat_id": chat_id})
        ref_watercolor = callback_registry.register("apply_style", {"prompt": prompt, "style": "watercolor painting, soft wash, artistic, textured paper", "chat_id": chat_id})
        ref_3d = callback_registry.register("apply_style", {"prompt": prompt, "style": "3D render, blender style, clay model, cute, toy aesthetic", "chat_id": chat_id})
        
        keyboard = build_result_keyboard([
            ("📸 Realistic", f"apply_style:{ref_realistic}"),
            ("🌸 Anime", f"apply_style:{ref_anime}"),
            ("🎨 Watercolor", f"apply_style:{ref_watercolor}"),
            ("🧸 3D Render", f"apply_style:{ref_3d}"),
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text="🎨 Choose a style to apply to your prompt:",
            reply_markup=keyboard
        )
        
    elif action == "apply_style":
        prompt = payload["prompt"]
        style = payload["style"]
        styled_prompt = f"{prompt} ({style})"
        await _handle_image(update, styled_prompt, chat_id, context)
        
    elif action == "regen_pdf":
        prompt = payload["prompt"]
        await _handle_pdf(update, chat_id, text=prompt, context=context)
        
    elif action == "regen_docx":
        prompt = payload["prompt"]
        await _handle_docx(update, chat_id, text=prompt, context=context)
        
    elif action == "regen_styled":
        prompt = payload["prompt"]
        ref_path = payload["ref_path"]
        ext = payload["ext"]
        topic = payload["topic"]
        use_style = payload["use_style"]
        
        if use_style:
            # We mock the pending reference in context.user_data to generate with style
            context.user_data["pending_reference"] = ref_path
            await _handle_styled_doc(update, prompt, chat_id, context)
        else:
            # Generate plain report
            if ext == ".docx":
                await _handle_docx(update, chat_id, text=prompt, context=context)
            else:
                await _handle_pdf(update, chat_id, text=prompt, context=context)
                
    elif action == "regen_excel":
        prompt = payload["prompt"]
        await _handle_excel(update, prompt, chat_id, context)
        
    elif action == "edit_excel":
        prompt = payload["prompt"]
        context.user_data["pending_excel_edit"] = {"original_prompt": prompt, "chat_id": chat_id}
        await context.bot.send_message(
            chat_id=chat_id,
            text="✏️ Please reply with the edits you'd like to make to the spreadsheet (e.g. 'add a column for Rating' or 'sort by Paradigm')."
        )
        
    elif action == "news_report":
        report_query = payload["query"]
        cache_key = f"{chat_id}:{report_query[:200]}"
        cached = _pending_reports.get(cache_key)
        if cached:
            cached_answer, cached_articles = cached
            await _send_news_report(chat_id, context, report_query, cached_answer, cached_articles)
        else:
            # Cache expired (e.g. bot restarted) — run a fresh search then report
            status_msg_id, _ = await send_status(update, context, f"🔎 Fetching fresh results for *{report_query}*…")
            try:
                fresh_response = await search_service.web_search_rich(report_query)
                fresh_images = await fetch_all_images(fresh_response.results, allow_ai_fallback=False)
                fresh_articles = [
                    ArticleCard(
                        title=item.title,
                        url=item.url,
                        snippet=item.snippet,
                        image=img,
                        published_date=item.published_date,
                    )
                    for item, img in zip(fresh_response.results, fresh_images)
                ]
                await _send_news_report(
                    chat_id, context, report_query, fresh_response.answer, fresh_articles
                )
            except Exception as exc:
                logger.exception("Callback report re-search failed: %s", exc)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ Could not regenerate the report. Please run the search again.",
                )
            finally:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
                except Exception:
                    pass

    elif action == "news_more":
        query_text = payload["query"]
        limit = payload.get("limit", 8)
        
        # Edit inline reply markup to prevent double-tap
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
            
        status_msg_id, _ = await send_status(update, context, f"🔍 Fetching {limit} search results for *{query_text}*…")
        try:
            response = await search_service.web_search_rich(query_text, max_results=limit)
            markdown_result = search_service.format_as_markdown(response)
            
            # Register new callback actions for report & even more results
            ref_report = callback_registry.register("news_report", {"query": query_text, "chat_id": chat_id})
            ref_more = callback_registry.register("news_more", {"query": query_text, "chat_id": chat_id, "limit": limit + 3})
            
            keyboard = build_result_keyboard([
                ("📄 Generate Report", f"news_report:{ref_report}"),
                ("🔍 More results", f"news_more:{ref_more}"),
            ])
            
            # Replace the search results message by deleting status and sending the text reply
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text=markdown_result,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard
            )
        except Exception as exc:
            logger.exception("Callback news_more failed: %s", exc)
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Could not retrieve more results. Please try again later.",
            )
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass
                
    elif action == "file_followup":
        file_path = payload["file_path"]
        filename = payload["filename"]
        context.user_data["pending_file_ask"] = {"file_path": file_path, "filename": filename, "chat_id": chat_id}
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❓ What question would you like to ask about `{filename}`?"
        )

    elif action == "tg_sug":
        user_id_recipient = payload["user_id"]
        name = payload["name"]
        text_to_send = payload["text"]
        user_id = update.effective_user.id

        pending = PendingSend(
            user_id=user_id,
            recipient_entity_id=user_id_recipient,
            recipient_display_name=name,
            match_confidence=None,
            final_text=text_to_send,
            source="quick_suggestion",
        )
        pending_send_registry.set(pending)

        card_text, card_keyboard = build_confirmation_card(pending)
        try:
            await query.edit_message_text(
                text=card_text,
                parse_mode=ParseMode.HTML,
                reply_markup=card_keyboard,
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text=card_text,
                parse_mode=ParseMode.HTML,
                reply_markup=card_keyboard,
            )

    elif action == "tg_custom":
        user_id = payload["user_id"]
        name = payload["name"]
        context.user_data["pending_tg_custom_target"] = {"user_id": user_id, "name": name, "match_confidence": None}
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"✍️ <b>Custom reply to {html.escape(name)}:</b>\n\nType your message or send a 🎙 voice note below.",
            parse_mode=ParseMode.HTML,
        )

    elif action == "tg_ai_long":
        from services import telethon_service  # noqa: PLC0415
        user_id_recipient = payload["user_id"]
        name = payload["name"]
        last_msg = payload.get("last_message", "")
        status_msg_id, _ = await send_status(update, context, f"🤖 Drafting detailed AI reply for {name}…")
        try:
            _, history = await telethon_service.get_contact_messages(name, limit=5)
            context_str = "\n".join(f"{m['sender']}: {m['text']}" for m in history)
            prompt = (
                f"You are helping me compose a thoughtful, conversational direct reply to {name}.\n\n"
                f"Recent chat context:\n{context_str}\n\n"
                f"Their latest message: \"{last_msg}\"\n\n"
                "Write a natural, friendly, and complete reply in the exact same language they spoke (Uzbek, Russian, or English). "
                "Do NOT include meta-commentary, greetings like 'Dear' unless natural, or quotation marks. Only output the exact reply text to send."
            )
            draft = await ollama_service.chat(prompt, history=[])
            draft = draft.strip().strip('"\'')

            user_id = update.effective_user.id
            pending = PendingSend(
                user_id=user_id,
                recipient_entity_id=user_id_recipient,
                recipient_display_name=name,
                match_confidence=None,
                final_text=draft,
                source="ai_draft",
            )
            pending_send_registry.set(pending)

            card_text, card_keyboard = build_confirmation_card(pending)
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass
            await context.bot.send_message(
                chat_id=chat_id,
                text=card_text,
                parse_mode=ParseMode.HTML,
                reply_markup=card_keyboard,
            )
        except Exception as exc:
            logger.exception("Failed to draft AI reply: %s", exc)
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Could not draft reply: {exc}")

    elif action == "confirm_send":
        await query.answer()
        user_id = update.effective_user.id
        pending = pending_send_registry.get(user_id)
        if pending is None:
            await query.edit_message_text("⌛ This confirmation expired — please try again.")
            return

        if not rate_limiter.allow(user_id):
            await query.edit_message_text("⏳ Too many requests — please wait a moment and try again.")
            return

        from services import telethon_service  # noqa: PLC0415
        try:
            result = await telethon_service.send_personal_message(pending.recipient_entity_id, pending.final_text)
            pending_send_registry.clear(user_id)
            log_send(
                user_id=user_id,
                recipient=pending.recipient_display_name,
                text=pending.final_text,
                source=pending.source,
                confidence=pending.match_confidence,
                success=True,
            )
            clean_name = html.escape(pending.recipient_display_name)
            clean_msg = html.escape(pending.final_text)
            await query.edit_message_text(
                f"✅ <b>Sent to {clean_name}:</b>\n\n💬 <i>\"{clean_msg}\"</i>",
                parse_mode=ParseMode.HTML,
            )
            await _append(chat_id, "user", f"[Confirmed send to {pending.recipient_display_name}] {pending.final_text}")
            await _append(chat_id, "assistant", f"Sent message to {pending.recipient_display_name}: \"{pending.final_text}\"")
        except Exception as exc:
            logger.exception("handle_confirm_send failed: %s", exc)
            log_send(
                user_id=user_id,
                recipient=pending.recipient_display_name,
                text=pending.final_text,
                source=pending.source,
                confidence=pending.match_confidence,
                success=False,
                error=exc,
            )
            await query.edit_message_text(
                "❌ Failed to send — the contact may be unreachable, or the session expired."
            )

    elif action == "edit_send":
        await query.answer()
        user_id = update.effective_user.id
        pending = pending_send_registry.get(user_id)
        if pending is None:
            await query.edit_message_text("⌛ This confirmation expired — please try again.")
            return

        context.user_data["pending_tg_custom_target"] = {
            "user_id": pending.recipient_entity_id,
            "name": pending.recipient_display_name,
            "match_confidence": pending.match_confidence,
        }
        await query.edit_message_text(
            f"✏️ <b>Editing message for {html.escape(pending.recipient_display_name)}:</b>\n\n"
            f"Type your updated message or send a 🎙 voice note below.",
            parse_mode=ParseMode.HTML,
        )

    elif action == "cancel_send":
        await query.answer()
        user_id = update.effective_user.id
        pending_send_registry.clear(user_id)
        await query.edit_message_text("❌ Cancelled.")


    elif action == "tg_dismiss":
        from services import telethon_service  # noqa: PLC0415
        telethon_service.clear_active_contact()
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=chat_id,
            text="🗑 Dismissed context. Normal bot mode active.",
        )



async def _run_chat_regeneration(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_message: str) -> None:
    history = await history_db.get_history(chat_id, _MAX_HISTORY)
    trimmed_history = [turn for turn in history if turn != history[-1]] if history else []

    target_model = settings.DEEP_MODEL if should_use_deep_model(user_message) else settings.DEFAULT_MODEL

    last_model = await history_db.get_last_model(chat_id)
    prev_model = last_model or settings.DEFAULT_MODEL
    if prev_model != target_model:
        icon = "🧠" if target_model == settings.DEEP_MODEL else "⚡"
        model_name = "deep reasoning model" if target_model == settings.DEEP_MODEL else "fast conversational model"
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{icon} Switching to the {model_name} (`{target_model}`), this may take a few extra seconds to load..."
        )

    try:
        async with TypingRefresher(context.bot, chat_id):
            async with _semaphores[chat_id]:
                persona_name = await history_db.get_setting(chat_id, "persona_name")
                reply = await ollama_service.chat(
                    user_message,
                    trimmed_history,
                    model=target_model,
                    persona_name=persona_name,
                )
    except Exception as exc:
        logger.exception("Ollama chat error during regeneration chat_id=%d: %s", chat_id, exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Couldn't reach the AI model. Make sure Ollama is running and try again.",
        )
        return

    await _append(chat_id, "assistant", reply, model=target_model)
    await _send_html_reply(update, context, chat_id, reply, reply_markup=_get_chat_keyboard())


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log all uncaught exceptions — never let a raw traceback reach the user."""
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text(
                "⚠️ An unexpected error occurred. Please try again in a moment."
            )
        except Exception:
            pass


async def _get_owner_chat_id() -> int | None:
    stored = await history_db.get_global_setting("owner_chat_id")
    if stored and stored.lstrip("-").isdigit():
        return int(stored)
    return None


async def post_init(application: Application) -> None:
    commands = [
        BotCommand("start", "Show what this bot can do"),
        BotCommand("tg", "Send message via personal Telegram account"),
        BotCommand("image", "Generate an image"),
        BotCommand("excel", "Generate a spreadsheet"),
        BotCommand("pdf", "Generate a PDF document"),
        BotCommand("doc", "Generate a Word document"),
        BotCommand("news", "Search news with images and reports"),
        BotCommand("help", "Show usage examples"),
    ]
    await application.bot.set_my_commands(commands)

    # Start personal Telegram UserBot incoming listener in the background
    from services import telethon_service  # noqa: PLC0415
    try:
        if await telethon_service.is_connected():
            asyncio.create_task(telethon_service.start_listener(application, _get_owner_chat_id))
            logger.info("Personal Telegram UserBot listener started ✅")
    except Exception as exc:
        logger.warning("Could not start Telethon listener: %s", exc)


async def post_shutdown(application: Application) -> None:
    """Clean up shared HTTP clients on bot shutdown."""
    from services.ollama_service import close_ollama_client  # noqa: PLC0415
    await close_ollama_client()
    logger.info("Ollama HTTP client closed ✅")


# ── Startup: clear stale Telegram polling session ─────────────────────────────

async def _clear_stale_session(token: str) -> None:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params={"timeout": 0, "offset": -1})
            if resp.json().get("ok"):
                logger.info("Stale Telegram session cleared ✅")
    except Exception as exc:
        logger.warning("Could not clear stale session (non-fatal): %s", exc)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def main() -> None:
    token = settings.TELEGRAM_BOT_TOKEN
    if not token or token == "your_telegram_bot_token_here":
        raise SystemExit(
            "❌  TELEGRAM_BOT_TOKEN is not set.\n"
            "   Copy .env.example → .env and fill in your token."
        )

    # Startup temp-file sweep
    sweep_tmp_dir()

    # Initialize SQLite database
    history_db.init_db()

    from telegram.request import HTTPXRequest
    req = HTTPXRequest(
        connection_pool_size=16,
        read_timeout=30.0,
        write_timeout=30.0,
        connect_timeout=30.0,
        pool_timeout=30.0,
    )

    app = (
        Application.builder()
        .token(token)
        .request(req)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("deep", cmd_deep))
    app.add_handler(CommandHandler("fast", cmd_fast))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("excel", cmd_excel))
    app.add_handler(CommandHandler("pdf", cmd_pdf))
    app.add_handler(CommandHandler("doc", cmd_doc))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("project", cmd_project))
    app.add_handler(CommandHandler("inspect", cmd_project))
    app.add_handler(CommandHandler("tg", cmd_tg))
    app.add_handler(CommandHandler("message", cmd_tg))

    # Callbacks
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Document handler — routed at handler layer, NOT through intent parser
    app.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.COMMAND, _handle_file_upload)
    )

    # Voice & Audio handler — transcribes speech in Uzbek/Russian/English and routes to dispatch
    app.add_handler(
        MessageHandler((filters.VOICE | filters.AUDIO) & ~filters.COMMAND, _handle_voice)
    )

    # Photo handler
    app.add_handler(
        MessageHandler(filters.PHOTO & ~filters.COMMAND, _handle_photo)
    )

    # Sticker handler — static stickers → Gemini Vision; animated/video → Ollama reaction
    app.add_handler(
        MessageHandler(filters.Sticker.ALL & ~filters.COMMAND, _handle_sticker)
    )

    # Text catch-all
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, dispatch)
    )

    # Global error handler
    app.add_error_handler(error_handler)

    # Schedule periodic cleanup every hour
    if app.job_queue:
        app.job_queue.run_repeating(periodic_sweep, interval=3600, first=3600)
        logger.info("Scheduled hourly temp-file cleanup via JobQueue ✅")

    logger.info(
        "Bot starting — model: %s  ollama: %s",
        settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL,
    )

    asyncio.get_event_loop().run_until_complete(_clear_stale_session(token))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
