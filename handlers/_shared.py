"""
handlers/_shared.py — Shared helper utilities used by multiple handler modules.

Contains:
- build_confirmation_card
- _detect_persona_change / _PERSONA_PATTERNS / _STOP_WORDS
- _get_history / _append
- _get_chat_keyboard / _send_html_reply
- clear_pending_states / _cleanup / _reply
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

