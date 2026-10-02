"""
handlers/commands.py — Telegram slash command handlers.

Registers: /start /help /clear /status /deep /fast
           /image /excel /pdf /doc /news /project /tg
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

from handlers._shared import _reply, clear_pending_states
from telegram import BotCommand

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

    from main import dispatch  # lazy import to avoid circular dep
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

    from main import dispatch  # lazy import to avoid circular dep
    await dispatch(update, context, forced_model=settings.DEFAULT_MODEL, forced_text=text)



# ── Intent sub-handlers ───────────────────────────────────────────────────────




# ── Inline command shortcuts ──────────────────────────────────────────────────

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

