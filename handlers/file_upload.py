"""
handlers/file_upload.py — File upload, photo, and sticker handlers.

Contains: _handle_file_upload, _handle_photo, _handle_sticker
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

from handlers._shared import _send_html_reply, _reply, _get_history, _append, _cleanup, _get_chat_keyboard
from handlers.document_gen import (
    _handle_pdf, _handle_docx, _handle_excel, _handle_styled_doc,
)

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

