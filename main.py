"""
main.py — Thin orchestrator for the AI Telegram Bot.

All handler logic now lives in handlers/:
  _shared.py       — shared utilities, helpers, singletons
  commands.py      — /start /help /clear /status /deep /fast /image /excel /pdf /doc /news /project /tg
  chat_handler.py  — _handle_chat, _handle_image, _handle_weather
  search_handler.py — _handle_web_search, _send_news_report
  document_gen.py  — _handle_pdf, _handle_docx, _handle_excel, _handle_styled_doc
  file_upload.py   — _handle_file_upload, _handle_photo, _handle_sticker
  telegram_relay.py — _handle_voice, _handle_send_telegram_message, _handle_read_telegram_messages
  callbacks.py     — handle_callback, _run_chat_regeneration

This file retains only:
  dispatch() — the central intent router (references all handler sub-functions)
  error_handler, _get_owner_chat_id, post_init, post_shutdown, _clear_stale_session, main()
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
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
from utils.cleanup import periodic_sweep, sweep_tmp_dir
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

# ── Handler imports ───────────────────────────────────────────────────────────
from handlers._shared import (
    build_confirmation_card,
    _detect_persona_change,
    _get_history,
    _append,
    _get_chat_keyboard,
    _send_html_reply,
    clear_pending_states,
    _cleanup,
    _reply,
)
from handlers.commands import (
    cmd_start, cmd_help, cmd_clear, cmd_status, cmd_deep, cmd_fast,
    cmd_image, cmd_excel, cmd_pdf, cmd_doc, cmd_news, cmd_project, cmd_tg,
)
from handlers.chat_handler import _handle_chat, _handle_image, _handle_weather
from handlers.search_handler import _handle_web_search, _send_news_report
from handlers.document_gen import _handle_pdf, _handle_docx, _handle_excel, _handle_styled_doc
from handlers.file_upload import _handle_file_upload, _handle_photo, _handle_sticker
from handlers.telegram_relay import (
    _handle_voice,
    _handle_send_telegram_message,
    _handle_read_telegram_messages,
    _handle_local_project,
)
from handlers.callbacks import handle_callback

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_UPLOAD_DIR = Path(settings.TMP_DIR) / "uploads"

# ── Central dispatcher ────────────────────────────────────────────────────────

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



# ── Lifecycle callbacks and main() ────────────────────────────────────────────

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
