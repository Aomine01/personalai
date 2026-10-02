"""
handlers/chat_handler.py — Core chat, image generation, and weather handlers.

Contains: _handle_chat, _handle_image, _handle_weather
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

from handlers._shared import (
    _get_history, _append, _get_chat_keyboard, _send_html_reply, _reply, _cleanup,
)

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






async def _handle_chat(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    forced_model: str | None = None,
    persona_name: str | None = None,
) -> None:
    assert update.message is not None

    # Safety net: redirect telegram contact questions to _handle_read_telegram_messages
    from services import telethon_service  # noqa: PLC0415
    if await telethon_service.is_connected():
        if re.search(r"\b(contact|kontakt|telegram|контакт)\b", text, re.IGNORECASE) and re.search(r"\b(ask|asking|say|saying|text|write|wrote|send|sent|want|wants|nima|что|просит|хочет)\b", text, re.IGNORECASE):
            from handlers.telegram_relay import _handle_read_telegram_messages  # noqa: PLC0415
            await _handle_read_telegram_messages(update, text, chat_id, context)
            return

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


