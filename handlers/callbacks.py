"""
handlers/callbacks.py — Global inline-keyboard callback dispatcher.

Contains: handle_callback, _run_chat_regeneration
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
    _get_history, _append, _get_chat_keyboard, _send_html_reply,
    _reply, build_confirmation_card, clear_pending_states,
)
from handlers.search_handler import _send_news_report, _pending_reports
from handlers.chat_handler import _handle_chat
from handlers.document_gen import _handle_pdf, _handle_docx, _handle_excel
from handlers.telegram_relay import _handle_send_telegram_message

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
        msg_hash = payload.get("msg_hash", hash(last_msg[-80:]))
        draft_key = (user_id_recipient, msg_hash)

        # ── Try to use the pre-generated draft (fastest path) ────────────────
        cached = telethon_service._AI_DRAFT_CACHE.get(draft_key)
        draft: str | None = None

        if cached and cached.get("ready"):
            # Best case: draft was pre-generated while user was reading the card
            draft = cached["draft"]
            logger.info("tg_ai_long: using pre-generated draft (%d chars)", len(draft or ""))
        elif cached and not cached.get("ready"):
            # Still generating: show spinner and wait up to 18s for it to finish
            status_msg_id, _ = await send_status(update, context, f"⚡ Almost ready… generating reply for {name}")
            for _ in range(18):
                await asyncio.sleep(1)
                cached = telethon_service._AI_DRAFT_CACHE.get(draft_key)
                if cached and cached.get("ready"):
                    draft = cached["draft"]
                    break
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

        if not draft:
            # Fallback: pre-generation wasn't triggered or failed — generate on-demand
            status_msg_id, _ = await send_status(update, context, f"🤖 Drafting AI reply for {name}…")
            try:
                _, history = await telethon_service.get_contact_messages(name, limit=telethon_service._AI_DRAFT_HISTORY_LIMIT)
                prompt = telethon_service._build_ai_draft_prompt(history, last_msg, name)
                draft = await ollama_service.chat(prompt, history=[])
                draft = (draft or "").strip().strip('"\'')
            except Exception as exc:
                logger.exception("Failed to draft AI reply: %s", exc)
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
                except Exception:
                    pass
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Could not draft reply: {exc}")
                return
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

        if not draft:
            await context.bot.send_message(chat_id=chat_id, text="❌ Couldn't generate a draft — please try Custom Reply instead.")
            return

        # ── Show confirmation card ────────────────────────────────────────────
        pending = PendingSend(
            user_id=update.effective_user.id,
            recipient_entity_id=user_id_recipient,
            recipient_display_name=name,
            match_confidence=None,
            final_text=draft,
            source="ai_draft",
        )
        pending_send_registry.set(pending)
        card_text, card_keyboard = build_confirmation_card(pending)
        await context.bot.send_message(
            chat_id=chat_id,
            text=card_text,
            parse_mode=ParseMode.HTML,
            reply_markup=card_keyboard,
        )
        # Evict from draft cache since it's now in pending_send
        telethon_service._AI_DRAFT_CACHE.pop(draft_key, None)

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


