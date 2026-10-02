"""
handlers/telegram_relay.py — Personal Telegram UserBot relay handlers.

Contains: _handle_voice, _clean_contact_name, _extract_contact_and_message_regex,
          _extract_contact_and_message, _reformulate_message,
          _handle_send_telegram_message, _extract_contact_to_read,
          _handle_read_telegram_messages, _handle_local_project
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
    _reply, _send_html_reply, _append, _get_history,
    build_confirmation_card, clear_pending_states, _cleanup,
)

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
            from main import dispatch  # lazy import to avoid circular dep
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

def _extract_contact_to_read(text: str, active_contact_name: str | None = None) -> str:
    """Extract the target contact name from a read messages request."""
    for q in ['“', '”', '«', '»', '„']:
        text = text.replace(q, '"')
    for q in ['‘', '’', '`']:
        text = text.replace(q, "'")

    # 1. If in quotes
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    if quotes:
        return quotes[0].strip()

    # 2. Contact named / called X
    m_named = re.search(
        r'(?:contact|person|user|kontakt|контакт)\s+(?:named|called|nomli|по\s+имени|под\s+именем|с\s+именем)?\s*([A-Za-zА-Яа-я0-9_\s]+?)(?:,|\.|\s+what|\s+did|\s+is|\s+and|\s+nima|\s+deb|\s+что|\?|$)',
        text, re.IGNORECASE
    )
    if m_named:
        cand = m_named.group(1).strip()
        if cand and cand.lower() not in {'named', 'called', 'nomli', 'the', 'a'}:
            return cand

    # 3. Russian: что написал X / что X написал / что хочет X / что X хочет
    m_ru1 = re.search(
        r'(?:что\s+(?:мне\s+)?(?:написал|написала|прислал|прислала|отправил|отправила|сказал|сказала|хочет|просит|спрашивает|пишет|говорит)|покажи\s+(?:сообщения\s+)?(?:от|с)|чат\s+с)\s+([A-Za-zА-Яа-я0-9_\s]+?)(?:\?|$|\.)',
        text, re.IGNORECASE
    )
    if m_ru1:
        return m_ru1.group(1).strip()

    m_ru2 = re.search(
        r'что\s+([A-Za-zА-Яа-я0-9_\s]+?)\s+(?:мне\s+)?(?:написал|написала|прислал|прислала|отправил|отправила|сказал|сказала|хочет|просит|спрашивает|пишет|говорит)',
        text, re.IGNORECASE
    )
    if m_ru2:
        return m_ru2.group(1).strip()

    # 4. Uzbek: X nima deb yozdi / X nima so'rayapti / Xdan kelgan xabarlar
    m_uz = re.search(
        r'([A-Za-zА-Яа-я0-9_\s]+?)(?:dan|bilan|\s+nima|\s+nimani)',
        text, re.IGNORECASE
    )
    if m_uz:
        cand = m_uz.group(1).strip()
        if cand.lower() not in {'telegram', 'kontakt', 'xabar', 'u', 'menda'}:
            return cand

    # 5. English: what did/is/does X ask/want/say/send...
    m_en1 = re.search(
        r'what\s+(?:did|is|does)\s+([A-Za-zА-Яа-я0-9_\s]+?)\s+(?:is\s+)?(?:ask|asking|want|wants|say|saying|text|texted|write|wrote|send|sent|mean|talking|request)',
        text, re.IGNORECASE
    )
    if m_en1:
        cand = m_en1.group(1).strip()
        if cand.lower() not in {'he', 'she', 'they', 'this', 'it'}:
            return cand

    m_en2 = re.search(
        r'(?:messages?\s+(?:from|with)\s+|chat\s+with\s+)(.+?)(?:\s+say|\s+tell|\?|$|\.)',
        text, re.IGNORECASE
    )
    if m_en2:
        return m_en2.group(1).strip()

    m_en3 = re.search(
        r'what\s+(?:did\s+)?([A-Za-zА-Яа-я0-9_\s]+?)\s+(?:send|sent|text|texted|write|wrote|say|said|ask|asked)',
        text, re.IGNORECASE
    )
    if m_en3:
        cand = m_en3.group(1).strip()
        if cand.lower() not in {'he', 'she', 'they', 'this', 'it'}:
            return cand

    # Pronoun or active contact fallback
    pronouns = {'he', 'she', 'they', 'him', 'her', 'u', 'он', 'она', 'они'}
    words = set(text.lower().split())
    if active_contact_name and (words & pronouns or not text.strip()):
        return active_contact_name

    return text.strip()


async def _handle_read_telegram_messages(
    update: Update,
    text: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Search for contact, fetch recent messages from personal Telegram, display them,
    and intelligently answer any user questions about the conversation.
    """
    import time  # noqa: PLC0415
    from services import telethon_service  # noqa: PLC0415
    assert update.message is not None

    if not await telethon_service.is_connected():
        await update.message.reply_text(
            "⚠️ UserBot is not logged in. Please run `setup_userbot.py`.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    active_c = telethon_service.get_active_contact()
    active_name = active_c.get("name") if active_c else None

    contact_query = _extract_contact_to_read(text, active_contact_name=active_name)
    if not contact_query and active_name:
        contact_query = active_name

    if not contact_query:
        await update.message.reply_text(
            "❓ Please specify which contact's messages you want to read (e.g. `what did \"Sherdor\" send me?`).",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    status_msg_id, _ = await send_status(update, context, f"🔍 Fetching recent messages with *{contact_query}*…")

    try:
        contact_info, msgs = await telethon_service.get_contact_messages(contact_query, limit=10)
        # Fallback to active contact if query didn't match and active contact exists
        if not contact_info and active_name and contact_query.lower() != active_name.lower():
            contact_info, msgs = await telethon_service.get_contact_messages(active_name, limit=10)

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
        last_incoming_text = ""
        for m in msgs:
            prefix = "🟢 **You**" if m["is_me"] else f"👤 **{contact_info['name']}**"
            time_tag = f" _({m['date']})_" if m["date"] else ""
            lines.append(f"{prefix}{time_tag}:\n\"{m['text']}\"\n")
            if not m["is_me"] and m["text"]:
                last_incoming_text = m["text"]

        # Check if user asked a substantive question or wants an analysis/summary
        is_question = bool(re.search(
            r"\b(what|why|how|when|where|ask|asking|want|wants|say|saying|mean|meaning|summar|detail|nima|nega|qanday|qachon|ayt|so\'?ra|что|почему|как|зачем|просит|хочет|пишет|говорит)\b",
            text, re.IGNORECASE
        ))

        if is_question and msgs:
            try:
                # Detect the language of the user's question
                text_lower = text.lower()
                if re.search(r"[а-яё]", text_lower):
                    user_lang = "Russian"
                elif any(w in text_lower for w in ["nima", "qanday", "nega", "so'ra", "sora", "ayt", "deb", "kim", "qachon"]):
                    user_lang = "Uzbek"
                else:
                    user_lang = "English"

                chat_log = "\n".join(
                    f"[{'You' if m['is_me'] else contact_info['name']}]: {m['text']}"
                    for m in msgs
                )
                prompt = (
                    f"You are an expert personal AI assistant analyzing Telegram chat messages.\n"
                    f"The conversation is between the user ('You') and '{contact_info['name']}'.\n\n"
                    f"Chat History (chronological, latest at the bottom):\n"
                    f"{chat_log}\n\n"
                    f"User's question: \"{text}\"\n\n"
                    f"CRITICAL LINGUISTIC & CONVERSATION RULES:\n"
                    f"1. Casual Telegram chats frequently use informal Uzbek, phonetic spelling, and Russian loanwords:\n"
                    f"   - 'no' is the Russian loanword 'но' meaning 'but' / 'however' (NOT English 'no' or negation). For example, 'saytga kirvotti no ishtirokchilar dgan joyiga kirmayapti' means 'the site loads, but the section called Ishtirokchilar (Participants) does not open'.\n"
                    f"   - Phonetic spellings: 'kirvotti' / 'ishlavotti' = kiryapti / ishlayapti (loading / working); 'dgan' = degan (named / called); '-tayu' / '-taku' = '-ta-ku' (stating a count or quantity, e.g. '120 tayu' = 'it is still 120'); 'haliyam' = haligacha (still / yet); 'qolib ketgandur' = might have remained / cached.\n"
                    f"2. Pay close attention to WHO said WHAT ('You' vs '{contact_info['name']}'). Do NOT invert the speaker: if 'You' said something (e.g. suggesting to check for an old version), do not attribute that suggestion to {contact_info['name']}.\n"
                    f"3. Focus heavily on the MOST RECENT messages at the bottom to determine what {contact_info['name']} is currently reporting, asking for, or experiencing.\n"
                    f"4. OUTPUT LANGUAGE: The user asked in {user_lang}. You MUST write your entire answer in {user_lang}.\n"
                    f"5. Provide a direct, factual 1-3 sentence summary explaining the issue or topic and what {contact_info['name']}'s latest status or request is. Output ONLY the answer text, no conversational filler or intro phrases."
                )
                ai_analysis = await ollama_service.chat(prompt, history=[], model=settings.DEFAULT_MODEL)
                if ai_analysis and ai_analysis.strip():
                    lines.append(f"💡 **AI Summary / Answer:**\n{ai_analysis.strip()}\n")
            except Exception as exc:
                logger.warning("Failed to generate AI analysis for chat with %s: %s", contact_info['name'], exc)

        full_reply = "\n".join(lines)

        # Store in history so user can ask follow-ups
        await _append(chat_id, "user", text)
        await _append(chat_id, "assistant", full_reply)

        # Set active contact so quick reply shortcuts work immediately
        telethon_service.set_active_contact({
            "user_id": contact_info["id"],
            "name": contact_info["name"],
            "username": contact_info.get("username"),
            "entity": contact_info.get("entity"),
            "last_message": last_incoming_text,
            "timestamp": time.time(),
        })

        # Pre-generate AI draft if incoming message exists
        if last_incoming_text:
            msg_hash = hash(last_incoming_text[-80:])
            asyncio.create_task(
                telethon_service._pre_generate_ai_draft(
                    contact_info["id"], contact_info["name"], last_incoming_text, msg_hash
                )
            )

        # Build inline action buttons
        ref_custom = callback_registry.register("tg_custom", {
            "user_id": contact_info["id"],
            "name": contact_info["name"],
        })
        action_buttons = [
            InlineKeyboardButton("✍️ Custom Reply", callback_data=f"tg_custom:{ref_custom}"),
        ]
        if last_incoming_text:
            ref_ai_long = callback_registry.register("tg_ai_long", {
                "user_id": contact_info["id"],
                "name": contact_info["name"],
                "last_message": last_incoming_text,
                "msg_hash": hash(last_incoming_text[-80:]),
            })
            action_buttons.append(
                InlineKeyboardButton("🤖 Detailed AI Reply", callback_data=f"tg_ai_long:{ref_ai_long}")
            )

        ref_dismiss = callback_registry.register("tg_dismiss", {})
        reply_markup = InlineKeyboardMarkup([
            action_buttons,
            [InlineKeyboardButton("❌ Dismiss", callback_data=f"tg_dismiss:{ref_dismiss}")]
        ])

        await _send_html_reply(update, context, chat_id, full_reply, reply_markup=reply_markup)
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

