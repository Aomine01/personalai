"""
handlers/document_gen.py — Document generation handlers: PDF, DOCX, Excel, styled docs.

Contains: _handle_pdf, _handle_docx, _handle_excel, _handle_styled_doc
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

from handlers._shared import _send_html_reply, _get_history, _append, _cleanup, _reply

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

