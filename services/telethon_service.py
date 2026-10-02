"""
services/telethon_service.py — Personal Telegram UserBot client (Telethon MTProto).

Allows the agent to:
- Search real personal Telegram contacts and dialogs
- Send messages from the user's personal account
- List recent conversations

Session is stored in userbot.session (never re-authenticates after first login).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from telethon import TelegramClient, functions, types
from telethon.tl.types import User, Chat, Channel

logger = logging.getLogger(__name__)

# ── Quick-suggestion TTL cache ──────────────────────────────────────────
# Maps (sender_id, text_hash) -> (suggestions_list, expiry_timestamp)
_SUGGESTION_CACHE: dict[tuple[int, int], tuple[list[str], float]] = {}
_SUGGESTION_CACHE_MAX = 50
_SUGGESTION_TTL = 600  # 10 minutes
_SUGGESTION_TIMEOUT = 20.0  # seconds before falling back to generic replies
_SUGGESTION_FALLBACK = ["👍", "Ok!", "I'll get back to you shortly."]

# ── Pre-generated AI draft cache ────────────────────────────────────────
# Maps (sender_id, msg_hash) -> {"draft": str, "ready": bool, "ts": float}
# Populated in background the moment a message arrives so tapping
# "Detailed AI Reply" is near-instant instead of making the user wait.
_AI_DRAFT_CACHE: dict[tuple[int, int], dict] = {}
_AI_DRAFT_TTL = 300   # 5 min — drafts expire if user never taps the button
_AI_DRAFT_MAX = 30    # LRU eviction cap
_AI_DRAFT_HISTORY_LIMIT = 15  # number of messages to feed the model



# Session file stored next to main.py
SESSION_PATH = str(Path(__file__).parent.parent / "userbot")


def _get_credentials() -> tuple[int, str]:
    from config import settings  # noqa: PLC0415
    api_id = settings.TELEGRAM_API_ID
    api_hash = settings.TELEGRAM_API_HASH
    if not api_id or not api_hash:
        raise RuntimeError(
            "Missing Telegram API credentials. Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env"
        )
    return api_id, api_hash


_client: TelegramClient | None = None


async def get_client() -> TelegramClient:
    global _client
    api_id, api_hash = _get_credentials()
    if _client is None:
        _client = TelegramClient(SESSION_PATH, api_id, api_hash)
    if not _client.is_connected():
        await _client.connect()
    if not await _client.is_user_authorized():
        raise RuntimeError(
            "UserBot not authenticated. Run `python setup_userbot.py` first to log in."
        )
    return _client


async def is_connected() -> bool:
    try:
        client = await get_client()
        return await client.is_user_authorized()
    except Exception:
        return False


def _entity_display_name(entity: Any) -> str:
    if isinstance(entity, User):
        parts = [entity.first_name or "", entity.last_name or ""]
        name = " ".join(p for p in parts if p).strip()
        if entity.username:
            name += f" (@{entity.username})"
        return name or f"User #{entity.id}"
    if isinstance(entity, (Chat, Channel)):
        return entity.title or f"Chat #{entity.id}"
    return str(entity)


CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "ў": "o", "ғ": "g", "қ": "q", "ҳ": "h"
}


STOPWORDS_SET = {
    "to", "the", "a", "an", "for", "in", "on", "at", "by", "from", "of",
    "contact", "contacts", "user", "users", "person", "people",
    "message", "messages", "text", "texts", "dm", "voice",
    "him", "her", "them", "me", "it",
    "kontakt", "kontaktga", "kontaktni", "odam", "kishi",
    "контакт", "контакту", "пользователь", "пользователю", "человек", "человеку",
    "сообщение", "текст", "смс",
}


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # Strip common prefix keywords
    cleaned = re.sub(
        r"^(?:контакту|контакт|пользователю|пользователь|человеку|человек|nomli\s+kontaktni?|kontakt(?:ini|ni|ga)?|"
        r"to\s+the\s+contact|the\s+contact|to\s+contact|contact|to\s+user|the\s+user|user|"
        r"named|called|with\s+name|с\s+именем|по\s+имени|ismidagi)\s+",
        "", s.strip(), flags=re.IGNORECASE
    )
    # Transliterate Cyrillic to Latin
    res = "".join(CYR_TO_LAT.get(c, c) for c in cleaned.lower())
    # Canonicalize digraphs and variations
    res = res.replace("kh", "h").replace("w", "v").replace("shch", "sh").replace("ts", "s")
    # Keep only alphanumeric and whitespace
    res = "".join(c for c in res if c.isalnum() or c.isspace())
    return " ".join(res.split())


def _score_match(query: str, name: str, username: str | None) -> int:
    import difflib  # noqa: PLC0415
    q_norm = _normalize_text(query)
    n_norm = _normalize_text(name)
    u_norm = _normalize_text(username or "")

    if not q_norm or not n_norm:
        return 0

    if q_norm in STOPWORDS_SET:
        return 0

    # 1. Exact full match
    if q_norm == n_norm:
        return 100

    # 2. Exact username match
    if u_norm and q_norm == u_norm:
        return 95

    # 3. Whole phrase substring (only if query is long enough to not be ambiguous)
    if len(q_norm) >= 4 and q_norm in n_norm:
        return 90

    q_words = q_norm.split()
    n_words = n_norm.split()

    score = 0

    # 4. Check for acronym / alias matches (e.g., TR <-> Turkish)
    ALIASES = {
        "tr": {"turkish", "turk", "turkey", "turkiya", "турция", "туркиш"},
        "turkish": {"tr"},
        "pc": {"computer", "компьютер"},
    }
    for qw in q_words:
        qw_aliases = ALIASES.get(qw, set())
        for nw in n_words:
            if qw == nw:
                score += 50
            elif nw in qw_aliases or qw in ALIASES.get(nw, set()):
                score += 45
            elif len(qw) >= 3 and (nw.startswith(qw) or qw.startswith(nw)):
                score += 40
            else:
                ratio = difflib.SequenceMatcher(None, qw, nw).ratio()
                if ratio >= 0.8:
                    score += int(ratio * 35)

    if u_norm:
        for qw in q_words:
            if qw in u_norm:
                score += 25

    return score



async def find_contact(query: str) -> list[dict[str, Any]]:
    """Search personal contacts and dialogs for query."""
    client = await get_client()
    results: list[dict[str, Any]] = []
    # Strip prefix words from query
    clean_q = re.sub(
        r"^(?:контакту|контакт|пользователю|пользователь|человеку|человек|nomli\s+kontaktni?|kontakt(?:ini|ni|ga)?|contact|user)\s+",
        "", query.strip(), flags=re.IGNORECASE
    ).strip()
    query_lower = clean_q or query.strip()

    # 1. Search contacts
    try:
        contacts_result = await client(functions.contacts.GetContactsRequest(hash=0))
        for user in contacts_result.users:
            name_parts = [user.first_name or "", user.last_name or ""]
            full_name = " ".join(p for p in name_parts if p).strip()
            score = _score_match(query_lower, full_name.lower(), user.username)
            if score > 0:
                results.append({
                    "id": user.id,
                    "name": full_name,
                    "username": user.username,
                    "phone": getattr(user, "phone", None),
                    "type": "contact",
                    "entity": user,
                    "_score": score,
                })
    except Exception as e:
        logger.warning("Contact search error: %s", e)

    # 2. Search dialogs
    try:
        async for dialog in client.iter_dialogs(limit=200):
            entity = dialog.entity
            name = _entity_display_name(entity)
            username = getattr(entity, "username", None)
            score = _score_match(query_lower, name.lower(), username)
            if score > 0:
                existing_ids = {r["id"] for r in results}
                if entity.id not in existing_ids:
                    results.append({
                        "id": entity.id,
                        "name": name,
                        "username": username,
                        "phone": getattr(entity, "phone", None),
                        "type": "user" if isinstance(entity, User) else "group",
                        "entity": entity,
                        "_score": score,
                    })
    except Exception as e:
        logger.warning("Dialog search error: %s", e)

    # 3. Telegram global search fallback
    try:
        search_result = await client(functions.contacts.SearchRequest(q=query, limit=10))
        for user in search_result.users:
            name_parts = [user.first_name or "", user.last_name or ""]
            full_name = " ".join(p for p in name_parts if p).strip()
            existing_ids = {r["id"] for r in results}
            if user.id not in existing_ids:
                results.append({
                    "id": user.id,
                    "name": full_name,
                    "username": user.username,
                    "phone": getattr(user, "phone", None),
                    "type": "search_result",
                    "entity": user,
                    "_score": 5,
                })
    except Exception as e:
        logger.warning("Global search error: %s", e)

    results.sort(key=lambda x: -x["_score"])
    return results[:10]


_bot_sent_msg_ids: set[int] = set()
_last_manual_outgoing: dict[int, float] = {}  # {user_id: timestamp}


class SendFailedError(Exception):
    """Raised when sending a message via UserBot fails."""
    pass


async def send_personal_message(recipient_entity: Any, text: str) -> dict[str, Any]:
    """Send a message from your personal Telegram account."""
    try:
        client = await get_client()
        msg = await client.send_message(recipient_entity, text)
        if msg and hasattr(msg, "id"):
            _bot_sent_msg_ids.add(msg.id)
            if len(_bot_sent_msg_ids) > 1000:
                _bot_sent_msg_ids.clear()
        return {
            "message_id": msg.id,
            "date": str(msg.date),
            "text": text,
        }
    except Exception as exc:
        logger.exception("Telethon send_personal_message failed: %s", exc)
        raise SendFailedError(f"Failed to send message: {exc}") from exc



async def get_contact_messages(contact_query: str, limit: int = 5) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Search for contact and return the last `limit` messages from the chat with them.
    """
    matches = await find_contact(contact_query)
    if not matches:
        return None, []

    best = matches[0]
    client = await get_client()
    me = await client.get_me()

    messages = []
    try:
        async for msg in client.iter_messages(best["entity"], limit=limit):
            is_me = (msg.sender_id == me.id) if msg.sender_id else bool(msg.out)
            sender_name = "You" if is_me else best["name"]

            # Format text or media placeholder
            body = msg.message or ""
            if msg.voice:
                dur = msg.voice.attributes[0].duration if (msg.voice.attributes and hasattr(msg.voice.attributes[0], "duration")) else ""
                body = f"[🎙 Voice message {dur}s] {body}".strip()
            elif msg.photo:
                body = f"[📷 Photo] {body}".strip()
            elif msg.video:
                body = f"[🎥 Video] {body}".strip()
            elif msg.document:
                doc_name = getattr(msg.file, "name", "Document") if hasattr(msg, "file") else "Document"
                body = f"[📄 {doc_name}] {body}".strip()
            elif msg.sticker:
                emoji = getattr(msg.file, "emoji", "") if hasattr(msg, "file") else ""
                body = f"[Sticker {emoji}]".strip()

            if not body:
                body = "[Media message]"

            time_str = msg.date.strftime("%H:%M") if msg.date else ""
            messages.append({
                "id": msg.id,
                "is_me": is_me,
                "sender": sender_name,
                "text": body,
                "date": time_str,
            })
    except Exception as e:
        logger.warning("Error fetching messages for %s: %s", best["name"], e)

    messages.reverse()  # chronological order (oldest to newest)
    return best, messages



async def list_recent_dialogs(limit: int = 15) -> list[dict[str, Any]]:
    """Return list of recent chats."""
    client = await get_client()
    dialogs = []
    async for dialog in client.iter_dialogs(limit=limit):
        entity = dialog.entity
        dialogs.append({
            "name": _entity_display_name(entity),
            "type": "user" if isinstance(entity, User) else "group",
            "unread": dialog.unread_count,
            "last_message": dialog.message.message[:80] if dialog.message else "",
        })
    return dialogs


async def disconnect() -> None:
    global _client
    if _client and _client.is_connected():
        await _client.disconnect()
        _client = None


# ── Active Incoming Contact State ─────────────────────────────────────────────

_active_contact: dict[str, Any] | None = None


def get_active_contact(max_age_seconds: int = 1800) -> dict[str, Any] | None:
    """
    Return the active contact that recently messaged the user if within max_age_seconds.
    Defaults to 30 minutes window.
    """
    import time  # noqa: PLC0415
    global _active_contact
    if not _active_contact:
        return None
    age = time.time() - _active_contact.get("timestamp", 0)
    if age > max_age_seconds:
        _active_contact = None
        return None
    return _active_contact


def set_active_contact(contact_data: dict[str, Any]) -> None:
    """Set the most recent contact who messaged the user."""
    import time  # noqa: PLC0415
    global _active_contact
    if "timestamp" not in contact_data:
        contact_data["timestamp"] = time.time()
    _active_contact = contact_data


def clear_active_contact() -> None:
    """Clear active incoming contact."""
    global _active_contact
    _active_contact = None


# ── AI Draft Pre-generator ────────────────────────────────────────────────────

def _build_ai_draft_prompt(history: list[dict[str, Any]], incoming_text: str, sender_name: str) -> str:
    """
    Build a high-quality prompt for the detailed AI reply.
    Uses the full raw conversation (no summarization) so the model sees
    actual words, tone, jokes, and references.
    """
    first_name = sender_name.split()[0]

    # Format messages as a natural chat log
    chat_log_lines = []
    for m in history:
        speaker = "Me" if m.get("sender") == "Me" else first_name
        text = m.get("text", "").strip()
        if text:
            chat_log_lines.append(f"[{speaker}]: {text}")

    chat_log = "\n".join(chat_log_lines) if chat_log_lines else f"[{first_name}]: {incoming_text}"

    return (
        f"You are ghostwriting a Telegram reply on my behalf to {first_name}.\n\n"
        f"Full conversation (most recent at the bottom):\n"
        f"{chat_log}\n\n"
        f"Their latest message: \"{incoming_text}\"\n\n"
        "Your job:\n"
        f"1. Read the entire conversation carefully — understand the relationship, "
        f"ongoing topics, tone, jokes, and what {first_name} actually needs right now.\n"
        "   Linguistic note for informal Uzbek/Russian chats:\n"
        "   - 'no' is Russian 'но' meaning 'but' / 'however' (NOT English 'no' or negation).\n"
        "   - 'dgan' = degan; 'kirvotti' / 'ishlavotti' = kiryapti / ishlayapti; '-tayu' = '-ta-ku' (e.g. '120 tayu' = 'it is still 120'); 'haliyam' = haligacha (still).\n"
        "2. Write ONE complete, natural reply from my perspective that:\n"
        f"   - Directly addresses what {first_name} just said (don't ignore it)\n"
        "   - Matches the conversational tone (casual if they're casual, serious if serious)\n"
        f"   - Is written in the EXACT same language {first_name} used (Uzbek → Uzbek, Russian → Russian, English → English). Do NOT translate.\n"
        "   - Sounds like a real person texting, not an AI. No formal openers, no 'Dear', no sign-offs.\n"
        "   - Is appropriately concise — if it's casual chat, 1-3 sentences; if they asked something detailed, answer fully.\n\n"
        "Output ONLY the reply text — nothing else. No quotes, no explanation, no prefix."
    )


async def _pre_generate_ai_draft(
    sender_id: int,
    sender_name: str,
    msg_text: str,
    msg_hash: int,
) -> None:
    """
    Generate an AI draft reply in the background as soon as a message arrives.
    Result is stored in _AI_DRAFT_CACHE so tapping '🤖 Detailed AI Reply' is instant.
    """
    from services import ollama_service  # noqa: PLC0415

    cache_key = (sender_id, msg_hash)
    now = time.time()

    # Mark as in-progress immediately so a second event doesn't double-generate
    _AI_DRAFT_CACHE[cache_key] = {"draft": None, "ready": False, "ts": now}

    try:
        _, history = await get_contact_messages(sender_name, limit=_AI_DRAFT_HISTORY_LIMIT)
        prompt = _build_ai_draft_prompt(history, msg_text, sender_name)

        # Use the fast model to keep pre-generation quick
        from config import settings  # noqa: PLC0415
        fast_model = getattr(settings, "DEFAULT_MODEL", "qwen3:8b")
        draft = await ollama_service.chat(prompt, history=[], model=fast_model)
        draft = draft.strip().strip('"\'')

        # Evict oldest if at capacity
        if len(_AI_DRAFT_CACHE) >= _AI_DRAFT_MAX:
            oldest = next(iter(_AI_DRAFT_CACHE))
            del _AI_DRAFT_CACHE[oldest]

        _AI_DRAFT_CACHE[cache_key] = {"draft": draft, "ready": True, "ts": now}
        logger.info("Pre-generated AI draft for %s (sender_id=%d): %d chars", sender_name, sender_id, len(draft))

    except Exception as exc:
        logger.warning("Failed to pre-generate AI draft for sender_id=%d: %s", sender_id, exc)
        # Remove in-progress marker so the callback falls back to on-demand generation
        _AI_DRAFT_CACHE.pop(cache_key, None)


# ── Quick Suggestions Generator via Ollama ────────────────────────────────────

async def generate_quick_suggestions(
    history: list[dict[str, Any]],
    incoming_text: str,
    sender_name: str,
) -> list[str]:
    """
    Generate 3 natural, short quick-reply options tailored to the contact's
    language, tone, and the specific message type.
    """
    import json
    import re as _re
    from services import ollama_service  # noqa: PLC0415

    # ── Pre-classify message type to help the LLM ─────────────────────────────
    text = incoming_text.strip()
    text_lower = text.lower()

    # Detect links
    url_pattern = _re.compile(r'https?://\S+', _re.IGNORECASE)
    urls = url_pattern.findall(text)
    has_url = bool(urls)
    text_without_url = url_pattern.sub('', text).strip()

    # Detect message type
    is_name_call = (
        len(text.split()) <= 2
        and not text.endswith('?')
        and not has_url
        and not any(c in text for c in ['!', '.'])
    )
    is_question = text.endswith('?') or text_lower.startswith(
        ('qachon', 'qayer', 'nima', 'kim', 'nega', 'qanday',
         'when', 'where', 'what', 'who', 'why', 'how',
         'когда', 'где', 'что', 'кто', 'почему', 'как')
    )
    is_greeting = text_lower in (
        'salom', 'salm', 'привет', 'hi', 'hello', 'hey',
        'assalomu alaykum', 'ассалому алайкум', 'хай', 'haloo'
    )

    # Detect link platform for context
    link_platform = ''
    if has_url:
        for url in urls:
            if 'instagram' in url: link_platform = 'Instagram'
            elif 'youtube' in url or 'youtu.be' in url: link_platform = 'YouTube'
            elif 'tiktok' in url: link_platform = 'TikTok'
            elif 't.me' in url or 'telegram' in url: link_platform = 'Telegram'
            elif 'twitter' in url or 'x.com' in url: link_platform = 'Twitter/X'

    # ── Format recent conversation context ────────────────────────────────────
    context_lines = []
    for m in history[-6:]:
        context_lines.append(f"{m['sender']}: {m['text']}")
    context_str = "\n".join(context_lines) if context_lines else "(no prior context)"

    # ── Build a type-aware prompt ──────────────────────────────────────────────
    first_name = sender_name.split()[0]

    # Special-case: someone just calling your name
    if is_name_call and not is_greeting:
        msg_type_hint = (
            f"{first_name} is just calling your name/attention (sent: \"{text}\"). "
            "They want your attention. Replies should be short acknowledgement responses "
            "like 'Ha?', 'Labbay?', 'Nima bo'ldi?', 'Aytingchi', or equivalent in their language. "
            "Do NOT suggest plans, times, or locations — that makes no sense here."
        )
    elif is_greeting:
        msg_type_hint = (
            f"{first_name} sent a greeting: \"{text}\". "
            "Replies should be greeting responses in the same language."
        )
    elif has_url and link_platform:
        caption = text_without_url if text_without_url else f"a {link_platform} link"
        msg_type_hint = (
            f"{first_name} shared a {link_platform} link"
            + (f" with caption: \"{caption}\"" if text_without_url else "") + ". "
            "Replies should react to the shared content naturally — "
            "e.g. express interest/reaction, say you'll watch/check it, or ask what it is. "
            "Do NOT ask where it was filmed unless that's clearly relevant."
        )
    elif is_question:
        msg_type_hint = (
            f"{first_name} asked a question: \"{text}\". "
            "Replies should directly answer the question with 3 different realistic answers."
        )
    else:
        msg_type_hint = (
            f"{first_name} said: \"{text}\". "
            "Replies should naturally respond to what they said."
        )

    prompt = (
        f"You are helping me draft 3 natural Telegram quick-replies to a message from {first_name}.\n\n"
        f"Recent conversation:\n{context_str}\n\n"
        f"--- LATEST MESSAGE ---\n"
        f"{msg_type_hint}\n\n"
        "RULES:\n"
        f"1. Match {first_name}'s exact language (Uzbek, Russian, English — whatever they used). "
        "Do NOT translate or switch languages.\n"
        "2. Each reply must be a genuine, natural response that fits the message above. "
        "Think: what would a real person say in this situation?\n"
        "3. Offer 3 distinct tones/choices (e.g. positive, neutral, curious) — "
        "but ONLY if they make sense for this message type. "
        "Do NOT force a 'busy' or 'propose another time' angle if the message is just someone calling your name or sharing a link.\n"
        "4. Keep each reply SHORT — 2 to 8 words maximum. No full sentences with conjunctions.\n"
        "5. Do NOT include emoji unless they're genuinely natural here.\n\n"
        'Output ONLY a JSON array of exactly 3 strings:\n["reply1", "reply2", "reply3"]'
    )

    raw_resp = ''
    try:
        raw_resp = await ollama_service.chat(prompt, history=[])
        cleaned = raw_resp.strip()

        # Extract JSON array even if wrapped in markdown or prose
        match = _re.search(r'\[.*?\]', cleaned, _re.DOTALL)
        if match:
            cleaned = match.group(0)

        # Fix missing commas between quotes e.g. ["A" "B"] → ["A", "B"]
        cleaned = _re.sub(r'"\s+"', '", "', cleaned)
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) >= 2:
            results = [str(s).strip().strip('"\'') for s in data[:3]]
            # Filter out empty or too-long results
            results = [r for r in results if 2 < len(r) < 120]
            if len(results) >= 2:
                return results
    except Exception as exc:
        logger.warning(
            "Failed to parse suggestions from Ollama: %s | raw: %r",
            exc, raw_resp[:200] if raw_resp else ''
        )

    # Heuristic fallback: pull numbered/bulleted lines from raw response
    if raw_resp:
        lines = [
            l.strip().lstrip("0123456789.-) ").strip('"\'')
            for l in raw_resp.split("\n") if l.strip()
        ]
        valid = [l for l in lines if 2 < len(l) < 120 and not l.startswith(("[", "{"))]
        if len(valid) >= 2:
            return valid[:3]

    # Last-resort fallback — language neutral
    return ["👍", "Ok!", "I'll get back to you shortly."]


# ── Background Telethon Event Listener ────────────────────────────────────────

_listener_registered = False


async def start_listener(bot_app: Any, get_owner_chat_id_func: Any) -> None:
    """
    Start listening for incoming direct messages on personal Telegram account.
    Forwards messages into the owner's Telegram AI Bot chat with 1-tap reply suggestions.
    """
    import html
    import time
    from telethon import events
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode
    from utils.callback_registry import callback_registry

    global _listener_registered
    if _listener_registered:
        logger.info("Telethon listener is already running.")
        return

    client = await get_client()
    me = await client.get_me()
    _listener_registered = True

    logger.info("Starting Telethon incoming message listener for user %s (@%s)...", me.first_name, me.username)

    @client.on(events.NewMessage(outgoing=True))
    async def _on_outgoing_message(event: Any) -> None:
        try:
            if not event.is_private:
                return
            # If message was typed directly in official Telegram (not via AI bot)
            if event.id not in _bot_sent_msg_ids:
                chat_id = event.chat_id
                if chat_id:
                    _last_manual_outgoing[chat_id] = time.time()
                    logger.info("Detected manual outgoing message in Telegram for chat %d. Pausing bot alerts for 5 mins.", chat_id)
        except Exception as e:
            logger.debug("Error in _on_outgoing_message: %s", e)

    @client.on(events.NewMessage(incoming=True))
    async def _on_incoming_message(event: Any) -> None:
        try:
            # Only listen to private 1-on-1 direct messages
            if not event.is_private:
                return

            if event.out or (event.sender_id == me.id):
                return

            sender = await event.get_sender()
            if not sender or getattr(sender, "bot", False):
                return

            sender_name = _entity_display_name(sender)
            username = getattr(sender, "username", None)
            sender_id = event.sender_id

            # Check if the user is actively chatting with this person in Telegram directly
            last_manual_sent = _last_manual_outgoing.get(sender_id, 0)
            elapsed = time.time() - last_manual_sent
            if elapsed < 300:  # 5 minutes window
                logger.info(
                    "User is actively texting %s in Telegram (last sent %.1fs ago). Skipping bot alert card.",
                    sender_name, elapsed
                )
                return

            # Extract message text or media description
            msg_text = event.raw_text or ""
            transcription = ""
            voice_file_to_send = None


            if event.voice:
                dur = ""
                try:
                    if event.voice.attributes and hasattr(event.voice.attributes[0], "duration"):
                        dur = f" {event.voice.attributes[0].duration}s"
                except Exception:
                    pass

                # Download voice audio and transcribe it
                try:
                    from pathlib import Path
                    from services.voice_service import transcribe_voice_message  # noqa: PLC0415
                    Path("/tmp/aiagent").mkdir(parents=True, exist_ok=True)
                    tmp_voice_path = f"/tmp/aiagent/incoming_voice_{sender_id}_{event.id}.ogg"
                    await client.download_media(event.message, file=tmp_voice_path)
                    transcription = await transcribe_voice_message(tmp_voice_path)
                    voice_file_to_send = tmp_voice_path
                except Exception as stt_err:
                    logger.warning("Failed to transcribe incoming voice note: %s", stt_err)

                if transcription:
                    msg_text = f"[🎙 Voice message{dur}]: \"{transcription}\""
                else:
                    msg_text = f"[🎙 Voice message{dur}] {msg_text}".strip()

            elif event.photo:
                msg_text = f"[📷 Photo] {msg_text}".strip()
            elif event.video:
                msg_text = f"[🎥 Video] {msg_text}".strip()
            elif event.document:
                doc_name = getattr(event.file, "name", "Document") if hasattr(event, "file") else "Document"
                msg_text = f"[📄 {doc_name}] {msg_text}".strip()
            elif event.sticker:
                emoji = getattr(event.file, "emoji", "") if hasattr(event, "file") else ""
                msg_text = f"[Sticker {emoji}]"

            if not msg_text:
                msg_text = "[Media message]"

            logger.info("Telethon received message from %s (%d): %s", sender_name, sender_id, msg_text)

            owner_chat_id = await get_owner_chat_id_func()
            if not owner_chat_id:
                logger.warning("No owner_chat_id registered for bot. Send /start to the bot first.")
                return

            # If there was a voice note, send the actual audio note into the owner's bot chat so they can listen to it!
            if voice_file_to_send and Path(voice_file_to_send).exists():
                try:
                    with open(voice_file_to_send, "rb") as vf:
                        caption_text = f"🎙️ <b>Voice message from {html.escape(sender_name)}:</b>"
                        if transcription:
                            caption_text += f"\n\n💬 <i>\"{html.escape(transcription)}\"</i>"
                        await bot_app.bot.send_voice(
                            chat_id=int(owner_chat_id),
                            voice=vf,
                            caption=caption_text,
                            parse_mode=ParseMode.HTML,
                        )
                except Exception as forward_voice_err:
                    logger.warning("Could not forward voice file to bot chat: %s", forward_voice_err)

            # Fetch recent history with this contact
            _, history = await get_contact_messages(sender_name, limit=4)

            # Generate 3 contextual quick-reply suggestions via Ollama
            # Check TTL cache first
            _cache_key = (sender_id, hash(msg_text[-50:]))
            _now = time.time()
            _cached = _SUGGESTION_CACHE.get(_cache_key)
            if _cached and _cached[1] > _now:
                suggestions = _cached[0]
                logger.debug("Suggestion cache hit for sender_id=%d", sender_id)
            else:
                try:
                    suggestions = await asyncio.wait_for(
                        generate_quick_suggestions(history, msg_text, sender_name),
                        timeout=_SUGGESTION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("generate_quick_suggestions timed out for sender_id=%d, using fallback", sender_id)
                    suggestions = list(_SUGGESTION_FALLBACK)
                except Exception as sug_exc:
                    logger.warning("generate_quick_suggestions failed: %s", sug_exc)
                    suggestions = list(_SUGGESTION_FALLBACK)
                # Store in TTL cache (evict oldest if at capacity)
                if len(_SUGGESTION_CACHE) >= _SUGGESTION_CACHE_MAX:
                    oldest_key = next(iter(_SUGGESTION_CACHE))
                    del _SUGGESTION_CACHE[oldest_key]
                _SUGGESTION_CACHE[_cache_key] = (suggestions, _now + _SUGGESTION_TTL)

            # ── Fire off AI draft pre-generation in background ─────────────────────
            # Start immediately so by the time user taps '🤖 Detailed AI Reply'
            # the draft is already ready — no waiting.
            _draft_cache_key = (sender_id, hash(msg_text[-80:]))
            existing_draft = _AI_DRAFT_CACHE.get(_draft_cache_key)
            draft_expired = (
                existing_draft is None
                or time.time() - existing_draft.get("ts", 0) > _AI_DRAFT_TTL
            )
            if draft_expired:
                asyncio.create_task(
                    _pre_generate_ai_draft(sender_id, sender_name, msg_text, hash(msg_text[-80:]))
                )

            set_active_contact({
                "user_id": sender_id,
                "name": sender_name,
                "username": username,
                "entity": sender,
                "last_message": msg_text,
                "timestamp": time.time(),
                "suggestions": suggestions,
            })

            # Build inline keyboard
            buttons: list[list[InlineKeyboardButton]] = []

            for idx, sug in enumerate(suggestions):
                ref_id = callback_registry.register("tg_sug", {
                    "user_id": sender_id,
                    "name": sender_name,
                    "text": sug,
                })
                # Truncate button label for UI cleanly
                btn_text = f"{idx+1}️⃣ {sug}"
                if len(btn_text) > 42:
                    btn_text = btn_text[:40] + "…"
                buttons.append([InlineKeyboardButton(btn_text, callback_data=f"tg_sug:{ref_id}")])

            ref_custom = callback_registry.register("tg_custom", {
                "user_id": sender_id,
                "name": sender_name,
            })
            ref_ai_long = callback_registry.register("tg_ai_long", {
                "user_id": sender_id,
                "name": sender_name,
                "last_message": msg_text,
                "msg_hash": hash(msg_text[-80:]),  # key to look up pre-generated draft
            })

            ref_dismiss = callback_registry.register("tg_dismiss", {
                "user_id": sender_id,
                "name": sender_name,
            })

            buttons.append([
                InlineKeyboardButton("✍️ Custom Reply", callback_data=f"tg_custom:{ref_custom}"),
                InlineKeyboardButton("🤖 Detailed AI Reply", callback_data=f"tg_ai_long:{ref_ai_long}"),
            ])
            buttons.append([
                InlineKeyboardButton("❌ Dismiss", callback_data=f"tg_dismiss:{ref_dismiss}"),
            ])

            reply_markup = InlineKeyboardMarkup(buttons)

            sug_list_text = "\n".join(f"{i+1}️⃣ <i>\"{html.escape(s)}\"</i>" for i, s in enumerate(suggestions))

            card_html = (
                f"📩 <b>New message from {html.escape(sender_name)}:</b>\n"
                f"<blockquote><i>\"{html.escape(msg_text)}\"</i></blockquote>\n\n"
                f"💡 <b>Quick Reply Suggestions:</b>\n"
                f"{sug_list_text}\n\n"
                f"<i>🎙️ Tip: Send a voice note or type below to reply directly!</i>"
            )

            await bot_app.bot.send_message(
                chat_id=owner_chat_id,
                text=card_html,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )

        except Exception as err:
            logger.exception("Error in telethon incoming message handler: %s", err)

