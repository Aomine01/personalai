"""
utils/intent_parser.py — Keyword-based intent detection for TEXT messages only.

File/document uploads have no text to classify — they are routed directly in
main.py via filters.Document.ALL, completely bypassing this module.

Intents (ordered most-specific → least-specific):
  GENERATE_IMAGE    — image / picture generation request
  WEATHER           — weather / forecast query
  WEB_SEARCH        — browse / search the internet
  CREATE_PDF        — PDF summary or report request
  WRITE_REPORT      — Word (.docx) report request
  CREATE_EXCEL      — Spreadsheet / xlsx request
  CREATE_STYLED_DOC — "match this style / use this as a template" request
                      NOTE: only meaningful if a pending_reference_file exists
                      in conversation state — the handler checks this, not here.
  CHAT              — fallback

Web-search sub-flags:
  parse_web_search_flags(text) returns (clean_query, wants_report: bool).
  wants_report=True when the user asks for a PDF summary document alongside
  the search (e.g. "give me a report", "save as pdf", "news report").
"""
from __future__ import annotations

import re
from enum import Enum, auto


class Intent(Enum):
    READ_TELEGRAM_MESSAGES = auto()
    SEND_TELEGRAM_MESSAGE = auto()
    LOCAL_PROJECT     = auto()
    GENERATE_IMAGE    = auto()
    WEATHER           = auto()
    WEB_SEARCH        = auto()
    CREATE_PDF        = auto()
    WRITE_REPORT      = auto()
    CREATE_EXCEL      = auto()
    CREATE_STYLED_DOC = auto()
    CHAT              = auto()


_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        Intent.READ_TELEGRAM_MESSAGES,
        re.compile(
            # English
            r"\b(?:what\s+(?:did|is|does)\s+[\w\s\"\'«»“”]+?\s+(?:ask|asking|want|wants|say|saying|text|texted|texting|write|wrote|writing|send|sent|sending|request|mean|talking\s+about))"
            r"|\b(what\s+(?:did\s+)?[\w\s\"\'«»“”]+?\s*(?:send|sent|text|texted|write|wrote|say|said|ask|asked|want|wanted)(?:\s+(?:to\s+)?me)?(?:\s+for)?)"
            r"|\b(?:i\s+have\s+a\s+)?contact\s+(?:named|called)\s+[\w\s\"\'«»“”]+"
            r"|\b(?:read|check|show|get|see|find|fetch)\s+(?:recent\s+)?(?:messages?|chat|conversation|texts?)\s+(?:from|with)\s+[\w\s\"\'«»“”]+"
            r"|\b(?:last|latest)\s+messages?\s+(?:from|with)\s+[\w\s\"\'«»“”]+"
            r"|\b(?:did|has)\s+[\w\s\"\'«»“”]+?\s+(?:message|text|send|write|say|reply|answer)(?:\s+me)?\b"
            r"|\bany\s+(?:new\s+)?messages?\s+(?:from|with)\s+[\w\s\"\'«»“”]+"
            # Uzbek
            r"|\b[\w\s\"\'«»“”]+?\s+(?:nima|nimani)\s+(?:deb\s+)?(?:yozdi|yozgan|jo\'?natdi|jo\'?natgan|yubordi|yuborgan|aytgan|aytdi|so\'?radi|so\'?rayapti|so\'?ragan|xohlayapti|demoqchi|deyapti)\b"
            r"|\b[\w\s\"\'«»“”]+?(?:dan|bilan)\s+(?:kelgan\s+)?(?:xabar|xabarlar|chat|suhbat)\b"
            r"|\b(?:menda\s+)?(?:kontakt|kontaktim|kontakt\s+bor)\b"
            # Russian
            r"|\bчто\s+(?:мне\s+)?(?:написал|написала|прислал|прислала|отправил|отправила|сказал|сказала|хочет|просит|спрашивает|пишет|говорит)\s+[\w\s\"\'«»“”]+"
            r"|\bчто\s+[\w\s\"\'«»“”]+?\s+(?:мне\s+)?(?:написал|написала|прислал|прислала|отправил|отправила|сказал|сказала|хочет|просит|спрашивает|пишет|говорит)"
            r"|\b(?:покажи|прочитай|посмотри|проверь|найди)\s+(?:сообщения?|переписку|чат)\s+(?:от|с)\s+[\w\s\"\'«»“”]+"
            r"|\b(?:у\s+меня\s+есть\s+)?контакт\s+(?:по\s+имени|под\s+именем|с\s+именем)?\s*[\w\s\"\'«»“”]+",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.SEND_TELEGRAM_MESSAGE,
        re.compile(
            # English — require contact name (capitalized word) immediately after verb
            r"\b(?:open\s+telegram\s+and|send\s+(?:a\s+)?(?:message|msg|text)\s+to)\b"
            r"|\btext\s+(?:him|her|them\b|[A-Z][a-z])"
            r"|\b(?:message|dm)\s+[A-Z][a-z]"
            r"|\bfind\s+(?:a\s+)?(?:contact|person|user)\s+(?:called|named)\s+[\w\s]+\s+and\s+(?:text|message|send|write)\b"
            r"|\b(?:write\s+to|chat\s+with)\s+(?:a\s+)?(?:contact\s+)?[A-Z][a-z]"
            # Uzbek
            r"|\b(?:kontakt|kontaktni|kontaktini|odamni|foydalanuvchini)\s+(?:top|topgin|qidir)\b"
            r"|\b(?:top|topgin|qidir)\s+va\s+(?:yoz|yozgin|yubor|yuborgin|jo\'?nat|jo\'?natgin)\b"
            r"|\b(?:xabar|sms|matn)\s+(?:yoz|yozgin|yubor|yuborgin|jo\'?nat|jo\'?natgin)\b"
            r"|\b[A-Za-zА-Яа-я]{3,}(?:ga|ka|qa)\b.{1,80}\b(?:deb\s+)?(?:yoz|yozgin|yubor|yuborgin|jo\'?nat|jo\'?natgin)\b"
            # Russian — must have a name/target after the verb OR uses pronoun after prior context
            r"|позвони |напомни |скажи |\b(?:найди|поищи)\s+(?:контакт|человека|пользователя)?\b.{0,30}\b(?:напиши|отправь|скинь)\b"
            r"|\b(?:напиши|отправь|скинь|пошли|напомни|скажи|сообщи|перешли)\s+[A-Za-zА-Яа-я]+\b"
            r"|\b(?:напомни|скажи|\u043fерешли)\s+(?:ей|ему|им)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.LOCAL_PROJECT,
        re.compile(
            r"\b(in\s+my\s+mac|on\s+my\s+mac|go\s+to\s+(?:the\s+)?folder|look\s+(?:in|for)\s+(?:the\s+)?(?:folder|project|repo|directory))\b"
            r"|\b(?:analyze|inspect|explore|check|scan|read|look\s+at)\s+(?:my\s+)?(?:local\s+)?(?:project|codebase|repo|folder|directory|app)\b"
            r"|\b(?:folder|project|directory|repo)\s+(?:called|named)\s+[\w\-]+"
            r"|\b(?:~/|/Users/|Desktop/|Documents/|Projects/|Developer/)[\w\-/\.]+",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.GENERATE_IMAGE,
        re.compile(
            r"\b(gen[a-z]*|create|make|draw|paint|show|generet|genrete)\b.{0,30}\b(image|img[a-z]*|picture|pic[a-z]*|photo|drawing|illustration|art|logo[a-z]*|icon[a-z]*|graphic[a-z]*|avatar[a-z]*)\b"
            r"|\b(image|img[a-z]*|picture|pic[a-z]*|photo|drawing|illustration|art|logo[a-z]*|icon[a-z]*|graphic[a-z]*|avatar[a-z]*)\b.{0,30}\b(of|showing|with)\b"
            r"|\b(genrete|generet|gen[a-z]*|draw|paint)\b\s+(it|this)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.WEATHER,
        re.compile(
            r"\b(weather|forecast|temperature|temp|rain|snow|sunny|cloudy|humid|wind speed|feels like)\b"
            r"|\bhow (hot|cold|warm|cool|rainy|sunny|windy) is\b"
            r"|\bwill it rain\b|\bis it (raining|snowing|sunny|cloudy)\b"
            r"|\b(today|tomorrow|this week)[\s\w]{0,20}(weather|rain|snow|sun|cloud)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.WEB_SEARCH,
        re.compile(
            # Explicit search commands
            r"\b(search|browse|look up|look it up|find|google|check online|search the (web|internet|net)|search for|look online|check the (web|internet|news))\b"
            # Questions about current/recent things (handling contractions like whats/what's, whos/who's, wheres/where's)
            r"|\b(what\s*\'?s|what\s+is|who\s*\'?s|who\s+is|who\s+was|what\s+are|when\s+did|where\s*\'?s|where\s+is|how\s+does|why\s+does|tell\s+me\s+about)\b.{0,60}\b(today|now|current|latest|recent|news|202[3-9])\b"
            # Latest news
            r"|\b(latest|recent|current|breaking) (news|updates?|info|information)\b"
            # Ratings, reviews, scores — always need live data
            r"|\b(rating|ratings|review|reviews|score|scores|imdb|rotten tomatoes|metacritic|box office|gross|grossing)\b"
            # Release / premiere info
            r"|\b(release date|when (is|does|did|will).{0,30}(release|come out|premiere|air|drop|launch))\b"
            r"|\b(out now|just released|newly released|new release|coming out|when.*out)\b"
            # Prices, standings, results — live facts
            r"|\b(price of|cost of|how much is|stock price|standings|results|who won|who lost|champion|winner)\b"
            # Explicit year mentions for factual queries (things in 2024/2025/2026 etc.)
            r"|\b(202[3-9])\b.{0,60}\b(movie|film|series|show|season|episode|game|album|song|book|award|oscar|grammy)\b"
            r"|\b(movie|film|series|show|season|episode|game|album|song|book|award|oscar|grammy)\b.{0,60}\b(202[3-9])\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CREATE_PDF,
        re.compile(
            # Must have an explicit action verb BEFORE 'pdf' (within 25 chars)
            r"\b(create|make|gen[a-z]*|generet|genrete|export|save|write|send|get|build|produce|give me)\b.{0,25}\bpdf\b"
            # OR 'pdf' directly followed by what kind of document
            r"|\bpdf\b.{0,20}\b(summary|report|document|file)\b"
            # Explicit combo phrases
            r"|\bpdf summary\b|\bexport.*to.*pdf\b|\bsave.*as.*pdf\b",
            re.IGNORECASE,
        ),
    ),
    (
        # ── CREATE_STYLED_DOC must come BEFORE WRITE_REPORT ──────────────────
        # so "generate a doc like this" doesn't get stolen by the broader doc pattern
        Intent.CREATE_STYLED_DOC,
        re.compile(
            r"\bmatch\b.{0,20}\bstyle\b"
            r"|\buse\b.{0,20}\btemplate\b"
            r"|\bsame style\b|\bsame format\b"
            # "generate a doc like this/that/it [for X]"
            r"|\b(gen[a-z]*|create|make|write)\b.{0,30}\b(doc|document|pdf)\b.{0,30}"
            r"\b(like|similar to|based on|matching|in this style|in the same style)\b"
            # "generate a doc like <filename>"
            r"|\b(gen[a-z]*|create|make|write)\b.{0,20}\b(doc|document)\b.{0,10}\blike\b"
            # "use this style / use this format / use this as template"
            r"|\buse this\b.{0,30}\b(style|format|template|as (a )?template)\b"
            # "generate a doc based on this file"
            r"|\b(based on|using).{0,20}\b(this|the) (file|document|doc|reference)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.WRITE_REPORT,
        re.compile(
            # Explicit "report" phrases
            r"\b(write|create|gen[a-z]*|make|produce|give me|send me)\b.{0,30}\breport\b"
            r"|\breport\b.{0,10}\b(docx|word|document)\b"
            r"|\b(docx|word document|word file)\b"
            # "generate/create/make/write a doc / document / word ..."
            r"|\b(gen[a-z]*|create|make|write|produce|give me|send me)\b.{0,15}"
            r"\b(doc|document|word)\b(?:\s+file)?"
            # "i want / i need a document/doc/word file"
            r"|\b(want|need)\b.{0,30}\b(doc|document|word)\b"
            # "i want you to generate a doc"
            r"|\bi want you to\b.{0,30}\b(doc|document|word)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CREATE_EXCEL,
        re.compile(
            r"\b(create|make|gen[a-z]*|generet|genrete|build|export|save|write|send|get)\b.{0,30}\b(excel|spreadsheet|xlsx)\b"
            r"|\b(excel|spreadsheet|xlsx)\b.{0,30}\b(of|for|with|showing|gen[a-z]*|create|make)\b"
            r"|\b(make|generate|gen[a-z]*)\b.{0,20}\btable\b.{0,20}\b(excel|spreadsheet)\b"
            r"|\bgen[a-z]* xlsx\b|\bcreate xlsx\b|\bexport to excel\b",
            re.IGNORECASE,
        ),
    ),
]


def parse_intent(text: str) -> Intent:
    """Return the most likely Intent for *text*. Runs in O(n_patterns) time."""
    # ── Conversational / instructional setup detection ──────────────────────────
    # If the user is teaching rules, naming the bot, asking if they understand,
    # or telling them to memorize, bypass trigger detection to keep it in CHAT.
    lower_text = text.lower().strip()
    
    setup_patterns = [
        r"^(right now\s+)?i\s+(will|am going to)\s+(give|send|text|write|list|provide|show)\b",
        r"\b(did you get it|do you understand|is that clear|is this clear|agree|okay|ok)\s*(\w+)?\??$",
        r"\bfrom now (on|onwards)\b",
        r"\b(i will call you|your name is|you are now called)\b"
    ]
    
    if any(re.search(pat, lower_text) for pat in setup_patterns):
        return Intent.CHAT

    for intent, pattern in _PATTERNS:
        if pattern.search(text):
            return intent
    return Intent.CHAT


def extract_image_subject(text: str) -> str:
    """
    Extract the subject description from an image-generation request.
    Handles spelling variations and strips action phrases and filler words cleanly.
    """
    # Compile pattern to match leading conversational filler and trigger verbs
    prefix_pat = re.compile(
        r"^(?:yeah|yes|ok|okay|sure|please|no|can\s+you|could\s+you|now|just|so|and|then|please|you|u|have\s+to|want|want\s+to|\s|,|-)*"
        r"\b(gen[a-z]*|create|make|draw|paint|show|genrete|generet)\b\s*"
        r"(?:me\s+)?(?:and\s+)?(?:an?\s+)?(?:image|img[a-z]*|pic[a-z]*|photo)?\s*"
        r"(?:of|showing|with|for)?\s*",
        re.IGNORECASE
    )
    cleaned = prefix_pat.sub("", text)

    # Strip trailing pronouns, visual request fillers, and follow-up directives
    trail_pat = re.compile(
        r"\b(?:it|this|to\s+me|for\s+me|visually|now|so\s+i\s+(?:can|will|could|be\s+able\s+to)\b.*|"
        r"so\s+i\s+can\s+see\s+it|please|\s)+\b$",
        re.IGNORECASE
    )
    cleaned = trail_pat.sub("", cleaned)
    return cleaned.strip()


def extract_search_query(text: str) -> str:
    """Strip search-trigger words and return the raw query."""
    prefix_pat = re.compile(
        r"^(?:please\s+)?(?:search(?:\s+for|\s+the\s+(?:web|internet|net))?|"
        r"look\s+(?:up|online|it\s+up)|browse|google|find|check(?:\s+online)?)\s*",
        re.IGNORECASE,
    )
    return prefix_pat.sub("", text).strip()


# ── News-report sub-flag detection ───────────────────────────────────────────

_REPORT_HINTS: re.Pattern[str] = re.compile(
    r"\b(give\s+me\s+a\s+report|summary\s+doc(ument)?|generate\s+summary"
    r"|news\s+report|save\s+(as\s+)?pdf|make\s+(a\s+)?report"
    r"|generate\s+(a\s+)?report|create\s+(a\s+)?report"
    r"|export\s+(to\s+)?pdf|send\s+as\s+pdf"
    r"|i\s+want\s+a\s+report|show\s+me\s+a\s+report)\b",
    re.IGNORECASE,
)


def parse_web_search_flags(text: str) -> tuple[str, bool]:
    """
    Parse a WEB_SEARCH text into (clean_query, wants_report).

    wants_report=True when the user asked for a PDF summary document alongside
    the search results (trigger phrases: "give me a report", "save as pdf",
    "news report", "generate summary", etc.).

    The returned clean_query has both the search-prefix verbs AND report-trigger
    phrases stripped so the raw query sent to Tavily is clean.

    Usage (in main.py):
        query, wants_report = parse_web_search_flags(text)
    """
    wants_report = bool(_REPORT_HINTS.search(text))
    # Strip report phrases from the query so they don't pollute Tavily results
    cleaned = _REPORT_HINTS.sub("", text).strip()
    # Also strip leading search-trigger verbs
    query = extract_search_query(cleaned)
    return query or text.strip(), wants_report
