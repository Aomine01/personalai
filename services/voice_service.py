"""
services/voice_service.py — Speech-to-Text (STT) voice message transcription.

Supports:
1. Google Gemini Flash (Cloud STT) — ultra-fast, handles Uzbek, Russian, English, accents with extreme accuracy.
2. Faster-Whisper (Local STT) — 100% offline fallback running on your Mac.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded local whisper model singleton
_whisper_model = None


def _get_local_whisper():
    """Lazy load faster-whisper tiny/base model for local offline transcription."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Loading local faster-whisper model (base)...")
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as e:
            logger.warning("Could not load faster-whisper: %s", e)
            return None
    return _whisper_model


async def _transcribe_with_gemini(audio_path: str) -> str:
    """Transcribe audio using Google Gemini multimodal audio API."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Determine MIME type based on file extension
    ext = Path(audio_path).suffix.lower()
    mime_type = "audio/ogg" if ext in {".ogg", ".oga"} else "audio/mp3" if ext == ".mp3" else "audio/wav"

    def _sync_call():
        # Real model names in order of reliability/speed.
        # Use the latest available Gemini models for STT (ordered by preference).
        # gemini-3.6-flash is the current recommended model per the API.
        for m in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        "Please accurately transcribe the speech in this audio word-for-word into text. "
                        "Pay close attention to contact names, abbreviations, and numbers (e.g. TR number, Turkish number, names in Uzbek, Russian, English). "
                        "Do NOT translate it, preserve the exact language spoken. "
                        "Do NOT add any conversational filler, quotes, explanation, or commentary. "
                        "Output ONLY the exact transcribed text."
                    ],
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                res = (response.text or "").strip()
                if res:
                    return res
            except Exception as exc:
                logger.warning("Gemini voice STT attempt on model %s failed: %s", m, exc)
        return ""

    return await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=20.0)



async def _transcribe_with_whisper(audio_path: str) -> str:
    """Transcribe audio locally using faster-whisper."""
    model = _get_local_whisper()
    if model is None:
        raise RuntimeError("Local Whisper model not available")

    def _sync_transcribe():
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text for s in segments).strip()

    return await asyncio.to_thread(_sync_transcribe)


async def transcribe_voice_message(audio_path: str) -> str:
    """
    Transcribe a voice note (.ogg / .mp3 / .wav) to text.
    Tries Google Gemini Flash first (fastest, high Uzbek accuracy),
    and falls back to local faster-whisper.
    """
    # 1. Try Gemini Flash STT
    if settings.GEMINI_API_KEY:
        try:
            logger.info("Transcribing voice note with Gemini Flash STT: %s", audio_path)
            text = await _transcribe_with_gemini(audio_path)
            if text:
                return text
        except Exception as e:
            logger.warning("Gemini voice transcription failed: %s, trying local fallback", e)

    # 2. Try Local Faster-Whisper
    try:
        logger.info("Transcribing voice note with local Faster-Whisper: %s", audio_path)
        text = await _transcribe_with_whisper(audio_path)
        if text:
            return text
    except Exception as e:
        logger.warning("Local Whisper transcription failed: %s", e)

    raise RuntimeError("Could not transcribe voice message (both Gemini and Whisper failed).")
