"""
config.py — Centralised settings loaded from environment / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Optional keys — kept empty so the image_service mock can detect absence
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    STABILITY_API_KEY: str = os.getenv("STABILITY_API_KEY", "")

    # Optional local Stable Diffusion API url (Automatic1111)
    LOCAL_SD_URL: str = os.getenv("LOCAL_SD_URL", "")

    # How many past messages to include in PDF / DOCX context
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "20"))

    # Temp directory for generated files
    TMP_DIR: str = os.getenv("TMP_DIR", "/tmp/aiagent")

    # Weather — OpenWeatherMap free tier
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")

    # Web search — Tavily
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Vision — Google Gemini Flash (free tier, better than llava for people/objects)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Personal Telegram UserBot (Telethon MTProto client)
    TELEGRAM_API_ID: int = int(os.getenv("TELEGRAM_API_ID", "0"))
    TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
    TELEGRAM_PHONE: str = os.getenv("TELEGRAM_PHONE", "")

    # SQLite database file path
    DB_PATH: str = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db"))

    # Dual-model routing configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen3:8b")
    DEEP_MODEL: str = os.getenv("DEEP_MODEL", "gpt-oss:20b")
    MODEL_SWITCH_KEEP_ALIVE: str = os.getenv("MODEL_SWITCH_KEEP_ALIVE", "5m")



settings = Settings()

# Ensure temp directory exists
os.makedirs(settings.TMP_DIR, exist_ok=True)

