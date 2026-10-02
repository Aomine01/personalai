"""
services/weather_service.py — OpenWeatherMap integration.

Free-tier endpoints used:
  Current weather:  GET /data/2.5/weather
  5-day forecast:   GET /data/2.5/forecast

Docs: https://openweathermap.org/api
"""
from __future__ import annotations

import httpx
import logging
import re
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.openweathermap.org/data/2.5"

# Weather condition code → emoji map (coarse)
_EMOJI: dict[str, str] = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Drizzle": "🌦️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️",
    "Fog": "🌫️",
    "Haze": "🌫️",
    "Smoke": "🌫️",
    "Dust": "🌪️",
    "Sand": "🌪️",
    "Ash": "🌋",
    "Squall": "💨",
    "Tornado": "🌪️",
}


def _emoji(main: str) -> str:
    return _EMOJI.get(main, "🌡️")


def _wind_dir(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


async def get_current_weather(city: str) -> str:
    """Return a formatted current-weather string for *city*."""
    if not settings.OPENWEATHER_API_KEY:
        return "❌ Weather API key not configured."

    params = {
        "q": city,
        "appid": settings.OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_BASE}/weather", params=params)

        if resp.status_code == 404:
            return f"❌ City *{city}* not found. Try a different spelling or add the country code (e.g. *Paris,FR*)."

        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

        weather_main = data["weather"][0]["main"]
        description = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        visibility = data.get("visibility", 0) // 1000  # m -> km
        wind_speed = data["wind"]["speed"]              # m/s
        wind_deg = data["wind"].get("deg", 0)
        wind_dir = _wind_dir(wind_deg)
        city_name = data["name"]
        country = data["sys"]["country"]
        emoji = _emoji(weather_main)

        import datetime
        sunrise = datetime.datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M")
        sunset = datetime.datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M")

        return (
            f"{emoji} *Weather in {city_name}, {country}*\n\n"
            f"🌡 *{temp:.1f}°C* (feels like {feels_like:.1f}°C)\n"
            f"📋 {description}\n"
            f"💧 Humidity: {humidity}%\n"
            f"🌬 Wind: {wind_speed} m/s {wind_dir}\n"
            f"👁 Visibility: {visibility} km\n"
            f"📊 Pressure: {pressure} hPa\n"
            f"🌅 Sunrise: {sunrise} · Sunset: {sunset}"
        )

    except httpx.HTTPStatusError as exc:
        logger.error("OpenWeatherMap HTTP error for %s: %s", city, exc)
        return f"⚠️ Couldn't fetch weather for *{city}*: server returned {exc.response.status_code}."
    except Exception as exc:
        logger.exception("Weather fetch failed for %s: %s", city, exc)
        return "⚠️ Weather service is temporarily unavailable. Please try again."


async def get_forecast(city: str) -> str:
    """Return a 5-day (3-hour interval) forecast summary for *city*."""
    if not settings.OPENWEATHER_API_KEY:
        return "❌ Weather API key not configured."

    params = {
        "q": city,
        "appid": settings.OPENWEATHER_API_KEY,
        "units": "metric",
        "cnt": 40,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_BASE}/forecast", params=params)

        if resp.status_code == 404:
            return f"❌ City *{city}* not found."

        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        city_name = data["city"]["name"]
        country = data["city"]["country"]

        import datetime
        from collections import defaultdict

        days: dict[str, list[dict]] = defaultdict(list)
        for item in data["list"]:
            date_str = item["dt_txt"][:10]
            days[date_str].append(item)

        lines = [f"📅 *5-Day Forecast for {city_name}, {country}*\n"]
        for date_str, slots in sorted(days.items())[:5]:
            slot = next((s for s in slots if "12:00" in s["dt_txt"]), slots[0])
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            day_name = dt.strftime("%a %d %b")
            temp_min = min(s["main"]["temp_min"] for s in slots)
            temp_max = max(s["main"]["temp_max"] for s in slots)
            description = slot["weather"][0]["description"].capitalize()
            emoji = _emoji(slot["weather"][0]["main"])
            lines.append(f"{emoji} *{day_name}*: {description}, {temp_min:.0f}–{temp_max:.0f}°C")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("Forecast fetch failed for %s: %s", city, exc)
        return "⚠️ Could not fetch forecast. Please try again."


# ── City extraction ──────────────────────────────────────────────────────────

_CITY_PATTERNS = [
    re.compile(
        r"\b(?:weather|forecast|temperature|temp|rain|snow|sunny|cloudy|humid|wind)\b.{0,20}"
        r"\b(?:in|at|for|of)\s+([A-Za-z][A-Za-z\s,]{1,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"^([A-Za-z][A-Za-z\s,]{1,30})\s+(?:weather|forecast|temperature|temp)\b",
        re.IGNORECASE,
    ),
]


def extract_city(text: str) -> str:
    """Best-effort city extraction from a natural language weather request."""
    for pat in _CITY_PATTERNS:
        m = pat.search(text)
        if m:
            city = m.group(1).strip().rstrip(".,!?")
            city = re.sub(
                r"\s+(?:today|tomorrow|now|currently|please|right now|this week)\s*$",
                "",
                city,
                flags=re.IGNORECASE,
            ).strip()
            if city:
                return city
    return ""
