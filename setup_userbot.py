"""
setup_userbot.py — One-time interactive login for the Telegram UserBot.

Run this once:
    .venv/bin/python setup_userbot.py

It will:
1. Ask for your phone number (if not in .env)
2. Send a login code to your Telegram app
3. Save the session to userbot.session

After that, main.py can use the UserBot without any further login prompts.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

API_ID = int(os.environ.get("TELEGRAM_API_ID", 0))
API_HASH = os.environ.get("TELEGRAM_API_HASH", "")
PHONE = os.environ.get("TELEGRAM_PHONE", "")
SESSION_PATH = str(Path(__file__).parent / "userbot")


async def main() -> None:
    from telethon import TelegramClient

    if not API_ID or not API_HASH:
        print("❌ TELEGRAM_API_ID or TELEGRAM_API_HASH not set in .env")
        return

    phone = PHONE or input("📱 Enter your Telegram phone number (e.g. +998901234567): ").strip()

    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"✅ Already logged in as: {me.first_name} (@{me.username})")
        await client.disconnect()
        return

    await client.send_code_request(phone)
    code = input("🔑 Enter the verification code sent to your Telegram app: ").strip()

    try:
        await client.sign_in(phone, code)
    except Exception as e:
        if "Two-steps" in str(e) or "password" in str(e).lower():
            password = input("🔒 Enter your 2FA password: ").strip()
            await client.sign_in(password=password)
        else:
            raise

    me = await client.get_me()
    print(f"\n✅ Successfully logged in as: {me.first_name} ({me.phone})")
    print(f"📁 Session saved to: {SESSION_PATH}.session")
    print("\nYou can now start the bot normally. The UserBot is ready!")
    await client.disconnect()

    # Save phone to .env if not there
    env_path = Path(__file__).parent / ".env"
    env_content = env_path.read_text()
    if "TELEGRAM_PHONE=" in env_content and not PHONE:
        env_content = env_content.replace("TELEGRAM_PHONE=", f"TELEGRAM_PHONE={phone}")
        env_path.write_text(env_content)
        print(f"✅ Phone number saved to .env")


if __name__ == "__main__":
    asyncio.run(main())
