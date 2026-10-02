import asyncio
import logging
from telegram.constants import ChatAction
from telegram import Bot

logger = logging.getLogger(__name__)

class TypingRefresher:
    """Async context manager to keep the typing indicator active during long calls."""
    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            finally:
                self.task = None

    async def _loop(self):
        while True:
            try:
                await self.bot.send_chat_action(chat_id=self.chat_id, action=ChatAction.TYPING)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Failed to send typing chat action: %s", e)
            
            try:
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
