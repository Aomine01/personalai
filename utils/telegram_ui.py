import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

def build_result_keyboard(buttons: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    """
    Constructs an inline keyboard layout from a list of (label, callback_data) tuples.
    Displays buttons sequentially (each on a new row to ensure readability on mobile screens).
    """
    rows = [[InlineKeyboardButton(text=label, callback_data=data)] for label, data in buttons]
    return InlineKeyboardMarkup(rows)

async def send_status(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> tuple[int, int]:
    """
    Sends an initial text-based status/progress message using context.bot.
    Returns (message_id, chat_id) for subsequent editing or replacement.
    """
    assert update.effective_chat is not None
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode=ParseMode.MARKDOWN
    )
    return msg.message_id, update.effective_chat.id

async def edit_to_result_text(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """
    Edits an existing text message in-place with new text and an optional keyboard.
    """
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
    except Exception as exc:
        logger.warning("Failed editing message in-place: %s, trying plain text fallback", exc)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=keyboard
            )
        except Exception as fallback_exc:
            logger.error("Failed final message edit: %s", fallback_exc)

async def replace_with_media(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    status_message_id: int,
    media_path: str,
    media_type: str,
    caption: str,
    keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """
    Deletes the original text status message and sends a new media message instead,
    attaching the results and action buttons. Works around Telegram's constraint
    disallowing text-to-media edits.
    media_type: 'photo' | 'document'
    """
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=status_message_id)
    except Exception as exc:
        logger.debug("Failed deleting status message during replacement: %s", exc)

    send_fn = context.bot.send_photo if media_type == "photo" else context.bot.send_document
    arg_name = "photo" if media_type == "photo" else "document"

    with open(media_path, "rb") as f:
        await send_fn(
            chat_id=chat_id,
            **{arg_name: f},
            caption=caption,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard
        )
