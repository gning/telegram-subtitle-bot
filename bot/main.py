"""
Entry point — set up the Telegram bot and start polling.
"""

import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    filters,
    MessageHandler,
)

from bot.config import LOCAL_BOT_API_URL, TELEGRAM_BOT_TOKEN
from bot.handlers import handle_video, start

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Telegram subtitle bot...")

    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    if LOCAL_BOT_API_URL:
        # Remote Bot API server: raises download limit from 20 MB to 2 GB.
        # Do NOT use local_mode(True) — that's only for when the server runs on
        # the same machine. With a remote server, file downloads are handled via
        # HTTP in handlers.py using a reconstructed URL.
        builder = (
            builder
            .base_url(f"{LOCAL_BOT_API_URL}/bot")
            .base_file_url(f"{LOCAL_BOT_API_URL}/file/bot")
            .local_mode(True)
        )
        logger.info("Using local Bot API server at %s", LOCAL_BOT_API_URL)
    app = builder.build()

    app.add_handler(CommandHandler("start", start))

    # Accept both compressed video messages and video files sent as documents
    video_filter = filters.VIDEO | (filters.Document.VIDEO)
    app.add_handler(MessageHandler(video_filter, handle_video))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
