"""
Entry point — set up the Telegram bot and start polling.
"""

import logging

from telegram import Update
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    filters,
    MessageHandler,
    PicklePersistence,
)

from bot import config
from bot.handlers import handle_video, start
from bot.polling import HeartbeatHTTPXRequest, PollingWatchdog
from bot.settings import (
    cmd_settings,
    cmd_set_whisper,
    cmd_set_whisper_url,
    cmd_set_whisper_model,
    cmd_set_mlx_model,
    cmd_set_translation,
    cmd_set_translation_url,
    cmd_set_translation_model,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
# HTTPX emits one synchronous INFO log for every long-poll response. Apart from
# exposing the bot token in the URL, a back-pressured stderr/logging pipe can
# block the asyncio event loop at that log call and stop polling indefinitely.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Telegram subtitle bot...")

    persistence = PicklePersistence(filepath="bot_persistence.pkl")
    watchdog = (
        PollingWatchdog(config.TELEGRAM_POLL_WATCHDOG_TIMEOUT_SECONDS)
        if config.TELEGRAM_POLL_WATCHDOG_TIMEOUT_SECONDS > 0
        else None
    )

    api_request = HTTPXRequest(
        read_timeout=config.TELEGRAM_API_READ_TIMEOUT_SECONDS,
        write_timeout=config.TELEGRAM_API_WRITE_TIMEOUT_SECONDS,
        connect_timeout=config.TELEGRAM_API_CONNECT_TIMEOUT_SECONDS,
        pool_timeout=config.TELEGRAM_API_POOL_TIMEOUT_SECONDS,
        media_write_timeout=config.TELEGRAM_API_MEDIA_WRITE_TIMEOUT_SECONDS,
    )
    get_updates_request = HeartbeatHTTPXRequest(
        connection_pool_size=1,
        read_timeout=config.TELEGRAM_GET_UPDATES_READ_TIMEOUT_SECONDS,
        write_timeout=config.TELEGRAM_API_WRITE_TIMEOUT_SECONDS,
        connect_timeout=config.TELEGRAM_API_CONNECT_TIMEOUT_SECONDS,
        pool_timeout=config.TELEGRAM_API_POOL_TIMEOUT_SECONDS,
        on_success=watchdog.record_success if watchdog else lambda: None,
    )

    async def start_watchdog(_application: Application) -> None:
        if watchdog:
            watchdog.start()
            logger.info(
                "Polling watchdog enabled (timeout %.0fs).",
                config.TELEGRAM_POLL_WATCHDOG_TIMEOUT_SECONDS,
            )

    async def stop_watchdog(_application: Application) -> None:
        if watchdog:
            watchdog.stop()

    builder = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .persistence(persistence)
        .request(api_request)
        .get_updates_request(get_updates_request)
        .post_init(start_watchdog)
        .post_shutdown(stop_watchdog)
    )
    if config.LOCAL_BOT_API_URL:
        # local_mode=True matches telegram-bot-api --local. It lets PTB copy
        # absolute file paths directly when the server shares this filesystem;
        # handlers.py falls back to HTTP for a remote local-mode server.
        builder = (
            builder
            .base_url(f"{config.LOCAL_BOT_API_URL}/bot")
            .base_file_url(f"{config.LOCAL_BOT_API_URL}/file/bot")
            .local_mode(True)
        )
        logger.info("Using local Bot API server at %s", config.LOCAL_BOT_API_URL)
    app = builder.build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings",              cmd_settings))
    app.add_handler(CommandHandler("set_whisper",           cmd_set_whisper))
    app.add_handler(CommandHandler("set_whisper_url",       cmd_set_whisper_url))
    app.add_handler(CommandHandler("set_whisper_model",     cmd_set_whisper_model))
    app.add_handler(CommandHandler("set_mlx_model",         cmd_set_mlx_model))
    app.add_handler(CommandHandler("set_translation",       cmd_set_translation))
    app.add_handler(CommandHandler("set_translation_url",   cmd_set_translation_url))
    app.add_handler(CommandHandler("set_translation_model", cmd_set_translation_model))

    # Accept both compressed video messages and video files sent as documents
    video_filter = filters.VIDEO | (filters.Document.VIDEO)
    app.add_handler(MessageHandler(video_filter, handle_video))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        timeout=config.TELEGRAM_POLL_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
