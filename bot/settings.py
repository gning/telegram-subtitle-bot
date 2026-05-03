import logging
from telegram import Update
from telegram.ext import ContextTypes
from bot.config import (
    WHISPER_BACKEND, WHISPER_API_URL, WHISPER_API_MODEL,
    TRANSLATION_BACKEND, OLLAMA_BASE_URL, OLLAMA_TRANSLATION_MODEL,
)

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "whisper_backend": WHISPER_BACKEND,
    "whisper_api_url": WHISPER_API_URL,
    "whisper_api_model": WHISPER_API_MODEL,
    "translation_backend": TRANSLATION_BACKEND,
    "translation_url": OLLAMA_BASE_URL,
    "translation_model": OLLAMA_TRANSLATION_MODEL,
}


def get_settings(user_data: dict) -> dict:
    return {**_DEFAULTS, **user_data.get("settings", {})}


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings(context.user_data)
    await update.message.reply_text(
        "Current settings:\n"
        "\nTranscription:\n"
        f"  Backend:   {s['whisper_backend']}\n"
        f"  API URL:   {s['whisper_api_url']}\n"
        f"  API Model: {s['whisper_api_model']}\n"
        "\nTranslation:\n"
        f"  Backend:   {s['translation_backend']}\n"
        f"  URL:       {s['translation_url']}\n"
        f"  Model:     {s['translation_model']}\n"
        "\nCommands:\n"
        "  /set_whisper local|api\n"
        "  /set_whisper_url <url>\n"
        "  /set_whisper_model <model>\n"
        "  /set_translation openrouter|ollama\n"
        "  /set_translation_url <url>\n"
        "  /set_translation_model <model>"
    )


async def cmd_set_whisper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if not args or args[0] not in ("local", "api"):
        await update.message.reply_text("Usage: /set_whisper local  or  /set_whisper api")
        return
    context.user_data.setdefault("settings", {})["whisper_backend"] = args[0]
    await update.message.reply_text(f"Whisper backend set to: {args[0]}")


async def cmd_set_whisper_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /set_whisper_url <url>")
        return
    url = context.args[0].rstrip("/")
    context.user_data.setdefault("settings", {})["whisper_api_url"] = url
    await update.message.reply_text(f"Whisper API URL set to: {url}")


async def cmd_set_whisper_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /set_whisper_model <model>")
        return
    context.user_data.setdefault("settings", {})["whisper_api_model"] = context.args[0]
    await update.message.reply_text(f"Whisper API model set to: {context.args[0]}")


async def cmd_set_translation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if not args or args[0] not in ("openrouter", "ollama"):
        await update.message.reply_text(
            "Usage: /set_translation openrouter  or  /set_translation ollama"
        )
        return
    context.user_data.setdefault("settings", {})["translation_backend"] = args[0]
    await update.message.reply_text(f"Translation backend set to: {args[0]}")


async def cmd_set_translation_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /set_translation_url <url>")
        return
    url = context.args[0].rstrip("/")
    context.user_data.setdefault("settings", {})["translation_url"] = url
    await update.message.reply_text(f"Translation URL set to: {url}")


async def cmd_set_translation_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /set_translation_model <model>")
        return
    context.user_data.setdefault("settings", {})["translation_model"] = context.args[0]
    await update.message.reply_text(f"Translation model set to: {context.args[0]}")
