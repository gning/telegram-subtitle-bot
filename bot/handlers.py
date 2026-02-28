"""
Telegram message handlers.

Workflow for each received video:
  1. Validate duration (from message metadata, then ffprobe post-download).
  2. Download to a temp directory.
  3. Extract audio with FFmpeg.
  4. Transcribe with faster-whisper.
  5. Detect language; translate accordingly.
  6. Generate ASS subtitle file.
  7. Burn subtitles into video with FFmpeg.
  8. Send result (or notify if >50 MB).
  9. Clean up temp files.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import httpx
from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot import config
from bot import subtitle, transcriber, translator, video
from bot.config import LOCAL_BOT_API_URL, TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_SEND_BYTES_DEFAULT = 50 * 1024 * 1024   # 50 MB  (hosted Bot API)
_TELEGRAM_MAX_SEND_BYTES_LOCAL   = 2 * 1024 * 1024 * 1024  # 2 GB  (local Bot API server)


# ---------------------------------------------------------------------------
# /start handler
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hello! I create bilingual (Chinese + English) subtitles for your videos.\n\n"
        "Just send me a video and I will:\n"
        "  1. Transcribe the speech\n"
        "  2. Translate it to the other language\n"
        "  3. Burn bilingual subtitles into the video\n"
        "  4. Send the result back to you\n\n"
        f"Maximum video length: {config.MAX_VIDEO_DURATION_SECONDS // 60} minutes.\n\n"
        "Send a video as a file (not compressed) for best quality.",
    )


# ---------------------------------------------------------------------------
# Video handler
# ---------------------------------------------------------------------------

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message: Message = update.message

    # Determine which kind of media was sent
    tg_file = None
    duration_hint: int | None = None  # seconds, from Telegram metadata
    original_filename: str | None = None

    try:
        if message.video:
            tg_file = await message.video.get_file()
            duration_hint = message.video.duration
            original_filename = message.video.file_name
        elif message.document:
            mime = message.document.mime_type or ""
            if not mime.startswith("video/"):
                return  # not a video document, ignore
            tg_file = await message.document.get_file()
            original_filename = message.document.file_name
            # Documents don't carry duration metadata; check after download
        else:
            return  # shouldn't happen given the filter in main.py
    except BadRequest as exc:
        if "too big" in str(exc).lower():
            if LOCAL_BOT_API_URL:
                tip = "The file exceeds the 2 GB limit even for a local Bot API server."
            else:
                tip = (
                    "The hosted Telegram Bot API only allows downloading files up to 20 MB. "
                    "Run a local Bot API server (set LOCAL_BOT_API_URL in .env) to lift this to 2 GB."
                )
            await message.reply_text(f"File is too large to download. {tip}")
        else:
            await message.reply_text(f"Could not fetch the video: {exc}")
        return

    # Quick pre-flight duration check from Telegram metadata
    if duration_hint is not None and duration_hint > config.MAX_VIDEO_DURATION_SECONDS:
        await message.reply_text(
            f"Video is too long ({duration_hint}s). "
            f"Maximum allowed duration is {config.MAX_VIDEO_DURATION_SECONDS}s "
            f"({config.MAX_VIDEO_DURATION_SECONDS // 60} min)."
        )
        return

    # Send initial status
    status_msg = await message.reply_text("Downloading video...")
    t_total_start = time.monotonic()
    timings: list[tuple[str, float]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = str(tmp / "input.mp4")
        audio_path = str(tmp / "audio.wav")
        ass_path   = str(tmp / "subs.ass")
        output_path = str(tmp / "output.mp4")

        try:
            # 1. Download
            t0 = time.monotonic()
            await context.bot.send_chat_action(message.chat_id, ChatAction.UPLOAD_VIDEO)
            await _download_file(tg_file, input_path)
            timings.append(("Download", time.monotonic() - t0))
            logger.info("Downloaded video to %s", input_path)

            # 2. Duration check via ffprobe (definitive)
            await _edit(status_msg, "Checking video...")
            try:
                duration = await video.get_duration(input_path)
            except RuntimeError as exc:
                logger.error("ffprobe failed: %s", exc)
                await _edit(status_msg, "Could not read video metadata. Is this a valid video file?")
                return

            if duration > config.MAX_VIDEO_DURATION_SECONDS:
                await _edit(
                    status_msg,
                    f"Video is too long ({int(duration)}s). "
                    f"Maximum allowed: {config.MAX_VIDEO_DURATION_SECONDS}s.",
                )
                return

            # 3. Extract audio
            await _edit(status_msg, "Extracting audio...")
            t0 = time.monotonic()
            try:
                await video.extract_audio(input_path, audio_path)
            except RuntimeError as exc:
                logger.error("Audio extraction failed: %s", exc)
                await _edit(
                    status_msg,
                    "Failed to extract audio. The video may have no audio track.",
                )
                return
            timings.append(("Audio extraction", time.monotonic() - t0))

            # 4. Transcribe
            await _edit(status_msg, "Transcribing speech... (this may take a while)")
            t0 = time.monotonic()
            try:
                segments, source_lang = await transcriber.transcribe(audio_path)
            except Exception as exc:
                logger.exception("Transcription failed: %s", exc)
                await _edit(status_msg, f"Transcription failed: {exc}")
                return
            timings.append(("Transcription", time.monotonic() - t0))

            if not segments:
                await _edit(
                    status_msg,
                    "No speech was detected in the video. "
                    "The audio may be silent or the language may not be recognised.",
                )
                return

            logger.info(
                "Transcribed %d segments; detected language: %s", len(segments), source_lang
            )

            # 5. Translate
            norm_lang = source_lang.lower()
            is_chinese = norm_lang in ("zh", "zh-cn", "zh-tw")
            is_english = norm_lang == "en"

            await _edit(status_msg, "Translating subtitles...")
            t0 = time.monotonic()
            try:
                if is_chinese:
                    translations = await translator.translate_segments(segments, "English")
                elif is_english:
                    translations = await translator.translate_segments(segments, "Simplified Chinese")
                else:
                    translations = await translator.translate_segments_dual(segments)
            except Exception as exc:
                logger.exception("Translation failed: %s", exc)
                await _edit(status_msg, f"Translation failed: {exc}")
                return
            timings.append(("Translation", time.monotonic() - t0))

            # 6. Generate subtitles
            await _edit(status_msg, "Generating subtitle file...")
            t0 = time.monotonic()
            subtitle.generate_ass(segments, source_lang, translations, ass_path)
            timings.append(("Subtitle generation", time.monotonic() - t0))

            # 7. Burn subtitles
            await _edit(status_msg, "Burning subtitles into video...")
            t0 = time.monotonic()
            try:
                await video.burn_subtitles(input_path, ass_path, output_path)
            except RuntimeError as exc:
                logger.error("Subtitle burning failed: %s", exc)
                await _edit(status_msg, f"Failed to burn subtitles: {exc}")
                return
            timings.append(("Subtitle burning", time.monotonic() - t0))

            # 8. Send result
            max_bytes = (
                _TELEGRAM_MAX_SEND_BYTES_LOCAL
                if LOCAL_BOT_API_URL
                else _TELEGRAM_MAX_SEND_BYTES_DEFAULT
            )
            limit_label = "2 GB" if LOCAL_BOT_API_URL else "50 MB"
            output_size = os.path.getsize(output_path)
            if output_size > max_bytes:
                await _edit(
                    status_msg,
                    f"Done! But the output file is {output_size // (1024*1024)} MB, "
                    f"which exceeds Telegram's {limit_label} limit. "
                    "Please use a shorter or lower-quality video.",
                )
                return

            # Derive output filename from original: <name>_subtitled.mp4
            if original_filename:
                stem = Path(original_filename).stem
                output_filename = f"{stem}_subtitled.mp4"
            else:
                output_filename = "subtitled.mp4"

            await _edit(status_msg, "Uploading result...")
            await context.bot.send_chat_action(message.chat_id, ChatAction.UPLOAD_VIDEO)
            t0 = time.monotonic()
            with open(output_path, "rb") as fh:
                await message.reply_document(
                    document=fh,
                    filename=output_filename,
                    caption="Here is your video with bilingual subtitles!",
                    write_timeout=600,
                    read_timeout=600,
                )
            timings.append(("Upload", time.monotonic() - t0))
            await _edit(status_msg, "Done!")

            total = time.monotonic() - t_total_start
            summary_lines = ["⏱ Processing time summary:"]
            for step, secs in timings:
                summary_lines.append(f"  • {step}: {int(secs)}s")
            summary_lines.append(f"  ─────────────────")
            summary_lines.append(f"  Total: {int(total)}s")
            await message.reply_text("\n".join(summary_lines))

        except Exception as exc:
            logger.exception("Unexpected error while processing video: %s", exc)
            await _edit(status_msg, f"An unexpected error occurred: {exc}")
        # temp directory and all files are cleaned up automatically on exit


async def _edit(msg: Message, text: str) -> None:
    """Helper: edit a status message, ignoring 'message not modified' errors."""
    try:
        await msg.edit_text(text)
    except Exception:
        pass  # Not critical if status update fails


async def _download_file(tg_file, dest_path: str) -> None:
    """
    Download a Telegram file to dest_path.

    Handles three cases:
    1. Local Bot API server (--local): getFile returns an absolute path on this
       machine.  PTB with local_mode=True copies the file directly from disk.
    2. Remote Bot API server (--local): getFile returns an absolute path on the
       *remote* machine.  We reconstruct the correct HTTP URL and stream it.
    3. Hosted Telegram API: standard PTB download via HTTPS.
    """
    file_path = tg_file.file_path or ""
    logger.debug("Downloading file_path=%r", file_path)

    if LOCAL_BOT_API_URL and file_path.startswith("/"):
        if Path(file_path).exists():
            # Case 1: truly local server — PTB copies from disk (fast, no HTTP).
            await tg_file.download_to_drive(dest_path)
        else:
            # Case 2: remote server with --local.  Reconstruct the correct URL
            # by extracting the relative portion after /<token>/.
            token_marker = f"/{TELEGRAM_BOT_TOKEN}/"
            idx = file_path.find(token_marker)
            relative = file_path[idx + len(token_marker):] if idx != -1 else file_path.lstrip("/")
            url = f"{LOCAL_BOT_API_URL}/file/bot{TELEGRAM_BOT_TOKEN}/{relative}"
            logger.info("Remote local-mode server: downloading via %s", url)
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(dest_path, "wb") as fh:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            fh.write(chunk)
    else:
        # Case 3: standard PTB download (hosted API or relative path).
        await tg_file.download_to_drive(dest_path)
