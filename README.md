# telegram-subtitle-bot

A Telegram bot that transcribes videos, translates the transcript, and returns the video with burned-in bilingual subtitles.

## How it works

1. Send the bot a video (or video file).
2. The bot extracts the audio and transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. The transcript is translated via [OpenRouter](https://openrouter.ai/) into a second language:
   - Chinese audio → English subtitles added
   - English audio → Chinese subtitles added
   - Any other language → both Chinese and English subtitles added
4. An ASS subtitle file is generated and burned into the video with FFmpeg.
5. The finished video is sent back.

## Requirements

- Python 3.10+
- FFmpeg (with libass support) available on `PATH`, or set `FFMPEG_BIN` / `FFPROBE_BIN`
- A [Telegram bot token](https://core.telegram.org/bots#botfather)
- An [OpenRouter API key](https://openrouter.ai/)

## Setup

```bash
git clone https://github.com/gning/telegram-subtitle-bot.git
cd telegram-subtitle-bot
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in your tokens
python -m bot.main
```

## Configuration

All configuration is done via environment variables (`.env` file).

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Bot token from @BotFather |
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key |
| `LOCAL_BOT_API_URL` | No | — | URL of a [local Bot API server](https://github.com/tdlib/telegram-bot-api) to lift the 20 MB download cap |
| `OPENROUTER_MODEL` | No | `google/gemini-2.0-flash-001` | Model used for translation |
| `WHISPER_MODEL_SIZE` | No | `large-v3` | faster-whisper model size |
| `WHISPER_DEVICE` | No | `auto` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | No | `float16` | Whisper compute type (e.g. `int8` for CPU) |
| `MAX_VIDEO_DURATION_SECONDS` | No | `600` | Videos longer than this are rejected |
| `FFMPEG_BIN` | No | system `ffmpeg` | Path to a custom ffmpeg binary |
| `FFPROBE_BIN` | No | system `ffprobe` | Path to a custom ffprobe binary |

## Notes

- Telegram's default Bot API limits uploads/downloads to **20 MB**. Run a [local Bot API server](https://github.com/tdlib/telegram-bot-api) and set `LOCAL_BOT_API_URL` to handle larger files.
- The Whisper model is downloaded on first run (~3 GB for `large-v3`). Smaller models (`medium`, `small`) are faster but less accurate.
- Whisper runs in a thread pool to avoid blocking the async event loop.
