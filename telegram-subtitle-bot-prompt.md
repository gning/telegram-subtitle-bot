Build a Telegram Bot that processes user-uploaded videos by extracting audio, transcribing it, generating bilingual (Chinese + English) subtitles, burning them into the video, and sending the result back.

## Tech Stack

- **Language**: Python 3.10+
- **Telegram Bot framework**: python-telegram-bot (async)
- **Speech-to-text**: faster-whisper with the `large-v3` model
- **Translation**: Any LLM via OpenRouter API (model name configurable)
- **Video/audio processing**: FFmpeg (called via subprocess)
- **Subtitle format**: ASS (for dual-language styling control)
- **Config management**: python-dotenv, all secrets and settings in a `.env` file

## Configuration (.env)

```
TELEGRAM_BOT_TOKEN=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemini-2.0-flash-001
WHISPER_MODEL_SIZE=large-v3
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=float16
MAX_VIDEO_DURATION_SECONDS=600
```

## Core Workflow

1. User sends a video (or video file) to the bot.
2. Bot replies with a "Processing..." status message.
3. Download the video to a temp directory.
4. Extract audio from the video using FFmpeg (`ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 audio.wav`).
5. Run faster-whisper on the audio to get transcribed segments with timestamps (start, end, text).
6. Detect the language of the transcription. If the source language is Chinese, translate each segment to English. If it's English, translate to Chinese. For other languages, translate to both Chinese and English.
7. Translation: Send each segment (or batched segments) to the OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`) with a system prompt instructing it to translate the text while preserving meaning and brevity suitable for subtitles. Use the model specified in `OPENROUTER_MODEL`.
8. Generate an ASS subtitle file with two lines per segment: the original language on top and the translated language on the bottom. Use appropriate fonts — a sans-serif CJK font (e.g., Noto Sans CJK SC) for Chinese and a standard sans-serif font for English. Set font size, outline, shadow, and positioning so subtitles are readable.
9. Burn the ASS subtitles into the video using FFmpeg (`ffmpeg -i input.mp4 -vf "ass=subs.ass" -c:a copy output.mp4`).
10. Send the output video back to the user. If the file exceeds Telegram's 50MB limit, notify the user.
11. Update the status message to "Done" or report any errors.
12. Clean up all temp files.

## Project Structure

```
telegram-subtitle-bot/
├── bot/
│   ├── __init__.py
│   ├── main.py              # Entry point, bot setup and handlers
│   ├── config.py            # Load .env and expose config constants
│   ├── handlers.py          # Telegram message handlers
│   ├── transcriber.py       # faster-whisper transcription logic
│   ├── translator.py        # OpenRouter translation logic
│   ├── subtitle.py          # ASS subtitle generation
│   └── video.py             # FFmpeg operations (extract audio, burn subs)
├── .env.example
├── requirements.txt
└── README.md
```

## Important Implementation Details

- Use `asyncio` properly — run faster-whisper and FFmpeg in a thread pool (`run_in_executor`) to avoid blocking the event loop.
- For translation, batch segments (e.g., 10 at a time) into a single API call to reduce latency and cost. The prompt should instruct the LLM to return translations in a structured JSON array matching the input order.
- Handle edge cases: empty transcription results, videos with no audio track, translation API failures (retry up to 3 times with exponential backoff).
- Add logging throughout (use Python's `logging` module) so issues can be debugged.
- The bot should handle multiple users concurrently without blocking.
- Add a `/start` command that explains what the bot does and how to use it.
- Validate video duration before processing — reject videos longer than `MAX_VIDEO_DURATION_SECONDS` to prevent resource abuse.
