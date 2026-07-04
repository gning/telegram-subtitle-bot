# telegram-subtitle-bot

A Telegram bot that transcribes videos, translates the transcript, and returns the video with burned-in bilingual subtitles.

## How it works

1. Send the bot a video (or video file).
2. The bot extracts the audio and transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), an OpenAI-compatible Whisper API, or MLX on Apple Silicon.
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
- Apple Silicon Mac for the MLX backend

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
| `WHISPER_BACKEND` | No | `local` | Transcription backend: `local`, `api`, or `mlx` |
| `WHISPER_MODEL_SIZE` | No | `large-v3` | faster-whisper model size |
| `WHISPER_DEVICE` | No | `auto` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | No | `float16` | Whisper compute type (e.g. `int8` for CPU) |
| `WHISPER_BEAM_SIZE` | No | `5` | faster-whisper beam size; `1` (greedy) is ~2x faster with slightly lower accuracy |
| `WHISPER_VAD_FILTER` | No | `1` | Skip silent audio before transcribing; set `0` to disable |
| `WHISPER_API_URL` | No | `http://localhost:11434` | OpenAI-compatible audio transcription API base URL |
| `WHISPER_API_MODEL` | No | `karanchopda333/whisper` | Model name sent to the API backend |
| `MLX_ASR_MODEL` | No | `mlx-community/Qwen3-ASR-1.7B-8bit` | MLX ASR model used when `WHISPER_BACKEND=mlx` |
| `MLX_ASR_MODEL_DIR` | No | `models/Qwen3-ASR-1.7B-8bit` | Pre-downloaded local MLX model directory, used before downloading from Hugging Face |
| `MLX_ASR_CHUNK_DURATION_SECONDS` | No | `30` | Audio chunk size for MLX ASR |
| `MLX_ASR_MAX_TOKENS` | No | `4096` | Maximum generated transcription tokens for MLX ASR |
| `MLX_ASR_PREFILL_STEP_SIZE` | No | `512` | MLX prompt prefill step size |
| `TRANSLATION_CONCURRENCY` | No | `4` | Translation batches sent to the API in parallel; `1` restores sequential behaviour |
| `MAX_VIDEO_DURATION_SECONDS` | No | `0` | Videos longer than this are rejected; `0` means unlimited |
| `FFMPEG_ENCODE_PRESET` | No | `veryfast` | x264 preset for the subtitle burn-in re-encode (`medium` for smaller files, `ultrafast` for speed) |
| `FFMPEG_BIN` | No | system `ffmpeg` | Path to a custom ffmpeg binary |
| `FFPROBE_BIN` | No | system `ffprobe` | Path to a custom ffprobe binary |

## Performance

The processing pipeline includes several optimizations:

- **Parallel translation** — subtitle batches are sent to the translation API concurrently (4 at a time by default, tunable via `TRANSLATION_CONCURRENCY`) instead of one after another, cutting translation time roughly 3–4x on long videos. Segment order is always preserved.
- **Connection reuse** — translation requests share a single HTTP client, avoiding a TLS handshake per batch.
- **Fast subtitle burn-in** — the re-encode uses the x264 `veryfast` preset (~3x faster than the default `medium` at near-identical visual quality; tune via `FFMPEG_ENCODE_PRESET`) and writes `+faststart` output so videos stream immediately in Telegram.
- **Voice-activity detection** — faster-whisper skips silent stretches instead of decoding them (disable with `WHISPER_VAD_FILTER=0`). Set `WHISPER_BEAM_SIZE=1` for ~2x faster transcription at slightly lower accuracy.

## Notes

- Telegram's default Bot API limits uploads/downloads to **20 MB**. Run a [local Bot API server](https://github.com/tdlib/telegram-bot-api) and set `LOCAL_BOT_API_URL` to handle larger files.
- The faster-whisper model is downloaded on first run (~3 GB for `large-v3`). Smaller models (`medium`, `small`) are faster but less accurate.
- The MLX model is downloaded on first use and requires `mlx-audio` on Apple Silicon. Use `/set_whisper mlx` and `/set_mlx_model mlx-community/Qwen3-ASR-1.7B-8bit` to select it per user.
- Transcription runs in a thread pool to avoid blocking the async event loop.
