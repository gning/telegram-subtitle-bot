import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
# Optional: point to a local Bot API server (removes the 20 MB download limit).
# Leave unset to use the default hosted Telegram Bot API.
LOCAL_BOT_API_URL: str | None = os.getenv("LOCAL_BOT_API_URL")  # e.g. http://localhost:8081

OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

MAX_VIDEO_DURATION_SECONDS: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "600"))
