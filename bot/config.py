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
WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
# Skip silent stretches before decoding — large speedup on videos with pauses.
WHISPER_VAD_FILTER: bool = os.getenv("WHISPER_VAD_FILTER", "1") not in ("0", "false", "no")
WHISPER_BACKEND: str = os.getenv("WHISPER_BACKEND", "local")   # "local" | "api" | "mlx"
WHISPER_API_URL: str = os.getenv("WHISPER_API_URL", "http://localhost:11434")
WHISPER_API_MODEL: str = os.getenv("WHISPER_API_MODEL", "karanchopda333/whisper")
MLX_ASR_MODEL: str = os.getenv("MLX_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-8bit")
MLX_ASR_MODEL_DIR: str = os.getenv("MLX_ASR_MODEL_DIR", "models/Qwen3-ASR-1.7B-8bit")
MLX_ASR_CHUNK_DURATION_SECONDS: float = float(os.getenv("MLX_ASR_CHUNK_DURATION_SECONDS", "30"))
MLX_ASR_MAX_TOKENS: int = int(os.getenv("MLX_ASR_MAX_TOKENS", "4096"))
MLX_ASR_PREFILL_STEP_SIZE: int = int(os.getenv("MLX_ASR_PREFILL_STEP_SIZE", "512"))

TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "openrouter")  # "openrouter" | "ollama"
# How many translation batches to send concurrently. 1 = sequential.
TRANSLATION_CONCURRENCY: int = max(1, int(os.getenv("TRANSLATION_CONCURRENCY", "4")))
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TRANSLATION_MODEL: str = os.getenv("OLLAMA_TRANSLATION_MODEL", "gemma4:31b")

MAX_VIDEO_DURATION_SECONDS: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "0"))  # 0 = unlimited

# x264 preset for the subtitle burn-in re-encode. "veryfast" is ~3x faster than
# the x264 default ("medium") at nearly identical visual quality for this use.
FFMPEG_ENCODE_PRESET: str = os.getenv("FFMPEG_ENCODE_PRESET", "veryfast")
