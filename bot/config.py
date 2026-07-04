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
WHISPER_BACKEND: str = os.getenv("WHISPER_BACKEND", "local")   # "local" | "api" | "mlx"
WHISPER_API_URL: str = os.getenv("WHISPER_API_URL", "http://localhost:11434")
WHISPER_API_MODEL: str = os.getenv("WHISPER_API_MODEL", "karanchopda333/whisper")
MLX_ASR_MODEL: str = os.getenv("MLX_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-8bit")
MLX_ASR_MODEL_DIR: str = os.getenv("MLX_ASR_MODEL_DIR", "models/Qwen3-ASR-1.7B-8bit")
MLX_ASR_CHUNK_DURATION_SECONDS: float = float(os.getenv("MLX_ASR_CHUNK_DURATION_SECONDS", "30"))
MLX_ASR_MAX_TOKENS: int = int(os.getenv("MLX_ASR_MAX_TOKENS", "4096"))
MLX_ASR_PREFILL_STEP_SIZE: int = int(os.getenv("MLX_ASR_PREFILL_STEP_SIZE", "512"))

TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "openrouter")  # "openrouter" | "ollama"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TRANSLATION_MODEL: str = os.getenv("OLLAMA_TRANSLATION_MODEL", "gemma4:31b")

MAX_VIDEO_DURATION_SECONDS: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "0"))  # 0 = unlimited
