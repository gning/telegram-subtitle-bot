import asyncio
import inspect
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

from bot.config import (
    MLX_ASR_CHUNK_DURATION_SECONDS,
    MLX_ASR_MAX_TOKENS,
    MLX_ASR_MODEL,
    MLX_ASR_MODEL_DIR,
    MLX_ASR_PREFILL_STEP_SIZE,
    WHISPER_API_KEY,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
)

# The Qwen3 MLX weights are large. On this machine the Hugging Face Xet
# transfer path has been observed to stall after creating an incomplete blob.
# Force the regular HTTP downloader before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once.
_model = None
_mlx_model = None
_mlx_model_name = None

_REQUIRED_MLX_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
            WHISPER_MODEL_SIZE,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded.")
    return _model


def _transcribe_sync(audio_path: str) -> tuple[list[dict], str]:
    """
    Transcribe audio and return (segments, detected_language).

    Each segment is a dict with keys: start, end, text.
    """
    model = _get_model()
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
    )
    logger.info("Detected language: %s (probability %.2f)", info.language, info.language_probability)

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})

    logger.info("Transcribed %d segments.", len(segments))
    return segments, info.language


def _normalise_language_code(language: str | None) -> str:
    if isinstance(language, (list, tuple)):
        language = next((item for item in language if item), None)
    if not language:
        return "und"
    language_lower = str(language).strip().lower()
    language_map = {
        "english": "en",
        "en-us": "en",
        "en-gb": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "simplified chinese": "zh",
        "traditional chinese": "zh",
        "cantonese": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
    }
    return language_map.get(language_lower, language_lower or "und")


def _get_mlx_model(model_name: str):
    global _mlx_model, _mlx_model_name
    if _mlx_model is None or _mlx_model_name != model_name:
        try:
            from huggingface_hub import snapshot_download
            from mlx_audio.stt.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "MLX transcription requires mlx-audio and huggingface_hub. "
                "Install them with: pip install -U mlx-audio huggingface_hub"
            ) from exc

        model_path = _resolve_mlx_model_path(model_name, snapshot_download)

        logger.info("Loading MLX ASR model from %s...", model_path)
        t0 = time.monotonic()
        _mlx_model = load_model(model_path)
        _mlx_model_name = model_name
        logger.info("MLX ASR model loaded in %.1fs.", time.monotonic() - t0)
    return _mlx_model


def _has_complete_mlx_model_dir(model_dir: Path) -> bool:
    return all((model_dir / filename).is_file() for filename in _REQUIRED_MLX_MODEL_FILES)


def _resolve_mlx_model_path(model_name: str, snapshot_download) -> str:
    configured_path = Path(model_name).expanduser()
    if configured_path.exists():
        if not _has_complete_mlx_model_dir(configured_path):
            missing = [
                filename
                for filename in _REQUIRED_MLX_MODEL_FILES
                if not (configured_path / filename).is_file()
            ]
            raise RuntimeError(
                f"MLX ASR model directory is incomplete: {configured_path}. "
                f"Missing: {', '.join(missing)}"
            )
        return str(configured_path)

    local_model_dir = Path(MLX_ASR_MODEL_DIR).expanduser()
    if _has_complete_mlx_model_dir(local_model_dir):
        logger.info("Using pre-downloaded MLX ASR model directory: %s", local_model_dir)
        return str(local_model_dir)

    logger.info("Downloading/checking MLX ASR model snapshot '%s'...", model_name)
    t0 = time.monotonic()
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            max_workers=1,
            local_files_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MLX ASR model '{model_name}'. "
            f"If the Hugging Face Python downloader stalls, pre-download it to "
            f"{local_model_dir} and set MLX_ASR_MODEL_DIR. Original error: {exc}"
        ) from exc
    logger.info(
        "MLX ASR model snapshot ready in %.1fs: %s",
        time.monotonic() - t0,
        model_path,
    )
    return model_path


def _get_audio_duration(audio_path: str) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _distribute_evenly(segments: list[dict], audio_path: str) -> list[dict]:
    """
    Assign start/end timestamps proportional to each segment's character count.
    Falls back to 1 second per segment if the total duration cannot be determined.
    """
    logger.warning(
        "Transcriber returned no timestamps; distributing evenly across audio duration."
    )
    total_chars = sum(len(s["text"]) for s in segments) or 1
    total_duration = _get_audio_duration(audio_path)
    if total_duration is None or total_duration <= 0:
        total_duration = float(len(segments))  # 1 second per segment fallback

    current = 0.0
    result = []
    for seg in segments:
        proportion = len(seg["text"]) / total_chars
        duration = proportion * total_duration
        result.append({"start": current, "end": current + duration, "text": seg["text"]})
        current += duration
    return result


def _parse_api_segments(raw_segments: list[dict], audio_path: str) -> list[dict]:
    """Normalise API response segments to {start, end, text} dicts."""
    segments = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        })

    # If all end values are 0.0, the API returned no timestamps — distribute evenly.
    if segments and all(s["end"] == 0.0 for s in segments):
        segments = _distribute_evenly(segments, audio_path)

    return segments


def _coerce_segment(segment) -> dict | None:
    if isinstance(segment, dict):
        text = (segment.get("text") or "").strip()
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
    else:
        text = (getattr(segment, "text", "") or "").strip()
        start = getattr(segment, "start", 0.0)
        end = getattr(segment, "end", 0.0)

    if not text:
        return None
    return {"start": float(start or 0.0), "end": float(end or 0.0), "text": text}


def _parse_mlx_result(transcription, audio_path: str) -> tuple[list[dict], str]:
    raw_segments = getattr(transcription, "segments", None)
    if raw_segments is None and isinstance(transcription, dict):
        raw_segments = transcription.get("segments")

    segments = []
    if raw_segments:
        for raw_segment in raw_segments:
            segment = _coerce_segment(raw_segment)
            if segment:
                segments.append(segment)

    if not segments:
        text = ""
        if isinstance(transcription, str):
            text = transcription
        elif isinstance(transcription, dict):
            text = transcription.get("text", "")
        else:
            text = getattr(transcription, "text", "")
        text = (text or "").strip()
        if text:
            segments = [{"start": 0.0, "end": 0.0, "text": text}]

    if segments and all(s["end"] == 0.0 for s in segments):
        segments = _distribute_evenly(segments, audio_path)

    language = None
    if isinstance(transcription, dict):
        language = transcription.get("language")
    else:
        language = getattr(transcription, "language", None)

    return segments, _normalise_language_code(language)


def _generate_mlx_transcription(model, audio_path: str, output_path: str):
    from mlx_audio.stt.generate import generate_transcription

    duration = _get_audio_duration(audio_path) or 0.0
    # Keep generation bounded. Qwen3-ASR defaults to 8192 tokens, which can make
    # a bad decode look like a hang. This budget is intentionally generous for
    # normal speech while still finite.
    max_tokens = min(
        MLX_ASR_MAX_TOKENS,
        max(128, int(duration * 12) + 128),
    )
    logger.info(
        "Starting MLX transcription (duration=%.1fs, chunk=%.1fs, max_tokens=%d, prefill_step=%d)...",
        duration,
        MLX_ASR_CHUNK_DURATION_SECONDS,
        max_tokens,
        MLX_ASR_PREFILL_STEP_SIZE,
    )

    kwargs = {
        "model": model,
        "output_path": output_path,
        "format": "txt",
        "verbose": False,
        "chunk_duration": MLX_ASR_CHUNK_DURATION_SECONDS,
        "max_tokens": max_tokens,
        "prefill_step_size": MLX_ASR_PREFILL_STEP_SIZE,
    }
    parameters = inspect.signature(generate_transcription).parameters
    if "audio_path" in parameters:
        kwargs["audio_path"] = audio_path
        return generate_transcription(**kwargs)
    if "audio" in parameters:
        kwargs["audio"] = audio_path
        return generate_transcription(**kwargs)

    try:
        kwargs["audio_path"] = audio_path
        return generate_transcription(**kwargs)
    except TypeError:
        kwargs.pop("audio_path", None)
        kwargs["audio"] = audio_path
        return generate_transcription(**kwargs)


def _transcribe_mlx_sync(audio_path: str, model_name: str) -> tuple[list[dict], str]:
    model = _get_mlx_model(model_name)
    output_path = str(Path(audio_path).with_name("mlx_transcript"))
    transcription = _generate_mlx_transcription(model, audio_path, output_path)
    segments, language = _parse_mlx_result(transcription, audio_path)
    logger.info(
        "MLX transcribed %d segments; detected language: %s",
        len(segments),
        language,
    )
    return segments, language


# Hosted Whisper APIs cap the request size (Groq: 25 MB free tier, 100 MB dev
# tier). Whisper only uses 16 kHz mono audio, so upload low-bitrate Opus
# instead of raw WAV and split anything that still exceeds the cap.
_API_UPLOAD_BITRATE_BPS = 32_000
_API_MAX_UPLOAD_BYTES = int(float(os.getenv("WHISPER_API_MAX_UPLOAD_MB", "24")) * 1024 * 1024)


def _prepare_api_upload(audio_path: str) -> list[tuple[str, float]]:
    """Compress *audio_path* to Opus chunks that fit under the API size cap.

    Returns [(chunk_path, start_offset_seconds), ...] in playback order.
    """
    from bot.video import _get_duration_sync, _run_ffmpeg

    opus_args = [
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", str(_API_UPLOAD_BITRATE_BPS),
    ]
    base = os.path.splitext(audio_path)[0]
    duration = _get_duration_sync(audio_path)
    # 0.9 leaves headroom for container overhead above the nominal bitrate.
    max_chunk_seconds = _API_MAX_UPLOAD_BYTES * 8 * 0.9 / _API_UPLOAD_BITRATE_BPS
    if duration <= max_chunk_seconds:
        out = f"{base}.api.ogg"
        _run_ffmpeg(["-i", audio_path, *opus_args, out])
        return [(out, 0.0)]

    count = math.ceil(duration / max_chunk_seconds)
    chunk_seconds = duration / count
    chunks = []
    for i in range(count):
        out = f"{base}.api.{i:03d}.ogg"
        start = i * chunk_seconds
        _run_ffmpeg([
            "-ss", f"{start:.3f}", "-t", f"{chunk_seconds:.3f}",
            "-i", audio_path, *opus_args, out,
        ])
        chunks.append((out, start))
    return chunks


async def _transcribe_api(
    audio_path: str,
    api_url: str,
    api_model: str,
) -> tuple[list[dict], str]:
    endpoint = f"{api_url.rstrip('/')}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {WHISPER_API_KEY}"} if WHISPER_API_KEY else {}

    loop = asyncio.get_event_loop()
    uploads = await loop.run_in_executor(None, _prepare_api_upload, audio_path)
    if len(uploads) > 1:
        logger.info("Audio exceeds API upload cap; split into %d chunks", len(uploads))

    segments: list[dict] = []
    language = "und"
    try:
        async with httpx.AsyncClient(timeout=300.0, http2=False) as client:
            for chunk_path, offset in uploads:
                with open(chunk_path, "rb") as f:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        files={"file": (Path(chunk_path).name, f, "audio/ogg")},
                        data={
                            "model": api_model,
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "segment",
                        },
                    )
                # raise_for_status inside the client context so the response
                # is fully buffered before the connection closes.
                if response.is_error:
                    raise RuntimeError(
                        f"Whisper API returned HTTP {response.status_code}: {response.text[:200]}"
                    )

                data = response.json()
                logger.debug("Whisper API raw response keys: %s", list(data.keys()))
                if language == "und":
                    language = data.get("language") or "und"
                raw_segments = data.get("segments", [])

                # Some Whisper API backends (e.g. Ollama) return only a
                # top-level "text" field with no "segments" array.  Fall back
                # to a single segment so the rest of the pipeline receives
                # something to work with.
                if not raw_segments:
                    full_text = (data.get("text") or "").strip()
                    logger.warning(
                        "Whisper API returned no segments (keys=%s). full_text present: %s",
                        list(data.keys()),
                        bool(full_text),
                    )
                    if full_text:
                        raw_segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

                for seg in _parse_api_segments(raw_segments, chunk_path):
                    seg["start"] += offset
                    seg["end"] += offset
                    segments.append(seg)
    except RuntimeError:
        raise
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Whisper API timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Whisper API connection error ({type(exc).__name__}): {exc}"
        ) from exc

    logger.info("API transcribed %d segments; detected language: %s", len(segments), language)
    return segments, _normalise_language_code(language)


async def transcribe(
    audio_path: str,
    settings: dict | None = None,
) -> tuple[list[dict], str]:
    """
    Async transcription.  Returns (segments, detected_language).

    If settings["whisper_backend"] == "api", calls the configured OpenAI-compatible
    Whisper API endpoint. If it is "mlx", runs mlx-audio with the configured MLX
    ASR model. Otherwise runs faster-whisper in a thread pool.
    """
    effective = settings or {}
    if effective.get("whisper_backend") == "api":
        return await _transcribe_api(
            audio_path,
            effective.get("whisper_api_url", ""),
            effective.get("whisper_api_model", ""),
        )
    if effective.get("whisper_backend") == "mlx":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _transcribe_mlx_sync,
            audio_path,
            effective.get("mlx_asr_model", MLX_ASR_MODEL),
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path)
