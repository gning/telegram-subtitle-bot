import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from bot.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once.
_model = None


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
    segments_iter, info = model.transcribe(audio_path, beam_size=5)
    logger.info("Detected language: %s (probability %.2f)", info.language, info.language_probability)

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": seg.start, "end": seg.end, "text": text})

    logger.info("Transcribed %d segments.", len(segments))
    return segments, info.language


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
        "Whisper API returned no timestamps — distributing evenly across audio duration."
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


async def _transcribe_api(
    audio_path: str,
    api_url: str,
    api_model: str,
) -> tuple[list[dict], str]:
    endpoint = f"{api_url.rstrip('/')}/v1/audio/transcriptions"
    try:
        async with httpx.AsyncClient(timeout=300.0, http2=False) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    endpoint,
                    files={"file": (Path(audio_path).name, f, "audio/wav")},
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
    except RuntimeError:
        raise
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Whisper API timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Whisper API connection error ({type(exc).__name__}): {exc}"
        ) from exc

    data = response.json()
    language = data.get("language", "und")
    segments = _parse_api_segments(data.get("segments", []), audio_path)
    logger.info("API transcribed %d segments; detected language: %s", len(segments), language)
    return segments, language


async def transcribe(
    audio_path: str,
    settings: dict | None = None,
) -> tuple[list[dict], str]:
    """
    Async transcription.  Returns (segments, detected_language).

    If settings["whisper_backend"] == "api", calls the configured OpenAI-compatible
    Whisper API endpoint.  Otherwise runs faster-whisper in a thread pool.
    """
    effective = settings or {}
    if effective.get("whisper_backend") == "api":
        return await _transcribe_api(
            audio_path,
            effective.get("whisper_api_url", ""),
            effective.get("whisper_api_model", ""),
        )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path)
