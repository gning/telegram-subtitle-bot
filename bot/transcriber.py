import asyncio
import logging
from typing import Optional

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


async def transcribe(audio_path: str) -> tuple[list[dict], str]:
    """
    Async transcription.  Returns (segments, detected_language).
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sync, audio_path)
