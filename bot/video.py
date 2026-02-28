import asyncio
import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _find_executable(name: str) -> str:
    """Return the path for *name*, checking FFMPEG_BIN / FFPROBE_BIN env vars first."""
    override = os.environ.get(f"{name.upper()}_BIN")
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(
        f"'{name}' not found on PATH. Install FFmpeg or set {name.upper()}_BIN."
    )


_FFMPEG = _find_executable("ffmpeg")
_FFPROBE = _find_executable("ffprobe")


def _run_ffmpeg(args: list[str]) -> None:
    """Run an FFmpeg command, raising RuntimeError on failure."""
    cmd = [_FFMPEG, "-y"] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")


def _get_duration_sync(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        _FFPROBE,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def _extract_audio_sync(video_path: str, audio_path: str) -> None:
    """Extract mono 16 kHz WAV audio from a video file."""
    _run_ffmpeg([
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ])


def _burn_subtitles_sync(video_path: str, ass_path: str, output_path: str) -> None:
    """Burn an ASS subtitle file into a video, copying the audio stream."""
    # FFmpeg filtergraph escaping: backslashes first, then colons.
    # Use `filename=` explicitly — required by FFmpeg 8+ (positional form removed).
    escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:")
    _run_ffmpeg([
        "-i", video_path,
        "-vf", f"ass=filename={escaped}",
        "-c:a", "copy",
        output_path,
    ])


async def get_duration(video_path: str) -> float:
    """Async wrapper around ffprobe duration check."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_duration_sync, video_path)


async def extract_audio(video_path: str, audio_path: str) -> None:
    """Async wrapper around FFmpeg audio extraction."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _extract_audio_sync, video_path, audio_path)


async def burn_subtitles(video_path: str, ass_path: str, output_path: str) -> None:
    """Async wrapper around FFmpeg subtitle burning."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _burn_subtitles_sync, video_path, ass_path, output_path)
