"""
Translation via OpenRouter API.

Supported modes:
  - translate_segments(segments, target_language) -> list[str]
      Translate each segment's text to the given language.

  - translate_segments_dual(segments) -> list[dict]
      Translate each segment's text to both Chinese and English.
      Returns list of {"zh": ..., "en": ...} dicts.
"""

import asyncio
import json
import logging

import httpx

from bot.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10
_MAX_RETRIES = 3
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Simplified Chinese",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}


def _language_display(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def translate_segments(segments: list[dict], target_language: str) -> list[str]:
    """
    Translate the text of each segment to *target_language* (e.g. "English",
    "Simplified Chinese").  Returns a list of translated strings in the same
    order as the input segments.
    """
    texts = [seg["text"] for seg in segments]
    results: list[str] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        translated = await _translate_batch_single(batch, target_language)
        results.extend(translated)
    return results


async def translate_segments_dual(segments: list[dict]) -> list[dict]:
    """
    Translate each segment to both Simplified Chinese and English.
    Returns list of {"zh": str, "en": str} dicts.
    """
    texts = [seg["text"] for seg in segments]
    results: list[dict] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        translated = await _translate_batch_dual(batch)
        results.extend(translated)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _translate_batch_single(texts: list[str], target_language: str) -> list[str]:
    for attempt in range(_MAX_RETRIES):
        try:
            return await _call_single(texts, target_language)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Translation attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("Translation failed after all retries")  # unreachable


async def _translate_batch_dual(texts: list[str]) -> list[dict]:
    for attempt in range(_MAX_RETRIES):
        try:
            return await _call_dual(texts)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Dual translation attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("Dual translation failed after all retries")  # unreachable


async def _call_single(texts: list[str], target_language: str) -> list[str]:
    system_prompt = (
        f"You are a subtitle translator. Translate the following subtitle texts to {target_language}. "
        "Return ONLY a valid JSON object in this exact format: "
        '{"translations": ["translated text 1", "translated text 2", ...]}. '
        "The array must have the same number of elements as the input. "
        "Keep translations brief and natural for subtitles. Preserve meaning and tone."
    )
    user_content = json.dumps(texts, ensure_ascii=False)

    data = await _post(system_prompt, user_content)
    return _extract_translations_single(data, len(texts))


async def _call_dual(texts: list[str]) -> list[dict]:
    system_prompt = (
        "You are a subtitle translator. Translate the following subtitle texts to both "
        "Simplified Chinese and English. "
        "Return ONLY a valid JSON object in this exact format: "
        '{"translations": [{"zh": "Chinese text", "en": "English text"}, ...]}. '
        "The array must have the same number of elements as the input. "
        "Keep translations brief and natural for subtitles. Preserve meaning and tone."
    )
    user_content = json.dumps(texts, ensure_ascii=False)

    data = await _post(system_prompt, user_content)
    return _extract_translations_dual(data, len(texts))


async def _post(system_prompt: str, user_content: str) -> dict:
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        return response.json()


def _extract_translations_single(data: dict, expected_count: int) -> list[str]:
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    translations = None
    if isinstance(parsed, list):
        translations = parsed
    elif isinstance(parsed, dict):
        for key in ("translations", "result", "data", "texts", "output"):
            val = parsed.get(key)
            if isinstance(val, list):
                translations = val
                break

    if translations is None:
        raise ValueError(f"Cannot find translations list in response: {content[:200]}")

    if len(translations) != expected_count:
        logger.warning(
            "Expected %d translations, got %d. Padding/truncating.",
            expected_count,
            len(translations),
        )
        # Pad with empty strings if the model returned fewer items
        while len(translations) < expected_count:
            translations.append("")
        translations = translations[:expected_count]

    return [str(t) for t in translations]


def _extract_translations_dual(data: dict, expected_count: int) -> list[dict]:
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    items = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("translations", "result", "data", "output"):
            val = parsed.get(key)
            if isinstance(val, list):
                items = val
                break

    if items is None:
        raise ValueError(f"Cannot find translations list in response: {content[:200]}")

    # Normalise each item into {"zh": ..., "en": ...}
    result = []
    for item in items[:expected_count]:
        if isinstance(item, dict):
            result.append(
                {
                    "zh": str(item.get("zh", item.get("chinese", item.get("Chinese", "")))),
                    "en": str(item.get("en", item.get("english", item.get("English", "")))),
                }
            )
        else:
            result.append({"zh": "", "en": str(item)})

    while len(result) < expected_count:
        result.append({"zh": "", "en": ""})

    return result
