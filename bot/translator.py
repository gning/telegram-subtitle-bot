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
from json import JSONDecodeError

import httpx

from bot.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TRANSLATION_BACKEND,
    TRANSLATION_CONCURRENCY,
    OLLAMA_BASE_URL,
    OLLAMA_TRANSLATION_MODEL,
)

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10
_MAX_BATCH_CHARS = 2500
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


class TranslationResponseError(ValueError):
    """Raised when the translation provider returns unusable message content."""


# Shared HTTP client so concurrent batches reuse connections instead of paying
# a TLS handshake per request. Created lazily on the running event loop.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=180.0)
    return _client


async def _gather_batches(batches: list[list[str]], worker) -> list:
    """Run *worker(batch)* over all batches with bounded concurrency,
    preserving input order in the flattened result."""
    semaphore = asyncio.Semaphore(TRANSLATION_CONCURRENCY)

    async def _run(batch: list[str]):
        async with semaphore:
            return await worker(batch)

    results: list = []
    for translated in await asyncio.gather(*(_run(b) for b in batches)):
        results.extend(translated)
    return results


def _language_display(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def translate_segments(
    segments: list[dict],
    target_language: str,
    settings: dict | None = None,
) -> list[str]:
    """
    Translate the text of each segment to *target_language* (e.g. "English",
    "Simplified Chinese").  Returns a list of translated strings in the same
    order as the input segments.
    """
    texts = [seg["text"] for seg in segments]
    return await _gather_batches(
        _iter_batches(texts),
        lambda batch: _translate_batch_single_adaptive(batch, target_language, settings),
    )


async def translate_segments_dual(
    segments: list[dict],
    settings: dict | None = None,
) -> list[dict]:
    """
    Translate each segment to both Simplified Chinese and English.
    Returns list of {"zh": str, "en": str} dicts.
    """
    texts = [seg["text"] for seg in segments]
    return await _gather_batches(
        _iter_batches(texts),
        lambda batch: _translate_batch_dual_adaptive(batch, settings),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_batches(texts: list[str]) -> list[list[str]]:
    """Batch by item count and text size so long videos do not produce huge JSON."""
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_chars = 0

    for text in texts:
        text_chars = len(text)
        would_exceed_count = len(batch) >= _BATCH_SIZE
        would_exceed_chars = batch and batch_chars + text_chars > _MAX_BATCH_CHARS
        if would_exceed_count or would_exceed_chars:
            batches.append(batch)
            batch = []
            batch_chars = 0

        batch.append(text)
        batch_chars += text_chars

    if batch:
        batches.append(batch)

    return batches


async def _translate_batch_single_adaptive(
    texts: list[str], target_language: str, settings: dict | None
) -> list[str]:
    try:
        return await _translate_batch_single(texts, target_language, settings)
    except TranslationResponseError:
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        logger.warning(
            "Malformed translation response for %d texts; retrying as %d + %d.",
            len(texts),
            mid,
            len(texts) - mid,
        )
        left = await _translate_batch_single_adaptive(texts[:mid], target_language, settings)
        right = await _translate_batch_single_adaptive(texts[mid:], target_language, settings)
        return left + right


async def _translate_batch_dual_adaptive(texts: list[str], settings: dict | None) -> list[dict]:
    try:
        return await _translate_batch_dual(texts, settings)
    except TranslationResponseError:
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        logger.warning(
            "Malformed dual translation response for %d texts; retrying as %d + %d.",
            len(texts),
            mid,
            len(texts) - mid,
        )
        left = await _translate_batch_dual_adaptive(texts[:mid], settings)
        right = await _translate_batch_dual_adaptive(texts[mid:], settings)
        return left + right


async def _translate_batch_single(
    texts: list[str], target_language: str, settings: dict | None
) -> list[str]:
    for attempt in range(_MAX_RETRIES):
        try:
            return await _call_single(texts, target_language, settings)
        except TranslationResponseError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Translation attempt %d/%d returned malformed JSON. Retrying in %ds...",
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            await asyncio.sleep(wait)
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


async def _translate_batch_dual(texts: list[str], settings: dict | None) -> list[dict]:
    for attempt in range(_MAX_RETRIES):
        try:
            return await _call_dual(texts, settings)
        except TranslationResponseError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Dual translation attempt %d/%d returned malformed JSON. Retrying in %ds...",
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            await asyncio.sleep(wait)
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


async def _call_single(texts: list[str], target_language: str, settings: dict | None) -> list[str]:
    system_prompt = (
        f"You are a subtitle translator. Translate the following subtitle texts to {target_language}. "
        "Return ONLY a valid JSON object in this exact format: "
        '{"translations": ["translated text 1", "translated text 2", ...]}. '
        "The array must have the same number of elements as the input. "
        "Keep translations brief and natural for subtitles. Preserve meaning and tone."
    )
    user_content = json.dumps(texts, ensure_ascii=False)

    data = await _post(system_prompt, user_content, settings)
    return _extract_translations_single(data, len(texts))


async def _call_dual(texts: list[str], settings: dict | None) -> list[dict]:
    system_prompt = (
        "You are a subtitle translator. Translate the following subtitle texts to both "
        "Simplified Chinese and English. "
        "Return ONLY a valid JSON object in this exact format: "
        '{"translations": [{"zh": "Chinese text", "en": "English text"}, ...]}. '
        "The array must have the same number of elements as the input. "
        "Keep translations brief and natural for subtitles. Preserve meaning and tone."
    )
    user_content = json.dumps(texts, ensure_ascii=False)

    data = await _post(system_prompt, user_content, settings)
    return _extract_translations_dual(data, len(texts))


async def _post(system_prompt: str, user_content: str, settings: dict | None) -> dict:
    s = settings or {}
    backend = s.get("translation_backend", TRANSLATION_BACKEND)

    if backend == "ollama":
        base_url = s.get("translation_url", OLLAMA_BASE_URL)
        model = s.get("translation_model", OLLAMA_TRANSLATION_MODEL)
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
    else:
        model = OPENROUTER_MODEL
        url = _OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

    logger.info("Translating via %s (model=%s)", backend, model)
    response = await _get_client().post(
        url,
        headers=headers,
        json={
            "model": model,
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
    content = _message_content(data)
    parsed = _parse_json_content(content)

    translations = None
    if isinstance(parsed, list):
        translations = parsed
    elif isinstance(parsed, dict):
        for key in ("translations", "result", "data", "texts", "output"):
            val = parsed.get(key)
            if isinstance(val, list):
                translations = val
                break
        if translations is None and expected_count == 1:
            for key in ("translation", "text", "result", "output"):
                val = parsed.get(key)
                if isinstance(val, str):
                    translations = [val]
                    break

    if translations is None:
        raise TranslationResponseError(f"Cannot find translations list in response: {content[:200]}")

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
    content = _message_content(data)
    parsed = _parse_json_content(content)

    items = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("translations", "result", "data", "output"):
            val = parsed.get(key)
            if isinstance(val, list):
                items = val
                break
        if items is None and expected_count == 1 and (
            any(key in parsed for key in ("zh", "chinese", "Chinese"))
            or any(key in parsed for key in ("en", "english", "English"))
        ):
            items = [parsed]

    if items is None:
        raise TranslationResponseError(f"Cannot find translations list in response: {content[:200]}")

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


def _message_content(data: dict) -> str:
    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason")
    content = choice["message"].get("content") or ""
    if finish_reason == "length":
        raise TranslationResponseError("Translation response was truncated by the model.")
    return content


def _parse_json_content(content: str):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(text)
    except JSONDecodeError as exc:
        start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
        if start_positions:
            start = min(start_positions)
            try:
                parsed, end = decoder.raw_decode(text[start:])
            except JSONDecodeError:
                pass
            else:
                trailing = text[start + end:].strip()
                if trailing:
                    logger.warning("Ignoring trailing text after translation JSON: %r", trailing[:120])
                return parsed

        snippet = text[:200].replace("\n", "\\n")
        raise TranslationResponseError(
            f"Could not parse translation JSON: {exc.msg} at char {exc.pos}. "
            f"Response starts with: {snippet!r}"
        ) from exc

    trailing = text[end:].strip()
    if trailing:
        logger.warning("Ignoring trailing text after translation JSON: %r", trailing[:120])
    return parsed
