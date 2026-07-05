"""
ASS subtitle file generation.

Layout (all cases):
  - Both Chinese and English subtitles sit in the bottom quarter of the screen.
  - Chinese subtitle (yellow) is stacked above the English subtitle (white).
  - Three-language (other source): original language on top, Chinese above English at bottom.
"""

from __future__ import annotations

import logging
import math
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASS colour constants  (&HAABBGGRR)
# ---------------------------------------------------------------------------
_WHITE = "&H00FFFFFF"
_YELLOW = "&H0000FFFF"   # yellow: R=255 G=255 B=0 → AABBGGRR = 0000FFFF
_TRANSPARENT = "&H00000000"
_BLACK_OUTLINE = "&H00000000"
_SHADOW_COLOUR = "&H64000000"  # 40 % opaque black

# ---------------------------------------------------------------------------
# Font names
# ---------------------------------------------------------------------------
_FONT_CJK = "Noto Sans CJK SC"
_FONT_LATIN = "Arial"

# ---------------------------------------------------------------------------
# ASS header template
# ---------------------------------------------------------------------------
_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# Style definition helper
# Alignment: 2=bottom-center, 8=top-center
def _style(name: str, font: str, size: int, alignment: int, margin_v: int,
           colour: str = _WHITE) -> str:
    return (
        f"Style: {name},{font},{size},"
        f"{colour},{_TRANSPARENT},{_BLACK_OUTLINE},{_SHADOW_COLOUR},"
        f"0,0,0,0,100,100,0,0,1,2,1,{alignment},10,10,{margin_v},1"
    )


# Predefined styles used across layouts.
# CJK styles are yellow; Latin styles are white.
_STYLES = "\n".join([
    # Top-aligned styles
    _style("LatinTop",       _FONT_LATIN, 32, 8, 20, _WHITE),
    _style("CJKTop",         _FONT_CJK,   36, 8, 20, _YELLOW),
    # Bottom-aligned styles
    _style("LatinBottom",    _FONT_LATIN, 32, 2, 12, _WHITE),
    _style("CJKBottom",      _FONT_CJK,   36, 2, 12, _YELLOW),
    # Mid-bottom styles (used when 3 lines are needed)
    # Positioned ~56 px above the bottom (font 36 * 1.3 line height ≈ 47 px + gap)
    _style("CJKMidBottom",   _FONT_CJK,   36, 2, 62, _YELLOW),
    _style("LatinMidBottom", _FONT_LATIN, 32, 2, 62, _WHITE),
])


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

def _ts(seconds: float) -> str:
    """Convert seconds to ASS timestamp h:mm:ss.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Dialogue line helper
# ---------------------------------------------------------------------------

def _dialogue(start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ass(
    segments: list[dict],
    source_lang: str,
    translations: list[str] | list[dict],
    output_path: str,
) -> None:
    """
    Write an ASS subtitle file.

    Parameters
    ----------
    segments      : list of {"start", "end", "text"} dicts from the transcriber
    source_lang   : ISO 639-1 language code detected by Whisper (e.g. "en", "zh")
    translations  : For single-target languages, a list of translated strings.
                    For dual targets ("other" languages), a list of
                    {"zh": ..., "en": ...} dicts.
    output_path   : where to write the .ass file
    """
    lines: list[str] = []
    norm_lang = source_lang.lower()
    is_chinese_source = norm_lang in ("zh", "zh-cn", "zh-tw")
    is_english_source = norm_lang == "en"

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        original = seg["text"]

        if is_chinese_source:
            # Chinese original above English translation, both in bottom quarter
            translation = translations[i] if i < len(translations) else ""
            lines.append(_dialogue(start, end, "CJKBottom", _stack_zh_en(original, str(translation))))

        elif is_english_source:
            # Chinese translation above English original, both in bottom quarter
            translation = translations[i] if i < len(translations) else ""
            lines.append(_dialogue(start, end, "CJKBottom", _stack_zh_en(str(translation), original)))

        else:
            # Other language: original on top, Chinese above English at bottom
            pair = translations[i] if i < len(translations) else {"zh": "", "en": ""}
            zh = pair.get("zh", "") if isinstance(pair, dict) else ""
            en = pair.get("en", "") if isinstance(pair, dict) else str(pair)
            lines.append(_dialogue(start, end, "LatinTop",   _escape(original)))
            lines.append(_dialogue(start, end, "CJKBottom", _stack_zh_en(zh, en)))

    content = _HEADER.format(styles=_STYLES) + "\n".join(lines) + "\n"

    with open(output_path, "w", encoding="utf-8-sig") as fh:
        fh.write(content)

    logger.info("ASS subtitle file written to %s (%d events)", output_path, len(lines))


# Inline override switching to the Latin look mid-event (font, size, white).
_LATIN_OVERRIDE = f"{{\\fn{_FONT_LATIN}\\fs32\\c&HFFFFFF&}}"


def _stack_zh_en(zh: str, en: str) -> str:
    """Combine Chinese-above-English into a single event's text.

    A single event keeps the stacking order fixed: with separate events,
    libass collision handling relocates whichever event overlaps once a
    wrapped line grows taller than the margin gap between them.
    """
    zh_part = _escape(_wrap_cjk(zh)) if zh else ""
    en_part = f"{_LATIN_OVERRIDE}{_escape(en)}" if en else ""
    if zh_part and en_part:
        return f"{zh_part}\\N{en_part}"
    return zh_part or en_part


# libass only wraps lines at spaces, so Chinese text never wraps on its own.
# Insert explicit \N breaks, balancing the lines. Width budget: PlayResX 1280
# minus margins ≈ 1260 px at font size 36 → ~35 full-width glyphs; use a bit
# less for the outline and glyph-width variance.
_CJK_MAX_LINE_UNITS = 32.0
# Characters that must not start a line (CJK punctuation rules).
_NO_LINE_START = "，。！？、；：）】》」』…—％℃"


def _char_units(ch: str) -> float:
    """Approximate display width: full-width glyphs 1.0, others 0.5."""
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5


def _wrap_cjk(text: str, max_units: float = _CJK_MAX_LINE_UNITS) -> str:
    """Break spaceless CJK text into balanced lines joined with \\N."""
    total = sum(_char_units(ch) for ch in text)
    if total <= max_units:
        return text

    line_count = math.ceil(total / max_units)
    target = total / line_count
    lines: list[str] = []
    current, current_units = "", 0.0
    for idx, ch in enumerate(text):
        current += ch
        current_units += _char_units(ch)
        if current_units >= target and len(lines) < line_count - 1:
            # Don't split inside an ASCII word (mixed CJK/Latin text).
            nxt = text[idx + 1] if idx + 1 < len(text) else ""
            if ch.isascii() and ch.isalnum() and nxt.isascii() and nxt.isalnum():
                continue
            lines.append(current)
            current, current_units = "", 0.0
    if current:
        lines.append(current)

    # Pull leading closing punctuation back onto the previous line.
    for i in range(1, len(lines)):
        while lines[i] and lines[i][0] in _NO_LINE_START:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    return "\\N".join(line.strip() for line in lines if line.strip())


def _escape(text: str) -> str:
    """Escape characters that have special meaning in ASS dialogue text."""
    # Replace real newlines with ASS soft line break
    text = text.replace("\n", "\\N")
    # Braces are used for override tags — escape literal ones
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text
