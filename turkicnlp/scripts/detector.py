"""
Automatic script detection for Turkic text.

Provides :func:`detect_script` for determining the dominant writing script,
and :func:`detect_script_segments` for segmenting mixed-script documents.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Optional

from turkicnlp.scripts import Script, ScriptConfig


# Unicode block ranges for script detection
SCRIPT_RANGES: dict[Script, list[tuple[int, int]]] = {
    Script.LATIN: [
        (0x0041, 0x024F),   # Basic Latin + Extended A/B
        (0x1E00, 0x1EFF),   # Latin Extended Additional
    ],
    Script.CYRILLIC: [
        (0x0400, 0x04FF),   # Cyrillic
        (0x0500, 0x052F),   # Cyrillic Supplement
        (0x2DE0, 0x2DFF),   # Cyrillic Extended-A
        (0xA640, 0xA69F),   # Cyrillic Extended-B
    ],
    Script.PERSO_ARABIC: [
        (0x0600, 0x06FF),   # Arabic
        (0x0750, 0x077F),   # Arabic Supplement
        (0x08A0, 0x08FF),   # Arabic Extended-A
        (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    ],
    Script.OLD_TURKIC_RUNIC: [
        (0x10C00, 0x10C4F),  # Old Turkic
    ],
}


def _char_to_script(char: str) -> Optional[Script]:
    """Map a single character to its script. Returns ``None`` for neutral chars."""
    cp = ord(char)
    cat = unicodedata.category(char)
    if cat.startswith(("N", "P", "Z", "S", "C")):
        return None
    for script, ranges in SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= cp <= end:
                return script
    return None


def detect_script(text: str) -> Script:
    """Detect the dominant script in a text string.

    Counts letter characters by script and returns the most common one.
    Ignores digits, punctuation, and whitespace.

    Args:
        text: Input text.

    Returns:
        The dominant :class:`Script`.

    Raises:
        ValueError: If no script characters are found.
    """
    counts: Counter[Script] = Counter()
    for char in text:
        script = _char_to_script(char)
        if script is not None:
            counts[script] += 1

    if not counts:
        raise ValueError("No script characters detected in text.")

    return counts.most_common(1)[0][0]


def detect_script_segments(text: str) -> list[tuple[str, Script]]:
    """Segment text into contiguous runs of the same script.

    Useful for mixed-script documents (e.g. Uzbek with Cyrillic paragraphs
    embedded in Latin text).

    Args:
        text: Input text.

    Returns:
        List of ``(segment_text, script)`` tuples.
    """
    if not text:
        return []

    segments: list[tuple[str, Script]] = []
    current_script: Optional[Script] = None
    current_start = 0

    for i, char in enumerate(text):
        char_script = _char_to_script(char)
        if char_script is None:
            continue
        if char_script != current_script:
            if current_script is not None:
                segments.append((text[current_start:i], current_script))
            current_start = i
            current_script = char_script

    if current_script is not None:
        segments.append((text[current_start:], current_script))

    return segments


def detect_script_for_language(
    text: str, lang: str, script_config: ScriptConfig
) -> Script:
    """Detect script with language-aware validation.

    Args:
        text: Input text.
        lang: ISO 639-3 language code.
        script_config: The language's :class:`ScriptConfig`.

    Returns:
        The detected :class:`Script`.

    Raises:
        ValueError: If detected script is not valid for the language.
    """
    detected = detect_script(text)

    if detected in script_config.available:
        return detected

    raise ValueError(
        f"Detected script '{detected}' is not a known script for language '{lang}'. "
        f"Expected one of: {[str(s) for s in script_config.available]}. "
        f"If the text is in a different script, use the `script` parameter to specify it, "
        f"or enable transliteration."
    )
