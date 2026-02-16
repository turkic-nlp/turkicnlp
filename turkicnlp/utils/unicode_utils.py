"""
Unicode normalization helpers for Turkic text processing.

Handles common issues like the Turkish İ/I dotted/dotless distinction,
NFC/NFD normalization, and character confusables.
"""

from __future__ import annotations

import unicodedata


def normalize_nfc(text: str) -> str:
    """Apply NFC Unicode normalization.

    Args:
        text: Input text.

    Returns:
        NFC-normalized text.
    """
    return unicodedata.normalize("NFC", text)


def normalize_turkish_i(text: str) -> str:
    """Handle Turkish İ/I dotted/dotless case folding correctly.

    Standard ``str.lower()`` maps ``I`` to ``i``, but in Turkish
    ``I`` should map to ``ı`` (dotless i) and ``İ`` to ``i``.

    Args:
        text: Input text.

    Returns:
        Text with Turkish-correct lowercase mapping.
    """
    result = text.replace("I", "ı").replace("İ", "i")
    return result


def strip_diacritics(text: str) -> str:
    """Remove combining diacritical marks from text.

    Useful for approximate matching and search.

    Args:
        text: Input text.

    Returns:
        Text with diacritics removed.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
