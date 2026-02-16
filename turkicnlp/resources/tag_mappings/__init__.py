"""
Apertium → Universal Dependencies tag mapping system.

Each language has its own mapping module that translates Apertium
morphological tags to UD UPOS and feature strings.
"""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.base import TagMapper


def load_tag_map(lang: str) -> TagMapper:
    """Load the tag mapper for a given language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        A :class:`TagMapper` instance for the language.

    Raises:
        ValueError: If no tag mapping exists for the language.
    """
    if lang == "kaz":
        from turkicnlp.resources.tag_mappings.kaz import KazakhTagMapper
        return KazakhTagMapper()
    elif lang == "tat":
        from turkicnlp.resources.tag_mappings.tat import TatarTagMapper
        return TatarTagMapper()
    elif lang == "tur":
        from turkicnlp.resources.tag_mappings.tur import TurkishTagMapper
        return TurkishTagMapper()
    else:
        return TagMapper()
