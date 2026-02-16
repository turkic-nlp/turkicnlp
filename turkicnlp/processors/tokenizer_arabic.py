"""
Tokenizer for Perso-Arabic script Turkic languages (Uyghur, Ottoman Turkish).

Handles RTL text, ZWNJ word boundaries, Arabic punctuation, and
kashida (tatweel) stripping.
"""

from __future__ import annotations

import re
from typing import Optional

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.base import Processor
from turkicnlp.scripts import Script


class ArabicScriptTokenizer(Processor):
    """Tokenizer for Perso-Arabic script Turkic languages.

    Key differences from Latin/Cyrillic:
    - RTL text direction
    - ZWNJ (U+200C) as word boundary in some contexts
    - No uppercase/lowercase distinction
    - Arabic punctuation (``،`` ``؛`` ``؟``)
    - Kashida (tatweel, U+0640) stripping
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []
    SUPPORTED_SCRIPTS = [Script.PERSO_ARABIC]

    SENT_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+")
    ZWNJ = "\u200c"
    KASHIDA = "\u0640"
    ARABIC_PUNCT = re.compile(r"([،؛؟!.\(\)\[\]«»\u201c\u201d'\"]+)")

    def __init__(
        self,
        lang: str,
        script: Optional[Script] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._normalize_map: dict[str, str] = {}

    def load(self, model_path: Optional[str] = None) -> None:
        """Load language-specific normalization rules."""
        self._normalize_map = self._load_normalization(self.lang)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Tokenize Perso-Arabic script text."""
        raise NotImplementedError(
            "ArabicScriptTokenizer.process not yet implemented."
        )

    def _normalize(self, text: str) -> str:
        """Normalize Arabic script text (kashida removal, alef variants, etc.)."""
        raise NotImplementedError

    @staticmethod
    def _load_normalization(lang: str) -> dict[str, str]:
        """Load language-specific Arabic character normalization rules."""
        if lang == "uig":
            return {}
        elif lang == "ota":
            return {}
        return {}
