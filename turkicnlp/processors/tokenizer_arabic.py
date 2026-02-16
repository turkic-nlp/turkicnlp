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
        text = self._normalize(doc.text)

        raw_sents = self.SENT_SPLIT.split(text)
        raw_sents = [s.strip() for s in raw_sents if s.strip()]

        char_offset = 0
        for sent_text in raw_sents:
            sentence = Sentence(text=sent_text)

            word_id = 1
            token_pattern = re.compile(rf"[^\s{self.ZWNJ}]+")
            parts_pattern = re.compile(r"[،؛؟!.\(\)\[\]«»\u201c\u201d'\"]+|[^،؛؟!.\(\)\[\]«»\u201c\u201d'\"]+")

            for match in token_pattern.finditer(sent_text):
                raw_token = match.group()
                raw_start = match.start()

                for part_match in parts_pattern.finditer(raw_token):
                    part_text = part_match.group()
                    part_start = char_offset + raw_start + part_match.start()
                    part_end = part_start + len(part_text)
                    is_punct = bool(self.ARABIC_PUNCT.fullmatch(part_text))

                    word = Word(
                        id=word_id,
                        text=part_text,
                        upos="PUNCT" if is_punct else None,
                        lemma=part_text if is_punct else None,
                        start_char=part_start,
                        end_char=part_end,
                    )
                    token = Token(
                        id=(word_id,),
                        text=part_text,
                        words=[word],
                        start_char=part_start,
                        end_char=part_end,
                    )
                    sentence.tokens.append(token)
                    sentence.words.append(word)
                    word_id += 1

            doc.sentences.append(sentence)
            char_offset += len(sent_text) + 1

        doc._processor_log.append("tokenize:arabic_script")
        return doc

    def _normalize(self, text: str) -> str:
        """Normalize Arabic script text (kashida removal, alef variants, etc.)."""
        text = text.replace(self.KASHIDA, "")
        text = re.sub(r"[أإآٱ]", "ا", text)
        for old, new in self._normalize_map.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _load_normalization(lang: str) -> dict[str, str]:
        """Load language-specific Arabic character normalization rules."""
        if lang == "uig":
            return {}
        elif lang == "ota":
            return {}
        return {}
