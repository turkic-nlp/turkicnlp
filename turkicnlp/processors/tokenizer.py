"""
Tokenizers for Latin and Cyrillic script Turkic languages.

Provides :class:`RegexTokenizer` (rule-based, handles ~80% of cases)
and :class:`NeuralTokenizer` (BiLSTM character-level, for edge cases).
"""

from __future__ import annotations

import re
from typing import Optional

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.base import Processor


class RegexTokenizer(Processor):
    """Rule-based tokenizer for Latin/Cyrillic Turkic languages.

    Most Turkic languages are space-delimited with predictable punctuation.
    This handles the majority of cases.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
    TOKEN_SPLIT = re.compile(r"(\S+)")
    PUNCT_DETACH = re.compile(r'^(.*?)([.,:;!?"\'\)\]\}]+)$')

    def __init__(
        self,
        lang: str,
        script: Optional[object] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._abbreviations: set[str] = set()

    def load(self, model_path: Optional[str] = None) -> None:
        """Load optional abbreviation lists to avoid false sentence splits."""
        self._abbreviations = set()
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Split text into sentences and tokens."""
        raise NotImplementedError("RegexTokenizer.process not yet implemented.")


class NeuralTokenizer(Processor):
    """Neural tokenizer based on a character-level BiLSTM.

    For languages where rule-based splitting is insufficient.
    Architecture follows Stanza's tokenizer.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    def load(self, model_path: str) -> None:
        """Load trained character-level BiLSTM tokenizer model."""
        raise NotImplementedError("NeuralTokenizer.load not yet implemented.")

    def process(self, doc: Document) -> Document:
        """Neural sentence splitting and tokenization."""
        raise NotImplementedError("NeuralTokenizer.process not yet implemented.")
