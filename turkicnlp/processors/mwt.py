"""
Multi-Word Token expansion processor.

Expands multi-word tokens (MWTs) into their constituent syntactic words.
MWTs are relatively rare in most Turkic languages but occur in some
constructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from turkicnlp.models.document import Document, Token, Word
from turkicnlp.processors.base import Processor
from turkicnlp.resources.registry import ModelRegistry


class MWTProcessor(Processor):
    """Multi-word token expander.

    Identifies tokens that span multiple syntactic words and expands
    them, updating token IDs accordingly.
    """

    NAME = "mwt"
    PROVIDES = ["mwt"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load MWT expansion model or rules.

        Args:
            model_path: Path to MWT expansion resources.
        """
        self._rules = self._load_rules(self.lang)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Expand multi-word tokens into syntactic words."""
        if not self._rules:
            doc._processor_log.append("mwt:rules")
            return doc

        expanded_tokens: list[Token] = []
        expanded_words: list[Word] = []
        word_id = 1

        for sent in doc.sentences:
            expanded_tokens = []
            expanded_words = []
            word_id = 1

            for token in sent.tokens:
                surface = token.text
                if surface in self._rules:
                    pieces = self._rules[surface]
                    start_id = word_id
                    words = []
                    for piece in pieces:
                        word = Word(
                            id=word_id,
                            text=piece,
                            start_char=token.start_char,
                            end_char=token.end_char,
                        )
                        words.append(word)
                        expanded_words.append(word)
                        word_id += 1
                    end_id = word_id - 1
                    expanded_tokens.append(
                        Token(
                            id=(start_id, end_id),
                            text=surface,
                            words=words,
                            start_char=token.start_char,
                            end_char=token.end_char,
                        )
                    )
                else:
                    if token.words:
                        token.words[0].id = word_id
                        expanded_words.append(token.words[0])
                    token.id = (word_id,)
                    expanded_tokens.append(token)
                    word_id += 1

            sent.tokens = list(expanded_tokens)
            sent.words = list(expanded_words)

        doc._processor_log.append("mwt:rules")
        return doc

    def _load_rules(self, lang: str) -> dict[str, list[str]]:
        rules: dict[str, list[str]] = {}
        base = Path(__file__).resolve().parents[1] / "resources"
        default_path = base / "mwt_rules" / f"{lang}.json"
        extracted_path = base / "mwt_rules_extracted" / f"{lang}.json"

        for path in [default_path, extracted_path]:
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for entry in data.get("rules", []):
                surface = entry.get("surface")
                pieces = entry.get("split") or entry.get("pieces")
                if surface and pieces:
                    rules[surface] = list(pieces)
        return rules
