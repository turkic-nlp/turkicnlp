"""
Dependency parsing processor.

Provides :class:`BiaffineDepParser` implementing biaffine attention
(Dozat & Manning, 2017), the same architecture as Stanza's parser.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor


class BiaffineDepParser(Processor):
    """Dependency parser using biaffine attention.

    Predicts head word and dependency relation for each word in the sentence.
    """

    NAME = "depparse"
    PROVIDES = ["depparse"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str) -> None:
        """Load dependency parser model checkpoint.

        Args:
            model_path: Path to directory containing ``depparse.pt``.
        """
        raise NotImplementedError(
            "BiaffineDepParser.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Predict head and deprel for each word."""
        raise NotImplementedError(
            "BiaffineDepParser.process not yet implemented."
        )

    def _predict(
        self, tokens: list[str], pos_tags: list[str]
    ) -> tuple[list[int], list[str]]:
        """Run biaffine parser inference.

        Returns:
            Tuple of (heads, deprels).
        """
        raise NotImplementedError
