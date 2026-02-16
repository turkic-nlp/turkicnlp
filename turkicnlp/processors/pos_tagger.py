"""
POS tagging processor.

Provides :class:`NeuralPOSTagger` using a BiLSTM-CRF or fine-tuned
transformer for joint UPOS + morphological feature prediction.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor


class NeuralPOSTagger(Processor):
    """Neural POS tagger with joint UPOS and morphological feature prediction.

    Architecture: BiLSTM-CRF or fine-tuned transformer over pre-trained
    word embeddings or contextual representations.
    """

    NAME = "pos"
    PROVIDES = ["pos", "feats"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load POS tagger model checkpoint.

        Args:
            model_path: Path to directory containing ``pos_tagger.pt``.
        """
        raise NotImplementedError("NeuralPOSTagger.load not yet implemented.")

    def process(self, doc: Document) -> Document:
        """Tag each word with UPOS and morphological features."""
        raise NotImplementedError(
            "NeuralPOSTagger.process not yet implemented."
        )

    def _predict(self, tokens: list[str]) -> list[dict]:
        """Run model inference on a list of tokens.

        Returns:
            List of dicts with keys ``upos``, ``feats``, and optionally ``xpos``.
        """
        raise NotImplementedError
