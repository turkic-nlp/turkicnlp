"""
Sentiment analysis processor.

Provides :class:`SentimentProcessor` for sentence-level sentiment
classification (positive / negative / neutral).
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor


class SentimentProcessor(Processor):
    """Sentence-level sentiment classifier.

    Assigns a sentiment label to each sentence in the document.
    """

    NAME = "sentiment"
    PROVIDES = ["sentiment"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load sentiment classification model.

        Args:
            model_path: Path to directory containing sentiment model files.
        """
        raise NotImplementedError(
            "SentimentProcessor.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Classify sentiment for each sentence."""
        raise NotImplementedError(
            "SentimentProcessor.process not yet implemented."
        )
