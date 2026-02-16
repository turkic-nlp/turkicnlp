"""
Named Entity Recognition processor.

Provides :class:`NERProcessor` using a sequence labeling model
(BiLSTM-CRF or transformer) with BIO tagging.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document, Span
from turkicnlp.processors.base import Processor


class NERProcessor(Processor):
    """Named Entity Recognition using a sequence labeling model.

    Assigns BIO tags to words and converts them to :class:`Span` entities
    on each sentence.
    """

    NAME = "ner"
    PROVIDES = ["ner"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load NER model checkpoint.

        Args:
            model_path: Path to directory containing ``ner.pt``.
        """
        raise NotImplementedError("NERProcessor.load not yet implemented.")

    def process(self, doc: Document) -> Document:
        """Predict NER tags and build entity spans."""
        raise NotImplementedError("NERProcessor.process not yet implemented.")

    def _predict(self, tokens: list[str]) -> list[str]:
        """Run NER model inference.

        Returns:
            BIO-format tags for each token.
        """
        raise NotImplementedError

    @staticmethod
    def _bio_to_spans(words: list, tags: list[str]) -> list[Span]:
        """Convert BIO-tagged words into :class:`Span` objects.

        Args:
            words: List of :class:`Word` objects.
            tags: Corresponding BIO tags.

        Returns:
            List of entity spans.
        """
        spans: list[Span] = []
        current_span: Optional[Span] = None
        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                if current_span:
                    spans.append(current_span)
                entity_type = tag[2:]
                current_span = Span(
                    text=word.text,
                    type=entity_type,
                    start_char=word.start_char or 0,
                    end_char=word.end_char or 0,
                    words=[word],
                )
            elif tag.startswith("I-") and current_span:
                current_span.text += " " + word.text
                current_span.end_char = word.end_char or 0
                current_span.words.append(word)
            else:
                if current_span:
                    spans.append(current_span)
                    current_span = None
        if current_span:
            spans.append(current_span)
        return spans
