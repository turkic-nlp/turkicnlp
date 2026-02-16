"""
Lemmatization processors.

Provides :class:`DictionaryLemmatizer` (lookup-based) and
:class:`NeuralLemmatizer` (seq2seq character-level model).
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor


class DictionaryLemmatizer(Processor):
    """Dictionary-based lemmatizer using precompiled lookup tables.

    Falls back to identity (surface form = lemma) for unknown words.
    """

    NAME = "lemma"
    PROVIDES = ["lemma"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str) -> None:
        """Load lemma dictionary from disk.

        Args:
            model_path: Path to directory containing lemma lookup files.
        """
        raise NotImplementedError(
            "DictionaryLemmatizer.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Assign lemmas to each word using dictionary lookup."""
        raise NotImplementedError(
            "DictionaryLemmatizer.process not yet implemented."
        )


class NeuralLemmatizer(Processor):
    """Neural lemmatizer using a character-level seq2seq model.

    Generates lemmas character-by-character, conditioned on the surface
    form and UPOS tag.
    """

    NAME = "lemma"
    PROVIDES = ["lemma"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str) -> None:
        """Load neural lemmatizer model."""
        raise NotImplementedError(
            "NeuralLemmatizer.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Generate lemmas for each word using the neural model."""
        raise NotImplementedError(
            "NeuralLemmatizer.process not yet implemented."
        )
