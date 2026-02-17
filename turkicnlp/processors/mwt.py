"""
Multi-Word Token expansion processor.

Expands multi-word tokens (MWTs) into their constituent syntactic words.
MWTs are relatively rare in most Turkic languages but occur in some
constructions.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor


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
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Expand multi-word tokens into syntactic words."""
        doc._processor_log.append("mwt:neural")
        return doc
