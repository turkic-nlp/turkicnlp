"""
Named Entity Recognition processor.

Provides :class:`NERProcessor` as a backward-compatible alias to the
Stanza-backed NER implementation.
"""

from __future__ import annotations

from turkicnlp.processors.stanza_backend import StanzaNERProcessor


class NERProcessor(StanzaNERProcessor):
    """Backward-compatible alias for Stanza-based NER."""

