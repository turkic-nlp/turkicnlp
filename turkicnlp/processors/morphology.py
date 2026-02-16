"""
Morphological analysis processors.

Provides :class:`ApertiumMorphProcessor` (HFST-native FST loading, no system
Apertium install needed) and :class:`NeuralMorphProcessor` (character-level
neural analyzer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor
from turkicnlp.scripts import Script, get_script_config


class ApertiumMorphProcessor(Processor):
    """Morphological analyzer using Apertium FST data loaded natively via ``hfst``.

    The compiled ``.hfst`` transducer is loaded via the ``hfst`` Python package.
    No system Apertium installation is required.

    License note:
        The ``.hfst`` data files are GPL-3.0 licensed and are downloaded
        separately from the Apache-2.0 turkicnlp library.
    """

    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]
    REQUIRES = ["tokenize"]

    def __init__(
        self,
        lang: str,
        script: Optional[Script] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self._analyzer = None       # hfst.HfstTransducer
        self._generator = None      # hfst.HfstTransducer (optional)
        self._tag_mapper = None
        self._to_fst_translit = None
        self._from_fst_translit = None

    def load(self, model_path: str | Path) -> None:
        """Load compiled HFST transducer from the downloaded data directory.

        Args:
            model_path: Path to the apertium data directory, e.g.
                ``~/.turkicnlp/models/kaz/Cyrl/morph/apertium/``.
        """
        raise NotImplementedError(
            "ApertiumMorphProcessor.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Run HFST morphological analysis on each word.

        Steps:
            1. Optionally transliterate to FST script
            2. Look up in HFST transducer
            3. Disambiguate (pick best reading)
            4. Map Apertium tags to UD tags
            5. Optionally transliterate lemma back
        """
        raise NotImplementedError(
            "ApertiumMorphProcessor.process not yet implemented."
        )

    def _analyze(self, surface: str) -> list[dict]:
        """Look up a surface form in the HFST transducer.

        Returns:
            List of readings, each as ``{"lemma": str, "pos": str, "feats": list[str]}``.
        """
        raise NotImplementedError

    def _disambiguate(self, readings: list[dict]) -> dict:
        """Pick the best reading from multiple analyses."""
        raise NotImplementedError

    def generate(self, lemma: str, tags: list[str]) -> Optional[str]:
        """Generate a surface form from a lemma and morphological tags.

        Args:
            lemma: Base form (e.g. ``мектеп``).
            tags: Apertium-format tags (e.g. ``["n", "dat", "sg"]``).

        Returns:
            Generated surface form or ``None`` if generation is unavailable.
        """
        raise NotImplementedError

    @property
    def available_for_generation(self) -> bool:
        """Whether this processor can generate surface forms."""
        return self._generator is not None


class NeuralMorphProcessor(Processor):
    """Neural morphological analyzer using a character-level seq2seq model.

    For languages without Apertium FSTs or as a higher-accuracy alternative.
    """

    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load neural morphological analysis model."""
        raise NotImplementedError(
            "NeuralMorphProcessor.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Run neural morphological analysis on each word."""
        raise NotImplementedError(
            "NeuralMorphProcessor.process not yet implemented."
        )
