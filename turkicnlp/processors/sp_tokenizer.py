"""MorphemeAwareSPTokenizer — SentencePiece tokenizer processor for TurkicNLP Pipeline.

Wraps a SentencePiece model trained on morpheme-boundary-annotated Turkic text.
Produces ``sp_tokens`` and ``sp_ids`` annotations on each sentence.

Usage via Pipeline::

    import turkicnlp
    nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "sp_tokenize"])
    doc = nlp("Мен мектептерімізден оқыдым")
    for sent in doc.sentences:
        print(sent.sp_tokens)

Usage standalone::

    from turkicnlp.processors.sp_tokenizer import MorphemeAwareSPTokenizer
    proc = MorphemeAwareSPTokenizer(lang="kaz", model_path="models/sp/monolingual/sp_kaz.model")
    proc.load()
    # Process raw text (no pipeline needed)
    tokens = proc.tokenize_text("мектептерімізден")
    # ['мектеп', '▁', 'тер', '▁', 'іміз', '▁', 'ден']
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor

logger = logging.getLogger(__name__)

BOUNDARY = "\u2581"  # ▁


class MorphemeAwareSPTokenizer(Processor):
    """SentencePiece tokenizer trained on morpheme-boundary-annotated Turkic text.

    Extends the TurkicNLP :class:`~turkicnlp.processors.base.Processor` interface.
    Produces ``sp_tokens`` and ``sp_ids`` on each sentence in the document.

    Attributes:
        NAME: ``"sp_tokenize"``
        REQUIRES: ``["tokenize"]``
        PROVIDES: ``["sp_tokens", "sp_ids"]``
    """

    NAME = "sp_tokenize"
    REQUIRES = ["tokenize"]
    PROVIDES = ["sp_tokens", "sp_ids"]

    def __init__(
        self,
        lang: str,
        script=None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang=lang, script=script, config=config)
        self._model_path: Optional[str] = (config or {}).get("model_path")
        self.sp = None

    def load(self, model_path: str = "") -> None:
        """Load the SentencePiece model from disk.

        Args:
            model_path: Path to the ``.model`` file.  If empty, falls back to
                the path set in ``config["model_path"]`` or the default registry.
        """
        import sentencepiece as spm

        resolved = model_path or self._model_path or self._default_model_path(self.lang)
        logger.info("Loading SP model: %s", resolved)
        self.sp = spm.SentencePieceProcessor(model_file=str(resolved))
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Annotate each sentence with SentencePiece tokens and IDs.

        Sets ``sentence.sp_tokens`` (list[str]) and ``sentence.sp_ids`` (list[int])
        on each sentence.
        """
        if self.sp is None:
            raise RuntimeError("Call load() before process().")

        for sentence in doc.sentences:
            sp_tokens: list[str] = []
            sp_ids: list[int] = []
            for word in sentence.words:
                tokens = self.sp.encode(word.text, out_type=str)
                ids = self.sp.encode(word.text)
                sp_tokens.extend(tokens)
                sp_ids.extend(ids)
            sentence.sp_tokens = sp_tokens  # type: ignore[attr-defined]
            sentence.sp_ids = sp_ids  # type: ignore[attr-defined]
        return doc

    def tokenize_text(self, text: str) -> list[str]:
        """Tokenize a raw string directly (no pipeline required).

        Args:
            text: Input text (may contain spaces; tokenized as-is by SP).

        Returns:
            List of SentencePiece token strings.
        """
        if self.sp is None:
            raise RuntimeError("Call load() before tokenize_text().")
        return self.sp.encode(text, out_type=str)

    def tokenize_ids(self, text: str) -> list[int]:
        """Tokenize a raw string and return integer IDs."""
        if self.sp is None:
            raise RuntimeError("Call load() before tokenize_ids().")
        return self.sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode integer IDs back to text."""
        if self.sp is None:
            raise RuntimeError("Call load() before decode().")
        return self.sp.decode(ids)

    def vocab_size(self) -> int:
        """Return the SP model vocabulary size."""
        if self.sp is None:
            raise RuntimeError("Call load() first.")
        return self.sp.get_piece_size()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_model_path(lang: str) -> str:
        """Return the default SP model path for *lang*.

        Tries the TurkicNLP model registry first; falls back to a
        conventional local path ``models/sp/monolingual/sp_{lang}.model``.
        """
        try:
            from turkicnlp.models.registry import ModelRegistry  # type: ignore

            return ModelRegistry.get_path(lang, "sp_tokenize")
        except Exception:
            fallback = Path("models/sp/monolingual") / f"sp_{lang}.model"
            logger.debug("ModelRegistry unavailable; using fallback path: %s", fallback)
            return str(fallback)
