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
# NOTE: "▁" is also SentencePiece's own word-boundary symbol. Models trained with
# this boundary cannot distinguish morpheme from word boundaries when decoding;
# for new models pass a dedicated boundary (e.g. annotate with boundary="\u2502"
# and configure sp_tokenize_boundary="\u2502" plus user_defined_symbols in SP).


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
        cfg = config or {}
        self._model_path: Optional[str] = cfg.get("model_path")
        # The SP models are trained on morpheme-boundary-annotated text
        # (tools/annotate_corpus.py). Encoding raw words would differ from the
        # training distribution, so by default words are segmented the same way
        # before encoding. The boundary must match the one used for training.
        self._segment_input: bool = bool(cfg.get("segment_input", True))
        self._boundary: str = str(cfg.get("boundary", BOUNDARY))
        self._segmenter = None
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
        if self._segment_input:
            try:
                from turkicnlp.processors.morpheme_tokenizer import MorphemeTokenizer

                self._segmenter = MorphemeTokenizer(lang=self.lang)
                self._segmenter.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Morpheme segmentation unavailable (%s); encoding raw words.", exc)
                self._segmenter = None
        self._loaded = True

    def _prepare(self, word: str) -> str:
        """Insert morpheme boundaries exactly as in the training corpus."""
        if self._segmenter is None or len(word) <= 1 or not any(c.isalpha() for c in word):
            return word
        try:
            segs = self._segmenter.segment(word).segments
        except Exception:  # noqa: BLE001
            return word
        return self._boundary.join(segs) if len(segs) > 1 else word

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
                text = self._prepare(word.text)
                tokens = self.sp.encode(text, out_type=str)
                ids = self.sp.encode(text)
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
        return self.sp.encode(" ".join(self._prepare(w) for w in text.split()), out_type=str)

    def tokenize_ids(self, text: str) -> list[int]:
        """Tokenize a raw string and return integer IDs."""
        if self.sp is None:
            raise RuntimeError("Call load() before tokenize_ids().")
        return self.sp.encode(" ".join(self._prepare(w) for w in text.split()))

    def decode(self, ids: list[int]) -> str:
        """Decode integer IDs back to text."""
        if self.sp is None:
            raise RuntimeError("Call load() before decode().")
        text = self.sp.decode(ids)
        # remove inserted morpheme boundaries (SentencePiece decodes "▁" as a space,
        # so with the default boundary morphemes of one word come back space-separated)
        return text.replace(self._boundary, "") if self._boundary != BOUNDARY else text

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
        from turkicnlp.resources.registry import ModelRegistry

        registry_path = ModelRegistry.default_dir() / "sp_tokenize" / f"sp_{lang}.model"
        if registry_path.exists():
            return str(registry_path)
        fallback = Path("models/sp/monolingual") / f"sp_{lang}.model"
        logger.debug("No SP model at %s; using fallback path: %s", registry_path, fallback)
        return str(fallback)
