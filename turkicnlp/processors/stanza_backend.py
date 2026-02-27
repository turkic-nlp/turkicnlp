"""
Stanza-backed processors for TurkicNLP.

Provides adapter classes that delegate to Stanford Stanza's pretrained
Universal Dependencies models for tokenization, MWT expansion, POS tagging,
lemmatization, and dependency parsing.

Stanza models are automatically downloaded on first use via ``stanza.download()``.
Install with: ``pip install turkicnlp[stanza]``
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from turkicnlp.models.document import Document, Sentence, Span, Token, Word
from turkicnlp.processors.base import Processor
from turkicnlp.resources.registry import ModelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code mapping: ISO 639-3 (turkicnlp) -> Stanza language code
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    "tur": "tr",   # UD treebanks: IMST, BOUN, FrameNet, KeNet, ATIS, Penn, Tourism
    "kaz": "kk",   # UD treebank: KTB
    "uig": "ug",   # UD treebank: UDT
    "kir": "ky",   # UD treebank: KTMU
    "ota": "ota",  # UD treebank: BOUN
    "uzb": "uz",   # Custom-trained on UzUDT (turkic-nlp/trained-stanza-models)
}

# Languages with custom-trained Stanza models (not from official Stanza hub).
# These require explicit model paths and allow_unknown_language=True.
_CUSTOM_STANZA_LANGS: set[str] = {"uzb"}

# Mapping from processor name to model path kwarg name used by stanza.Pipeline
_CUSTOM_MODEL_PATH_KWARGS: dict[str, tuple[str, str]] = {
    "tokenize": ("tokenize_model_path", "tokenizer.pt"),
    "pos":      ("pos_model_path",      "tagger.pt"),
    "lemma":    ("lemma_model_path",     "lemmatizer.pt"),
    "depparse": ("depparse_model_path",  "parser.pt"),
}

# Processors that need a pretrain embedding path
_CUSTOM_PRETRAIN_KWARGS: dict[str, str] = {
    "pos":      "pos_pretrain_path",
    "depparse": "depparse_pretrain_path",
}

STANZA_SUPPORTED_LANGUAGES: set[str] = set(_LANG_MAP.keys())


def _require_stanza() -> Any:
    """Import and return the ``stanza`` package.

    Raises:
        ImportError: If stanza is not installed.
    """
    try:
        import stanza

        return stanza
    except ImportError:
        raise ImportError(
            "Stanza is required for the 'stanza' backend. "
            "Install it with: pip install turkicnlp[stanza]"
        ) from None


def _get_stanza_lang(lang: str) -> str:
    """Map ISO 639-3 code to Stanza's language identifier.

    Raises:
        ValueError: If language is not supported by the Stanza backend.
    """
    if lang not in _LANG_MAP:
        raise ValueError(
            f"Language '{lang}' is not supported by the Stanza backend. "
            f"Supported: {sorted(STANZA_SUPPORTED_LANGUAGES)}"
        )
    return _LANG_MAP[lang]


def _is_custom_stanza(lang: str) -> bool:
    """Check if a language uses custom-trained Stanza models."""
    return lang in _CUSTOM_STANZA_LANGS


def _get_custom_model_dir(lang: str) -> "Path":
    """Return the directory containing custom Stanza model files.

    Uses ISO 639-3 code (e.g. ``uzb``) as directory name, consistent
    with the rest of the library.
    """
    from pathlib import Path
    return ModelRegistry.default_dir() / "stanza_custom" / lang


def _build_custom_kwargs(lang: str, processors: list[str]) -> dict[str, Any]:
    """Build stanza.Pipeline kwargs for custom-trained models.

    Args:
        lang: ISO 639-3 language code.
        processors: List of processor names to include.

    Returns:
        Dict of keyword arguments for stanza.Pipeline.
    """
    model_dir = _get_custom_model_dir(lang)
    kwargs: dict[str, Any] = {
        "lang": _LANG_MAP[lang],
        "allow_unknown_language": True,
    }
    stanza_procs: list[str] = []
    for proc_name in processors:
        if proc_name in _CUSTOM_MODEL_PATH_KWARGS:
            kwarg_name, filename = _CUSTOM_MODEL_PATH_KWARGS[proc_name]
            model_path = model_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Custom Stanza model not found: {model_path}. "
                    f"Run: turkicnlp.download('{lang}')"
                )
            kwargs[kwarg_name] = str(model_path)
            stanza_procs.append(proc_name)
        if proc_name in _CUSTOM_PRETRAIN_KWARGS:
            pretrain_path = model_dir / "pretrain.pt"
            if not pretrain_path.exists():
                raise FileNotFoundError(
                    f"Custom Stanza pretrain not found: {pretrain_path}. "
                    f"Run: turkicnlp.download('{lang}')"
                )
            kwargs[_CUSTOM_PRETRAIN_KWARGS[proc_name]] = str(pretrain_path)
    kwargs["processors"] = ",".join(stanza_procs)
    return kwargs


def download_stanza_model(lang: str) -> None:
    """Download Stanza models for a Turkic language.

    For custom-trained models (e.g. Uzbek), this is a no-op since their
    download is handled by :func:`_download_stanza_custom_model` in the
    downloader module.

    Args:
        lang: ISO 639-3 language code (e.g. ``tur``, ``kaz``).
    """
    if _is_custom_stanza(lang):
        return
    stanza = _require_stanza()
    stanza_lang = _get_stanza_lang(lang)
    stanza_dir = ModelRegistry.default_dir() / "stanza"
    try:
        stanza.download(stanza_lang, model_dir=str(stanza_dir))
    except TypeError:
        stanza.download(stanza_lang, dir=str(stanza_dir))


# ---------------------------------------------------------------------------
# Shared pipeline manager
# ---------------------------------------------------------------------------


class _StanzaManager:
    """Manages shared Stanza pipeline instances and per-document caching.

    Pipelines are created lazily and cached by ``(language, use_gpu)`` key.
    Stanza results are cached on :class:`Document` objects to avoid redundant
    processing when multiple adapter processors extract from the same run.

    Two pipeline modes:

    * **Full mode**: Stanza handles tokenization. Used when
      ``StanzaTokenizer`` is in the pipeline.
    * **Pretokenized mode**: Input is already tokenized by a non-Stanza
      tokenizer. Stanza runs POS/lemma/depparse on the given tokens.
    """

    _full_pipelines: ClassVar[dict[tuple, Any]] = {}
    _pretok_pipelines: ClassVar[dict[tuple, Any]] = {}
    _full_ner_pipelines: ClassVar[dict[tuple, Any]] = {}
    _pretok_ner_pipelines: ClassVar[dict[tuple, Any]] = {}

    @classmethod
    def clear(cls) -> None:
        """Clear all cached pipelines (useful for testing)."""
        cls._full_pipelines.clear()
        cls._pretok_pipelines.clear()
        cls._full_ner_pipelines.clear()
        cls._pretok_ner_pipelines.clear()

    @classmethod
    def get_full_pipeline(cls, lang: str, use_gpu: bool = False) -> Any:
        """Get or create a full Stanza pipeline (with neural tokenization).

        Uses Stanza's default processor set for the language, which typically
        includes tokenize, mwt (if applicable), pos, lemma, and depparse.
        For custom-trained models, uses explicit model paths.
        """
        key = (lang, use_gpu)
        if key not in cls._full_pipelines:
            stanza = _require_stanza()
            stanza_lang = _get_stanza_lang(lang)

            if _is_custom_stanza(lang):
                kwargs = _build_custom_kwargs(
                    lang, ["tokenize", "pos", "lemma", "depparse"]
                )
                logger.info(
                    "Creating custom Stanza pipeline for '%s'", stanza_lang
                )
                # Use ERROR level during construction to suppress Stanza's
                # "unsupported language" warning, then restore to WARNING.
                cls._full_pipelines[key] = stanza.Pipeline(
                    use_gpu=use_gpu,
                    logging_level="ERROR",
                    **kwargs,
                )
                logging.getLogger("stanza").setLevel(logging.WARNING)
            else:
                stanza_dir = ModelRegistry.default_dir() / "stanza"
                try:
                    stanza.download(
                        stanza_lang, model_dir=str(stanza_dir), logging_level="WARNING"
                    )
                except TypeError:
                    stanza.download(stanza_lang, dir=str(stanza_dir), logging_level="WARNING")
                logger.info("Creating Stanza pipeline for '%s'", stanza_lang)
                try:
                    cls._full_pipelines[key] = stanza.Pipeline(
                        stanza_lang,
                        model_dir=str(stanza_dir),
                        use_gpu=use_gpu,
                        logging_level="WARNING",
                    )
                except TypeError:
                    cls._full_pipelines[key] = stanza.Pipeline(
                        stanza_lang,
                        dir=str(stanza_dir),
                        use_gpu=use_gpu,
                        logging_level="WARNING",
                    )
        return cls._full_pipelines[key]

    @classmethod
    def get_pretokenized_pipeline(cls, lang: str, use_gpu: bool = False) -> Any:
        """Get or create a pretokenized Stanza pipeline.

        Runs POS tagging, lemmatization, and dependency parsing on
        pre-tokenized input from a non-Stanza tokenizer.
        """
        key = (lang, use_gpu)
        if key not in cls._pretok_pipelines:
            stanza = _require_stanza()
            stanza_lang = _get_stanza_lang(lang)

            if _is_custom_stanza(lang):
                kwargs = _build_custom_kwargs(
                    lang, ["tokenize", "pos", "lemma", "depparse"]
                )
                kwargs["tokenize_pretokenized"] = True
                logger.info(
                    "Creating pretokenized custom Stanza pipeline for '%s'",
                    stanza_lang,
                )
                cls._pretok_pipelines[key] = stanza.Pipeline(
                    use_gpu=use_gpu,
                    logging_level="ERROR",
                    **kwargs,
                )
                logging.getLogger("stanza").setLevel(logging.WARNING)
            else:
                stanza_dir = ModelRegistry.default_dir() / "stanza"
                try:
                    stanza.download(
                        stanza_lang, model_dir=str(stanza_dir), logging_level="WARNING"
                    )
                except TypeError:
                    stanza.download(stanza_lang, dir=str(stanza_dir), logging_level="WARNING")
                logger.info(
                    "Creating pretokenized Stanza pipeline for '%s'", stanza_lang
                )
                try:
                    cls._pretok_pipelines[key] = stanza.Pipeline(
                        stanza_lang,
                        processors="tokenize,pos,lemma,depparse",
                        tokenize_pretokenized=True,
                        model_dir=str(stanza_dir),
                        use_gpu=use_gpu,
                        logging_level="WARNING",
                    )
                except TypeError:
                    cls._pretok_pipelines[key] = stanza.Pipeline(
                        stanza_lang,
                        processors="tokenize,pos,lemma,depparse",
                        tokenize_pretokenized=True,
                        dir=str(stanza_dir),
                        use_gpu=use_gpu,
                        logging_level="WARNING",
                    )
        return cls._pretok_pipelines[key]

    @classmethod
    def get_full_ner_pipeline(cls, lang: str, use_gpu: bool = False) -> Any:
        """Get or create a full Stanza pipeline for NER."""
        if _is_custom_stanza(lang):
            raise ValueError(
                f"NER is not available for custom Stanza language '{lang}'. "
                "No NER model has been trained for this language yet."
            )
        key = (lang, use_gpu)
        if key not in cls._full_ner_pipelines:
            stanza = _require_stanza()
            stanza_lang = _get_stanza_lang(lang)
            stanza_dir = ModelRegistry.default_dir() / "stanza"
            try:
                stanza.download(
                    stanza_lang, model_dir=str(stanza_dir), logging_level="WARNING"
                )
            except TypeError:
                stanza.download(stanza_lang, dir=str(stanza_dir), logging_level="WARNING")
            logger.info("Creating Stanza NER pipeline for '%s'", stanza_lang)
            try:
                cls._full_ner_pipelines[key] = stanza.Pipeline(
                    stanza_lang,
                    processors="tokenize,ner",
                    model_dir=str(stanza_dir),
                    use_gpu=use_gpu,
                    logging_level="WARNING",
                )
            except TypeError:
                cls._full_ner_pipelines[key] = stanza.Pipeline(
                    stanza_lang,
                    processors="tokenize,ner",
                    dir=str(stanza_dir),
                    use_gpu=use_gpu,
                    logging_level="WARNING",
                )
        return cls._full_ner_pipelines[key]

    @classmethod
    def get_pretokenized_ner_pipeline(cls, lang: str, use_gpu: bool = False) -> Any:
        """Get or create a pretokenized Stanza pipeline for NER."""
        if _is_custom_stanza(lang):
            raise ValueError(
                f"NER is not available for custom Stanza language '{lang}'. "
                "No NER model has been trained for this language yet."
            )
        key = (lang, use_gpu)
        if key not in cls._pretok_ner_pipelines:
            stanza = _require_stanza()
            stanza_lang = _get_stanza_lang(lang)
            stanza_dir = ModelRegistry.default_dir() / "stanza"
            try:
                stanza.download(
                    stanza_lang, model_dir=str(stanza_dir), logging_level="WARNING"
                )
            except TypeError:
                stanza.download(stanza_lang, dir=str(stanza_dir), logging_level="WARNING")
            logger.info(
                "Creating pretokenized Stanza NER pipeline for '%s'", stanza_lang
            )
            try:
                cls._pretok_ner_pipelines[key] = stanza.Pipeline(
                    stanza_lang,
                    processors="tokenize,ner",
                    tokenize_pretokenized=True,
                    model_dir=str(stanza_dir),
                    use_gpu=use_gpu,
                    logging_level="WARNING",
                )
            except TypeError:
                cls._pretok_ner_pipelines[key] = stanza.Pipeline(
                    stanza_lang,
                    processors="tokenize,ner",
                    tokenize_pretokenized=True,
                    dir=str(stanza_dir),
                    use_gpu=use_gpu,
                    logging_level="WARNING",
                )
        return cls._pretok_ner_pipelines[key]

    @classmethod
    def run_full(cls, doc: Document, use_gpu: bool = False) -> Any:
        """Run full Stanza pipeline on document text, with caching."""
        cached = getattr(doc, "_stanza_full_cache", None)
        if cached is not None:
            return cached
        pipeline = cls.get_full_pipeline(doc.lang, use_gpu)
        result = pipeline(doc.text)
        doc._stanza_full_cache = result  # type: ignore[attr-defined]
        return result

    @classmethod
    def run_pretokenized(cls, doc: Document, use_gpu: bool = False) -> Any:
        """Run Stanza on pre-tokenized input, with caching."""
        cached = getattr(doc, "_stanza_pretok_cache", None)
        if cached is not None:
            return cached
        pipeline = cls.get_pretokenized_pipeline(doc.lang, use_gpu)
        tokens = [[w.text for w in sent.words] for sent in doc.sentences]
        result = pipeline(tokens)
        doc._stanza_pretok_cache = result  # type: ignore[attr-defined]
        return result

    @classmethod
    def run_full_ner(cls, doc: Document, use_gpu: bool = False) -> Any:
        """Run full Stanza NER pipeline on document text, with caching."""
        cached = getattr(doc, "_stanza_full_ner_cache", None)
        if cached is not None:
            return cached
        pipeline = cls.get_full_ner_pipeline(doc.lang, use_gpu)
        result = pipeline(doc.text)
        doc._stanza_full_ner_cache = result  # type: ignore[attr-defined]
        return result

    @classmethod
    def run_pretokenized_ner(cls, doc: Document, use_gpu: bool = False) -> Any:
        """Run Stanza NER on pre-tokenized input, with caching."""
        cached = getattr(doc, "_stanza_pretok_ner_cache", None)
        if cached is not None:
            return cached
        pipeline = cls.get_pretokenized_ner_pipeline(doc.lang, use_gpu)
        tokens = [[w.text for w in sent.words] for sent in doc.sentences]
        result = pipeline(tokens)
        doc._stanza_pretok_ner_cache = result  # type: ignore[attr-defined]
        return result


def _run_stanza(doc: Document, use_gpu: bool = False) -> Any:
    """Run Stanza in the appropriate mode based on tokenizer provenance.

    If the document was tokenized by ``StanzaTokenizer`` (full mode),
    reuses the cached full-pipeline result. Otherwise, runs Stanza in
    pretokenized mode on the existing tokens.
    """
    if "tokenize:stanza" in doc._processor_log:
        return _StanzaManager.run_full(doc, use_gpu)
    else:
        return _StanzaManager.run_pretokenized(doc, use_gpu)


def _run_stanza_ner(doc: Document, use_gpu: bool = False) -> Any:
    """Run Stanza NER in full or pretokenized mode based on provenance."""
    if "tokenize:stanza" in doc._processor_log:
        return _StanzaManager.run_full_ner(doc, use_gpu)
    else:
        return _StanzaManager.run_pretokenized_ner(doc, use_gpu)


def _bioes_to_bio(tag: Optional[str]) -> str:
    """Normalize BIOES or other Stanza tags to BIO."""
    if not tag or tag == "O":
        return "O"
    if "-" not in tag:
        return "O"
    prefix, label = tag.split("-", 1)
    if prefix == "S":
        return f"B-{label}"
    if prefix == "E":
        return f"I-{label}"
    if prefix in {"B", "I"}:
        return f"{prefix}-{label}"
    return "O"


def _bio_to_spans(words: list[Word], tags: list[str]) -> list[Span]:
    """Convert BIO tags to entity spans."""
    spans: list[Span] = []
    current: Optional[Span] = None
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current is not None:
                spans.append(current)
            current = Span(
                text=word.text,
                type=tag[2:],
                start_char=word.start_char or 0,
                end_char=word.end_char or 0,
                words=[word],
            )
            continue

        if tag.startswith("I-") and current is not None:
            current.text += f" {word.text}"
            current.end_char = word.end_char or current.end_char
            current.words.append(word)
            continue

        if current is not None:
            spans.append(current)
            current = None

    if current is not None:
        spans.append(current)
    return spans


# ---------------------------------------------------------------------------
# Adapter processors
# ---------------------------------------------------------------------------


class StanzaTokenizer(Processor):
    """Stanza-backed tokenizer with sentence splitting and MWT expansion.

    Uses Stanza's neural tokenizer trained on UD treebank data. Handles
    sentence segmentation, tokenization, and multi-word token expansion
    in a single pass.
    """

    NAME = "tokenize"
    PROVIDES = ["tokenize"]
    REQUIRES = []

    def load(self, model_path: str = "") -> None:
        """Validate language support and ensure Stanza models are available."""
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._use_gpu = self.config.get("use_gpu", False)
        _StanzaManager.get_full_pipeline(self.lang, self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Tokenize text using Stanza, creating sentences, tokens, and words."""
        stanza_doc = _StanzaManager.run_full(doc, self._use_gpu)

        doc.sentences = []
        for stanza_sent in stanza_doc.sentences:
            sent = Sentence(text=stanza_sent.text)
            word_id = 1

            for stanza_token in stanza_sent.tokens:
                if len(stanza_token.words) > 1:
                    # Multi-word token
                    mwt_start = word_id
                    mwt_end = word_id + len(stanza_token.words) - 1
                    words: list[Word] = []
                    for stanza_word in stanza_token.words:
                        w = Word(
                            id=word_id,
                            text=stanza_word.text,
                            start_char=stanza_token.start_char,
                            end_char=stanza_token.end_char,
                        )
                        words.append(w)
                        sent.words.append(w)
                        word_id += 1

                    token = Token(
                        id=(mwt_start, mwt_end),
                        text=stanza_token.text,
                        words=words,
                        start_char=stanza_token.start_char,
                        end_char=stanza_token.end_char,
                    )
                    sent.tokens.append(token)
                else:
                    # Regular token (single word)
                    stanza_word = stanza_token.words[0]
                    w = Word(
                        id=word_id,
                        text=stanza_word.text,
                        start_char=stanza_token.start_char,
                        end_char=stanza_token.end_char,
                    )
                    token = Token(
                        id=(word_id,),
                        text=stanza_word.text,
                        words=[w],
                        start_char=stanza_token.start_char,
                        end_char=stanza_token.end_char,
                    )
                    sent.tokens.append(token)
                    sent.words.append(w)
                    word_id += 1

            doc.sentences.append(sent)

        doc._processor_log.append("tokenize:stanza")
        return doc


class StanzaMWTExpander(Processor):
    """Stanza-backed multi-word token expander.

    When used with ``StanzaTokenizer``, MWT expansion is already handled
    during tokenization. This processor is a pass-through in that case.
    In mixed mode (non-Stanza tokenizer), this is a no-op since
    pre-tokenized input does not contain MWTs.
    """

    NAME = "mwt"
    PROVIDES = ["mwt"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        doc._processor_log.append("mwt:stanza")
        return doc


class StanzaPOSTagger(Processor):
    """Stanza-backed POS tagger with UPOS, XPOS, and morphological features.

    Uses Stanza's neural tagger trained on UD treebank data. Supports
    both full mode (Stanza tokenizer) and pretokenized mode (any tokenizer).
    """

    NAME = "pos"
    PROVIDES = ["pos", "feats"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._use_gpu = self.config.get("use_gpu", False)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        stanza_doc = _run_stanza(doc, self._use_gpu)

        for sent, stanza_sent in zip(doc.sentences, stanza_doc.sentences):
            for word, stanza_word in zip(sent.words, stanza_sent.words):
                word.upos = stanza_word.upos
                word.xpos = getattr(stanza_word, "xpos", None)
                word.feats = stanza_word.feats if stanza_word.feats else None

        doc._processor_log.append("pos:stanza")
        return doc


class StanzaLemmatizer(Processor):
    """Stanza-backed lemmatizer.

    Uses Stanza's neural lemmatizer trained on UD treebank data.
    """

    NAME = "lemma"
    PROVIDES = ["lemma"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str = "") -> None:
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._use_gpu = self.config.get("use_gpu", False)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        stanza_doc = _run_stanza(doc, self._use_gpu)

        for sent, stanza_sent in zip(doc.sentences, stanza_doc.sentences):
            for word, stanza_word in zip(sent.words, stanza_sent.words):
                word.lemma = stanza_word.lemma

        doc._processor_log.append("lemma:stanza")
        return doc


class StanzaDepParser(Processor):
    """Stanza-backed dependency parser.

    Uses Stanza's biaffine attention parser trained on UD treebank data.
    Predicts head word indices and dependency relation labels.
    """

    NAME = "depparse"
    PROVIDES = ["depparse"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str = "") -> None:
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._use_gpu = self.config.get("use_gpu", False)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        stanza_doc = _run_stanza(doc, self._use_gpu)

        for sent, stanza_sent in zip(doc.sentences, stanza_doc.sentences):
            for word, stanza_word in zip(sent.words, stanza_sent.words):
                word.head = stanza_word.head
                word.deprel = stanza_word.deprel

        doc._processor_log.append("depparse:stanza")
        return doc


class StanzaNERProcessor(Processor):
    """Stanza-backed named entity recognizer."""

    NAME = "ner"
    PROVIDES = ["ner"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        _require_stanza()
        _get_stanza_lang(self.lang)
        self._use_gpu = self.config.get("use_gpu", False)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        stanza_doc = _run_stanza_ner(doc, self._use_gpu)

        for sent, stanza_sent in zip(doc.sentences, stanza_doc.sentences):
            tags: list[str] = []
            for stanza_token in stanza_sent.tokens:
                tag = _bioes_to_bio(getattr(stanza_token, "ner", None))
                if len(stanza_token.words) > 1:
                    if tag.startswith("B-"):
                        tags.append(tag)
                        tags.extend(f"I-{tag[2:]}" for _ in stanza_token.words[1:])
                    else:
                        tags.extend([tag] * len(stanza_token.words))
                else:
                    tags.append(tag)

            tags = tags[: len(sent.words)]
            if len(tags) < len(sent.words):
                tags.extend(["O"] * (len(sent.words) - len(tags)))

            for word, tag in zip(sent.words, tags):
                word.ner = tag
            sent.entities = _bio_to_spans(sent.words, tags)

        doc._processor_log.append("ner:stanza")
        return doc
