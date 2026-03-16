"""
Multilingual Glot500-based morphological analyzer and lemmatizer backend.

Uses a TurkicMorphAnalyzer model (trained on 10 Turkic languages with
shared BiLSTM over a frozen Glot500 backbone) for joint:
  - UPOS tagging
  - UD morphological feature prediction (multi-label)
  - Lemmatization via edit-script classification

Supports:
  - All 10 trained languages (tur, aze, uzb, tuk, kaz, kir, bak, tat, uig, ota)
  - Zero-shot inference on unseen languages (kaa, kum, sah, krc, nog) via proxy embeddings

The checkpoint and Glot500 backbone are downloaded once and shared
across all languages.

Install with: ``pip install turkicnlp[transformers]``
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any, ClassVar, Optional

import torch

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor
from turkicnlp.resources.registry import ModelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MORPH_CHECKPOINT_URL = (
    "https://github.com/turkic-nlp/trained-stanza-models/releases/download/v0.1.6/multilingual_morph_glot500.pt"
)
MORPH_CHECKPOINT_SHA256 = (
    "d33e9cfca6263c6d99be18847c8214a42127babdf0c0d7f8098630444f1ba8fe"
)
MORPH_CHECKPOINT_FILENAME = "multilingual_morph_glot500.pt"
BACKBONE_HF_NAME = "cis-lmu/glot500-base"

# Morph feature threshold for sigmoid activation
FEAT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Download helpers (reuse backbone from parser if already downloaded)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    print(f"\r  {min(percent, 100)}%", end="", flush=True)
    if percent >= 100:
        print()


def _ensure_morph_checkpoint(base_dir: Path) -> Path:
    """Download the morph model checkpoint if needed."""
    dest_dir = base_dir / "multilingual"
    dest_file = dest_dir / MORPH_CHECKPOINT_FILENAME
    if dest_file.exists():
        actual_sha = _sha256(dest_file)
        if actual_sha == MORPH_CHECKPOINT_SHA256:
            return dest_file
        logger.warning(
            "Morph checkpoint SHA-256 mismatch (corrupted?), re-downloading: %s",
            dest_file,
        )
        dest_file.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("  ↓ Downloading multilingual Glot500 morph analyzer checkpoint...")
    urllib.request.urlretrieve(
        MORPH_CHECKPOINT_URL, dest_file, reporthook=_progress_hook
    )
    actual_sha = _sha256(dest_file)
    if actual_sha != MORPH_CHECKPOINT_SHA256:
        dest_file.unlink()
        raise ValueError(
            f"SHA-256 mismatch for {dest_file}: "
            f"expected {MORPH_CHECKPOINT_SHA256}, got {actual_sha}"
        )
    return dest_file


def _ensure_backbone(base_dir: Path) -> Path:
    """Download the Glot500 backbone from HuggingFace if needed.

    Shares the backbone with the parser model — if already downloaded,
    this is a no-op.
    """
    shared_dir = base_dir / "huggingface" / BACKBONE_HF_NAME.replace("/", "--")
    if (shared_dir / "config.json").exists():
        return shared_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"  ↓ Downloading {BACKBONE_HF_NAME} backbone...")
            shared_dir.mkdir(parents=True, exist_ok=True)
            AutoTokenizer.from_pretrained(BACKBONE_HF_NAME).save_pretrained(
                str(shared_dir)
            )
            AutoModel.from_pretrained(BACKBONE_HF_NAME).save_pretrained(
                str(shared_dir)
            )
            return shared_dir
        except ImportError:
            raise ImportError(
                "Downloading the Glot500 backbone requires `transformers` "
                "or `huggingface_hub`. "
                "Install with: pip install turkicnlp[transformers]"
            )
    print(f"  ↓ Downloading {BACKBONE_HF_NAME} backbone...")
    shared_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BACKBONE_HF_NAME,
        local_dir=str(shared_dir),
    )
    return shared_dir


def _resolve_device(use_gpu: bool) -> torch.device:
    """Resolve device: GPU if requested and available, else CPU."""
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model manager (singleton pattern)
# ---------------------------------------------------------------------------


class _MorphAnalyzerManager:
    """Manages shared TurkicMorphAnalyzer model loading and inference caching.

    The model is loaded once and cached by (device,) key. Inference results
    are cached on the Document to avoid redundant forward passes when multiple
    morph-related processors access the same document.
    """

    _models: ClassVar[dict[tuple, Any]] = {}

    @classmethod
    def clear(cls) -> None:
        """Clear all cached models (useful for testing)."""
        cls._models.clear()

    @classmethod
    def ensure_downloaded(cls) -> tuple[Path, Path]:
        """Ensure checkpoint and backbone are downloaded.

        Returns:
            (checkpoint_path, backbone_dir) tuple.
        """
        base_dir = ModelRegistry.default_dir()
        checkpoint_path = _ensure_morph_checkpoint(base_dir)
        backbone_dir = _ensure_backbone(base_dir)
        return checkpoint_path, backbone_dir

    @classmethod
    def get_model(cls, use_gpu: bool = False) -> Any:
        """Get or create the TurkicMorphAnalyzer model.

        The model is shared across all languages — only the lang_id and
        script routing differ per call.
        """
        device = _resolve_device(use_gpu)
        key = (str(device),)
        if key not in cls._models:
            from turkicnlp.processors.multilingual_morph_model import (
                TurkicMorphAnalyzer,
                EditScriptVocab,
            )

            checkpoint_path, backbone_dir = cls.ensure_downloaded()

            logger.info("Loading multilingual TurkicMorphAnalyzer on %s", device)
            state = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            # Load edit script vocabulary from checkpoint (required)
            edit_vocab = EditScriptVocab()
            if "edit_vocab" not in state or state["edit_vocab"] is None:
                raise ValueError(
                    "Morph checkpoint is missing 'edit_vocab'. "
                    "The checkpoint may be corrupted or from an incompatible version."
                )
            edit_vocab.load_state_dict(state["edit_vocab"])
            num_edit_scripts = len(edit_vocab)

            model = TurkicMorphAnalyzer(
                hf_cache_dir=str(backbone_dir),
                num_edit_scripts=num_edit_scripts,
            )
            if "trained_state_dict" in state:
                missing, unexpected = model.load_state_dict(
                    state["trained_state_dict"], strict=False
                )
                # Backbone keys are expected to be missing (frozen, not saved)
                non_backbone_missing = [
                    k for k in missing if not k.startswith("backbone.")
                ]
                if non_backbone_missing:
                    logger.warning(
                        "Morph model missing non-backbone keys: %s",
                        non_backbone_missing,
                    )
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            model.to(device)
            model.eval()
            cls._models[key] = (model, device, edit_vocab)
        return cls._models[key]

    @classmethod
    def run(cls, doc: Document, use_gpu: bool = False) -> dict:
        """Run morph inference on a Document, caching results.

        Returns a dict with keys:
            - "pos_tags": list[list[str]]   (per sentence, per word)
            - "feats": list[list[str]]      (per sentence, per word — UD feat strings)
            - "lemmas": list[list[str]]     (per sentence, per word)
        """
        cached = getattr(doc, "_morph_analyzer_cache", None)
        if cached is not None:
            return cached

        from turkicnlp.processors.multilingual_morph_model import (
            resolve_morph_lang,
            tokenize_words,
            encode_chars,
            apply_edit_script,
            UPOS_TAGS,
            UD_MORPH_FEATS,
            LANG_ID_MAP,
        )

        model, device, edit_vocab = cls.get_model(use_gpu)
        model.eval()

        short_code, script = resolve_morph_lang(doc.lang)
        lang_id = LANG_ID_MAP[short_code]

        result: dict[str, list] = {"pos_tags": [], "feats": [], "lemmas": []}

        for sentence in doc.sentences:
            words = [w.text for w in sentence.words]
            if not words:
                result["pos_tags"].append([])
                result["feats"].append([])
                result["lemmas"].append([])
                continue

            input_ids, attention_mask, word_starts, word_lengths = tokenize_words(
                model.tokenizer, [words], device,
            )
            lang_ids = torch.tensor([lang_id], dtype=torch.long, device=device)

            # Encode characters for the lemma head
            n = len(words)
            char_ids = encode_chars([words], max_word_len=n).to(device)

            with torch.no_grad():
                pos_logits, feat_logits, edit_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_starts=word_starts,
                    word_lengths=word_lengths,
                    lang_ids=lang_ids,
                    script=script,
                    char_ids=char_ids,
                )

            # Decode POS
            pos_ids = pos_logits[0, :n].argmax(-1).tolist()
            result["pos_tags"].append([UPOS_TAGS[p] for p in pos_ids])

            # Decode morph features (multi-label sigmoid)
            feat_probs = torch.sigmoid(feat_logits[0, :n])  # [n, NUM_MORPH_FEATS]
            sent_feats = []
            for i in range(n):
                active = (feat_probs[i] > FEAT_THRESHOLD).nonzero(as_tuple=True)[0]
                if len(active) == 0:
                    sent_feats.append("_")
                else:
                    feats = sorted(
                        [UD_MORPH_FEATS[idx.item()] for idx in active]
                    )
                    sent_feats.append("|".join(feats))
            result["feats"].append(sent_feats)

            # Decode lemmas via edit scripts
            edit_ids = edit_logits[0, :n].argmax(-1).tolist()
            sent_lemmas = []
            for i, eid in enumerate(edit_ids):
                edit_str = edit_vocab.decode(eid)
                if edit_str == "<unk>":
                    # Fallback: identity (lowercase)
                    sent_lemmas.append(words[i].lower())
                else:
                    sent_lemmas.append(apply_edit_script(words[i], edit_str))
            result["lemmas"].append(sent_lemmas)

        doc._morph_analyzer_cache = result  # type: ignore[attr-defined]
        return result


# ---------------------------------------------------------------------------
# Processor implementations
# ---------------------------------------------------------------------------


class MultilingualMorphAnalyzer(Processor):
    """Neural morphological analyzer using the multilingual Glot500 model.

    Provides UPOS, morphological features, and lemmas in a single forward pass.
    Supports 10 trained Turkic languages and zero-shot inference on
    Karakalpak, Kumyk, Sakha, Karachay-Balkar, and Nogai via proxy language embeddings.
    """

    NAME = "morph_neural"
    PROVIDES = ["pos", "feats", "lemma"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        self._use_gpu = self.config.get("use_gpu", False)
        _MorphAnalyzerManager.ensure_downloaded()
        _MorphAnalyzerManager.get_model(self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        result = _MorphAnalyzerManager.run(doc, self._use_gpu)

        for sent, pos_tags, feats, lemmas in zip(
            doc.sentences,
            result["pos_tags"],
            result["feats"],
            result["lemmas"],
        ):
            for word, tag, feat, lemma in zip(sent.words, pos_tags, feats, lemmas):
                word.upos = tag
                word.feats = feat
                word.lemma = lemma

        doc._processor_log.append("morph_neural:multilingual_glot500_morph")
        return doc


class MultilingualMorphFeats(Processor):
    """Morphological feature predictor using the multilingual Glot500 morph model.

    Only sets feats (not POS or lemma). Useful when POS is provided by
    another processor and only morph features are needed.
    """

    NAME = "feats"
    PROVIDES = ["feats"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        self._use_gpu = self.config.get("use_gpu", False)
        _MorphAnalyzerManager.ensure_downloaded()
        _MorphAnalyzerManager.get_model(self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        result = _MorphAnalyzerManager.run(doc, self._use_gpu)

        for sent, feats in zip(doc.sentences, result["feats"]):
            for word, feat in zip(sent.words, feats):
                word.feats = feat

        doc._processor_log.append("feats:multilingual_glot500_morph")
        return doc


class MultilingualMorphLemmatizer(Processor):
    """Lemmatizer using the multilingual Glot500 morph model.

    Only sets lemma. Useful when POS/feats are provided by other processors.
    """

    NAME = "lemma"
    PROVIDES = ["lemma"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        self._use_gpu = self.config.get("use_gpu", False)
        _MorphAnalyzerManager.ensure_downloaded()
        _MorphAnalyzerManager.get_model(self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        result = _MorphAnalyzerManager.run(doc, self._use_gpu)

        for sent, lemmas in zip(doc.sentences, result["lemmas"]):
            for word, lemma in zip(sent.words, lemmas):
                word.lemma = lemma

        doc._processor_log.append("lemma:multilingual_glot500_morph")
        return doc
