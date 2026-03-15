"""
Multilingual Glot500-based POS tagger and dependency parser backend.

Uses a single TurkicParser model (trained on 10 Turkic languages with
shared BiLSTM + biaffine parser over a frozen Glot500 backbone) for
joint POS tagging and dependency parsing.

Supports:
  - All 10 trained languages (tur, aze, uzb, tuk, kaz, kir, bak, tat, uig, ota)
  - Zero-shot inference on unseen languages (kaa, kum, sah) via proxy embeddings

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

CHECKPOINT_URL = (
    "https://github.com/turkic-nlp/trained-stanza-models/releases/download/v0.1.5/multilingual_pos_depparse_glot500.pt"
)
CHECKPOINT_SHA256 = "aef71dbc01dd68c81e672fea1d38f4509e343f8bc72714f68ed0f458ec29853d"
CHECKPOINT_FILENAME = "multilingual_pos_depparse_glot500.pt"
BACKBONE_HF_NAME = "cis-lmu/glot500-base"


# ---------------------------------------------------------------------------
# Download helpers
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


def _ensure_checkpoint(base_dir: Path) -> Path:
    """Download the model checkpoint if it doesn't exist or is corrupted."""
    dest_dir = base_dir / "multilingual"
    dest_file = dest_dir / CHECKPOINT_FILENAME
    if dest_file.exists():
        actual_sha = _sha256(dest_file)
        if actual_sha == CHECKPOINT_SHA256:
            return dest_file
        logger.warning(
            "Checkpoint SHA-256 mismatch (corrupted?), re-downloading: %s",
            dest_file,
        )
        dest_file.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading multilingual Glot500 checkpoint...")
    urllib.request.urlretrieve(CHECKPOINT_URL, dest_file, reporthook=_progress_hook)
    actual_sha = _sha256(dest_file)
    if actual_sha != CHECKPOINT_SHA256:
        dest_file.unlink()
        raise ValueError(
            f"SHA-256 mismatch for {dest_file}: "
            f"expected {CHECKPOINT_SHA256}, got {actual_sha}"
        )
    return dest_file


def _ensure_backbone(base_dir: Path) -> Path:
    """Download the Glot500 backbone from HuggingFace if needed. Returns path."""
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
            # Download and save directly into shared_dir (not as HF cache)
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
# Model manager (singleton pattern, like _StanzaManager)
# ---------------------------------------------------------------------------


class _MultilingualParserManager:
    """Manages shared TurkicParser model loading and inference caching.

    The model is loaded once and cached by (device,) key. Inference results
    are cached on the Document to avoid redundant forward passes when both
    the POS tagger and dep parser processors access the same document.
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
        checkpoint_path = _ensure_checkpoint(base_dir)
        backbone_dir = _ensure_backbone(base_dir)
        return checkpoint_path, backbone_dir

    @classmethod
    def get_model(cls, use_gpu: bool = False) -> Any:
        """Get or create the TurkicParser model.

        The model is shared across all languages — only the lang_id and
        script routing differ per call.
        """
        device = _resolve_device(use_gpu)
        key = (str(device),)
        if key not in cls._models:
            from turkicnlp.processors.multilingual_model import TurkicParser

            checkpoint_path, backbone_dir = cls.ensure_downloaded()

            logger.info("Loading multilingual TurkicParser on %s", device)
            model = TurkicParser(hf_cache_dir=str(backbone_dir))
            state = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if "trained_state_dict" in state:
                model.load_state_dict(state["trained_state_dict"], strict=False)
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            model.to(device)
            model.eval()
            cls._models[key] = (model, device)
        return cls._models[key]

    @classmethod
    def run(cls, doc: Document, use_gpu: bool = False) -> dict:
        """Run inference on a Document, caching results.

        Returns a dict with keys:
            - "pos_tags": list[list[str]]  (per sentence, per word)
            - "heads": list[list[int]]
            - "deprels": list[list[str]]
        """
        cached = getattr(doc, "_multilingual_cache", None)
        if cached is not None:
            return cached

        from turkicnlp.processors.multilingual_model import (
            resolve_lang, tokenize_words,
            UPOS_TAGS, DEPREL_TAGS, LANG_ID_MAP,
        )

        model, device = cls.get_model(use_gpu)
        model.eval()  # Defensive: ensure no dropout during inference

        short_code, script = resolve_lang(doc.lang)
        lang_id = LANG_ID_MAP[short_code]

        result: dict[str, list] = {"pos_tags": [], "heads": [], "deprels": []}

        # Process each sentence
        for sentence in doc.sentences:
            words = [w.text for w in sentence.words]
            if not words:
                result["pos_tags"].append([])
                result["heads"].append([])
                result["deprels"].append([])
                continue

            input_ids, attention_mask, word_starts, word_lengths = tokenize_words(
                model.tokenizer, [words], device,
            )
            lang_ids = torch.tensor([lang_id], dtype=torch.long, device=device)

            with torch.no_grad():
                pos_logits, arc_logits, label_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_starts=word_starts,
                    word_lengths=word_lengths,
                    lang_ids=lang_ids,
                    script=script,
                )

            n = len(words)
            pos_ids = pos_logits[0, :n].argmax(-1).tolist()
            head_ids = arc_logits[0, :n].argmax(-1).tolist()
            label_ids = [
                label_logits[0, i, head_ids[i]].argmax(-1).item()
                for i in range(n)
            ]

            result["pos_tags"].append([UPOS_TAGS[p] for p in pos_ids])
            result["heads"].append(head_ids)
            result["deprels"].append([DEPREL_TAGS[lid] for lid in label_ids])

        doc._multilingual_cache = result  # type: ignore[attr-defined]
        return result


# ---------------------------------------------------------------------------
# Processor implementations
# ---------------------------------------------------------------------------


class MultilingualPOSTagger(Processor):
    """POS tagger using the multilingual Glot500 TurkicParser.

    Supports 10 trained Turkic languages and zero-shot inference on
    Karakalpak, Kumyk, and Sakha via proxy language embeddings.
    """

    NAME = "pos"
    PROVIDES = ["pos"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str = "") -> None:
        self._use_gpu = self.config.get("use_gpu", False)
        _MultilingualParserManager.ensure_downloaded()
        _MultilingualParserManager.get_model(self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        result = _MultilingualParserManager.run(doc, self._use_gpu)

        for sent, pos_tags in zip(doc.sentences, result["pos_tags"]):
            for word, tag in zip(sent.words, pos_tags):
                word.upos = tag

        doc._processor_log.append("pos:multilingual_glot500_model")
        return doc


class MultilingualDepParser(Processor):
    """Dependency parser using the multilingual Glot500 TurkicParser.

    Supports 10 trained Turkic languages and zero-shot inference on
    Karakalpak, Kumyk, and Sakha via proxy language embeddings.
    """

    NAME = "depparse"
    PROVIDES = ["depparse"]
    REQUIRES = ["tokenize", "pos"]

    def load(self, model_path: str = "") -> None:
        self._use_gpu = self.config.get("use_gpu", False)
        _MultilingualParserManager.ensure_downloaded()
        _MultilingualParserManager.get_model(self._use_gpu)
        self._loaded = True

    def process(self, doc: Document) -> Document:
        self.check_requirements(doc)
        result = _MultilingualParserManager.run(doc, self._use_gpu)

        for sent, heads, deprels in zip(
            doc.sentences, result["heads"], result["deprels"]
        ):
            for word, head, deprel in zip(sent.words, heads, deprels):
                word.head = head
                word.deprel = deprel

        doc._processor_log.append("depparse:multilingual_glot500_model")
        return doc
