"""
NLLB-based sentence/document embedding processor.
"""

from __future__ import annotations

import json
from pathlib import Path

from turkicnlp.models.document import Document, Sentence
from turkicnlp.processors.base import Processor

_NLLB_LANG_MAP: dict[str, str] = {
    "aze": "azj_Latn",
    "azb": "azb_Arab",
    "bak": "bak_Cyrl",
    "crh": "crh_Latn",
    "kaz": "kaz_Cyrl",
    "kir": "kir_Cyrl",
    "tat": "tat_Cyrl",
    "tuk": "tuk_Latn",
    "tur": "tur_Latn",
    "uig": "uig_Arab",
    "uzb": "uzn_Latn",
}

_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


class NLLBEmbeddingsProcessor(Processor):
    """Generate sentence and document embeddings using NLLB encoder states."""

    NAME = "embeddings"
    PROVIDES = ["embeddings"]
    REQUIRES = []

    def load(self, model_path: str) -> None:
        """Load tokenizer + seq2seq model and keep encoder for embeddings."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM
            try:
                from transformers import NllbTokenizer as _NLLBTokenizer
            except ImportError:
                from transformers import AutoTokenizer as _NLLBTokenizer
        except ImportError as exc:
            raise ImportError(
                "NLLB embeddings require `transformers` and `torch`. "
                "Install with: pip install turkicnlp[transformers,torch]"
            ) from exc

        self._torch = torch
        self._normalize = bool(self.config.get("normalize", True))

        model_name = self.config.get("model_name", _DEFAULT_MODEL)
        src_lang = self.config.get("src_lang") or _NLLB_LANG_MAP.get(self.lang)
        if not src_lang:
            raise ValueError(
                f"No NLLB language mapping found for '{self.lang}'. "
                "Set `embeddings_src_lang=...` in Pipeline config."
            )

        local_model_dir = self._resolve_local_model_dir(model_name, Path(model_path))
        load_from = str(local_model_dir) if local_model_dir else model_name

        print(
            f"  → Loading NLLB model '{model_name}' for embeddings from {load_from}"
        )
        self._tokenizer = _NLLBTokenizer.from_pretrained(load_from, src_lang=src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(load_from)
        self._encoder = self._model.get_encoder()
        self._src_lang = str(src_lang)
        self._loaded = True

    def _resolve_local_model_dir(self, model_name: str, model_path: Path) -> Path | None:
        # Primary location for NLLB in TurkicNLP: shared Hugging Face cache.
        try:
            from turkicnlp.resources.registry import ModelRegistry

            shared = (
                ModelRegistry.default_dir() / "huggingface" / str(model_name).replace("/", "--")
            )
            if (shared / "config.json").exists():
                return shared
        except Exception:
            pass

        # Backward compatibility with previously downloaded per-language metadata.
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            shared = metadata.get("shared_model_dir")
            if shared and Path(shared).exists():
                return Path(shared)
            local = metadata.get("local_model_dir")
            if local and Path(local).exists():
                return Path(local)
        model_subdir = model_path / "model"
        if model_subdir.exists():
            return model_subdir
        return None

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with self._torch.no_grad():
            out = self._encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
            hidden = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            if self._normalize:
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [[float(x) for x in row] for row in pooled.tolist()]

    def process(self, doc: Document) -> Document:
        """Attach sentence-level and document-level embeddings to the Document."""
        if not self._loaded:
            raise RuntimeError("NLLBEmbeddingsProcessor must be loaded before use.")

        if doc.sentences:
            texts = [s.text for s in doc.sentences]
        else:
            texts = [doc.text]
            doc.sentences = [Sentence(text=doc.text)]

        sent_vectors = self._encode_texts(texts)
        for sent, vec in zip(doc.sentences, sent_vectors):
            sent.embedding = vec

        doc.embedding = self._encode_texts([doc.text])[0]
        doc._processor_log.append("embeddings:nllb")
        return doc
