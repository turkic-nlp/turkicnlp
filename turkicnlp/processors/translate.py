"""
NLLB-based machine translation processor.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from turkicnlp.models.document import Document, Sentence
from turkicnlp.processors.base import Processor

_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
_NLLB_LANGS_CACHE: dict | None = None


def _load_nllb_langs() -> dict:
    global _NLLB_LANGS_CACHE
    if _NLLB_LANGS_CACHE is None:
        with resources.open_text("turkicnlp.resources", "nllb_flores200_languages.json") as f:
            _NLLB_LANGS_CACHE = json.load(f)
    return _NLLB_LANGS_CACHE


def _resolve_nllb_lang(value: str, param_name: str) -> str:
    langs = _load_nllb_langs()
    codes = set(langs.get("codes", []))
    by_iso3 = langs.get("by_iso3", {})

    if "_" in value:
        if value not in codes:
            raise ValueError(
                f"Unsupported {param_name}='{value}'. "
                "Use an NLLB code from FLORES-200 (e.g. tur_Latn, kaz_Cyrl, eng_Latn)."
            )
        return value

    entry = by_iso3.get(value)
    if not entry:
        raise ValueError(
            f"Unsupported {param_name}='{value}' for NLLB/FLORES-200. "
            "Use ISO-639-3 code (e.g. 'tuk', 'kaz', 'eng') or explicit NLLB code "
            "(e.g. 'tuk_Latn', 'kaz_Cyrl', 'eng_Latn')."
        )
    return str(entry["default"])


class NLLBTranslateProcessor(Processor):
    """Translate text with NLLB and attach outputs on sentence/document level."""

    NAME = "translate"
    PROVIDES = ["translation"]
    REQUIRES = []

    def load(self, model_path: str) -> None:
        """Load NLLB tokenizer + seq2seq model for generation."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "NLLB translation requires `transformers` and `torch`. "
                "Install with: pip install turkicnlp[transformers,torch]"
            ) from exc

        self._torch = torch
        self._max_new_tokens = int(self.config.get("max_new_tokens", 256))
        self._num_beams = int(self.config.get("num_beams", 4))

        model_name = self.config.get("model_name", _DEFAULT_MODEL)
        src_lang_raw = str(self.config.get("src_lang", self.lang))
        tgt_lang_raw = str(self.config.get("tgt_lang", "eng"))
        src_lang = _resolve_nllb_lang(src_lang_raw, "translate_src_lang")
        tgt_lang = _resolve_nllb_lang(tgt_lang_raw, "translate_tgt_lang")

        local_model_dir = self._resolve_local_model_dir(model_name, Path(model_path))
        load_from = str(local_model_dir) if local_model_dir else model_name

        self._tokenizer = AutoTokenizer.from_pretrained(load_from, src_lang=src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(load_from)
        self._forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(str(tgt_lang))
        if self._forced_bos_token_id is None or self._forced_bos_token_id < 0:
            raise ValueError(
                f"Unknown NLLB target language token '{tgt_lang}'. "
                "Set `translate_tgt_lang=...` to a valid ISO-639-3 code (e.g. eng) "
                "or explicit NLLB code (e.g. eng_Latn)."
            )
        self._loaded = True

    def _resolve_local_model_dir(self, model_name: str, model_path: Path) -> Path | None:
        try:
            from turkicnlp.resources.registry import ModelRegistry

            shared = (
                ModelRegistry.default_dir() / "huggingface" / str(model_name).replace("/", "--")
            )
            if (shared / "config.json").exists():
                return shared
        except Exception:
            pass

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

    def _translate_texts(self, texts: list[str]) -> list[str]:
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with self._torch.no_grad():
            generated = self._model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                forced_bos_token_id=self._forced_bos_token_id,
                num_beams=self._num_beams,
                max_new_tokens=self._max_new_tokens,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def process(self, doc: Document) -> Document:
        """Attach sentence and document translation output."""
        if not self._loaded:
            raise RuntimeError("NLLBTranslateProcessor must be loaded before use.")

        if doc.sentences:
            texts = [s.text for s in doc.sentences]
        else:
            texts = [doc.text]
            doc.sentences = [Sentence(text=doc.text)]

        translated = self._translate_texts(texts)
        for sentence, text in zip(doc.sentences, translated):
            sentence.translation = text

        doc.translation = "\n".join(translated)
        doc._processor_log.append("translate:nllb")
        return doc
