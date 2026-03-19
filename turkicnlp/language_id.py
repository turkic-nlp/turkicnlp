"""
GlotLID-based language identification.

Wraps the GlotLID FastText model with optional label filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from turkicnlp.resources.registry import ModelRegistry

_DEFAULT_REPO_ID = "cis-lmu/glotlid"
_DEFAULT_FILENAME = "model_v3.bin"

# GlotLID uses Glottolog-style language codes in labels for some languages.
# Add overrides here when TurkicNLP uses ISO-639-3 but GlotLID expects a
# different code.
_LANG_CODE_OVERRIDES: dict[str, str] = {
    "uzb": "uzn",  # Uzbek (Latn) in GlotLID is labeled as uzn_Latn
}


def glotlid_label_for(lang: str, script: str) -> str:
    """Return the expected GlotLID label for a language/script pair."""
    glot_code = _LANG_CODE_OVERRIDES.get(lang, lang)
    return f"__label__{glot_code}_{script}"


class LanguageDetection:
    """Language identification using the GlotLID FastText model.

    Args:
        model_path: Optional local path to the FastText model file.
        languages: List of label strings (e.g. ``__label__eng_Latn``) to limit
            predictions to. Use ``-1`` or ``None`` to enable the default
            Turkic-language label set.
        mode: ``before`` limits labels before softmax (recommended), ``after``
            computes softmax across all labels and then filters.
        repo_id: Hugging Face repo id (default: ``cis-lmu/glotlid``).
        filename: Model filename in the repo (default: ``model_v3.bin``).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        languages: Iterable[str] | int | None = -1,
        mode: str = "before",
        repo_id: str = _DEFAULT_REPO_ID,
        filename: str = _DEFAULT_FILENAME,
    ) -> None:
        try:
            import fasttext
        except ImportError as exc:
            raise ImportError(
                "GlotLID requires `fasttext`. Install with: pip install turkicnlp[lid]"
            ) from exc

        if model_path is None:
            model_path = self._download_model(repo_id, filename)

        self.model = fasttext.load_model(model_path)
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()

        if languages in (-1, None):
            languages = self._default_turkic_labels()
        self.language_indices = self._resolve_language_indices(languages)
        self.labels = list(np.array(self.labels)[self.language_indices])

        if mode == "after":
            self.predict = self._predict_limit_after_softmax
        else:
            self.predict = self._predict_limit_before_softmax

    def _resolve_language_indices(self, languages: Iterable[str] | int | None) -> list[int]:
        if isinstance(languages, (list, tuple, set)):
            unique_labels = list(dict.fromkeys(languages))
            resolved_labels = [self._normalize_label(l) for l in unique_labels]
            indices = [self.labels.index(l) for l in resolved_labels if l in self.labels]
            if not indices:
                raise ValueError(
                    "None of the requested labels are present in the GlotLID model."
                )
            return indices

        raise ValueError("languages must be a list/tuple/set of labels or -1/None.")

    def _default_turkic_labels(self) -> list[str]:
        from turkicnlp.scripts import LANGUAGE_SCRIPTS

        labels: list[str] = []
        for lang, config in LANGUAGE_SCRIPTS.items():
            for script in config.available:
                labels.append(glotlid_label_for(lang, script.value))
        return labels

    def _normalize_label(self, label: str) -> str:
        if label in self.labels:
            return label
        if not label.startswith("__label__"):
            return label
        payload = label[len("__label__") :]
        if "_" not in payload:
            return label
        lang, script = payload.split("_", 1)
        mapped = _LANG_CODE_OVERRIDES.get(lang)
        if not mapped:
            return label
        candidate = f"__label__{mapped}_{script}"
        return candidate

    def _download_model(self, repo_id: str, filename: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Downloading GlotLID requires `huggingface_hub`. "
                "Install with: pip install turkicnlp[lid]"
            ) from exc

        cache_dir = ModelRegistry.default_dir() / "huggingface" / repo_id.replace("/", "--")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
        )

    def _predict_limit_before_softmax(self, text: str, k: int = 1) -> tuple[tuple[str, ...], np.ndarray]:
        sentence_vector = self.model.get_sentence_vector(text)
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)
        softmax_result = np.exp(result_vector - np.max(result_vector))
        softmax_result = softmax_result / np.sum(softmax_result)

        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = tuple(self.labels[i] for i in top_k_indices)
        top_k_probs = softmax_result[top_k_indices]
        return top_k_labels, top_k_probs

    def _predict_limit_after_softmax(self, text: str, k: int = 1) -> tuple[tuple[str, ...], np.ndarray]:
        sentence_vector = self.model.get_sentence_vector(text)
        result_vector = np.dot(self.output_matrix, sentence_vector)
        softmax_result = np.exp(result_vector - np.max(result_vector))
        softmax_result = softmax_result / np.sum(softmax_result)

        softmax_result = softmax_result[self.language_indices]
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = tuple(self.labels[i] for i in top_k_indices)
        top_k_probs = softmax_result[top_k_indices]
        return top_k_labels, top_k_probs

    def available_labels(self) -> list[str]:
        """Return the available labels after optional filtering."""
        return list(self.labels)


# Backwards compatibility
GlotLID = LanguageDetection
