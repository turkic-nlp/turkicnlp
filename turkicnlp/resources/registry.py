"""
Processor and Model registries for TurkicNLP.

:class:`ProcessorRegistry` maps processor names + backends to their
implementing classes. :class:`ModelRegistry` manages model discovery,
download, and path resolution with script-aware directory layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_DIR = Path.home() / ".turkicnlp" / "models"


class ModelRegistry:
    """Manages model discovery, download, and path resolution.

    Models are stored at::

        ~/.turkicnlp/models/{lang}/{script}/{processor}/{backend}/

    The :envvar:`TURKICNLP_MODELS_DIR` environment variable overrides
    the default location.
    """

    _catalog: Optional[dict] = None
    CATALOG_URL = "https://turkicnlp.github.io/models/catalog.json"

    @classmethod
    def default_dir(cls) -> Path:
        """Return the default model directory, respecting env override."""
        env_dir = os.environ.get("TURKICNLP_MODELS_DIR")
        return Path(env_dir) if env_dir else DEFAULT_MODEL_DIR

    @classmethod
    def get_model_path(
        cls,
        lang: str,
        processor: str,
        backend: str,
        script: Optional[str] = None,
    ) -> Path:
        """Return local path to a model. Raises if not downloaded.

        Args:
            lang: Language code.
            processor: Processor name.
            backend: Backend name.
            script: Script code (e.g. ``Cyrl``). If ``None``, omits script level.

        Raises:
            FileNotFoundError: If the model is not present locally.
        """
        if script:
            path = cls.default_dir() / lang / script / processor / backend
        else:
            path = cls.default_dir() / lang / processor / backend
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. "
                f"Run: turkicnlp.download('{lang}'"
                + (f", script='{script}'" if script else "")
                + ")"
            )
        return path

    @classmethod
    def load_catalog(cls) -> dict:
        """Load the model catalog, preferring local cache.

        Returns:
            The catalog dictionary.
        """
        if cls._catalog is not None:
            return cls._catalog
        catalog_path = cls.default_dir() / "catalog.json"
        if catalog_path.exists():
            with open(catalog_path) as f:
                cls._catalog = json.load(f)
        else:
            cls._catalog = cls._load_packaged_catalog()
            if cls._catalog is None:
                cls._catalog = cls._fetch_remote_catalog()
        return cls._catalog

    @classmethod
    def _fetch_remote_catalog(cls) -> dict:
        """Download catalog from remote."""
        import urllib.request

        catalog_url = os.environ.get("TURKICNLP_CATALOG_URL", cls.CATALOG_URL)
        catalog_path = cls.default_dir() / "catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(catalog_url, catalog_path)
        with open(catalog_path) as f:
            return json.load(f)

    @classmethod
    def _load_packaged_catalog(cls) -> Optional[dict]:
        """Load catalog bundled in the package, if present."""
        try:
            from importlib import resources
        except ImportError:
            return None

        try:
            with resources.open_text("turkicnlp.resources", "catalog.json") as f:
                return json.load(f)
        except (FileNotFoundError, ModuleNotFoundError):
            return None


class ProcessorRegistry:
    """Maps processor names + backends to their implementing classes.

    New processors self-register via :meth:`register`.
    """

    _registry: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, name: str, backend: str, proc_class: type) -> None:
        """Register a processor implementation.

        Args:
            name: Processor name (e.g. ``tokenize``, ``pos``).
            backend: Backend identifier (e.g. ``rule``, ``neural``, ``apertium``).
            proc_class: The processor class.
        """
        if name not in cls._registry:
            cls._registry[name] = {}
        cls._registry[name][backend] = proc_class

    @classmethod
    def get(cls, name: str, backend: str) -> type:
        """Get a specific processor class by name and backend.

        Raises:
            ValueError: If processor or backend is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown processor: {name}")
        if backend not in cls._registry[name]:
            raise ValueError(
                f"Backend '{backend}' not available for processor '{name}'. "
                f"Available: {list(cls._registry[name].keys())}"
            )
        return cls._registry[name][backend]

    @classmethod
    def get_any(cls, name: str) -> type:
        """Get any implementation of a processor (for dependency checking)."""
        if name not in cls._registry:
            raise ValueError(f"Unknown processor: {name}")
        return next(iter(cls._registry[name].values()))

    @classmethod
    def available_for(
        cls, lang: str, script: Optional[str] = None
    ) -> dict[str, list[str]]:
        """Return which processors are available for a language/script.

        Returns:
            Dict mapping processor names to lists of available backends.
        """
        catalog = ModelRegistry.load_catalog()
        if lang not in catalog:
            raise ValueError(
                f"Language '{lang}' not found. "
                f"Available: {list(catalog.keys())}"
            )
        lang_info = catalog[lang]
        processors_section = lang_info.get("processors", {})

        if script and script in processors_section:
            proc_data = processors_section[script]
        elif processors_section:
            first_key = next(iter(processors_section))
            proc_data = processors_section[first_key]
        else:
            return {}

        result: dict[str, list[str]] = {}
        for proc_name, proc_info in proc_data.items():
            if isinstance(proc_info, dict) and "backends" in proc_info:
                backends = list(proc_info["backends"].keys())
                result[proc_name] = backends
        return result


def _register_builtins() -> None:
    """Register all built-in processor implementations."""
    from turkicnlp.processors.tokenizer import RegexTokenizer, NeuralTokenizer
    from turkicnlp.processors.tokenizer_arabic import ArabicScriptTokenizer
    from turkicnlp.processors.morphology import (
        ApertiumMorphProcessor,
        NeuralMorphProcessor,
    )
    from turkicnlp.processors.pos_tagger import NeuralPOSTagger
    from turkicnlp.processors.lemmatizer import DictionaryLemmatizer, NeuralLemmatizer
    from turkicnlp.processors.dep_parser import BiaffineDepParser
    from turkicnlp.processors.ner import NERProcessor
    from turkicnlp.processors.sentiment import SentimentProcessor
    from turkicnlp.processors.mwt import MWTProcessor

    ProcessorRegistry.register("tokenize", "rule", RegexTokenizer)
    ProcessorRegistry.register("tokenize", "rule_arabic", ArabicScriptTokenizer)
    ProcessorRegistry.register("tokenize", "neural", NeuralTokenizer)
    ProcessorRegistry.register("mwt", "neural", MWTProcessor)
    ProcessorRegistry.register("morph", "apertium", ApertiumMorphProcessor)
    ProcessorRegistry.register("morph", "neural", NeuralMorphProcessor)
    ProcessorRegistry.register("pos", "neural", NeuralPOSTagger)
    ProcessorRegistry.register("lemma", "dictionary", DictionaryLemmatizer)
    ProcessorRegistry.register("lemma", "neural", NeuralLemmatizer)
    ProcessorRegistry.register("depparse", "neural", BiaffineDepParser)
    ProcessorRegistry.register("ner", "neural", NERProcessor)
    ProcessorRegistry.register("sentiment", "neural", SentimentProcessor)


_register_builtins()
