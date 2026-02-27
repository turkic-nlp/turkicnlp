"""
Pipeline orchestrator for TurkicNLP.

The :class:`Pipeline` resolves processor dependencies, loads script-appropriate
models, optionally inserts transliteration steps, and chains processors
in canonical order.
"""

from __future__ import annotations

from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor, ProcessorRequirementsError
from turkicnlp.scripts import Script, get_script_config, ScriptConfig
from turkicnlp.scripts.detector import detect_script
from turkicnlp.scripts.transliterator import Transliterator

# Canonical processor execution order
PROCESSOR_ORDER: list[str] = [
    "script_detect",
    "transliterate",
    "tokenize",
    "mwt",
    "morph",
    "pos",
    "lemma",
    "depparse",
    "ner",
    "embeddings",
    "sentiment",
    "transliterate_back",
]


class Pipeline:
    """Main entry point. Constructs a chain of Processors for a language
    and runs documents through them.

    Usage::

        nlp = Pipeline("kaz", processors=["tokenize", "pos", "lemma", "depparse"])
        doc = nlp("Мен мектепке бардым")

    Args:
        lang: ISO 639-3 language code.
        processors: Processor names to use. ``None`` uses all available.
        script: Script code (``Cyrl``, ``Latn``, ``Arab``) or ``auto``.
        transliterate_to: Target script for transliteration bridging.
        model_dir: Override default model directory.
        use_gpu: Whether to use GPU for neural processors.
        **processor_configs: Per-processor config overrides
            (e.g. ``morph_backend="apertium"``).
    """

    def __init__(
        self,
        lang: str,
        processors: Optional[list[str]] = None,
        script: Optional[str] = None,
        transliterate_to: Optional[str] = None,
        model_dir: Optional[str] = None,
        use_gpu: bool = False,
        **processor_configs: object,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._processors: list[Processor] = []
        self._script_config = get_script_config(lang)
        self._requested_processors = processors
        self._processor_configs = processor_configs
        self._autoload = processors is not None

        if script is None or script == "auto":
            self._script: Optional[Script] = None
            self._explicit_script = False
        else:
            self._script = Script(script)
            self._explicit_script = True
            if self._script not in self._script_config.available:
                raise ValueError(
                    f"Script '{script}' not available for {lang}. "
                    f"Available: {[str(s) for s in self._script_config.available]}"
                )

        self._transliterator: Optional[Transliterator] = None
        self._transliterate_back_enabled = False

        if transliterate_to:
            target_script = Script(transliterate_to)
            source_script = self._script or self._script_config.primary
            if target_script != source_script:
                self._transliterator = Transliterator(lang, source_script, target_script)
                self._transliterate_back_enabled = True
                self._model_script = target_script
            else:
                self._model_script = source_script
        else:
            self._model_script = self._script or self._script_config.primary

        if self._autoload:
            self._build_processors()

    def _build_processors(self) -> None:
        """Resolve, download, and load processors."""
        if self._processors:
            return

        from turkicnlp.resources.registry import ProcessorRegistry, ModelRegistry
        from turkicnlp.resources.downloader import download

        requested = self._requested_processors or PROCESSOR_ORDER
        resolved = self._resolve_dependencies(requested)

        catalog = ModelRegistry.load_catalog()
        lang_info = catalog.get(self.lang, {})
        processors_section = lang_info.get("processors", {})
        script_key = str(self._model_script)
        if script_key in processors_section:
            proc_catalog = processors_section[script_key]
        elif processors_section:
            primary = lang_info.get("scripts", {}).get("primary")
            proc_catalog = processors_section.get(primary, next(iter(processors_section.values())))
        else:
            proc_catalog = {}

        for proc_name in resolved:
            if proc_name in ("script_detect", "transliterate", "transliterate_back"):
                continue

            backend_key = f"{proc_name}_backend"
            backend = (
                self._processor_configs.get(backend_key)
                if backend_key in self._processor_configs
                else None
            )

            proc_info = proc_catalog.get(proc_name, {})
            if backend is None and isinstance(proc_info, dict):
                backend = proc_info.get("default")
                if backend is None and "backends" in proc_info:
                    backend = next(iter(proc_info["backends"].keys()), None)

            if backend is None:
                if proc_name == "tokenize":
                    if self._model_script == Script.PERSO_ARABIC:
                        backend = "rule_arabic"
                    else:
                        backend = "rule"
                else:
                    backend = next(iter(ProcessorRegistry._registry.get(proc_name, {})), None)

            if backend is None:
                raise ValueError(f"No backend available for processor '{proc_name}'.")

            config: dict[str, object] = {"use_gpu": self.use_gpu}
            prefix = f"{proc_name}_"
            for key, value in self._processor_configs.items():
                if key.startswith(prefix) and key != backend_key:
                    config[key[len(prefix):]] = value

            proc_class = ProcessorRegistry.get(proc_name, str(backend))
            proc = proc_class(lang=self.lang, script=self._model_script, config=config)

            backend_info = {}
            if isinstance(proc_info, dict):
                backend_info = proc_info.get("backends", {}).get(str(backend), {}) or {}

            model_path = None
            backend_type = backend_info.get("type")
            if backend_type == "huggingface_seq2seq":
                model_name = backend_info.get("model_name")
                shared_exists = False
                if model_name:
                    shared_dir = (
                        ModelRegistry.default_dir()
                        / "huggingface"
                        / str(model_name).replace("/", "--")
                    )
                    shared_exists = (shared_dir / "config.json").exists()
                if not shared_exists:
                    download(self.lang, processors=[proc_name], script=str(self._model_script))
            elif backend_type == "stanza_custom":
                # Custom-trained Stanza models: ensure files are downloaded
                custom_dir = ModelRegistry.default_dir() / "stanza_custom" / self.lang
                filename = backend_info.get("filename", f"{proc_name}.pt")
                pretrain_filename = backend_info.get("pretrain_filename")
                needs_download = not (custom_dir / filename).exists()
                if not needs_download and pretrain_filename:
                    needs_download = not (custom_dir / pretrain_filename).exists()
                if needs_download:
                    download(self.lang, processors=[proc_name], script=str(self._model_script))
            elif backend_type in ("apertium_fst", "neural_model"):
                try:
                    model_path = ModelRegistry.get_model_path(
                        self.lang, proc_name, str(backend), script=str(self._model_script)
                    )
                except FileNotFoundError:
                    download(self.lang, processors=[proc_name], script=str(self._model_script))
                    if backend_type in ("apertium_fst", "neural_model"):
                        model_path = ModelRegistry.get_model_path(
                            self.lang, proc_name, str(backend), script=str(self._model_script)
                        )

            proc.load(model_path if model_path is not None else "")
            self._processors.append(proc)

    def _resolve_dependencies(self, requested: list[str]) -> list[str]:
        """Ensure all processor dependencies are satisfied.

        Auto-adds missing dependencies. E.g. requesting ``["depparse"]``
        auto-adds ``["tokenize", "pos", "depparse"]``.
        """
        from turkicnlp.resources.registry import ProcessorRegistry

        needed: set[str] = set()
        for proc_name in requested:
            try:
                proc_class = ProcessorRegistry.get_any(proc_name)
                for req in proc_class.REQUIRES:
                    needed.add(req)
            except ValueError:
                pass
            needed.add(proc_name)

        resolved = [p for p in PROCESSOR_ORDER if p in needed]
        for proc_name in requested:
            if proc_name in needed and proc_name not in resolved:
                resolved.append(proc_name)
        return resolved

    def __call__(self, text: str) -> Document:
        """Process a single text string.

        Args:
            text: Input text.

        Returns:
            Annotated :class:`Document`.
        """
        if not self._processors:
            self._build_processors()

        doc = Document(text=text, lang=self.lang)

        if self._explicit_script:
            doc.script = str(self._script)
        else:
            try:
                detected = detect_script(text)
                doc.script = str(detected)
            except ValueError:
                doc.script = str(self._script_config.primary)

        original_text = text
        if self._transliterator:
            transliterated = self._transliterator.transliterate(text)
            doc._original_text = original_text
            doc.text = transliterated

        for processor in self._processors:
            doc = processor.process(doc)

        if self._transliterate_back_enabled and self._transliterator:
            reverse_translit = Transliterator(
                self.lang, self._transliterator.target, self._transliterator.source
            )
            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.lemma:
                        word.lemma = reverse_translit.transliterate(word.lemma)
            doc.text = original_text
            doc.script = str(self._script or detect_script(original_text))

        return doc

    def batch(self, texts: list[str], batch_size: int = 32) -> list[Document]:
        """Process multiple texts.

        Args:
            texts: List of input strings.
            batch_size: Batch size for neural processors.

        Returns:
            List of annotated documents.
        """
        return [self(text) for text in texts]

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "conllu",
    ) -> Optional[str]:
        """Process a text file, optionally writing output.

        Args:
            input_path: Path to input text file.
            output_path: Path to write output. ``None`` returns as string.
            output_format: ``conllu`` or ``json``.

        Returns:
            Formatted output string, or ``None`` if written to file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = self(text)

        if output_format == "conllu":
            result = doc.to_conllu()
        elif output_format == "json":
            import json

            result = json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unknown format: {output_format}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            return None
        return result

    @property
    def processors(self) -> list[str]:
        """List active processor names."""
        return [p.NAME for p in self._processors]

    @property
    def license_info(self) -> dict:
        """Return license information for all loaded components."""
        from turkicnlp.processors.morphology import ApertiumMorphProcessor

        info: dict = {"library": "Apache-2.0", "components": {}}
        for proc in self._processors:
            if isinstance(proc, ApertiumMorphProcessor):
                info["components"][proc.NAME] = {
                    "license": "GPL-3.0",
                    "note": "Apertium FST data downloaded separately; not bundled with library",
                }
            else:
                info["components"][proc.NAME] = {"license": "Apache-2.0"}
        return info

    def __repr__(self) -> str:
        procs = " -> ".join(self.processors)
        return f"<Pipeline(lang={self.lang}, script={self._model_script}, processors=[{procs}])>"
