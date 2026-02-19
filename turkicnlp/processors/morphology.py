"""
Morphological analysis processors.

Provides :class:`ApertiumMorphProcessor` (HFST-native FST loading, no system
Apertium install needed) and :class:`NeuralMorphProcessor` (character-level
neural analyzer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from turkicnlp.models.document import Document
from turkicnlp.processors.base import Processor
from turkicnlp.scripts import Script, get_script_config
from turkicnlp.scripts.transliterator import Transliterator


class ApertiumMorphProcessor(Processor):
    """Morphological analyzer using Apertium FST data loaded natively via ``hfst``.

    The compiled ``.hfst`` transducer is loaded via the ``hfst`` Python package.
    No system Apertium installation is required.

    License note:
        The ``.hfst`` data files are GPL-3.0 licensed and are downloaded
        separately from the Apache-2.0 turkicnlp library.
    """

    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]
    REQUIRES = ["tokenize"]

    def __init__(
        self,
        lang: str,
        script: Optional[Script] = None,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(lang, script, config)
        self.apertium_lang = self.config.get("apertium_lang", lang)
        self._analyzer = None       # hfst.HfstTransducer
        self._generator = None      # hfst.HfstTransducer (optional)
        self._tag_mapper = None
        self._to_fst_translit = None
        self._from_fst_translit = None
        self._script_config = None
        self._apertium_script = None
        self._needs_translit = False
        try:
            self._script_config = get_script_config(lang)
            self._apertium_script = self._script_config.apertium_script
            if self.config.get("apertium_script"):
                self._apertium_script = Script(self.config["apertium_script"])
            self._needs_translit = (
                script is not None
                and self._apertium_script is not None
                and script != self._apertium_script
            )
        except ValueError:
            self._script_config = None
            self._apertium_script = None
            self._needs_translit = False

    def load(self, model_path: str | Path) -> None:
        """Load compiled HFST transducer from the downloaded data directory.

        Args:
            model_path: Path to the apertium data directory, e.g.
                ``~/.turkicnlp/models/kaz/Cyrl/morph/apertium/``.
        """
        model_path = Path(model_path)

        analyzer_files = list(model_path.glob("*.automorf.hfst"))
        if not analyzer_files:
            raise FileNotFoundError(
                f"No .automorf.hfst file found in {model_path}. "
                f"Run: turkicnlp.download('{self.lang}') to download Apertium data."
            )

        try:
            import hfst
        except ImportError as exc:
            raise ImportError(
                "The 'hfst' package is required for Apertium morphological analysis. "
                "Install it with: pip install hfst"
            ) from exc

        analyzer_path = analyzer_files[0]
        istream = hfst.HfstInputStream(str(analyzer_path))
        self._analyzer = istream.read()

        generator_files = list(model_path.glob("*.autogen.hfst"))
        if generator_files:
            gstream = hfst.HfstInputStream(str(generator_files[0]))
            self._generator = gstream.read()

        from turkicnlp.resources.tag_mappings import load_tag_map

        self._tag_mapper = load_tag_map(self.lang)

        fst_script = self._apertium_script
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            import json

            try:
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("script"):
                    fst_script = Script(metadata["script"])
            except json.JSONDecodeError:
                pass

        self._apertium_script = fst_script
        self._needs_translit = (
            self.script is not None and fst_script is not None and self.script != fst_script
        )
        self._to_fst_translit = None
        self._from_fst_translit = None
        if self._needs_translit:
            self._to_fst_translit = Transliterator(self.lang, self.script, fst_script)
            self._from_fst_translit = Transliterator(self.lang, fst_script, self.script)

        self._loaded = True

    def process(self, doc: Document) -> Document:
        """Run HFST morphological analysis on each word.

        Steps:
            1. Optionally transliterate to FST script
            2. Look up in HFST transducer
            3. Disambiguate (pick best reading)
            4. Map Apertium tags to UD tags
            5. Optionally transliterate lemma back
        """
        self.check_requirements(doc)
        if self._analyzer is None:
            raise RuntimeError(
                "ApertiumMorphProcessor is not loaded. "
                "Call load() with a valid model path first."
            )

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == "PUNCT":
                    continue

                surface = word.text
                if self._needs_translit and self._to_fst_translit:
                    surface = self._to_fst_translit.transliterate(surface)

                readings = self._analyze(surface)
                if not readings:
                    readings = self._analyze(surface.lower())

                if not readings:
                    word.lemma = word.text
                    if word.upos is None:
                        word.upos = "X"
                    if word.feats is None:
                        word.feats = "_"
                    continue

                best = self._disambiguate(readings)
                lemma = best["lemma"]
                if self._needs_translit and self._from_fst_translit:
                    lemma = self._from_fst_translit.transliterate(lemma)

                word.lemma = lemma
                word.upos = self._tag_mapper.to_ud_pos(best["pos"])
                word.feats = self._tag_mapper.to_ud_feats(best["feats"])

        log_extra = ""
        if self._needs_translit and self.script and self._apertium_script:
            log_extra = f"(translit:{self.script}->{self._apertium_script})"
        doc._processor_log.append(f"morph:apertium-hfst-{self.lang}{log_extra}")
        return doc

    def _analyze(self, surface: str) -> list[dict]:
        """Look up a surface form in the HFST transducer.

        Returns:
            List of readings, each as ``{"lemma": str, "pos": str, "feats": list[str]}``.
        """
        if self._analyzer is None:
            return []

        import re

        try:
            results = self._analyzer.lookup(surface)
        except Exception:
            return []

        readings: list[dict] = []
        for output_str, weight in results:
            if isinstance(output_str, bytes):
                output_str = output_str.decode("utf-8", errors="ignore")
            clean = output_str.strip()
            if not clean:
                continue
            if "\t" in clean:
                clean = clean.split("\t")[-1]
            if clean.startswith("^") and clean.endswith("$"):
                clean = clean[1:-1]
            if "/" in clean:
                clean = clean.split("/")[-1]
            # Some HFST outputs include explicit epsilon markers in analyses.
            # Remove them so lemmas/tags are user-facing and stable.
            clean = clean.replace("@_EPSILON_SYMBOL_@", "")

            lemma_match = re.match(r"^([^<]+)", clean)
            if not lemma_match:
                continue
            lemma = lemma_match.group(1)

            tags = re.findall(r"<([^>]+)>", clean)
            if not tags:
                continue

            readings.append(
                {
                    "lemma": lemma,
                    "pos": tags[0],
                    "feats": tags[1:],
                    "weight": weight,
                    "raw": output_str,
                }
            )

        return readings

    def _disambiguate(self, readings: list[dict]) -> dict:
        """Pick the best reading from multiple analyses."""
        if not readings:
            return {"lemma": "", "pos": "x", "feats": []}
        if len(readings) == 1:
            return readings[0]

        readings.sort(key=lambda r: r.get("weight", 0.0))
        best_weight = readings[0].get("weight", 0.0)
        eps = 1e-9
        equal_weight = [r for r in readings if abs(r.get("weight", 0.0) - best_weight) <= eps]
        if len(equal_weight) == 1:
            return equal_weight[0]

        pos_priority = {
            "n": 0,
            "np": 1,
            "v": 2,
            "adj": 3,
            "adv": 4,
            "prn": 5,
            "det": 6,
        }
        equal_weight.sort(
            key=lambda r: (
                pos_priority.get(r["pos"], 99),
                len(r.get("feats", [])),
            )
        )
        return equal_weight[0]

    def generate(self, lemma: str, tags: list[str]) -> Optional[str]:
        """Generate a surface form from a lemma and morphological tags.

        Args:
            lemma: Base form (e.g. ``мектеп``).
            tags: Apertium-format tags (e.g. ``["n", "dat", "sg"]``).

        Returns:
            Generated surface form or ``None`` if generation is unavailable.
        """
        if self._generator is None:
            return None

        lemma_text = lemma
        if self._needs_translit and self._to_fst_translit:
            lemma_text = self._to_fst_translit.transliterate(lemma_text)

        tag_str = "".join(f"<{t}>" for t in tags)
        input_form = f"{lemma_text}{tag_str}"

        try:
            results = self._generator.lookup(input_form)
            if results:
                results.sort(key=lambda r: r[1])
                surface = results[0][0]
                if self._needs_translit and self._from_fst_translit:
                    surface = self._from_fst_translit.transliterate(surface)
                return surface
        except Exception:
            return None
        return None

    @property
    def available_for_generation(self) -> bool:
        """Whether this processor can generate surface forms."""
        return self._generator is not None


class NeuralMorphProcessor(Processor):
    """Neural morphological analyzer using a character-level seq2seq model.

    For languages without Apertium FSTs or as a higher-accuracy alternative.
    """

    NAME = "morph"
    PROVIDES = ["lemma", "pos", "feats"]
    REQUIRES = ["tokenize"]

    def load(self, model_path: str) -> None:
        """Load neural morphological analysis model."""
        raise NotImplementedError(
            "NeuralMorphProcessor.load not yet implemented."
        )

    def process(self, doc: Document) -> Document:
        """Run neural morphological analysis on each word."""
        raise NotImplementedError(
            "NeuralMorphProcessor.process not yet implemented."
        )
