"""
Morphological analysis processors.

Provides :class:`ApertiumMorphProcessor` (HFST-native FST loading, no system
Apertium install needed) and :class:`NeuralMorphProcessor` (character-level
neural analyzer).
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
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
    _HYPHEN_CHARS = {"-", "‐", "‑", "‒", "–", "—", "―"}

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
        # Optional external Apertium tagger integration (command + args).
        self._tagger_cmd = self.config.get("apertium_tagger_cmd")
        self._tagger_args = self.config.get("apertium_tagger_args", [])
        # Closed-class lexicon: form_lower → [(upos, feats), ...]
        # Populated by load() from resources/lexicons/<lang>.json.
        self._lexicon: dict[str, list[tuple[str, str]]] = {}
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

        # Load closed-class lexicon for disambiguation and fallback.
        lexicon_path = (
            Path(__file__).resolve().parents[1] / "resources" / "lexicons" / f"{self.lang}.json"
        )
        self._lexicon = {}
        if lexicon_path.exists():
            try:
                data = json.loads(lexicon_path.read_text(encoding="utf-8"))
                for entry in data.get("entries", []):
                    upos = entry.get("upos", "")
                    feats = entry.get("feats", "_")
                    for form in entry.get("forms", []):
                        key = form.lower()
                        if key not in self._lexicon:
                            self._lexicon[key] = []
                        self._lexicon[key].append((upos, feats))
            except Exception:
                pass

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
            readings_by_word: list[Optional[list[dict]]] = []
            for idx, word in enumerate(sentence.words):
                if word.upos == "PUNCT" or self._is_punctuation_token(word.text):
                    word.lemma = word.text
                    word.upos = "PUNCT"
                    word.feats = "_"
                    readings_by_word.append(None)
                    continue

                surface = word.text
                if self._needs_translit and self._to_fst_translit:
                    surface = self._to_fst_translit.transliterate(surface)

                readings_by_word.append(self._analyze_with_fallback(surface))

            tagged_readings = self._tagger_disambiguate(sentence.words, readings_by_word)

            for idx, word in enumerate(sentence.words):
                readings = readings_by_word[idx]
                if readings is None:
                    continue

                if not readings:
                    fallback = self._fallback_for_unknown(
                        word.text,
                        sentence_words=sentence.words,
                        word_index=idx,
                    )
                    if fallback:
                        word.lemma = fallback["lemma"]
                        word.upos = fallback["upos"]
                        word.feats = fallback["feats"]
                    else:
                        word.lemma = word.text
                        if word.upos is None:
                            word.upos = "X"
                        if word.feats is None:
                            word.feats = "_"
                    continue

                best = None
                if tagged_readings is not None:
                    best = tagged_readings[idx]

                if best is None:
                    best = self._disambiguate(
                        readings,
                        sentence_words=sentence.words,
                        word_index=idx,
                        surface_text=word.text,
                    )
                lemma = best["lemma"]
                if self._needs_translit and self._from_fst_translit:
                    lemma = self._from_fst_translit.transliterate(lemma)

                word.lemma = lemma
                word.upos = self._tag_mapper.to_ud_pos(best["pos"])
                raw_feats = self._tag_mapper.to_ud_feats(best["feats"])
                word.feats = self._normalize_ud_feats_for_upos(word.upos, raw_feats)

        log_extra = ""
        if self._needs_translit and self.script and self._apertium_script:
            log_extra = f"(translit:{self.script}->{self._apertium_script})"
        doc._processor_log.append(f"morph:apertium-hfst-{self.lang}{log_extra}")
        return doc

    @staticmethod
    def _is_punctuation_token(text: str) -> bool:
        if not text:
            return False
        return all(unicodedata.category(ch).startswith("P") for ch in text)

    @staticmethod
    def _is_question_particle(text: str) -> bool:
        """Return True if *text* looks like a Turkic question particle.

        Covers the common vowel-harmony sets across all 21 supported languages:
        Latin-script (mi/mı/mu/mü/my, ma/me, ba/be/pa/pe) and Cyrillic-script
        (мы/ме/мо/мү, ма/ме, ба/бе/па/пе, дуо, бе).
        """
        _QUESTION_PARTICLES = {
            # Latin-script question particles
            "mi", "mı", "mu", "mü", "my",
            "ma", "me",
            "ba", "be", "pa", "pe",
            # Cyrillic-script question particles
            "мы", "ме", "мо", "мү",
            "ма", "ба", "бе", "па", "пе", "по",
            # Yakut / Tuvan
            "дуо", "дë", "до", "ду",
        }
        return text.lower() in _QUESTION_PARTICLES

    def _normalize_for_lookup(self, surface: str) -> str:
        normalized = unicodedata.normalize("NFKC", surface)
        # Remove zero-width characters that can appear from copy/paste.
        normalized = "".join(ch for ch in normalized if ch not in {"\u200b", "\u200c", "\u200d", "\ufeff"})

        apostrophe_variants = {
            "\u2019",  # right single quotation mark
            "\u2018",  # left single quotation mark
            "\u02bc",  # modifier letter apostrophe
            "\u02bb",  # modifier letter turned comma
            "\u02b9",  # modifier letter prime
            "\u0060",  # grave accent
            "\u00b4",  # acute accent
            "\u02ca",  # modifier letter acute accent
            "\u02cb",  # modifier letter grave accent
            "\u2032",  # prime
            "\u02bd",  # modifier letter reversed comma
        }
        normalized = "".join("'" if ch in apostrophe_variants else ch for ch in normalized)
        normalized = re.sub(r"'+", "'", normalized)
        if normalized.endswith("'") and len(normalized) > 1:
            normalized = normalized.rstrip("'")
        return normalized

    def _normalize_hyphens(self, text: str) -> str:
        return "".join("-" if ch in self._HYPHEN_CHARS else ch for ch in text)

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return unicodedata.normalize("NFC", stripped)

    def _lookup_variants(self, surface: str) -> list[str]:
        candidates: list[str] = []

        def add(value: str) -> None:
            if value and value not in candidates:
                candidates.append(value)

        normalized = self._normalize_hyphens(self._normalize_for_lookup(surface))
        add(surface)
        add(surface.lower())
        add(normalized)
        add(normalized.lower())

        stripped = self._strip_diacritics(normalized)
        add(stripped)
        add(stripped.lower())

        return candidates

    def _analyze_with_fallback(self, surface: str) -> list[dict]:
        for variant in self._lookup_variants(surface):
            readings = self._analyze(variant)
            if readings:
                return readings
        return []

    def _lexeme_key(self, text: str) -> str:
        normalized = self._normalize_hyphens(self._normalize_for_lookup(text))
        normalized = self._strip_diacritics(normalized)
        return normalized.lower()

    @staticmethod
    def _is_cased_script(text: str) -> bool:
        for ch in text:
            if ch.isalpha():
                return ch.lower() != ch.upper()
        return False

    @staticmethod
    def _is_numeric_token(text: str) -> bool:
        has_digit = False
        for ch in text:
            if unicodedata.category(ch) == "Nd":
                has_digit = True
                continue
            if ch in {".", ",", "/", "-", "%", "+"}:
                continue
            return False
        return has_digit

    def _fallback_for_unknown(
        self,
        text: str,
        sentence_words: Optional[list] = None,
        word_index: Optional[int] = None,
    ) -> Optional[dict[str, str]]:
        if not text:
            return None
        if self._is_numeric_token(text):
            return {"lemma": text, "upos": "NUM", "feats": "NumType=Card"}

        normalized = self._normalize_for_lookup(text)
        if "'" in normalized and self._is_cased_script(text) and text[:1].isupper():
            stem, suffix = normalized.split("'", 1)
            if stem and suffix.isalpha() and len(suffix) <= 4:
                return {"lemma": stem, "upos": "PROPN", "feats": "_"}

        hyphenated = self._normalize_hyphens(normalized)
        if "-" in hyphenated:
            parts = [p for p in hyphenated.split("-") if p]
            if len(parts) == 2:
                left_key = self._lexeme_key(parts[0])
                right_key = self._lexeme_key(parts[1])
                if left_key == right_key:
                    return {"lemma": parts[0].lower(), "upos": "ADV", "feats": "_"}

        if self._is_cased_script(text) and text[:1].isupper():
            if not self._is_sentence_initial(sentence_words, word_index):
                return {"lemma": self._normalize_for_lookup(text), "upos": "PROPN", "feats": "_"}

        # Lexicon fallback: known closed-class words not handled by FST heuristics.
        lex_entries = self._lookup_lexicon(text)
        if lex_entries:
            upos, feats = lex_entries[0]
            return {"lemma": text.lower(), "upos": upos, "feats": feats}

        return None

    def _reciprocal_lemma(self, text: str) -> str:
        normalized = self._normalize_hyphens(self._normalize_for_lookup(text))
        head = normalized.split("-", 1)[0] if "-" in normalized else normalized
        return head.lower()

    @staticmethod
    def _is_sentence_initial(sentence_words: Optional[list], word_index: Optional[int]) -> bool:
        if sentence_words is None or word_index is None:
            return False
        for i, word in enumerate(sentence_words):
            if word.text and not all(unicodedata.category(ch).startswith("P") for ch in word.text):
                return i == word_index
        return False

    def _lookup_lexicon(self, text: str) -> list[tuple[str, str]]:
        """Look up *text* in the closed-class lexicon.

        Returns a list of ``(upos, feats)`` tuples, or an empty list.
        When transliteration is active (e.g. Latin Kazakh → Cyrillic FST),
        also checks the transliterated form so the Cyrillic lexicon matches.
        """
        entries = self._lexicon.get(text.lower())
        if entries:
            return entries
        if self._needs_translit and self._to_fst_translit:
            try:
                translit_key = self._to_fst_translit.transliterate(text).lower()
                entries = self._lexicon.get(translit_key)
                if entries:
                    return entries
            except Exception:
                pass
        return []

    def _tagger_disambiguate(
        self,
        sentence_words: list,
        readings_by_word: list[Optional[list[dict]]],
    ) -> Optional[list[Optional[dict]]]:
        if not self._tagger_cmd or not self._tagger_args:
            return None
        if not any(readings for readings in readings_by_word if readings is not None):
            return None

        stream = self._build_apertium_stream(sentence_words, readings_by_word)
        if stream is None:
            return None

        result = self._run_apertium_tagger(stream)
        if result is None:
            return None

        return self._parse_tagger_output(result, readings_by_word)

    def _build_apertium_stream(
        self,
        sentence_words: list,
        readings_by_word: list[Optional[list[dict]]],
    ) -> Optional[str]:
        units: list[str] = []
        for word, readings in zip(sentence_words, readings_by_word):
            surface = word.text
            if readings is None:
                units.append(f"^{surface}/{surface}<punct>$")
                continue
            if not readings:
                units.append(f"^{surface}/*{surface}$")
                continue
            analyses = "/".join(self._analysis_to_stream(r) for r in readings)
            units.append(f"^{surface}/{analyses}$")
        return " ".join(units) if units else None

    @staticmethod
    def _analysis_to_stream(reading: dict) -> str:
        tags = "".join(f"<{t}>" for t in [reading["pos"]] + reading.get("feats", []))
        return f"{reading['lemma']}{tags}"

    def _run_apertium_tagger(self, stream: str) -> Optional[str]:
        cmd = [self._tagger_cmd, *self._tagger_args]
        try:
            result = subprocess.run(
                cmd,
                input=stream,
                text=True,
                capture_output=True,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    def _parse_tagger_output(
        self,
        output: str,
        readings_by_word: list[Optional[list[dict]]],
    ) -> Optional[list[Optional[dict]]]:
        units = re.findall(r"\\^([^$]+)\\$", output)
        if not units or len(units) != len(readings_by_word):
            return None

        selected: list[Optional[dict]] = []
        for unit, readings in zip(units, readings_by_word):
            if readings is None:
                selected.append(None)
                continue
            parts = unit.split("/")
            if len(parts) < 2:
                selected.append(None)
                continue
            analysis = parts[1]
            parsed = self._parse_stream_analysis(analysis)
            if parsed is None:
                selected.append(None)
                continue
            lemma, pos, feats = parsed
            match = self._match_reading(readings, lemma, pos, feats)
            selected.append(match)
        return selected

    @staticmethod
    def _parse_stream_analysis(analysis: str) -> Optional[tuple[str, str, list[str]]]:
        lemma_match = re.match(r"^([^<]+)", analysis)
        if not lemma_match:
            return None
        lemma = lemma_match.group(1)
        tags = [t for t in re.findall(r"<([^>]+)>", analysis) if t]
        if not tags:
            return None
        return lemma, tags[0], tags[1:]

    @staticmethod
    def _match_reading(readings: list[dict], lemma: str, pos: str, feats: list[str]) -> Optional[dict]:
        for reading in readings:
            if reading.get("lemma") == lemma and reading.get("pos") == pos and reading.get("feats") == feats:
                return reading
        return None

    @staticmethod
    def _normalize_ud_feats_for_upos(upos: str, feats: str) -> str:
        if not feats or feats == "_":
            return "_"

        allowed_prefixes = {
            "VERB": (
                "Aspect=",
                "Evident=",
                "Mood=",
                "Number=",
                "Person=",
                "Clusivity=",
                "Polarity=",
                "Tense=",
                "VerbForm=",
                "Voice=",
            ),
            "AUX": (
                "Aspect=",
                "Evident=",
                "Mood=",
                "Number=",
                "Person=",
                "Clusivity=",
                "Polarity=",
                "Tense=",
                "VerbForm=",
                "Voice=",
            ),
            "NOUN": (
                "Case=",
                "Number=",
                "Gender=",
                "Number[psor]=",
                "Person[psor]=",
            ),
            "PROPN": (
                "Case=",
                "Number=",
                "Gender=",
                "Number[psor]=",
                "Person[psor]=",
            ),
            "PRON": (
                "Case=",
                "Number=",
                "Person=",
                "PronType=",
                "Reflex=",
            ),
            "DET": (
                "Case=",
                "Number=",
                "Person=",
                "PronType=",
                "Reflex=",
            ),
            "ADJ": (
                "Case=",
                "Degree=",
                "Number=",
                "Gender=",
            ),
            "ADV": (
                "Degree=",
                "Polarity=",
                "AdvType=",
                "PronType=",
            ),
            "NUM": (
                "Case=",
                "Number=",
                "NumType=",
                "Gender=",
            ),
            "PART": (
                "PartType=",
                "Polarity=",
            ),
            "ADP": ("AdpType=", "Case="),
            "CCONJ": (),
            "SCONJ": (),
            "INTJ": (),
            "PUNCT": (),
            "SYM": (),
            "X": (),
        }

        prefixes = allowed_prefixes.get(upos)
        if prefixes is None:
            return feats

        kept = [f for f in feats.split("|") if any(f.startswith(p) for p in prefixes)]
        return "|".join(sorted(kept)) if kept else "_"

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
            # Strip HFST/internal marker tokens from analyses.
            clean = re.sub(r"@[^@]+@", "", clean)

            lemma_match = re.match(r"^([^<]+)", clean)
            if not lemma_match:
                continue
            lemma = lemma_match.group(1)

            tags = [t for t in re.findall(r"<([^>]+)>", clean) if t]
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

    def _disambiguate(
        self,
        readings: list[dict],
        sentence_words: Optional[list] = None,
        word_index: Optional[int] = None,
        surface_text: Optional[str] = None,
    ) -> dict:
        """Pick the best reading from multiple analyses."""
        if not readings:
            return {"lemma": "", "pos": "x", "feats": []}
        if len(readings) == 1:
            return readings[0]

        pos_priority = {
            "v": 0,
            "vaux": 1,
            "n": 2,
            "np": 3,
            "adj": 4,
            "adv": 5,
            "prn": 6,
            "det": 7,
        }

        def next_is_question_particle() -> bool:
            if sentence_words is None or word_index is None:
                return False
            for j in range(word_index + 1, len(sentence_words)):
                if self._is_punctuation_token(sentence_words[j].text):
                    continue
                return self._is_question_particle(sentence_words[j].text)
            return False

        def is_final_lexical_position() -> bool:
            if sentence_words is None or word_index is None:
                return False
            for j in range(len(sentence_words) - 1, -1, -1):
                if not self._is_punctuation_token(sentence_words[j].text):
                    return j == word_index
            return False

        final_lexical = is_final_lexical_position()
        def context_score(reading: dict) -> int:
            score = 0
            pos = reading.get("pos")
            feats = set(reading.get("feats", []))
            verbal_tags = {
                "past",
                "pres",
                "fut",
                "aor",
                "ifi",
                "ind",
                "imp",
                "opt",
                "cond",
                "neces",
                "prog",
                "pass",
                "rcp",
                "inf",
                "ger",
                "part",
                "cvb",
            }
            case_tags = {"nom", "gen", "dat", "acc", "abl", "loc", "ins", "equ"}
            has_verbal_tags = bool(feats.intersection(verbal_tags))
            has_case_tags = bool(feats.intersection(case_tags))
            if final_lexical:
                if pos in {"v", "vaux"}:
                    score += 3
                if pos in {"n", "np"}:
                    score -= 2
                if feats.intersection({"past", "pres", "fut", "aor", "ifi"}):
                    score += 1
            else:
                # In Turkic SOV, finite verbs are rarely non-final.
                if (
                    pos in {"v", "vaux"}
                    and feats.intersection({"past", "pres", "fut", "aor", "ifi", "ind", "imp", "opt", "cond", "neces"})
                    and not next_is_question_particle()
                ):
                    score -= 2
            if pos in {"n", "np", "adj", "adv", "prn", "det", "num"} and has_verbal_tags:
                score -= 3
            if pos in {"v", "vaux"} and has_case_tags and "part" not in feats:
                score -= 1
            # Prefer proper-noun analyses for title-cased tokens in cased scripts.
            if surface_text and surface_text[:1].isupper():
                if pos == "np":
                    score += 1
                if pos in {"v", "vaux"} and "imp" in feats:
                    score -= 2
            # Lexicon boost: reward readings whose UPOS matches the lexicon entry.
            if surface_text is not None and self._lexicon:
                lex_entries = self._lookup_lexicon(surface_text)
                if lex_entries:
                    ud_pos = (
                        self._tag_mapper.to_ud_pos(pos)
                        if self._tag_mapper is not None
                        else None
                    )
                    if ud_pos is not None and ud_pos in {e[0] for e in lex_entries}:
                        score += 2
            return score

        readings.sort(
            key=lambda r: (
                r.get("weight", 0.0),
                -context_score(r),
                pos_priority.get(r.get("pos", ""), 99),
                len(r.get("feats", [])),
            )
        )
        return readings[0]

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
