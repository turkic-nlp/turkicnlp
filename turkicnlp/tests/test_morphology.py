"""Tests for morphological analysis processors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.morphology import ApertiumMorphProcessor, NeuralMorphProcessor
from turkicnlp.resources.tag_mappings import load_tag_map
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator


@dataclass
class _FakeTransducer:
    results: list[tuple[str, float]]

    def lookup(self, surface: str) -> list[tuple[str, float]]:
        return self.results


@dataclass
class _EchoTransducer:
    expected_surface: str
    results: list[tuple[str, float]]
    last_surface: str | None = None

    def lookup(self, surface: str) -> list[tuple[str, float]]:
        self.last_surface = surface
        if surface != self.expected_surface:
            return []
        return self.results


class TestApertiumMorphProcessor:
    def test_instantiation(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz")
        assert proc.NAME == "morph"
        assert "tokenize" in proc.REQUIRES

    def test_provides(self) -> None:
        assert "lemma" in ApertiumMorphProcessor.PROVIDES
        assert "pos" in ApertiumMorphProcessor.PROVIDES
        assert "feats" in ApertiumMorphProcessor.PROVIDES

    def test_analyze_parses_readings(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz")
        proc._analyzer = _FakeTransducer(
            results=[
                ("мектеп<n><dat><sg>", 0.5),
                ("мектеп/мектеп<n><nom><sg>", 1.0),
            ]
        )
        readings = proc._analyze("мектеп")
        assert len(readings) == 2
        assert readings[0]["lemma"] == "мектеп"
        assert readings[0]["pos"] == "n"
        assert readings[0]["feats"] == ["dat", "sg"]

    def test_analyze_strips_hfst_epsilon_marker(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz")
        proc._analyzer = _FakeTransducer(
            results=[
                ("@_EPSILON_SYMBOL_@бар<v><past><p3><sg>", 0.0),
            ]
        )
        readings = proc._analyze("барды")
        assert len(readings) == 1
        assert readings[0]["lemma"] == "бар"
        assert readings[0]["pos"] == "v"
        assert readings[0]["feats"] == ["past", "p3", "sg"]

    def test_disambiguate_prefers_weight_then_pos(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz")
        readings = [
            {"lemma": "a", "pos": "v", "feats": ["pres"], "weight": 0.0},
            {"lemma": "b", "pos": "n", "feats": ["sg"], "weight": 0.0},
            {"lemma": "c", "pos": "adj", "feats": [], "weight": 1.0},
        ]
        best = proc._disambiguate(readings)
        assert best["lemma"] == "b"

    def test_process_with_transliteration(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz", script=Script.LATIN)
        proc._tag_mapper = load_tag_map("kaz")
        proc._apertium_script = Script.CYRILLIC
        proc._needs_translit = True
        proc._to_fst_translit = Transliterator("kaz", Script.LATIN, Script.CYRILLIC)
        proc._from_fst_translit = Transliterator("kaz", Script.CYRILLIC, Script.LATIN)
        proc._analyzer = _EchoTransducer(
            expected_surface="мектеп",
            results=[("мектеп<n><nom><sg>", 0.0)],
        )

        word = Word(id=1, text="mektep")
        doc = Document(
            text="mektep",
            sentences=[
                Sentence(
                    text="mektep",
                    tokens=[Token(id=(1,), text="mektep", words=[word])],
                    words=[word],
                )
            ],
            script="Latn",
        )

        proc.process(doc)
        word = doc.sentences[0].words[0]
        assert word.lemma == "mektep"
        assert word.upos == "NOUN"
        assert word.feats == "Case=Nom|Number=Sing"
        assert proc._analyzer.last_surface == "мектеп"


class TestNeuralMorphProcessor:
    def test_instantiation(self) -> None:
        proc = NeuralMorphProcessor(lang="kaz")
        assert proc.NAME == "morph"


def test_load_requires_hfst(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "morph" / "apertium"
    model_dir.mkdir(parents=True)
    (model_dir / "kaz.automorf.hfst").write_text("stub")

    class _FakeStream:
        def __init__(self, path: str) -> None:
            self.path = path

        def read(self) -> _FakeTransducer:
            return _FakeTransducer(results=[])

    class _FakeHfst:
        HfstInputStream = _FakeStream

    monkeypatch.setitem(__import__("sys").modules, "hfst", _FakeHfst)

    proc = ApertiumMorphProcessor(lang="kaz")
    proc.load(model_dir)
    assert proc._analyzer is not None
