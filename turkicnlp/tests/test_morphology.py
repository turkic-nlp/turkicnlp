"""Tests for morphological analysis processors."""

from __future__ import annotations

from turkicnlp.processors.morphology import ApertiumMorphProcessor, NeuralMorphProcessor


class TestApertiumMorphProcessor:
    def test_instantiation(self) -> None:
        proc = ApertiumMorphProcessor(lang="kaz")
        assert proc.NAME == "morph"
        assert "tokenize" in proc.REQUIRES

    def test_provides(self) -> None:
        assert "lemma" in ApertiumMorphProcessor.PROVIDES
        assert "pos" in ApertiumMorphProcessor.PROVIDES
        assert "feats" in ApertiumMorphProcessor.PROVIDES


class TestNeuralMorphProcessor:
    def test_instantiation(self) -> None:
        proc = NeuralMorphProcessor(lang="kaz")
        assert proc.NAME == "morph"
