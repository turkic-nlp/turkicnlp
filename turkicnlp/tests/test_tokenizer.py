"""Tests for tokenizers."""

from __future__ import annotations

from turkicnlp.processors.tokenizer import RegexTokenizer, NeuralTokenizer
from turkicnlp.processors.tokenizer_arabic import ArabicScriptTokenizer


class TestRegexTokenizer:
    def test_instantiation(self) -> None:
        tok = RegexTokenizer(lang="kaz")
        assert tok.NAME == "tokenize"
        assert tok.REQUIRES == []

    def test_provides(self) -> None:
        assert RegexTokenizer.PROVIDES == ["tokenize"]


class TestArabicScriptTokenizer:
    def test_instantiation(self) -> None:
        tok = ArabicScriptTokenizer(lang="uig")
        assert tok.NAME == "tokenize"
