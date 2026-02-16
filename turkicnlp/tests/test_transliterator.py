"""Tests for script transliteration."""

from __future__ import annotations

import pytest

from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator


class TestKazakhTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("kaz", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("мен")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_table_raises(self) -> None:
        with pytest.raises(ValueError, match="No transliteration table"):
            Transliterator("kaz", Script.PERSO_ARABIC, Script.LATIN)


class TestUzbekTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("uzb", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("салом")
        assert isinstance(result, str)
