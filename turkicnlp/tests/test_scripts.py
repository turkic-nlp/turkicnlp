"""Tests for the script configuration system."""

from __future__ import annotations

import pytest

from turkicnlp.scripts import (
    Script,
    ScriptConfig,
    LANGUAGE_SCRIPTS,
    get_script_config,
)


class TestScript:
    def test_script_values(self) -> None:
        assert str(Script.LATIN) == "Latn"
        assert str(Script.CYRILLIC) == "Cyrl"
        assert str(Script.PERSO_ARABIC) == "Arab"


class TestLanguageScripts:
    def test_all_languages_have_config(self) -> None:
        expected_langs = {"tur", "kaz", "uzb", "aze", "kir", "tat", "uig", "bak"}
        for lang in expected_langs:
            assert lang in LANGUAGE_SCRIPTS

    def test_turkish_latin_only(self) -> None:
        cfg = get_script_config("tur")
        assert cfg.primary == Script.LATIN
        assert cfg.available == [Script.LATIN]

    def test_kazakh_dual_script(self) -> None:
        cfg = get_script_config("kaz")
        assert Script.CYRILLIC in cfg.available
        assert Script.LATIN in cfg.available

    def test_uyghur_rtl(self) -> None:
        cfg = get_script_config("uig")
        assert cfg.direction == "rtl"

    def test_unknown_language_raises(self) -> None:
        with pytest.raises(ValueError):
            get_script_config("xxx")
