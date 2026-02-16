"""Tests for script detection."""

from __future__ import annotations

import pytest

from turkicnlp.scripts import Script
from turkicnlp.scripts.detector import detect_script, detect_script_segments


class TestDetectScript:
    def test_cyrillic(self) -> None:
        assert detect_script("Мен мектепке бардым") == Script.CYRILLIC

    def test_latin(self) -> None:
        assert detect_script("Men mektepke bardym") == Script.LATIN

    def test_arabic(self) -> None:
        assert detect_script("مەن مەكتەپكە باردىم") == Script.PERSO_ARABIC

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_script("123 !?")


class TestDetectScriptSegments:
    def test_empty(self) -> None:
        assert detect_script_segments("") == []

    def test_single_script(self) -> None:
        segments = detect_script_segments("Hello world")
        assert len(segments) == 1
        assert segments[0][1] == Script.LATIN

    def test_mixed(self) -> None:
        segments = detect_script_segments("Hello мир")
        assert len(segments) == 2
