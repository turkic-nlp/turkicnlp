"""Tests for Apertium → UD tag mappings."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings import load_tag_map
from turkicnlp.resources.tag_mappings.base import TagMapper


class TestTagMapper:
    def test_default_pos_mapping(self) -> None:
        mapper = TagMapper()
        assert mapper.to_ud_pos("n") == "NOUN"
        assert mapper.to_ud_pos("v") == "VERB"
        assert mapper.to_ud_pos("unknown") == "X"

    def test_empty_feats(self) -> None:
        mapper = TagMapper()
        assert mapper.to_ud_feats([]) == "_"


class TestKazakhMapper:
    def test_load(self) -> None:
        mapper = load_tag_map("kaz")
        assert mapper.to_ud_pos("n") == "NOUN"
        assert "Case=Dat" in mapper.to_ud_feats(["dat"])


class TestTurkishMapper:
    def test_load(self) -> None:
        mapper = load_tag_map("tur")
        assert mapper.to_ud_pos("v") == "VERB"
