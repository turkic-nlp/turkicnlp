"""Tests for Apertium → UD tag mappings."""

from __future__ import annotations

import pytest

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

    def test_map_ud_feats_reports_unknown(self) -> None:
        mapper = TagMapper()
        mapped, unknown = mapper.map_ud_feats(["dat", "sg", "mystery"])
        assert mapped == []
        assert unknown == ["dat", "sg", "mystery"]


class TestKazakhMapper:
    def test_load(self) -> None:
        mapper = load_tag_map("kaz")
        assert mapper.to_ud_pos("n") == "NOUN"
        assert "Case=Dat" in mapper.to_ud_feats(["dat"])


class TestTurkishMapper:
    def test_load(self) -> None:
        mapper = load_tag_map("tur")
        assert mapper.to_ud_pos("v") == "VERB"


class TestTurkmenMapper:
    def test_load(self) -> None:
        mapper = load_tag_map("tuk")
        assert mapper.to_ud_pos("n") == "NOUN"
        assert "Case=Dat" in mapper.to_ud_feats(["dat"])
        assert "Number=Plur" in mapper.to_ud_feats(["pl"])
        assert "Tense=Past" in mapper.to_ud_feats(["past"])

    def test_psor_mapping(self) -> None:
        mapper = load_tag_map("tuk")
        feats = mapper.to_ud_feats(["px1sg"])
        assert "Person[psor]=1" in feats
        assert "Number[psor]=Sing" in feats

    def test_unknown_feat_reporting(self) -> None:
        mapper = load_tag_map("tuk")
        mapped, unknown = mapper.map_ud_feats(["dat", "unknown_tag"])
        assert "Case=Dat" in mapped
        assert unknown == ["unknown_tag"]


@pytest.mark.parametrize(
    "lang",
    ["aze", "uzb", "uig", "kir", "bak", "crh", "kaa", "nog", "kum", "krc", "alt", "tyv", "kjh", "chv", "gag", "sah"],
)
def test_common_turkic_mappers(lang: str) -> None:
    mapper = load_tag_map(lang)
    # Should use a non-empty mapper (not bare default fallback) for Apertium languages.
    assert mapper.__class__.__name__ != "TagMapper"
    assert "Case=Dat" in mapper.to_ud_feats(["dat"])
    assert "Number=Plur" in mapper.to_ud_feats(["pl"])


@pytest.mark.parametrize(
    ("lang", "feat", "expected"),
    [
        ("tur", "ifi", "Evident=Nfh"),
        ("tuk", "qst", "PartType=Int"),
        ("kaz", "evid", "Evident=Nfh"),
        ("chv", "prl", "Case=Prol"),
        ("sah", "par", "Case=Par"),
        ("tyv", "cvb", "VerbForm=Conv"),
        ("uzb", "prog", "Aspect=Prog"),
    ],
)
def test_language_specific_feat_overrides(lang: str, feat: str, expected: str) -> None:
    mapper = load_tag_map(lang)
    assert expected in mapper.to_ud_feats([feat])
