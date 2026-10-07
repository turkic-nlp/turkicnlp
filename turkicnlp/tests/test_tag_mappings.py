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
        ("tur", "ifi", "Tense=Past"),
        ("tur", "ifi", "Evident=Fh"),
        ("tur", "past", "Evident=Nfh"),
        ("kaz", "ifi", "Tense=Past"),
        ("uzb", "ifi", "Tense=Past"),
        ("kir", "gna_perf", "VerbForm=Conv"),
        ("tur", "gna_impf", "VerbForm=Conv"),
        ("kaz", "gpr_past", "VerbForm=Part"),
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


@pytest.mark.parametrize("lang", ["tur", "aze", "uzb", "kaz", "kir", "tat", "gag", "crh", "tuk", "uig"])
def test_ifi_is_definite_past_not_evidential(lang: str) -> None:
    """<ifi> is the -DI past in every Apertium Turkic analyser (geldi, келді)."""
    feats = load_tag_map(lang).to_ud_feats(["ifi", "p3", "sg"])
    assert "Tense=Past" in feats
    assert "Evident=Nfh" not in feats


@pytest.mark.parametrize(
    ("pos", "upos"), [("qst", "PART"), ("encl", "PART"), ("cop", "AUX")]
)
def test_clitic_and_copula_pos(pos: str, upos: str) -> None:
    for lang in ("tur", "gag", "kaz", "alt"):
        assert load_tag_map(lang).to_ud_pos(pos) == upos


def test_composite_feature_values_are_flattened_and_sorted() -> None:
    feats = load_tag_map("kaz").to_ud_feats(["gpr_past", "pl"])
    assert feats == "Number=Plur|Tense=Past|VerbForm=Part"
