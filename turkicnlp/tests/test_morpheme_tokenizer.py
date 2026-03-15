"""Tests for MorphemeTokenizer — phonology, suffix tables, and segmentation."""

from __future__ import annotations

import pytest

from turkicnlp.processors.morpheme_tokenizer import (
    MorphemeTokenizer,
    Morpheme,
    SegmentationResult,
    _Phonology,
    _get_harmony,
    _get_harmony4,
    _char_class,
    _resolve_allomorph,
    _find_stem_boundary,
    _segment_by_tags,
    _segment_by_ud_features,
    _LANG_SUFFIX_MAP,
    _KAZ_SUFFIXES,
    _TUR_SUFFIXES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Phonological classification tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPhonology:
    """Test vowel harmony and character classification."""

    def test_cyrillic_back_harmony(self):
        phon = _Phonology("Cyrl")
        assert _get_harmony("балалар", phon) == "B"
        assert _get_harmony("ата", phon) == "B"

    def test_cyrillic_front_harmony(self):
        phon = _Phonology("Cyrl")
        assert _get_harmony("үй", phon) == "F"
        assert _get_harmony("мектеп", phon) == "F"

    def test_latin_back_harmony(self):
        phon = _Phonology("Latn")
        assert _get_harmony("araba", phon) == "B"
        assert _get_harmony("okul", phon) == "B"

    def test_latin_front_harmony(self):
        phon = _Phonology("Latn")
        assert _get_harmony("ev", phon) == "F"
        assert _get_harmony("göz", phon) == "F"

    def test_4way_harmony_latin(self):
        phon = _Phonology("Latn")
        assert _get_harmony4("araba", phon) == "BU"
        assert _get_harmony4("ev", phon) == "FU"
        assert _get_harmony4("okul", phon) == "BR"
        assert _get_harmony4("göz", phon) == "FR"

    def test_char_class_cyrillic(self):
        phon = _Phonology("Cyrl")
        assert _char_class("а", phon) == "V"
        assert _char_class("р", phon) == "S"
        assert _char_class("м", phon) == "N"
        assert _char_class("б", phon) == "D"
        assert _char_class("п", phon) == "T"
        assert _char_class("к", phon) == "T"

    def test_char_class_latin(self):
        phon = _Phonology("Latn")
        assert _char_class("a", phon) == "V"
        assert _char_class("r", phon) == "S"
        assert _char_class("m", phon) == "N"
        assert _char_class("b", phon) == "D"
        assert _char_class("p", phon) == "T"

    def test_empty_string_defaults(self):
        phon = _Phonology("Cyrl")
        assert _get_harmony("", phon) == "B"
        assert _get_harmony4("", phon) == "BU"


# ═══════════════════════════════════════════════════════════════════════════
# Suffix allomorph resolution tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAllomorphResolution:
    """Test suffix allomorph selection based on phonological context."""

    def test_kazakh_plural_after_vowel(self):
        phon = _Phonology("Cyrl")
        # бала (back, ends in vowel) → лар
        result = _resolve_allomorph("pl", "бала", _KAZ_SUFFIXES, phon)
        assert result == "лар"

    def test_kazakh_plural_after_front_vowel(self):
        phon = _Phonology("Cyrl")
        # үй (front, ends in sonorant й) → лер
        result = _resolve_allomorph("pl", "үй", _KAZ_SUFFIXES, phon)
        assert result == "лер"

    def test_kazakh_plural_after_voiceless(self):
        phon = _Phonology("Cyrl")
        # мектеп (front, ends in voiceless п) → тер
        result = _resolve_allomorph("pl", "мектеп", _KAZ_SUFFIXES, phon)
        assert result == "тер"

    def test_kazakh_plural_after_nasal(self):
        phon = _Phonology("Cyrl")
        # алан (back, ends in nasal н) → дар
        result = _resolve_allomorph("pl", "алан", _KAZ_SUFFIXES, phon)
        assert result == "дар"

    def test_kazakh_dative_after_vowel(self):
        phon = _Phonology("Cyrl")
        # бала (back, vowel) → ға
        result = _resolve_allomorph("dat", "бала", _KAZ_SUFFIXES, phon)
        assert result == "ға"

    def test_kazakh_dative_after_voiceless(self):
        phon = _Phonology("Cyrl")
        # мектеп (front, voiceless) → ке
        result = _resolve_allomorph("dat", "мектеп", _KAZ_SUFFIXES, phon)
        assert result == "ке"

    def test_kazakh_locative_back(self):
        phon = _Phonology("Cyrl")
        result = _resolve_allomorph("loc", "бала", _KAZ_SUFFIXES, phon)
        assert result == "да"

    def test_kazakh_ablative_after_nasal(self):
        phon = _Phonology("Cyrl")
        # алан → нан (special nasal rule)
        result = _resolve_allomorph("abl", "алан", _KAZ_SUFFIXES, phon)
        assert result == "нан"

    def test_kazakh_instrumental(self):
        phon = _Phonology("Cyrl")
        result = _resolve_allomorph("ins", "бала", _KAZ_SUFFIXES, phon)
        assert result == "мен"

    def test_kazakh_possessive_1sg_after_vowel(self):
        phon = _Phonology("Cyrl")
        result = _resolve_allomorph("px1sg", "бала", _KAZ_SUFFIXES, phon)
        assert result == "м"

    def test_kazakh_possessive_1sg_after_consonant(self):
        phon = _Phonology("Cyrl")
        result = _resolve_allomorph("px1sg", "мектеп", _KAZ_SUFFIXES, phon)
        assert result == "ім"

    def test_turkish_plural(self):
        phon = _Phonology("Latn")
        result = _resolve_allomorph("pl", "araba", _TUR_SUFFIXES, phon)
        assert result == "lar"
        result = _resolve_allomorph("pl", "ev", _TUR_SUFFIXES, phon)
        assert result == "ler"

    def test_turkish_accusative_4way(self):
        phon = _Phonology("Latn")
        # After consonant: 4-way
        assert _resolve_allomorph("acc", "ev", _TUR_SUFFIXES, phon) == "i"
        assert _resolve_allomorph("acc", "okul", _TUR_SUFFIXES, phon) == "u"

    def test_turkish_locative(self):
        phon = _Phonology("Latn")
        assert _resolve_allomorph("loc", "ev", _TUR_SUFFIXES, phon) == "de"
        assert _resolve_allomorph("loc", "kitap", _TUR_SUFFIXES, phon) == "ta"


# ═══════════════════════════════════════════════════════════════════════════
# Stem boundary detection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStemBoundary:
    """Test stem boundary finding."""

    def test_exact_prefix(self):
        # үй is exact prefix of үйлер
        assert _find_stem_boundary("үйлер", "үй", "kaz") == 2

    def test_exact_prefix_latin(self):
        assert _find_stem_boundary("evler", "ev", "tur") == 2

    def test_consonant_alternation_turkish(self):
        # kitap → kitab- (p→b before vowel suffix)
        assert _find_stem_boundary("kitabı", "kitap", "tur") == 5

    def test_consonant_alternation_kazakh(self):
        # мектеп → мектеб- (п→б)
        assert _find_stem_boundary("мектебі", "мектеп", "kaz") == 6

    def test_no_suffix(self):
        # Word equals lemma
        assert _find_stem_boundary("бала", "бала", "kaz") == 4

    def test_fallback_lcp(self):
        # Partial match via longest common prefix
        assert _find_stem_boundary("gidiyor", "gitmek", "tur") >= 1

    def test_no_common_prefix_returns_1(self):
        # Edge case: completely different lemma → at least 1 char stem
        assert _find_stem_boundary("xyz", "abc", "tur") == 1

    def test_turkish_dotted_i(self):
        """Turkish İstanbul → İ should lowercase to i, not ı."""
        from turkicnlp.processors.morpheme_tokenizer import _turkic_lower
        assert _turkic_lower("İstanbul") == "istanbul"
        assert _turkic_lower("IRAK") == "ırak"  # I → ı in Turkic


# ═══════════════════════════════════════════════════════════════════════════
# Tag-based segmentation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSegmentByTags:
    """Test segmentation using Apertium tag sequences."""

    def test_kazakh_simple_plural(self):
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_tags(
            "үйлер", 2, ["pl"], _KAZ_SUFFIXES, phon,
        )
        assert [m.surface for m in morphemes] == ["үй", "лер"]
        assert morphemes[0].label == "STEM"
        assert morphemes[1].label == "PLUR"

    def test_kazakh_dative(self):
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_tags(
            "мектепке", 6, ["dat"], _KAZ_SUFFIXES, phon,
        )
        assert [m.surface for m in morphemes] == ["мектеп", "ке"]

    def test_kazakh_plural_possessive_case(self):
        phon = _Phonology("Cyrl")
        # үйлеріңіздегілерден
        # stem=үй, tags: pl, px2sg_frm, loc, subst, pl, abl
        morphemes = _segment_by_tags(
            "үйлеріңіздегілерден",
            2,  # stem boundary after "үй"
            ["pl", "px2sg_frm", "loc", "subst", "pl", "abl"],
            _KAZ_SUFFIXES,
            phon,
        )
        segments = [m.surface for m in morphemes]
        assert segments[0] == "үй"
        assert segments[1] == "лер"
        # The remaining segments should cover the rest
        assert "".join(segments) == "үйлеріңіздегілерден"

    def test_turkish_simple(self):
        phon = _Phonology("Latn")
        morphemes = _segment_by_tags(
            "evler", 2, ["pl"], _TUR_SUFFIXES, phon,
        )
        assert [m.surface for m in morphemes] == ["ev", "ler"]

    def test_turkish_locative(self):
        phon = _Phonology("Latn")
        morphemes = _segment_by_tags(
            "evde", 2, ["loc"], _TUR_SUFFIXES, phon,
        )
        assert [m.surface for m in morphemes] == ["ev", "de"]


class TestSegmentByUDFeatures:
    """Test segmentation using UD features (neural model fallback)."""

    def test_noun_plural_ablative(self):
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_ud_features(
            "үйлерден", 2, "NOUN", "Case=Abl|Number=Plur",
            _KAZ_SUFFIXES, phon,
        )
        segments = [m.surface for m in morphemes]
        assert segments[0] == "үй"
        assert "лер" in segments
        assert "ден" in segments

    def test_noun_possessive(self):
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_ud_features(
            "үйім", 2, "NOUN",
            "Number[psor]=Sing|Person[psor]=1",
            _KAZ_SUFFIXES, phon,
        )
        segments = [m.surface for m in morphemes]
        assert segments[0] == "үй"
        assert "ім" in segments

    def test_noun_possessive_compound_features(self):
        """Compound UD features like Number[psor]=Sing|Person[psor]=2 must match."""
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_ud_features(
            "үйің", 2, "NOUN",
            "Number[psor]=Sing|Person[psor]=2",
            _KAZ_SUFFIXES, phon,
        )
        segments = [m.surface for m in morphemes]
        assert segments[0] == "үй"
        assert "ің" in segments

    def test_noun_plural_possessive_case(self):
        """Multiple UD features: plural + possessive + case."""
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_ud_features(
            "үйлерімнен", 2, "NOUN",
            "Case=Abl|Number=Plur|Number[psor]=Sing|Person[psor]=1",
            _KAZ_SUFFIXES, phon,
        )
        segments = [m.surface for m in morphemes]
        assert segments[0] == "үй"
        assert "лер" in segments

    def test_no_features(self):
        phon = _Phonology("Cyrl")
        morphemes = _segment_by_ud_features(
            "бала", 4, "NOUN", "_", _KAZ_SUFFIXES, phon,
        )
        assert len(morphemes) == 1
        assert morphemes[0].surface == "бала"


# ═══════════════════════════════════════════════════════════════════════════
# MorphemeTokenizer class tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMorphemeTokenizerInit:
    """Test MorphemeTokenizer initialization."""

    def test_supported_langs(self):
        # All 23 entries in LANG_SUFFIX_MAP
        assert len(MorphemeTokenizer.SUPPORTED_LANGS) >= 20

    def test_init_kazakh(self):
        tok = MorphemeTokenizer(lang="kaz")
        assert tok.lang == "kaz"
        assert tok._script == "Cyrl"

    def test_init_turkish(self):
        tok = MorphemeTokenizer(lang="tur")
        assert tok.lang == "tur"
        assert tok._script == "Latn"

    def test_init_uzbek(self):
        tok = MorphemeTokenizer(lang="uzb")
        assert tok.lang == "uzb"
        assert tok._script == "Latn"

    def test_init_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            MorphemeTokenizer(lang="xxx")

    def test_not_loaded_raises(self):
        tok = MorphemeTokenizer(lang="kaz")
        with pytest.raises(RuntimeError, match="load"):
            tok.segment("үйлер")


class TestSegmentationResult:
    """Test SegmentationResult dataclass."""

    def test_segments_property(self):
        result = SegmentationResult(
            word="үйлер",
            morphemes=[
                Morpheme(surface="үй", label="STEM"),
                Morpheme(surface="лер", label="PLUR"),
            ],
            source="test",
        )
        assert result.segments == ["үй", "лер"]

    def test_labeled_property(self):
        result = SegmentationResult(
            word="үйлер",
            morphemes=[
                Morpheme(surface="үй", label="STEM"),
                Morpheme(surface="лер", label="PLUR"),
            ],
            source="test",
        )
        assert result.labeled == [("үй", "STEM"), ("лер", "PLUR")]


# ═══════════════════════════════════════════════════════════════════════════
# Suffix table coverage tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSuffixTableCoverage:
    """Ensure all languages have suffix tables and basic entries."""

    @pytest.mark.parametrize("lang", list(_LANG_SUFFIX_MAP.keys()))
    def test_language_has_suffix_table(self, lang):
        table, script = _LANG_SUFFIX_MAP[lang]
        assert isinstance(table, dict)
        assert script in ("Cyrl", "Latn", "Arab")

    @pytest.mark.parametrize("lang", list(_LANG_SUFFIX_MAP.keys()))
    def test_language_has_plural(self, lang):
        table, _ = _LANG_SUFFIX_MAP[lang]
        assert "pl" in table, f"{lang} missing plural suffix"

    @pytest.mark.parametrize("lang", list(_LANG_SUFFIX_MAP.keys()))
    def test_language_has_cases(self, lang):
        table, _ = _LANG_SUFFIX_MAP[lang]
        for case in ["acc", "dat", "loc", "abl"]:
            assert case in table, f"{lang} missing {case} case suffix"
