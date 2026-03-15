"""Tests for the multilingual neural morphological analyzer backend."""

import pytest


class TestMorphModelImports:
    """Verify model and backend imports work correctly."""

    def test_import_morph_model(self):
        from turkicnlp.processors.multilingual_morph_model import (
            TurkicMorphAnalyzer,
            MorphFeatureHead,
            LemmaHead,
            CharCNN,
            EditScriptVocab,
            encode_chars,
            apply_edit_script,
            UD_MORPH_FEATS,
            NUM_MORPH_FEATS,
            SUPPORTED_LANGS,
        )
        assert NUM_MORPH_FEATS == 69
        assert len(UD_MORPH_FEATS) == 69
        assert len(SUPPORTED_LANGS) == 21

    def test_import_morph_backend(self):
        from turkicnlp.processors.multilingual_morph_backend import (
            MultilingualMorphAnalyzer,
            MultilingualMorphFeats,
            MultilingualMorphLemmatizer,
            _MorphAnalyzerManager,
        )
        assert MultilingualMorphAnalyzer.NAME == "morph_neural"
        assert MultilingualMorphFeats.NAME == "feats"
        assert MultilingualMorphLemmatizer.NAME == "lemma"

    def test_morph_analyzer_provides(self):
        from turkicnlp.processors.multilingual_morph_backend import (
            MultilingualMorphAnalyzer,
        )
        assert "pos" in MultilingualMorphAnalyzer.PROVIDES
        assert "feats" in MultilingualMorphAnalyzer.PROVIDES
        assert "lemma" in MultilingualMorphAnalyzer.PROVIDES

    def test_morph_analyzer_requires(self):
        from turkicnlp.processors.multilingual_morph_backend import (
            MultilingualMorphAnalyzer,
        )
        assert "tokenize" in MultilingualMorphAnalyzer.REQUIRES


class TestEditScriptVocab:
    """Test edit script vocabulary and application."""

    def test_edit_script_vocab_basics(self):
        from turkicnlp.processors.multilingual_morph_model import EditScriptVocab
        vocab = EditScriptVocab()
        assert len(vocab) == 2  # <unk> and 0
        assert vocab.decode(0) == "<unk>"
        assert vocab.decode(1) == "0"
        assert vocab.decode(999) == "<unk>"

    def test_edit_script_vocab_load_state(self):
        from turkicnlp.processors.multilingual_morph_model import EditScriptVocab
        vocab = EditScriptVocab()
        state = {"script_to_id": {"<unk>": 0, "0": 1, "-3+mek": 2, "-2+": 3}}
        vocab.load_state_dict(state)
        assert len(vocab) == 4
        assert vocab.decode(2) == "-3+mek"
        assert vocab.decode(3) == "-2+"

    def test_apply_edit_script_identity(self):
        from turkicnlp.processors.multilingual_morph_model import apply_edit_script
        assert apply_edit_script("geldi", "0") == "geldi"

    def test_apply_edit_script_suffix_strip(self):
        from turkicnlp.processors.multilingual_morph_model import apply_edit_script
        # "geldi" - remove 2 chars ("di"), add nothing → "gel"
        assert apply_edit_script("geldi", "-2+") == "gel"

    def test_apply_edit_script_suffix_replace(self):
        from turkicnlp.processors.multilingual_morph_model import apply_edit_script
        # "geliyorum" - remove 6 chars ("iyorum"), add "mek" → "gelmek"
        assert apply_edit_script("geliyorum", "-6+mek") == "gelmek"

    def test_apply_edit_script_turkish_i(self):
        from turkicnlp.processors.multilingual_morph_model import apply_edit_script
        # Should lowercase and NFC normalize
        result = apply_edit_script("İstanbul", "0")
        assert result == "i\u0307stanbul" or result == "istanbul" or "istanbul" in result.lower()


class TestEncodeChars:
    """Test character encoding for the lemma head."""

    def test_encode_chars_basic(self):
        from turkicnlp.processors.multilingual_morph_model import encode_chars
        words = [["hello", "world"]]
        char_ids = encode_chars(words, max_word_len=2, max_char_len=10)
        assert char_ids.shape == (1, 2, 10)
        # First char of "hello" is 'h' (ord 104)
        assert char_ids[0, 0, 0].item() == (ord('h') % 511) + 1
        # Padding should be 0
        assert char_ids[0, 0, 5].item() == 0

    def test_encode_chars_padding(self):
        from turkicnlp.processors.multilingual_morph_model import encode_chars
        words = [["a"]]
        char_ids = encode_chars(words, max_word_len=3, max_char_len=5)
        assert char_ids.shape == (1, 3, 5)
        # Second word position should be all zeros (padding)
        assert char_ids[0, 1, :].sum().item() == 0


class TestMorphLangResolution:
    """Test morph-specific language resolution for all 20 supported languages."""

    def test_resolve_base_languages(self):
        from turkicnlp.processors.multilingual_morph_model import resolve_morph_lang
        # Base 10 trained languages
        short, script = resolve_morph_lang("tur")
        assert short == "tr" and script == "Latn"
        short, script = resolve_morph_lang("kaz")
        assert short == "kk" and script == "Cyrl"
        short, script = resolve_morph_lang("uig")
        assert short == "ug" and script == "Arab"

    def test_resolve_extended_languages(self):
        from turkicnlp.processors.multilingual_morph_model import resolve_morph_lang
        # Extended 10 trained via UniMorph/Wiktionary
        short, script = resolve_morph_lang("crh")
        assert short == "crh" and script == "Latn"
        short, script = resolve_morph_lang("chv")
        assert short == "chv" and script == "Cyrl"
        short, script = resolve_morph_lang("gag")
        assert short == "gag" and script == "Latn"
        short, script = resolve_morph_lang("tyv")
        assert short == "tyv" and script == "Cyrl"
        short, script = resolve_morph_lang("alt")
        assert short == "alt" and script == "Cyrl"
        short, script = resolve_morph_lang("atv")
        assert short == "atv" and script == "Cyrl"
        short, script = resolve_morph_lang("klj")
        assert short == "klj" and script == "Latn"

    def test_resolve_karakalpak_zero_shot(self):
        from turkicnlp.processors.multilingual_morph_model import (
            resolve_morph_lang, LANG_ID_MAP,
        )
        short, script = resolve_morph_lang("kaa")
        assert short == "kaa" and script == "Latn"
        # Karakalpak uses Uzbek proxy embedding
        assert LANG_ID_MAP["kaa"] == LANG_ID_MAP["uz"]

    def test_resolve_unsupported_raises(self):
        from turkicnlp.processors.multilingual_morph_model import resolve_morph_lang
        import pytest
        with pytest.raises(ValueError):
            resolve_morph_lang("eng")

    def test_lang_id_map_has_all_20(self):
        from turkicnlp.processors.multilingual_morph_model import (
            LANG_ID_MAP, SUPPORTED_LANGS, ISO3_TO_SHORT,
        )
        for iso3 in SUPPORTED_LANGS:
            short = ISO3_TO_SHORT[iso3]
            assert short in LANG_ID_MAP, f"{iso3} ({short}) missing from LANG_ID_MAP"

    def test_proxy_embeddings_are_valid(self):
        from turkicnlp.processors.multilingual_morph_model import (
            LANG_ID_MAP, NUM_LANGS,
        )
        for short, lid in LANG_ID_MAP.items():
            assert 0 <= lid < NUM_LANGS, f"{short} has invalid lang_id {lid} (max {NUM_LANGS})"


class TestRegistration:
    """Test that morph processors are registered in the ProcessorRegistry."""

    def test_morph_neural_registered(self):
        from turkicnlp.resources.registry import ProcessorRegistry
        proc_class = ProcessorRegistry.get("morph_neural", "multilingual_glot500_morph")
        assert proc_class is not None
        assert proc_class.NAME == "morph_neural"

    def test_feats_registered(self):
        from turkicnlp.resources.registry import ProcessorRegistry
        proc_class = ProcessorRegistry.get("feats", "multilingual_glot500_morph")
        assert proc_class is not None
        assert proc_class.NAME == "feats"

    def test_lemma_morph_registered(self):
        from turkicnlp.resources.registry import ProcessorRegistry
        proc_class = ProcessorRegistry.get("lemma", "multilingual_glot500_morph")
        assert proc_class is not None
        assert proc_class.NAME == "lemma"


class TestCatalog:
    """Test that catalog has morph_neural entries for supported languages."""

    def test_catalog_has_morph_neural(self):
        from turkicnlp.resources.registry import ModelRegistry
        catalog = ModelRegistry.load_catalog()
        # Turkish should have morph_neural
        tur = catalog["tur"]["processors"]["Latn"]
        assert "morph_neural" in tur
        assert "multilingual_glot500_morph" in tur["morph_neural"]["backends"]

    def test_catalog_has_lemma_morph_backend(self):
        from turkicnlp.resources.registry import ModelRegistry
        catalog = ModelRegistry.load_catalog()
        tur = catalog["tur"]["processors"]["Latn"]
        assert "multilingual_glot500_morph" in tur["lemma"]["backends"]

    def test_catalog_kazakh_has_morph_neural(self):
        from turkicnlp.resources.registry import ModelRegistry
        catalog = ModelRegistry.load_catalog()
        kaz = catalog["kaz"]["processors"]["Cyrl"]
        assert "morph_neural" in kaz
        assert "multilingual_glot500_morph" in kaz["morph_neural"]["backends"]
