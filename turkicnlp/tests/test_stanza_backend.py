"""Tests for Stanza backend processors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from turkicnlp.models.document import Document, Sentence, Token, Word
from turkicnlp.processors.stanza_backend import (
    STANZA_SUPPORTED_LANGUAGES,
    StanzaDepParser,
    StanzaLemmatizer,
    StanzaMWTExpander,
    StanzaNERProcessor,
    StanzaPOSTagger,
    StanzaTokenizer,
    _StanzaManager,
    _get_stanza_lang,
    _is_custom_stanza,
    _build_custom_kwargs,
    _get_custom_model_dir,
    _LANG_MAP,
    _CUSTOM_STANZA_LANGS,
    _CUSTOM_MODEL_PATH_KWARGS,
    _CUSTOM_PRETRAIN_KWARGS,
)


# ---------------------------------------------------------------------------
# Unit tests (no Stanza installation required)
# ---------------------------------------------------------------------------


class TestLangMapping:
    def test_supported_languages(self):
        assert "tur" in STANZA_SUPPORTED_LANGUAGES
        assert "kaz" in STANZA_SUPPORTED_LANGUAGES
        assert "uig" in STANZA_SUPPORTED_LANGUAGES
        assert "kir" in STANZA_SUPPORTED_LANGUAGES
        assert "ota" in STANZA_SUPPORTED_LANGUAGES
        assert "uzb" in STANZA_SUPPORTED_LANGUAGES

    def test_get_stanza_lang(self):
        assert _get_stanza_lang("tur") == "tr"
        assert _get_stanza_lang("kaz") == "kk"
        assert _get_stanza_lang("uig") == "ug"
        assert _get_stanza_lang("kir") == "ky"
        assert _get_stanza_lang("ota") == "ota"
        assert _get_stanza_lang("uzb") == "uz"

    def test_get_stanza_lang_unsupported(self):
        with pytest.raises(ValueError, match="not supported"):
            _get_stanza_lang("eng")

    def test_lang_map_consistency(self):
        assert set(_LANG_MAP.keys()) == STANZA_SUPPORTED_LANGUAGES


class TestCustomStanzaUnit:
    """Unit tests for the custom-trained Stanza model infrastructure."""

    def test_custom_stanza_langs(self):
        assert "uzb" in _CUSTOM_STANZA_LANGS
        assert "tur" not in _CUSTOM_STANZA_LANGS
        assert "kaz" not in _CUSTOM_STANZA_LANGS
        assert "kir" not in _CUSTOM_STANZA_LANGS

    def test_is_custom_stanza(self):
        assert _is_custom_stanza("uzb") is True
        assert _is_custom_stanza("tur") is False
        assert _is_custom_stanza("kaz") is False
        assert _is_custom_stanza("kir") is False
        assert _is_custom_stanza("uig") is False

    def test_custom_model_dir(self):
        model_dir = _get_custom_model_dir("uzb")
        assert model_dir.name == "uzb"
        assert model_dir.parent.name == "stanza_custom"

    def test_custom_model_path_kwargs_complete(self):
        """All four processor types have model path kwargs."""
        assert "tokenize" in _CUSTOM_MODEL_PATH_KWARGS
        assert "pos" in _CUSTOM_MODEL_PATH_KWARGS
        assert "lemma" in _CUSTOM_MODEL_PATH_KWARGS
        assert "depparse" in _CUSTOM_MODEL_PATH_KWARGS

    def test_custom_pretrain_kwargs(self):
        """pos and depparse need pretrain paths."""
        assert "pos" in _CUSTOM_PRETRAIN_KWARGS
        assert "depparse" in _CUSTOM_PRETRAIN_KWARGS
        assert "tokenize" not in _CUSTOM_PRETRAIN_KWARGS
        assert "lemma" not in _CUSTOM_PRETRAIN_KWARGS

    def test_build_custom_kwargs_with_files(self, tmp_path):
        """Test _build_custom_kwargs produces correct pipeline kwargs."""
        # Create fake model files
        model_dir = tmp_path / "stanza_custom" / "uzb"
        model_dir.mkdir(parents=True)
        (model_dir / "tokenizer.pt").write_bytes(b"fake")
        (model_dir / "tagger.pt").write_bytes(b"fake")
        (model_dir / "lemmatizer.pt").write_bytes(b"fake")
        (model_dir / "parser.pt").write_bytes(b"fake")
        (model_dir / "pretrain.pt").write_bytes(b"fake")

        with patch(
            "turkicnlp.processors.stanza_backend.ModelRegistry"
        ) as mock_reg:
            mock_reg.default_dir.return_value = tmp_path
            kwargs = _build_custom_kwargs(
                "uzb", ["tokenize", "pos", "lemma", "depparse"]
            )

        assert kwargs["lang"] == "uz"
        assert kwargs["allow_unknown_language"] is True
        assert kwargs["processors"] == "tokenize,pos,lemma,depparse"
        assert kwargs["tokenize_model_path"] == str(model_dir / "tokenizer.pt")
        assert kwargs["pos_model_path"] == str(model_dir / "tagger.pt")
        assert kwargs["lemma_model_path"] == str(model_dir / "lemmatizer.pt")
        assert kwargs["depparse_model_path"] == str(model_dir / "parser.pt")
        assert kwargs["pos_pretrain_path"] == str(model_dir / "pretrain.pt")
        assert kwargs["depparse_pretrain_path"] == str(model_dir / "pretrain.pt")

    def test_build_custom_kwargs_missing_model(self, tmp_path):
        """FileNotFoundError when model file is missing."""
        model_dir = tmp_path / "stanza_custom" / "uzb"
        model_dir.mkdir(parents=True)
        # No files created

        with patch(
            "turkicnlp.processors.stanza_backend.ModelRegistry"
        ) as mock_reg:
            mock_reg.default_dir.return_value = tmp_path
            with pytest.raises(FileNotFoundError, match="tokenizer.pt"):
                _build_custom_kwargs("uzb", ["tokenize"])

    def test_build_custom_kwargs_missing_pretrain(self, tmp_path):
        """FileNotFoundError when pretrain file is missing."""
        model_dir = tmp_path / "stanza_custom" / "uzb"
        model_dir.mkdir(parents=True)
        (model_dir / "tagger.pt").write_bytes(b"fake")
        # pretrain.pt not created

        with patch(
            "turkicnlp.processors.stanza_backend.ModelRegistry"
        ) as mock_reg:
            mock_reg.default_dir.return_value = tmp_path
            with pytest.raises(FileNotFoundError, match="pretrain.pt"):
                _build_custom_kwargs("uzb", ["pos"])

    def test_build_custom_kwargs_subset(self, tmp_path):
        """Only requested processors appear in the kwargs."""
        model_dir = tmp_path / "stanza_custom" / "uzb"
        model_dir.mkdir(parents=True)
        (model_dir / "tokenizer.pt").write_bytes(b"fake")

        with patch(
            "turkicnlp.processors.stanza_backend.ModelRegistry"
        ) as mock_reg:
            mock_reg.default_dir.return_value = tmp_path
            kwargs = _build_custom_kwargs("uzb", ["tokenize"])

        assert kwargs["processors"] == "tokenize"
        assert "tokenize_model_path" in kwargs
        assert "pos_model_path" not in kwargs
        assert "pos_pretrain_path" not in kwargs

    def test_ner_not_available_for_custom(self):
        """NER pipelines raise ValueError for custom stanza languages."""
        with pytest.raises(ValueError, match="NER is not available"):
            _StanzaManager.get_full_ner_pipeline("uzb")

        with pytest.raises(ValueError, match="NER is not available"):
            _StanzaManager.get_pretokenized_ner_pipeline("uzb")


class TestStanzaTokenizerUnit:
    def test_class_attributes(self):
        assert StanzaTokenizer.NAME == "tokenize"
        assert StanzaTokenizer.PROVIDES == ["tokenize"]
        assert StanzaTokenizer.REQUIRES == []

    def test_process_maps_sentences_and_tokens_turkish(self):
        """Test that StanzaTokenizer correctly maps Stanza output to Document."""
        tokenizer = StanzaTokenizer(lang="tur")
        tokenizer._use_gpu = False
        tokenizer._loaded = True

        # Build a mock Stanza Document
        mock_word = SimpleNamespace(text="Merhaba")
        mock_token = SimpleNamespace(
            words=[mock_word],
            start_char=0,
            end_char=7,
        )
        mock_sent = SimpleNamespace(
            text="Merhaba",
            tokens=[mock_token],
        )
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        doc = Document(text="Merhaba", lang="tur")

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tokenizer.process(doc)

        assert len(result.sentences) == 1
        assert len(result.sentences[0].words) == 1
        assert result.sentences[0].words[0].text == "Merhaba"
        assert result.sentences[0].words[0].id == 1
        assert result.sentences[0].words[0].start_char == 0
        assert result.sentences[0].words[0].end_char == 7
        assert len(result.sentences[0].tokens) == 1
        assert result.sentences[0].tokens[0].id == (1,)
        assert "tokenize:stanza" in result._processor_log

    def test_process_maps_sentences_kazakh(self):
        """Test tokenizer with Kazakh Cyrillic text."""
        tokenizer = StanzaTokenizer(lang="kaz")
        tokenizer._use_gpu = False
        tokenizer._loaded = True

        mock_w1 = SimpleNamespace(text="Мен")
        mock_w2 = SimpleNamespace(text="бардым")
        mock_t1 = SimpleNamespace(words=[mock_w1], start_char=0, end_char=3)
        mock_t2 = SimpleNamespace(words=[mock_w2], start_char=4, end_char=10)
        mock_sent = SimpleNamespace(
            text="Мен бардым", tokens=[mock_t1, mock_t2]
        )
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        doc = Document(text="Мен бардым", lang="kaz")
        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tokenizer.process(doc)

        assert len(result.sentences) == 1
        assert len(result.sentences[0].words) == 2
        assert result.sentences[0].words[0].text == "Мен"
        assert result.sentences[0].words[1].text == "бардым"

    def test_process_maps_sentences_kyrgyz(self):
        """Test tokenizer with Kyrgyz Cyrillic text."""
        tokenizer = StanzaTokenizer(lang="kir")
        tokenizer._use_gpu = False
        tokenizer._loaded = True

        mock_w1 = SimpleNamespace(text="Мен")
        mock_w2 = SimpleNamespace(text="бардым")
        mock_t1 = SimpleNamespace(words=[mock_w1], start_char=0, end_char=3)
        mock_t2 = SimpleNamespace(words=[mock_w2], start_char=4, end_char=10)
        mock_sent = SimpleNamespace(
            text="Мен бардым", tokens=[mock_t1, mock_t2]
        )
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        doc = Document(text="Мен бардым", lang="kir")
        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tokenizer.process(doc)

        assert len(result.sentences) == 1
        assert len(result.sentences[0].words) == 2
        assert result.sentences[0].words[0].text == "Мен"
        assert result.sentences[0].words[1].text == "бардым"

    def test_process_maps_sentences_uzbek(self):
        """Test tokenizer with Uzbek Latin text."""
        tokenizer = StanzaTokenizer(lang="uzb")
        tokenizer._use_gpu = False
        tokenizer._loaded = True

        mock_w1 = SimpleNamespace(text="Men")
        mock_w2 = SimpleNamespace(text="ketdim")
        mock_w3 = SimpleNamespace(text=".")
        mock_t1 = SimpleNamespace(words=[mock_w1], start_char=0, end_char=3)
        mock_t2 = SimpleNamespace(words=[mock_w2], start_char=4, end_char=10)
        mock_t3 = SimpleNamespace(words=[mock_w3], start_char=10, end_char=11)
        mock_sent = SimpleNamespace(
            text="Men ketdim.",
            tokens=[mock_t1, mock_t2, mock_t3],
        )
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        doc = Document(text="Men ketdim.", lang="uzb")
        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tokenizer.process(doc)

        assert len(result.sentences) == 1
        assert len(result.sentences[0].words) == 3
        assert result.sentences[0].words[0].text == "Men"
        assert result.sentences[0].words[1].text == "ketdim"
        assert result.sentences[0].words[2].text == "."

    def test_process_maps_mwt(self):
        """Test MWT handling: one token with multiple words."""
        tokenizer = StanzaTokenizer(lang="tur")
        tokenizer._use_gpu = False
        tokenizer._loaded = True

        mock_word1 = SimpleNamespace(text="ev")
        mock_word2 = SimpleNamespace(text="de")
        mock_mwt_token = SimpleNamespace(
            text="evde",
            words=[mock_word1, mock_word2],
            start_char=0,
            end_char=4,
        )
        mock_sent = SimpleNamespace(
            text="evde",
            tokens=[mock_mwt_token],
        )
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        doc = Document(text="evde", lang="tur")

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tokenizer.process(doc)

        assert len(result.sentences[0].tokens) == 1
        token = result.sentences[0].tokens[0]
        assert token.is_mwt
        assert token.id == (1, 2)
        assert token.text == "evde"
        assert len(token.words) == 2
        assert token.words[0].text == "ev"
        assert token.words[0].id == 1
        assert token.words[1].text == "de"
        assert token.words[1].id == 2
        assert len(result.sentences[0].words) == 2


class TestStanzaPOSTaggerUnit:
    def test_class_attributes(self):
        assert StanzaPOSTagger.NAME == "pos"
        assert StanzaPOSTagger.PROVIDES == ["pos", "feats"]
        assert StanzaPOSTagger.REQUIRES == ["tokenize"]

    def test_process_full_mode_turkish(self):
        """Test POS tagger in full Stanza mode (tokenize:stanza in log)."""
        tagger = StanzaPOSTagger(lang="tur")
        tagger._use_gpu = False
        tagger._loaded = True

        # Pre-populate document with tokenization
        w1 = Word(id=1, text="Merhaba")
        w2 = Word(id=2, text="dünya")
        sent = Sentence(text="Merhaba dünya", tokens=[], words=[w1, w2])
        doc = Document(text="Merhaba dünya", lang="tur", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        # Mock Stanza output
        mock_w1 = SimpleNamespace(upos="INTJ", xpos="Interj", feats="")
        mock_w2 = SimpleNamespace(upos="NOUN", xpos="Noun", feats="Case=Nom|Number=Sing")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tagger.process(doc)

        assert result.sentences[0].words[0].upos == "INTJ"
        assert result.sentences[0].words[0].xpos == "Interj"
        assert result.sentences[0].words[1].upos == "NOUN"
        assert result.sentences[0].words[1].feats == "Case=Nom|Number=Sing"
        assert "pos:stanza" in result._processor_log

    def test_process_full_mode_kazakh(self):
        """Test POS tagger with Kazakh text."""
        tagger = StanzaPOSTagger(lang="kaz")
        tagger._use_gpu = False
        tagger._loaded = True

        w1 = Word(id=1, text="Мен")
        w2 = Word(id=2, text="бардым")
        sent = Sentence(text="Мен бардым", tokens=[], words=[w1, w2])
        doc = Document(text="Мен бардым", lang="kaz", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(upos="PRON", xpos=None, feats="Case=Nom|Number=Sing|Person=1|PronType=Prs")
        mock_w2 = SimpleNamespace(upos="VERB", xpos=None, feats="Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tagger.process(doc)

        assert result.sentences[0].words[0].upos == "PRON"
        assert result.sentences[0].words[1].upos == "VERB"

    def test_process_full_mode_uzbek(self):
        """Test POS tagger with Uzbek text (custom stanza model)."""
        tagger = StanzaPOSTagger(lang="uzb")
        tagger._use_gpu = False
        tagger._loaded = True

        w1 = Word(id=1, text="Men")
        w2 = Word(id=2, text="maktabga")
        w3 = Word(id=3, text="ketdim")
        sent = Sentence(text="Men maktabga ketdim", tokens=[], words=[w1, w2, w3])
        doc = Document(text="Men maktabga ketdim", lang="uzb", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(upos="PRON", xpos=None, feats="Case=Nom|Number=Sing|Person=1|PronType=Prs")
        mock_w2 = SimpleNamespace(upos="NOUN", xpos=None, feats="Case=Dat|Number=Sing")
        mock_w3 = SimpleNamespace(upos="VERB", xpos=None, feats="Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2, mock_w3])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = tagger.process(doc)

        assert result.sentences[0].words[0].upos == "PRON"
        assert result.sentences[0].words[1].upos == "NOUN"
        assert result.sentences[0].words[2].upos == "VERB"
        assert "pos:stanza" in result._processor_log

    def test_process_pretokenized_mode(self):
        """Test POS tagger in pretokenized mode (non-Stanza tokenizer)."""
        tagger = StanzaPOSTagger(lang="tur")
        tagger._use_gpu = False
        tagger._loaded = True

        w1 = Word(id=1, text="Merhaba", upos=None)
        sent = Sentence(text="Merhaba", tokens=[], words=[w1])
        doc = Document(text="Merhaba", lang="tur", sentences=[sent])
        doc._processor_log.append("tokenize:regex")

        mock_w1 = SimpleNamespace(upos="INTJ", xpos=None, feats="")
        mock_sent = SimpleNamespace(words=[mock_w1])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_pretokenized", return_value=mock_stanza_doc
        ):
            result = tagger.process(doc)

        assert result.sentences[0].words[0].upos == "INTJ"


class TestStanzaLemmatizerUnit:
    def test_class_attributes(self):
        assert StanzaLemmatizer.NAME == "lemma"
        assert StanzaLemmatizer.PROVIDES == ["lemma"]

    def test_process_turkish(self):
        lemmatizer = StanzaLemmatizer(lang="tur")
        lemmatizer._use_gpu = False
        lemmatizer._loaded = True

        w1 = Word(id=1, text="evleri", upos="NOUN")
        sent = Sentence(text="evleri", tokens=[], words=[w1])
        doc = Document(text="evleri", lang="tur", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(lemma="ev")
        mock_sent = SimpleNamespace(words=[mock_w1])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = lemmatizer.process(doc)

        assert result.sentences[0].words[0].lemma == "ev"
        assert "lemma:stanza" in result._processor_log

    def test_process_kazakh(self):
        lemmatizer = StanzaLemmatizer(lang="kaz")
        lemmatizer._use_gpu = False
        lemmatizer._loaded = True

        w1 = Word(id=1, text="мектепке", upos="NOUN")
        sent = Sentence(text="мектепке", tokens=[], words=[w1])
        doc = Document(text="мектепке", lang="kaz", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(lemma="мектеп")
        mock_sent = SimpleNamespace(words=[mock_w1])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = lemmatizer.process(doc)

        assert result.sentences[0].words[0].lemma == "мектеп"

    def test_process_kyrgyz(self):
        lemmatizer = StanzaLemmatizer(lang="kir")
        lemmatizer._use_gpu = False
        lemmatizer._loaded = True

        w1 = Word(id=1, text="мектепке", upos="NOUN")
        sent = Sentence(text="мектепке", tokens=[], words=[w1])
        doc = Document(text="мектепке", lang="kir", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(lemma="мектеп")
        mock_sent = SimpleNamespace(words=[mock_w1])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = lemmatizer.process(doc)

        assert result.sentences[0].words[0].lemma == "мектеп"

    def test_process_uzbek(self):
        lemmatizer = StanzaLemmatizer(lang="uzb")
        lemmatizer._use_gpu = False
        lemmatizer._loaded = True

        w1 = Word(id=1, text="maktabga", upos="NOUN")
        sent = Sentence(text="maktabga", tokens=[], words=[w1])
        doc = Document(text="maktabga", lang="uzb", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(lemma="maktab")
        mock_sent = SimpleNamespace(words=[mock_w1])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = lemmatizer.process(doc)

        assert result.sentences[0].words[0].lemma == "maktab"


class TestStanzaDepParserUnit:
    def test_class_attributes(self):
        assert StanzaDepParser.NAME == "depparse"
        assert StanzaDepParser.PROVIDES == ["depparse"]

    def test_process_turkish(self):
        parser = StanzaDepParser(lang="tur")
        parser._use_gpu = False
        parser._loaded = True

        w1 = Word(id=1, text="Merhaba", upos="INTJ")
        w2 = Word(id=2, text="dünya", upos="NOUN")
        sent = Sentence(text="Merhaba dünya", tokens=[], words=[w1, w2])
        doc = Document(text="Merhaba dünya", lang="tur", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(head=0, deprel="root")
        mock_w2 = SimpleNamespace(head=1, deprel="vocative")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = parser.process(doc)

        assert result.sentences[0].words[0].head == 0
        assert result.sentences[0].words[0].deprel == "root"
        assert result.sentences[0].words[1].head == 1
        assert result.sentences[0].words[1].deprel == "vocative"
        assert "depparse:stanza" in result._processor_log

    def test_process_kazakh(self):
        parser = StanzaDepParser(lang="kaz")
        parser._use_gpu = False
        parser._loaded = True

        w1 = Word(id=1, text="Мен", upos="PRON")
        w2 = Word(id=2, text="бардым", upos="VERB")
        sent = Sentence(text="Мен бардым", tokens=[], words=[w1, w2])
        doc = Document(text="Мен бардым", lang="kaz", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(head=2, deprel="nsubj")
        mock_w2 = SimpleNamespace(head=0, deprel="root")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = parser.process(doc)

        assert result.sentences[0].words[0].head == 2
        assert result.sentences[0].words[0].deprel == "nsubj"
        assert result.sentences[0].words[1].head == 0
        assert result.sentences[0].words[1].deprel == "root"

    def test_process_uzbek(self):
        """Test depparse with Uzbek (custom stanza model)."""
        parser = StanzaDepParser(lang="uzb")
        parser._use_gpu = False
        parser._loaded = True

        w1 = Word(id=1, text="Men", upos="PRON")
        w2 = Word(id=2, text="maktabga", upos="NOUN")
        w3 = Word(id=3, text="ketdim", upos="VERB")
        sent = Sentence(
            text="Men maktabga ketdim", tokens=[], words=[w1, w2, w3]
        )
        doc = Document(
            text="Men maktabga ketdim", lang="uzb", sentences=[sent]
        )
        doc._processor_log.append("tokenize:stanza")

        mock_w1 = SimpleNamespace(head=3, deprel="nsubj")
        mock_w2 = SimpleNamespace(head=3, deprel="obl")
        mock_w3 = SimpleNamespace(head=0, deprel="root")
        mock_sent = SimpleNamespace(words=[mock_w1, mock_w2, mock_w3])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(
            _StanzaManager, "run_full", return_value=mock_stanza_doc
        ):
            result = parser.process(doc)

        assert result.sentences[0].words[0].deprel == "nsubj"
        assert result.sentences[0].words[1].deprel == "obl"
        assert result.sentences[0].words[2].deprel == "root"
        assert result.sentences[0].words[2].head == 0
        assert "depparse:stanza" in result._processor_log


class TestStanzaNERUnit:
    def test_class_attributes(self):
        assert StanzaNERProcessor.NAME == "ner"
        assert StanzaNERProcessor.PROVIDES == ["ner"]
        assert StanzaNERProcessor.REQUIRES == ["tokenize"]

    def test_process_maps_bio_and_entities(self):
        ner = StanzaNERProcessor(lang="tur")
        ner._use_gpu = False
        ner._loaded = True

        w1 = Word(id=1, text="Ahmet", start_char=0, end_char=5)
        w2 = Word(id=2, text="Yılmaz", start_char=6, end_char=12)
        w3 = Word(id=3, text="geldi", start_char=13, end_char=18)
        sent = Sentence(text="Ahmet Yılmaz geldi", tokens=[], words=[w1, w2, w3])
        doc = Document(text=sent.text, lang="tur", sentences=[sent])
        doc._processor_log.append("tokenize:stanza")

        tok1 = SimpleNamespace(ner="B-PER", words=[SimpleNamespace(text="Ahmet")])
        tok2 = SimpleNamespace(ner="E-PER", words=[SimpleNamespace(text="Yılmaz")])
        tok3 = SimpleNamespace(ner="O", words=[SimpleNamespace(text="geldi")])
        mock_sent = SimpleNamespace(tokens=[tok1, tok2, tok3])
        mock_stanza_doc = SimpleNamespace(sentences=[mock_sent])

        with patch.object(_StanzaManager, "run_full_ner", return_value=mock_stanza_doc):
            result = ner.process(doc)

        assert result.sentences[0].words[0].ner == "B-PER"
        assert result.sentences[0].words[1].ner == "I-PER"
        assert result.sentences[0].words[2].ner == "O"
        assert len(result.sentences[0].entities) == 1
        assert result.sentences[0].entities[0].text == "Ahmet Yılmaz"
        assert result.sentences[0].entities[0].type == "PER"
        assert "ner:stanza" in result._processor_log


class TestStanzaMWTExpanderUnit:
    def test_process_is_passthrough(self):
        expander = StanzaMWTExpander(lang="tur")
        expander._loaded = True

        doc = Document(text="test", lang="tur", sentences=[])
        result = expander.process(doc)
        assert "mwt:stanza" in result._processor_log


class TestStanzaManagerUnit:
    def test_clear(self):
        _StanzaManager._full_pipelines[("test", False)] = "pipeline"
        _StanzaManager._pretok_pipelines[("test", False)] = "pipeline"
        _StanzaManager.clear()
        assert len(_StanzaManager._full_pipelines) == 0
        assert len(_StanzaManager._pretok_pipelines) == 0

    def test_run_full_uses_cache(self):
        """Test that repeated calls reuse the cached Stanza result."""
        doc = Document(text="test", lang="tur")
        mock_result = SimpleNamespace(sentences=[])
        doc._stanza_full_cache = mock_result  # type: ignore[attr-defined]

        result = _StanzaManager.run_full(doc)
        assert result is mock_result

    def test_run_pretokenized_uses_cache(self):
        doc = Document(text="test", lang="tur", sentences=[])
        mock_result = SimpleNamespace(sentences=[])
        doc._stanza_pretok_cache = mock_result  # type: ignore[attr-defined]

        result = _StanzaManager.run_pretokenized(doc)
        assert result is mock_result


class TestCatalogIntegration:
    """Test that the catalog lists Stanza backends for all supported languages."""

    def test_uzbek_stanza_in_catalog(self):
        from turkicnlp.resources.downloader import list_processors

        procs = list_processors("uzb")
        assert "tokenize" in procs
        assert "stanza" in procs["tokenize"]
        assert "pos" in procs
        assert "stanza" in procs["pos"]
        assert "lemma" in procs
        assert "stanza" in procs["lemma"]
        assert "depparse" in procs
        assert "stanza" in procs["depparse"]

    def test_uzbek_stanza_custom_type_in_catalog(self):
        from turkicnlp.resources.registry import ModelRegistry

        catalog = ModelRegistry.load_catalog()
        uzb_procs = catalog["uzb"]["processors"]["Latn"]
        for proc_name in ("tokenize", "pos", "lemma", "depparse"):
            backend_info = uzb_procs[proc_name]["backends"]["stanza"]
            assert backend_info["type"] == "stanza_custom"
            assert "url" in backend_info
            assert "sha256" in backend_info

    def test_turkish_stanza_standard_type(self):
        from turkicnlp.resources.registry import ModelRegistry

        catalog = ModelRegistry.load_catalog()
        tur_procs = catalog["tur"]["processors"]["Latn"]
        assert tur_procs["tokenize"]["backends"]["stanza"]["type"] == "stanza"

    def test_kazakh_stanza_standard_type(self):
        from turkicnlp.resources.registry import ModelRegistry

        catalog = ModelRegistry.load_catalog()
        kaz_procs = catalog["kaz"]["processors"]["Cyrl"]
        assert kaz_procs["tokenize"]["backends"]["stanza"]["type"] == "stanza"

    def test_kyrgyz_stanza_standard_type(self):
        from turkicnlp.resources.registry import ModelRegistry

        catalog = ModelRegistry.load_catalog()
        kir_procs = catalog["kir"]["processors"]["Cyrl"]
        assert kir_procs["tokenize"]["backends"]["stanza"]["type"] == "stanza"


class TestRegistryIntegration:
    def test_stanza_backends_registered(self):
        from turkicnlp.resources.registry import ProcessorRegistry

        assert ProcessorRegistry.get("tokenize", "stanza") is StanzaTokenizer
        assert ProcessorRegistry.get("mwt", "stanza") is StanzaMWTExpander
        assert ProcessorRegistry.get("pos", "stanza") is StanzaPOSTagger
        assert ProcessorRegistry.get("lemma", "stanza") is StanzaLemmatizer
        assert ProcessorRegistry.get("depparse", "stanza") is StanzaDepParser
        assert ProcessorRegistry.get("ner", "stanza") is StanzaNERProcessor


# ---------------------------------------------------------------------------
# Integration tests (require Stanza to be installed + model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStanzaIntegration:
    """Integration tests that require Stanza and model downloads.

    Run with: pytest -m slow
    """

    @pytest.fixture(autouse=True)
    def clear_manager(self):
        _StanzaManager.clear()
        yield
        _StanzaManager.clear()

    def test_full_pipeline_turkish(self):
        """End-to-end test with Turkish text."""
        stanza = pytest.importorskip("stanza")

        doc = Document(text="Merhaba dünya.", lang="tur")

        tokenizer = StanzaTokenizer(lang="tur")
        tokenizer.load()

        pos_tagger = StanzaPOSTagger(lang="tur")
        pos_tagger.load()

        lemmatizer = StanzaLemmatizer(lang="tur")
        lemmatizer.load()

        dep_parser = StanzaDepParser(lang="tur")
        dep_parser.load()

        doc = tokenizer.process(doc)
        assert len(doc.sentences) >= 1
        assert len(doc.words) >= 2

        doc = pos_tagger.process(doc)
        assert all(w.upos is not None for w in doc.words)

        doc = lemmatizer.process(doc)
        assert all(w.lemma is not None for w in doc.words)

        doc = dep_parser.process(doc)
        assert all(w.head is not None for w in doc.words)
        assert all(w.deprel is not None for w in doc.words)

    def test_full_pipeline_kazakh(self):
        """End-to-end test with Kazakh Cyrillic text."""
        stanza = pytest.importorskip("stanza")

        doc = Document(text="Мен мектепке бардым.", lang="kaz")

        for ProcessorClass in [
            StanzaTokenizer,
            StanzaPOSTagger,
            StanzaLemmatizer,
            StanzaDepParser,
        ]:
            proc = ProcessorClass(lang="kaz")
            proc.load()
            doc = proc.process(doc)

        assert len(doc.sentences) >= 1
        assert len(doc.words) >= 3
        assert all(w.upos is not None for w in doc.words)
        assert all(w.lemma is not None for w in doc.words)
        assert all(w.head is not None for w in doc.words)

    def test_full_pipeline_kyrgyz(self):
        """End-to-end test with Kyrgyz Cyrillic text."""
        stanza = pytest.importorskip("stanza")

        doc = Document(text="Мен мектепке бардым.", lang="kir")

        for ProcessorClass in [
            StanzaTokenizer,
            StanzaPOSTagger,
            StanzaLemmatizer,
            StanzaDepParser,
        ]:
            proc = ProcessorClass(lang="kir")
            proc.load()
            doc = proc.process(doc)

        assert len(doc.sentences) >= 1
        assert len(doc.words) >= 3
        assert all(w.upos is not None for w in doc.words)
        assert all(w.lemma is not None for w in doc.words)
        assert all(w.head is not None for w in doc.words)

    def test_full_pipeline_uzbek(self):
        """End-to-end test with Uzbek Latin text (custom Stanza model)."""
        stanza = pytest.importorskip("stanza")

        # Ensure custom models are downloaded
        from turkicnlp.resources.downloader import download

        download("uzb", processors=["tokenize", "pos", "lemma", "depparse"])

        doc = Document(text="Men maktabga ketdim.", lang="uzb")

        for ProcessorClass in [
            StanzaTokenizer,
            StanzaPOSTagger,
            StanzaLemmatizer,
            StanzaDepParser,
        ]:
            proc = ProcessorClass(lang="uzb")
            proc.load()
            doc = proc.process(doc)

        assert len(doc.sentences) >= 1
        assert len(doc.words) >= 3
        assert all(w.upos is not None for w in doc.words)
        assert all(w.lemma is not None for w in doc.words)
        assert all(w.head is not None for w in doc.words)
        assert all(w.deprel is not None for w in doc.words)

    def test_pretokenized_mode_turkish(self):
        """Test Stanza POS/lemma/depparse with rule-based tokenizer."""
        stanza = pytest.importorskip("stanza")
        from turkicnlp.processors.tokenizer import RegexTokenizer

        doc = Document(text="Merhaba dünya.", lang="tur")

        # Tokenize with rule-based tokenizer
        tokenizer = RegexTokenizer(lang="tur")
        tokenizer.load()
        doc = tokenizer.process(doc)
        assert "tokenize:regex" in doc._processor_log

        # POS tag with Stanza (pretokenized mode)
        pos_tagger = StanzaPOSTagger(lang="tur")
        pos_tagger.load()
        doc = pos_tagger.process(doc)
        assert all(w.upos is not None for w in doc.words if w.upos != "PUNCT")

    def test_conllu_export_after_stanza_turkish(self):
        """Test that CoNLL-U export works after Stanza processing."""
        stanza = pytest.importorskip("stanza")

        doc = Document(text="Merhaba dünya.", lang="tur")

        for ProcessorClass in [
            StanzaTokenizer,
            StanzaPOSTagger,
            StanzaLemmatizer,
            StanzaDepParser,
        ]:
            proc = ProcessorClass(lang="tur")
            proc.load()
            doc = proc.process(doc)

        conllu = doc.to_conllu()
        assert "# text = " in conllu
        assert "NOUN" in conllu or "INTJ" in conllu

    def test_conllu_export_after_stanza_uzbek(self):
        """Test CoNLL-U export after custom Stanza processing for Uzbek."""
        stanza = pytest.importorskip("stanza")
        from turkicnlp.resources.downloader import download

        download("uzb", processors=["tokenize", "pos", "lemma", "depparse"])

        doc = Document(text="Men maktabga ketdim.", lang="uzb")

        for ProcessorClass in [
            StanzaTokenizer,
            StanzaPOSTagger,
            StanzaLemmatizer,
            StanzaDepParser,
        ]:
            proc = ProcessorClass(lang="uzb")
            proc.load()
            doc = proc.process(doc)

        conllu = doc.to_conllu()
        assert "# text = " in conllu
        assert "VERB" in conllu or "NOUN" in conllu

    def test_kazakh_pipeline(self):
        """Test Stanza with Kazakh text."""
        stanza = pytest.importorskip("stanza")

        doc = Document(text="Сәлем әлем.", lang="kaz")

        tokenizer = StanzaTokenizer(lang="kaz")
        tokenizer.load()
        doc = tokenizer.process(doc)
        assert len(doc.sentences) >= 1
        assert len(doc.words) >= 1
