"""Tests for tokenizers."""

from __future__ import annotations

import pytest

from turkicnlp.models.document import Document
from turkicnlp.processors.tokenizer import RegexTokenizer, NeuralTokenizer
from turkicnlp.processors.tokenizer_arabic import ArabicScriptTokenizer
from turkicnlp.scripts import Script


class TestRegexTokenizer:
    def test_instantiation(self) -> None:
        tok = RegexTokenizer(lang="kaz")
        assert tok.NAME == "tokenize"
        assert tok.REQUIRES == []

    def test_provides(self) -> None:
        assert RegexTokenizer.PROVIDES == ["tokenize"]

    def test_process_splits_punct_and_offsets(self) -> None:
        tok = RegexTokenizer(lang="kaz")
        tok.load()
        doc = Document(text="Men bardım, ok? Boladi!")
        tok.process(doc)

        assert [s.text for s in doc.sentences] == ["Men bardım, ok?", "Boladi!"]

        words = doc.sentences[0].words
        assert [w.text for w in words] == ["Men", "bardım", ",", "ok", "?"]
        assert [(w.start_char, w.end_char) for w in words] == [
            (0, 3),   # Men
            (4, 10),  # bardım
            (10, 11), # ,
            (12, 14), # ok
            (14, 15), # ?
        ]
        assert words[2].upos == "PUNCT"
        assert words[4].upos == "PUNCT"

        words2 = doc.sentences[1].words
        assert [w.text for w in words2] == ["Boladi", "!"]
        assert [(w.start_char, w.end_char) for w in words2] == [
            (16, 22), # Boladi
            (22, 23), # !
        ]
        assert words2[1].upos == "PUNCT"


class TestArabicScriptTokenizer:
    def test_instantiation(self) -> None:
        tok = ArabicScriptTokenizer(lang="uig")
        assert tok.NAME == "tokenize"

    def test_process_zwnj_and_punct(self) -> None:
        tok = ArabicScriptTokenizer(lang="uig")
        tok.load()
        text = "مەكتەپ\u200cكە؟ باردىم."
        doc = Document(text=text)
        tok.process(doc)

        assert [s.text for s in doc.sentences] == ["مەكتەپ\u200cكە؟", "باردىم."]

        words = doc.sentences[0].words
        assert [w.text for w in words] == ["مەكتەپ", "كە", "؟"]
        assert [(w.start_char, w.end_char) for w in words] == [
            (0, 6),
            (7, 9),
            (9, 10),
        ]
        assert words[2].upos == "PUNCT"

        words2 = doc.sentences[1].words
        assert [w.text for w in words2] == ["باردىم", "."]
        assert [(w.start_char, w.end_char) for w in words2] == [
            (11, 17),
            (17, 18),
        ]
        assert words2[1].upos == "PUNCT"


@pytest.mark.parametrize(
    "lang,script,text,expected",
    [
        ("tur", Script.LATIN, "Ben geldim.", ["Ben", "geldim", "."]),
        ("aze", Script.LATIN, "Mən gəldim!", ["Mən", "gəldim", "!"]),
        ("azb", Script.PERSO_ARABIC, "من گلدیم؟", ["من", "گلدیم", "؟"]),
        ("kaz", Script.CYRILLIC, "Мен келдім.", ["Мен", "келдім", "."]),
        ("uzb", Script.LATIN, "Men keldim.", ["Men", "keldim", "."]),
        ("kir", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("tuk", Script.LATIN, "Men geldim.", ["Men", "geldim", "."]),
        ("tat", Script.CYRILLIC, "Мин килдем.", ["Мин", "килдем", "."]),
        ("uig", Script.PERSO_ARABIC, "مەن كەلدىم.", ["مەن", "كەلدىم", "."]),
        ("bak", Script.CYRILLIC, "Мин килдем.", ["Мин", "килдем", "."]),
        ("crh", Script.LATIN, "Men keldim.", ["Men", "keldim", "."]),
        ("chv", Script.CYRILLIC, "Эпӗ килтем.", ["Эпӗ", "килтем", "."]),
        ("sah", Script.CYRILLIC, "Мин кэллим.", ["Мин", "кэллим", "."]),
        ("kaa", Script.LATIN, "Men keldim.", ["Men", "keldim", "."]),
        ("gag", Script.LATIN, "Ben geldim.", ["Ben", "geldim", "."]),
        ("nog", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("kum", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("krc", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("alt", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("tyv", Script.CYRILLIC, "Мен келдим.", ["Мен", "келдим", "."]),
        ("kjh", Script.CYRILLIC, "Мин килдем.", ["Мин", "килдем", "."]),
        ("ota", Script.PERSO_ARABIC, "من گلدیم.", ["من", "گلدیم", "."]),
        ("otk", Script.LATIN, "Men keldim.", ["Men", "keldim", "."]),
    ],
)
def test_tokenization_examples(lang: str, script: Script, text: str, expected: list[str]) -> None:
    if script == Script.PERSO_ARABIC:
        tok = ArabicScriptTokenizer(lang=lang, script=script)
    else:
        tok = RegexTokenizer(lang=lang, script=script)
    tok.load()
    doc = Document(text=text, script=script.value, lang=lang)
    tok.process(doc)
    tokens = [w.text for s in doc.sentences for w in s.words]
    assert tokens == expected


@pytest.mark.parametrize(
    "lang,script,text,expected",
    [
        ("tur", Script.LATIN, "Korkut ata geldi.", ["Korkut", "ata", "geldi", "."]),
        ("aze", Script.LATIN, "Oguz han geldi.", ["Oguz", "han", "geldi", "."]),
        ("azb", Script.PERSO_ARABIC, "قورقوت اتا آمد.", ["قورقوت", "اتا", "امد", "."]),
        ("kaz", Script.CYRILLIC, "Қорқыт ата келді.", ["Қорқыт", "ата", "келді", "."]),
        ("uzb", Script.LATIN, "Oguz han keldi.", ["Oguz", "han", "keldi", "."]),
        ("kir", Script.CYRILLIC, "Көркүт ата келди.", ["Көркүт", "ата", "келди", "."]),
        ("tuk", Script.LATIN, "Oguz han geldi.", ["Oguz", "han", "geldi", "."]),
        ("tat", Script.CYRILLIC, "Коркыт ата килде.", ["Коркыт", "ата", "килде", "."]),
        ("uig", Script.PERSO_ARABIC, "قورقوت ئاتا كەلدى.", ["قورقوت", "ئاتا", "كەلدى", "."]),
        ("bak", Script.CYRILLIC, "Ҡорҡот ата килде.", ["Ҡорҡот", "ата", "килде", "."]),
        ("crh", Script.LATIN, "Korkut ata keldi.", ["Korkut", "ata", "keldi", "."]),
        ("chv", Script.CYRILLIC, "Кӑркыт ата килче.", ["Кӑркыт", "ата", "килче", "."]),
        ("sah", Script.CYRILLIC, "Көркүот ата кэллэ.", ["Көркүот", "ата", "кэллэ", "."]),
        ("kaa", Script.LATIN, "Korkyt ata keldi.", ["Korkyt", "ata", "keldi", "."]),
        ("gag", Script.LATIN, "Korkut ata geldi.", ["Korkut", "ata", "geldi", "."]),
        ("nog", Script.CYRILLIC, "Коркыт ата келди.", ["Коркыт", "ата", "келди", "."]),
        ("kum", Script.CYRILLIC, "Коркыт ата келди.", ["Коркыт", "ата", "келди", "."]),
        ("krc", Script.CYRILLIC, "Коркыт ата келди.", ["Коркыт", "ата", "келди", "."]),
        ("alt", Script.CYRILLIC, "Коркыт ата келди.", ["Коркыт", "ата", "келди", "."]),
        ("tyv", Script.CYRILLIC, "Коркыт ата келди.", ["Коркыт", "ата", "келди", "."]),
        ("kjh", Script.CYRILLIC, "Коркыт ата килде.", ["Коркыт", "ата", "килде", "."]),
        ("ota", Script.PERSO_ARABIC, "قورقوت اتا آمد.", ["قورقوت", "اتا", "امد", "."]),
        ("otk", Script.LATIN, "Korkut ata geldi.", ["Korkut", "ata", "geldi", "."]),
    ],
)
def test_tokenization_historic_names(lang: str, script: Script, text: str, expected: list[str]) -> None:
    if script == Script.PERSO_ARABIC:
        tok = ArabicScriptTokenizer(lang=lang, script=script)
    else:
        tok = RegexTokenizer(lang=lang, script=script)
    tok.load()
    doc = Document(text=text, script=script.value, lang=lang)
    tok.process(doc)
    tokens = [w.text for s in doc.sentences for w in s.words]
    assert tokens == expected
