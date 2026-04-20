"""Tests for script transliteration."""

from __future__ import annotations

import pytest

from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator, TRANSLITERATION_TABLES


class TestKazakhTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("kaz", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("мен")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cyrl_to_latn_sentence(self) -> None:
        t = Transliterator("kaz", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("Қазақстан") == "Qazaqstan"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("kaz", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("men") == "мен"

    def test_unknown_table_raises(self) -> None:
        with pytest.raises(ValueError, match="No transliteration table"):
            Transliterator("kaz", Script.PERSO_ARABIC, Script.LATIN)


class TestUzbekTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("uzb", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("салом")
        assert isinstance(result, str)

    def test_cyrl_to_latn_sentence(self) -> None:
        t = Transliterator("uzb", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("Ўзбекистон") == "O'zbekiston"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("uzb", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("salom") == "салом"

    def test_latn_to_cyrl_apostrophe(self) -> None:
        t = Transliterator("uzb", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("O'zbekiston") == "Ўзбекистон"

    def test_latn_to_cyrl_digraphs(self) -> None:
        t = Transliterator("uzb", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("shaxar") == "шахар"
        assert t.transliterate("choy") == "чой"

    def test_cyrl_to_latn_contextual_e(self) -> None:
        t = Transliterator("uzb", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ел") == "yel"
        assert t.transliterate("Океан") == "Okean"
        assert t.transliterate("ае") == "aye"

    def test_latn_to_cyrl_apostrophe_variants(self) -> None:
        t = Transliterator("uzb", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("O‘zbekiston") == "Ўзбекистон"
        assert t.transliterate("G‘ijduvon") == "Ғиждувон"
        assert t.transliterate("gʻazal") == "ғазал"


class TestUyghurTransliteration:
    def test_arab_to_latn(self) -> None:
        t = Transliterator("uig", Script.PERSO_ARABIC, Script.LATIN)
        result = t.transliterate("سالام")
        assert isinstance(result, str)
        assert result == "salam"

    def test_latn_to_arab(self) -> None:
        t = Transliterator("uig", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("salam") == "سالام"

    def test_latn_to_arab_digraphs(self) -> None:
        t = Transliterator("uig", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("ch") == "چ"
        assert t.transliterate("sh") == "ش"
        assert t.transliterate("zh") == "ژ"
        assert t.transliterate("gh") == "غ"
        assert t.transliterate("ng") == "ڭ"

    def test_latn_to_arab_special_vowels(self) -> None:
        t = Transliterator("uig", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("ö") == "ئۆ"
        assert t.transliterate("ü") == "ئۈ"
        assert t.transliterate("é") == "ئې"
        assert t.transliterate("bö") == "بۆ"

    def test_latn_to_arab_initial_hamza_vowels(self) -> None:
        t = Transliterator("uig", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("alma") == "ئالما"
        assert t.transliterate("öy") == "ئۆي"
        assert t.transliterate("isim") == "ئىسىم"


class TestCrimeanTatarTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("crh", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("мектеп")
        assert result == "mektep"

    def test_cyrl_to_latn_digraphs(self) -> None:
        t = Transliterator("crh", Script.CYRILLIC, Script.LATIN)
        # гъ → ğ, дж → c, къ → q
        assert t.transliterate("гъ") == "ğ"
        assert t.transliterate("дж") == "c"
        assert t.transliterate("къ") == "q"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("crh", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("mektep") == "мектеп"

    def test_latn_to_cyrl_special_chars(self) -> None:
        t = Transliterator("crh", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ğ") == "гъ"
        assert t.transliterate("q") == "къ"
        assert t.transliterate("ñ") == "нъ"
        assert t.transliterate("c") == "дж"

    def test_latn_to_cyrl_cedilla(self) -> None:
        t = Transliterator("crh", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ç") == "ч"
        assert t.transliterate("ş") == "ш"


class TestAzerbaijaniTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("aze", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("Бакы")
        assert result == "Bakı"

    def test_cyrl_to_latn_special_letters(self) -> None:
        t = Transliterator("aze", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ә") == "ə"
        assert t.transliterate("ғ") == "ğ"
        assert t.transliterate("ө") == "ö"
        assert t.transliterate("ү") == "ü"
        assert t.transliterate("ҹ") == "c"
        assert t.transliterate("ҝ") == "g"
        assert t.transliterate("һ") == "h"

    def test_cyrl_to_latn_q_mapping(self) -> None:
        """Azerbaijani Cyrillic Г maps to Latin Q (uvular stop)."""
        t = Transliterator("aze", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("г") == "q"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("aze", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("Bakı") == "Бакы"

    def test_latn_to_cyrl_special_letters(self) -> None:
        t = Transliterator("aze", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ə") == "ә"
        assert t.transliterate("ğ") == "ғ"
        assert t.transliterate("ö") == "ө"
        assert t.transliterate("ü") == "ү"
        assert t.transliterate("ç") == "ч"
        assert t.transliterate("ş") == "ш"

    def test_latn_to_cyrl_reverse_mapping(self) -> None:
        """Latin q → Г, g → Ҝ, c → Ҹ in Azerbaijani."""
        t = Transliterator("aze", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("q") == "г"
        assert t.transliterate("g") == "ҝ"
        assert t.transliterate("c") == "ҹ"


class TestTurkmenTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("мен") == "men"

    def test_cyrl_to_latn_special_letters(self) -> None:
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ә") == "ä"
        assert t.transliterate("ө") == "ö"
        assert t.transliterate("ү") == "ü"
        assert t.transliterate("җ") == "j"
        assert t.transliterate("ң") == "ň"

    def test_cyrl_to_latn_v_to_w(self) -> None:
        """Turkmen Cyrillic В maps to Latin W."""
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("в") == "w"

    def test_cyrl_to_latn_y_and_zh(self) -> None:
        """Turkmen й → ý, ж → ž."""
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("й") == "ý"
        assert t.transliterate("ж") == "ž"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("tuk", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("men") == "мен"

    def test_latn_to_cyrl_special_letters(self) -> None:
        t = Transliterator("tuk", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ä") == "ә"
        assert t.transliterate("ö") == "ө"
        assert t.transliterate("ü") == "ү"
        assert t.transliterate("ň") == "ң"
        assert t.transliterate("ç") == "ч"
        assert t.transliterate("ş") == "ш"
        assert t.transliterate("ž") == "ж"
        assert t.transliterate("ý") == "й"
        assert t.transliterate("j") == "җ"
        assert t.transliterate("w") == "в"

    def test_latn_to_cyrl_sentence(self) -> None:
        t = Transliterator("tuk", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("Türkmenistan") == "Түркменистан"

    def test_cyrl_e_word_initial_to_ye(self) -> None:
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ел") == "ýel"
        assert t.transliterate("елкен") == "ýelken"

    def test_cyrl_e_after_i_oe_ue_ae_to_ye(self) -> None:
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("дүе") == "düýe"
        assert t.transliterate("өе") == "öýe"
        assert t.transliterate("диен") == "diýen"

    def test_cyrl_hard_sign_before_e_and_drop_signs(self) -> None:
        t = Transliterator("tuk", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("гүзъетим") == "güzýetim"
        assert t.transliterate("весъет") == "wesýet"
        assert t.transliterate("семья") == "semýa"

    def test_latn_to_cyrl_ye_to_e(self) -> None:
        t = Transliterator("tuk", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ýel") == "ел"
        assert t.transliterate("diýen") == "диен"
        assert t.transliterate("öýe") == "өе"


class TestTatarTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("tat", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("мин")
        assert result == "min"

    def test_cyrl_to_latn_special_letters(self) -> None:
        t = Transliterator("tat", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ә") == "ä"
        assert t.transliterate("ө") == "ö"
        assert t.transliterate("ү") == "ü"
        assert t.transliterate("җ") == "c"
        assert t.transliterate("ң") == "ñ"
        assert t.transliterate("һ") == "h"

    def test_cyrl_to_latn_v_to_w(self) -> None:
        """Tatar Cyrillic В maps to Latin W (not V)."""
        t = Transliterator("tat", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("в") == "w"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("tat", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("min") == "мин"

    def test_latn_to_cyrl_special_letters(self) -> None:
        t = Transliterator("tat", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("ä") == "ә"
        assert t.transliterate("ö") == "ө"
        assert t.transliterate("ü") == "ү"
        assert t.transliterate("ñ") == "ң"
        assert t.transliterate("c") == "җ"
        assert t.transliterate("w") == "в"


class TestKarakalpakTransliteration:
    def test_cyrl_to_latn(self) -> None:
        t = Transliterator("kaa", Script.CYRILLIC, Script.LATIN)
        result = t.transliterate("мен")
        assert result == "men"

    def test_cyrl_to_latn_special_letters(self) -> None:
        t = Transliterator("kaa", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ә") == "á"
        assert t.transliterate("ғ") == "ǵ"
        assert t.transliterate("қ") == "q"
        assert t.transliterate("ң") == "ń"
        assert t.transliterate("ө") == "ó"
        assert t.transliterate("ү") == "ú"
        assert t.transliterate("ў") == "w"
        assert t.transliterate("ҳ") == "h"
        assert t.transliterate("ы") == "í"

    def test_cyrl_to_latn_digraphs(self) -> None:
        t = Transliterator("kaa", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate("ш") == "sh"
        assert t.transliterate("ч") == "ch"

    def test_latn_to_cyrl(self) -> None:
        t = Transliterator("kaa", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("men") == "мен"

    def test_latn_to_cyrl_special_letters(self) -> None:
        t = Transliterator("kaa", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("á") == "ә"
        assert t.transliterate("ǵ") == "ғ"
        assert t.transliterate("q") == "қ"
        assert t.transliterate("ń") == "ң"
        assert t.transliterate("ó") == "ө"
        assert t.transliterate("ú") == "ү"
        assert t.transliterate("w") == "ў"
        assert t.transliterate("í") == "ы"

    def test_latn_to_cyrl_digraphs(self) -> None:
        t = Transliterator("kaa", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate("sh") == "ш"
        assert t.transliterate("ch") == "ч"


class TestOttomanTurkishTransliteration:
    def test_latn_to_arab(self) -> None:
        t = Transliterator("ota", Script.LATIN, Script.PERSO_ARABIC)
        result = t.transliterate("sultan")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_latn_to_arab_common_letters(self) -> None:
        t = Transliterator("ota", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("b") == "ب"
        assert t.transliterate("s") == "س"
        assert t.transliterate("k") == "ك"
        assert t.transliterate("l") == "ل"

    def test_latn_to_arab_digraphs(self) -> None:
        t = Transliterator("ota", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate("sh") == "ش"
        assert t.transliterate("ch") == "چ"


class TestOldTurkicTransliteration:
    def test_orkh_to_latn(self) -> None:
        t = Transliterator("otk", Script.OLD_TURKIC_RUNIC, Script.LATIN)
        # Test vowels
        assert t.transliterate("\U00010C00") == "a"
        assert t.transliterate("\U00010C03") == "ı"
        assert t.transliterate("\U00010C06") == "o"
        assert t.transliterate("\U00010C07") == "ö"

    def test_orkh_to_latn_consonants(self) -> None:
        t = Transliterator("otk", Script.OLD_TURKIC_RUNIC, Script.LATIN)
        assert t.transliterate("\U00010C09") == "b"
        assert t.transliterate("\U00010C0D") == "g"
        assert t.transliterate("\U00010C11") == "d"
        assert t.transliterate("\U00010C32") == "t"
        assert t.transliterate("\U00010C25") == "m"
        assert t.transliterate("\U00010C26") == "n"

    def test_orkh_to_latn_ligatures(self) -> None:
        t = Transliterator("otk", Script.OLD_TURKIC_RUNIC, Script.LATIN)
        assert t.transliterate("\U00010C36") == "lt"
        assert t.transliterate("\U00010C38") == "sh"
        assert t.transliterate("\U00010C3B") == "nt"
        assert t.transliterate("\U00010C3C") == "nch"


class TestAllTablesPresent:
    """Verify that every can_transliterate pair in LANGUAGE_SCRIPTS has a table."""

    def test_all_transliterate_pairs_have_tables(self) -> None:
        from turkicnlp.scripts import LANGUAGE_SCRIPTS

        missing = []
        for lang, config in LANGUAGE_SCRIPTS.items():
            if config.can_transliterate:
                for src, tgt in config.can_transliterate:
                    key = f"{lang}_{src.value}_to_{tgt.value}"
                    if key not in TRANSLITERATION_TABLES:
                        missing.append(key)
        assert missing == [], f"Missing transliteration tables: {missing}"


class TestUyghurMultiScript:
    """Uyghur multi-script transliteration tests.

    Test data sourced from the Uyghur Multi-Script Converter:
    https://github.com/neouyghur/ScriptConverter4Uyghur (Apache-2.0)

    Note: cases involving inter-vowel hamza (ئ) apostrophe preservation
    are intentionally excluded — that nuance is not part of our implementation.
    """

    # ------------------------------------------------------------------
    # Arabic (UAS) ↔ Latin (ULS)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("arab,latn", [
        ("قول", "qol"),
        ("باش", "bash"),
        ("پۇت", "put"),
        ("كۆز", "köz"),
        ("جەڭچى", "jengchi"),
        ("جۇدې", "judé"),
        ("سان", "san"),
        ("شىر", "shir"),
        ("شاڭخەي", "shangxey"),
        ("كىتاب", "kitab"),
        ("ۋەتەن", "weten"),
        ("تومۇر", "tomur"),
        ("كۆمۈر", "kömür"),
        ("ئېلىكتىر", "éliktir"),
        ("شىنجاڭ", "shinjang"),
        ("ئانار", "anar"),
        ("ئەنجۈر", "enjür"),
        ("ئوردا", "orda"),
        ("ئۇرۇش", "urush"),
        ("ئۆردەك", "ördek"),
        ("ئۈزۈم", "üzüm"),
        ("ئېلان", "élan"),
        ("ئىنكاس", "inkas"),
        ("ئۆزخان", "özxan"),
        ("پاسخا", "pasxa"),
        ("بايرىمى", "bayrimi"),
        ("گاڭگىراپ", "ganggirap"),
        ("جۇڭخۇا", "jungxua"),
        ("ئەدەب-ئەخلاق", "edeb-exlaq"),
        # inter-vowel ئ → apostrophe in Latin output
        ("ئىنىكئانا", "inik'ana"),
        ("ئەسئەت", "es'et"),
        ("رادىئو", "radi'o"),
        ("مەسئۇل", "mes'ul"),
        ("قارىئۆرۈك", "qari'örük"),
        ("نائۈمىد", "na'ümid"),
        ("ئىتئېيىق", "it'éyiq"),
        ("جەمئىي", "jem'iy"),
        ("مائارىپ", "ma'arip"),
        ("مۇئەللىم", "mu'ellim"),
        ("دائىرە", "da'ire"),
        ("مۇئەييەن", "mu'eyyen"),
        ("تەبىئىي", "tebi'iy"),
        ("پائالىيەت", "pa'aliyet"),
        ("جەمئىيەت", "jem'iyet"),
    ])
    def test_arab_to_latn(self, arab: str, latn: str) -> None:
        t = Transliterator("uig", Script.PERSO_ARABIC, Script.LATIN)
        assert t.transliterate(arab) == latn

    @pytest.mark.parametrize("latn,arab", [
        ("qol", "قول"),
        ("bash", "باش"),
        ("put", "پۇت"),
        ("köz", "كۆز"),
        ("jengchi", "جەڭچى"),
        ("judé", "جۇدې"),
        ("san", "سان"),
        ("shir", "شىر"),
        ("shangxey", "شاڭخەي"),
        ("kitab", "كىتاب"),
        ("weten", "ۋەتەن"),
        ("tomur", "تومۇر"),
        ("kömür", "كۆمۈر"),
        ("éliktir", "ئېلىكتىر"),
        ("shinjang", "شىنجاڭ"),
        ("anar", "ئانار"),
        ("enjür", "ئەنجۈر"),
        ("orda", "ئوردا"),
        ("urush", "ئۇرۇش"),
        ("ördek", "ئۆردەك"),
        ("üzüm", "ئۈزۈم"),
        ("élan", "ئېلان"),
        ("inkas", "ئىنكاس"),
        ("özxan", "ئۆزخان"),
        ("pasxa", "پاسخا"),
        ("bayrimi", "بايرىمى"),
        ("ganggirap", "گاڭگىراپ"),
        # vowel-after-vowel needs ئ in Arabic output
        ("radio", "رادىئو"),
        ("qariörük", "قارىئۆرۈك"),
        ("naümid", "نائۈمىد"),
        ("maarip", "مائارىپ"),
        ("muellim", "مۇئەللىم"),
        ("daire", "دائىرە"),
        ("mueyyen", "مۇئەييەن"),
        ("tebiiy", "تەبىئىي"),
        ("paaliyet", "پائالىيەت"),
        # apostrophe in input marks syllable boundary → stripped in Arabic output
        ("inik'ana", "ئىنىكئانا"),
        ("es'et", "ئەسئەت"),
        ("mes'ul", "مەسئۇل"),
        ("it'éyiq", "ئىتئېيىق"),
        ("cem'iy", "جەمئىي"),
        ("cem'iyet", "جەمئىيەت"),
    ])
    def test_latn_to_arab(self, latn: str, arab: str) -> None:
        t = Transliterator("uig", Script.LATIN, Script.PERSO_ARABIC)
        assert t.transliterate(latn) == arab

    # ------------------------------------------------------------------
    # Arabic (UAS) ↔ Cyrillic (UCS)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("arab,cyrl", [
        ("قول", "қол"),
        ("باش", "баш"),
        ("پۇت", "пут"),
        ("كۆز", "көз"),
        ("جەڭچى", "җәңчи"),
        ("جۇدې", "җуде"),
        ("سان", "сан"),
        ("شىر", "шир"),
        ("شاڭخەي", "шаңхәй"),
        ("كىتاب", "китаб"),
        ("ۋەتەن", "вәтән"),
        ("تومۇر", "томур"),
        ("كۆمۈر", "көмүр"),
        ("ئېلىكتىر", "еликтир"),
        ("شىنجاڭ", "шинҗаң"),
        ("ئانار", "анар"),
        ("ئەنجۈر", "әнҗүр"),
        ("ئوردا", "орда"),
        ("ئۇرۇش", "уруш"),
        ("ئۆردەك", "өрдәк"),
        ("ئۈزۈم", "үзүм"),
        ("ئېلان", "елан"),
        ("ئىنكاس", "инкас"),
        ("ئۆزخان", "өзхан"),
        ("پاسخا", "пасха"),
        ("بايرىمى", "байрими"),
        ("ئىسھاق", "исһақ"),
        ("ئۆزبېكىستانغا", "өзбекистанға"),
        ("ھىنگان", "һинган"),
        ("چەكلەنگەن", "чәкләнгән"),
        ("گاڭگىراپ", "гаңгирап"),
        ("باشلانغۇچ", "башланғуч"),
        # inter-vowel ئ → apostrophe in Cyrillic output
        ("ئىنىكئانا", "иник'ана"),
        ("ئەسئەت", "әс'әт"),
        ("رادىئو", "ради'о"),
        ("مەسئۇل", "мәс'ул"),
        ("قارىئۆرۈك", "қари'өрүк"),
        ("نائۈمىد", "на'үмид"),
        ("ئىتئېيىق", "ит'ейиқ"),
        ("جەمئىي", "җәм'ий"),
        ("جەمئىيەت", "җәм'ийәт"),
    ])
    def test_arab_to_cyrl(self, arab: str, cyrl: str) -> None:
        t = Transliterator("uig", Script.PERSO_ARABIC, Script.CYRILLIC)
        assert t.transliterate(arab) == cyrl

    @pytest.mark.parametrize("cyrl,arab", [
        ("қол", "قول"),
        ("баш", "باش"),
        ("пут", "پۇت"),
        ("көз", "كۆز"),
        ("җәңчи", "جەڭچى"),
        ("сан", "سان"),
        ("шир", "شىر"),
        ("шаңхәй", "شاڭخەي"),
        ("китаб", "كىتاب"),
        ("вәтән", "ۋەتەن"),
        ("томур", "تومۇر"),
        ("көмүр", "كۆمۈر"),
        ("еликтир", "ئېلىكتىر"),
        ("шинҗаң", "شىنجاڭ"),
        ("анар", "ئانار"),
        ("әнҗүр", "ئەنجۈر"),
        ("орда", "ئوردا"),
        ("уруш", "ئۇرۇش"),
        ("өрдәк", "ئۆردەك"),
        ("үзүм", "ئۈزۈم"),
        ("елан", "ئېلان"),
        ("инкас", "ئىنكاس"),
        ("өзхан", "ئۆزخان"),
        ("пасха", "پاسخا"),
        ("байрими", "بايرىمى"),
        ("исһақ", "ئىسھاق"),
        ("өзбекистанға", "ئۆزبېكىستانغا"),
        ("һинган", "ھىنگان"),
        ("чәкләнгән", "چەكلەنگەن"),
        ("гаңгирап", "گاڭگىراپ"),
        ("башланғуч", "باشلانغۇچ"),
    ])
    def test_cyrl_to_arab(self, cyrl: str, arab: str) -> None:
        t = Transliterator("uig", Script.CYRILLIC, Script.PERSO_ARABIC)
        assert t.transliterate(cyrl) == arab

    # ------------------------------------------------------------------
    # Latin (ULS) ↔ Cyrillic (UCS)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("latn,cyrl", [
        ("qol", "қол"),
        ("bash", "баш"),
        ("put", "пут"),
        ("köz", "көз"),
        ("jengchi", "җәңчи"),
        ("san", "сан"),
        ("sey", "сәй"),
        ("shir", "шир"),
        ("kitab", "китаб"),
        ("weten", "вәтән"),
        ("tomur", "томур"),
        ("kömür", "көмүр"),
        ("éliktir", "еликтир"),
        ("shinjang", "шинҗаң"),
        ("anar", "анар"),
        ("enjür", "әнҗүр"),
        ("orda", "орда"),
        ("urush", "уруш"),
        ("ördek", "өрдәк"),
        ("üzüm", "үзүм"),
        ("élan", "елан"),
        ("inkas", "инкас"),
        ("özxan", "өзхан"),
        ("pasxa", "пасха"),
        ("bayrimi", "байрими"),
    ])
    def test_latn_to_cyrl(self, latn: str, cyrl: str) -> None:
        t = Transliterator("uig", Script.LATIN, Script.CYRILLIC)
        assert t.transliterate(latn) == cyrl

    @pytest.mark.parametrize("cyrl,latn", [
        ("қол", "qol"),
        ("баш", "bash"),
        ("пут", "put"),
        ("көз", "köz"),
        ("җәңчи", "jengchi"),
        ("сан", "san"),
        ("сәй", "sey"),
        ("шир", "shir"),
        ("китаб", "kitab"),
        ("вәтән", "weten"),
        ("томур", "tomur"),
        ("көмүр", "kömür"),
        ("еликтир", "éliktir"),
        ("шинҗаң", "shinjang"),
        ("анар", "anar"),
        ("әнҗүр", "enjür"),
        ("орда", "orda"),
        ("уруш", "urush"),
        ("өрдәк", "ördek"),
        ("үзүм", "üzüm"),
        ("елан", "élan"),
        ("инкас", "inkas"),
        ("өзхан", "özxan"),
        ("пасха", "pasxa"),
        ("байрими", "bayrimi"),
    ])
    def test_cyrl_to_latn(self, cyrl: str, latn: str) -> None:
        t = Transliterator("uig", Script.CYRILLIC, Script.LATIN)
        assert t.transliterate(cyrl) == latn

    # ------------------------------------------------------------------
    # Arabic (UAS) ↔ CTS
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("arab,cts", [
        ("قول", "qol"),
        ("باش", "baş"),
        ("پۇت", "put"),
        ("كۆز", "köz"),
        ("جەڭچى", "ceñçi"),
        ("جۇدې", "cudé"),
        ("سان", "san"),
        ("سەي", "sey"),
        ("ئې", "é"),
        ("شىر", "şir"),
        ("شاڭخەي", "şañxey"),
        ("كىتاب", "kitab"),
        ("ۋەتەن", "veten"),
        ("تومۇر", "tomur"),
        ("كۆمۈر", "kömür"),
        ("ئېلىكتىر", "éliktir"),
        ("ۋيېتنام", "vyétnam"),
        ("شىنجاڭ", "şincañ"),
        ("ئانار", "anar"),
        ("ئەنجۈر", "encür"),
        ("ئوردا", "orda"),
        ("ئۇرۇش", "uruş"),
        ("ئۆردەك", "ördek"),
        ("ئۈزۈم", "üzüm"),
        ("ئېلان", "élan"),
        ("ئىنكاس", "inkas"),
        ("نەمەنگان", "nemengan"),
        ("ئۆزخان", "özxan"),
        ("پاسخا", "pasxa"),
        ("بايرىمى", "bayrimi"),
        ("مائارىپ", "maarip"),
        ("مۇئەللىم", "muellim"),
        ("دائىرە", "daire"),
        ("مۇئەييەن", "mueyyen"),
        ("تەبىئىي", "tebiiy"),
        ("پائالىيەت", "paaliyet"),
        ("ئىسھاق", "ishaq"),
        ("ئۆزبېكىستانغا", "özbékistanğa"),
        ("ھىنگان", "hingan"),
        ("چەكلەنگەن", "çeklengen"),
        ("گاڭگىراپ", "gañgirap"),
        ("باشلانغۇچ", "başlanğuç"),
    ])
    def test_arab_to_cts(self, arab: str, cts: str) -> None:
        t = Transliterator("uig", Script.PERSO_ARABIC, Script.COMMON_TURKIC)
        assert t.transliterate(arab) == cts

    @pytest.mark.parametrize("cts,arab", [
        ("qol", "قول"),
        ("baş", "باش"),
        ("put", "پۇت"),
        ("köz", "كۆز"),
        ("ceñçi", "جەڭچى"),
        ("cudé", "جۇدې"),
        ("san", "سان"),
        ("sey", "سەي"),
        ("é", "ئې"),
        ("şir", "شىر"),
        ("şañxey", "شاڭخەي"),
        ("kitab", "كىتاب"),
        ("veten", "ۋەتەن"),
        ("tomur", "تومۇر"),
        ("kömür", "كۆمۈر"),
        ("éliktir", "ئېلىكتىر"),
        ("vyétnam", "ۋيېتنام"),
        ("şincañ", "شىنجاڭ"),
        ("anar", "ئانار"),
        ("encür", "ئەنجۈر"),
        ("orda", "ئوردا"),
        ("uruş", "ئۇرۇش"),
        ("ördek", "ئۆردەك"),
        ("üzüm", "ئۈزۈم"),
        ("élan", "ئېلان"),
        ("inkas", "ئىنكاس"),
        ("nemengan", "نەمەنگان"),
        ("özxan", "ئۆزخان"),
        ("pasxa", "پاسخا"),
        ("bayrimi", "بايرىمى"),
        ("maarip", "مائارىپ"),
        ("muellim", "مۇئەللىم"),
        ("daire", "دائىرە"),
        ("mueyyen", "مۇئەييەن"),
        ("tebiiy", "تەبىئىي"),
        ("paaliyet", "پائالىيەت"),
        ("ishaq", "ئىسھاق"),
        ("özbékistanğa", "ئۆزبېكىستانغا"),
        ("hingan", "ھىنگان"),
        ("çeklengen", "چەكلەنگەن"),
        ("gañgirap", "گاڭگىراپ"),
        ("başlanğuç", "باشلانغۇچ"),
    ])
    def test_cts_to_arab(self, cts: str, arab: str) -> None:
        t = Transliterator("uig", Script.COMMON_TURKIC, Script.PERSO_ARABIC)
        assert t.transliterate(cts) == arab

    # ------------------------------------------------------------------
    # Latin (ULS) ↔ CTS
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("latn,cts", [
        ("qol", "qol"),
        ("bash", "baş"),
        ("put", "put"),
        ("köz", "köz"),
        ("jengchi", "ceñçi"),
        ("san", "san"),
        ("sey", "sey"),
        ("shir", "şir"),
        ("kitab", "kitab"),
        ("weten", "veten"),
        ("tomur", "tomur"),
        ("kömür", "kömür"),
        ("éliktir", "éliktir"),
        ("shinjang", "şincañ"),
        ("anar", "anar"),
        ("enjür", "encür"),
        ("orda", "orda"),
        ("urush", "uruş"),
        ("ördek", "ördek"),
        ("üzüm", "üzüm"),
        ("élan", "élan"),
        ("inkas", "inkas"),
        ("özxan", "özxan"),
        ("pasxa", "pasxa"),
        ("bayrimi", "bayrimi"),
    ])
    def test_latn_to_cts(self, latn: str, cts: str) -> None:
        t = Transliterator("uig", Script.LATIN, Script.COMMON_TURKIC)
        assert t.transliterate(latn) == cts

    @pytest.mark.parametrize("cts,latn", [
        ("qol", "qol"),
        ("baş", "bash"),
        ("put", "put"),
        ("köz", "köz"),
        ("ceñçi", "jengchi"),
        ("san", "san"),
        ("sey", "sey"),
        ("şir", "shir"),
        ("kitab", "kitab"),
        ("veten", "weten"),
        ("tomur", "tomur"),
        ("kömür", "kömür"),
        ("éliktir", "éliktir"),
        ("şincañ", "shinjang"),
        ("anar", "anar"),
        ("encür", "enjür"),
        ("orda", "orda"),
        ("uruş", "urush"),
        ("ördek", "ördek"),
        ("üzüm", "üzüm"),
        ("élan", "élan"),
        ("inkas", "inkas"),
        ("özxan", "özxan"),
        ("pasxa", "pasxa"),
        ("bayrimi", "bayrimi"),
    ])
    def test_cts_to_latn(self, cts: str, latn: str) -> None:
        t = Transliterator("uig", Script.COMMON_TURKIC, Script.LATIN)
        assert t.transliterate(cts) == latn

    # ------------------------------------------------------------------
    # Cyrillic (UCS) ↔ CTS
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("cyrl,cts", [
        ("қол", "qol"),
        ("баш", "baş"),
        ("пут", "put"),
        ("көз", "köz"),
        ("җәңчи", "ceñçi"),
        ("сан", "san"),
        ("сәй", "sey"),
        ("шир", "şir"),
        ("китаб", "kitab"),
        ("вәтән", "veten"),
        ("томур", "tomur"),
        ("көмүр", "kömür"),
        ("еликтир", "éliktir"),
        ("шинҗаң", "şincañ"),
        ("анар", "anar"),
        ("әнҗүр", "encür"),
        ("орда", "orda"),
        ("уруш", "uruş"),
        ("өрдәк", "ördek"),
        ("үзүм", "üzüm"),
        ("елан", "élan"),
        ("инкас", "inkas"),
        ("нәмәнган", "nemengan"),
        ("өзхан", "özxan"),
        ("пасха", "pasxa"),
        ("байрими", "bayrimi"),
        ("һинган", "hingan"),
        ("чәкләнгән", "çeklengen"),
        ("башланғуч", "başlanğuç"),
    ])
    def test_cyrl_to_cts(self, cyrl: str, cts: str) -> None:
        t = Transliterator("uig", Script.CYRILLIC, Script.COMMON_TURKIC)
        assert t.transliterate(cyrl) == cts

    @pytest.mark.parametrize("cts,cyrl", [
        ("qol", "қол"),
        ("baş", "баш"),
        ("put", "пут"),
        ("köz", "көз"),
        ("ceñçi", "җәңчи"),
        ("san", "сан"),
        ("sey", "сәй"),
        ("şir", "шир"),
        ("kitab", "китаб"),
        ("veten", "вәтән"),
        ("tomur", "томур"),
        ("kömür", "көмүр"),
        ("éliktir", "еликтир"),
        ("şincañ", "шинҗаң"),
        ("anar", "анар"),
        ("encür", "әнҗүр"),
        ("orda", "орда"),
        ("uruş", "уруш"),
        ("ördek", "өрдәк"),
        ("üzüm", "үзүм"),
        ("élan", "елан"),
        ("inkas", "инкас"),
        ("nemengan", "нәмәнган"),
        ("özxan", "өзхан"),
        ("pasxa", "пасха"),
        ("bayrimi", "байрими"),
        ("hingan", "һинган"),
        ("çeklengen", "чәкләнгән"),
        ("başlanğuç", "башланғуч"),
    ])
    def test_cts_to_cyrl(self, cts: str, cyrl: str) -> None:
        t = Transliterator("uig", Script.COMMON_TURKIC, Script.CYRILLIC)
        assert t.transliterate(cts) == cyrl
