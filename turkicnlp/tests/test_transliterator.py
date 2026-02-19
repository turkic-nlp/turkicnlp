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
                for src, tgt in config.can_transliterate.items():
                    key = f"{lang}_{src.value}_to_{tgt.value}"
                    if key not in TRANSLITERATION_TABLES:
                        missing.append(key)
        assert missing == [], f"Missing transliteration tables: {missing}"
