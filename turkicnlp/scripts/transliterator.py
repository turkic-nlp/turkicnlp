"""
Bidirectional script transliteration for Turkic languages.

Each language has its own mapping because the same Cyrillic letter
maps to different Latin letters across languages. Transliteration
tables are defined in :data:`TRANSLITERATION_TABLES`.
"""

from __future__ import annotations

from turkicnlp.scripts import Script


class Transliterator:
    """Convert text between writing scripts for a given Turkic language.

    Args:
        lang: ISO 639-3 language code.
        source: Source :class:`Script`.
        target: Target :class:`Script`.

    Raises:
        ValueError: If no transliteration table exists for the given combination.
    """

    def __init__(self, lang: str, source: Script, target: Script) -> None:
        self.lang = lang
        self.source = source
        self.target = target
        self._forward_map, self._reverse_map = self._load_mapping(lang, source, target)

    def transliterate(self, text: str) -> str:
        """Convert *text* from *source* script to *target* script.

        Uses greedy longest-match on the forward mapping table.

        Args:
            text: Input text in the source script.

        Returns:
            Text converted to the target script.
        """
        result: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                chunk = text[i : i + length]
                if chunk in self._forward_map:
                    result.append(self._forward_map[chunk])
                    i += length
                    matched = True
                    break
                if chunk.lower() in self._forward_map:
                    mapped = self._forward_map[chunk.lower()]
                    if chunk[0].isupper():
                        mapped = (
                            mapped[0].upper() + mapped[1:]
                            if len(mapped) > 1
                            else mapped.upper()
                        )
                    result.append(mapped)
                    i += length
                    matched = True
                    break
            if not matched:
                result.append(text[i])
                i += 1
        return "".join(result)

    @staticmethod
    def _load_mapping(
        lang: str, source: Script, target: Script
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Load character mapping for a language and script pair."""
        key = f"{lang}_{source.value}_to_{target.value}"
        if key not in TRANSLITERATION_TABLES:
            raise ValueError(
                f"No transliteration table for {lang} {source} -> {target}. "
                f"Available: {list(TRANSLITERATION_TABLES.keys())}"
            )
        forward = TRANSLITERATION_TABLES[key]
        reverse = {v: k for k, v in forward.items()}
        return forward, reverse


# ---------------------------------------------------------------------------
# Transliteration tables
# ---------------------------------------------------------------------------

TRANSLITERATION_TABLES: dict[str, dict[str, str]] = {
    # Kazakh Cyrillic → Latin (2021 official Latin alphabet)
    "kaz_Cyrl_to_Latn": {
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ұ": "ū", "Ұ": "Ū",
        "ү": "ü", "Ү": "Ü", "і": "ı", "І": "I", "һ": "h", "Һ": "H",
        "ш": "sh", "Ш": "Sh", "ч": "ch", "Ч": "Ch", "ж": "j", "Ж": "J",
        "щ": "shch", "Щ": "Shch",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "ı", "И": "I",
        "й": "ı", "Й": "I", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "h", "Х": "H", "ц": "ts", "Ц": "Ts", "э": "e", "Э": "E",
        "ъ": "", "ь": "", "ы": "y", "Ы": "Y", "ю": "yu", "Ю": "Yu",
        "я": "ya", "Я": "Ya",
    },

    # Kazakh Latin → Cyrillic
    "kaz_Latn_to_Cyrl": {
        "shch": "щ", "Shch": "Щ", "sh": "ш", "Sh": "Ш", "SH": "Ш",
        "ch": "ч", "Ch": "Ч", "CH": "Ч", "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я", "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        "ä": "ә", "Ä": "Ә", "ğ": "ғ", "Ğ": "Ғ", "q": "қ", "Q": "Қ",
        "ñ": "ң", "Ñ": "Ң", "ö": "ө", "Ö": "Ө", "ū": "ұ", "Ū": "Ұ",
        "ü": "ү", "Ü": "Ү", "ı": "і", "I": "І",
        "a": "а", "A": "А", "b": "б", "B": "Б", "v": "в", "V": "В",
        "g": "г", "G": "Г", "d": "д", "D": "Д", "e": "е", "E": "Е",
        "z": "з", "Z": "З", "j": "ж", "J": "Ж", "k": "к", "K": "К",
        "l": "л", "L": "Л", "m": "м", "M": "М", "n": "н", "N": "Н",
        "o": "о", "O": "О", "p": "п", "P": "П", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "f": "ф", "F": "Ф", "h": "х", "H": "Х", "y": "ы", "Y": "Ы",
    },

    # Uzbek Cyrillic → Latin (1995 official Latin alphabet)
    "uzb_Cyrl_to_Latn": {
        "ш": "sh", "Ш": "Sh", "ч": "ch", "Ч": "Ch", "ғ": "g'", "Ғ": "G'",
        "қ": "q", "Қ": "Q", "ҳ": "h", "Ҳ": "H", "ў": "o'", "Ў": "O'",
        "нг": "ng", "Нг": "Ng", "ж": "j", "Ж": "J",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "I",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "'", "ь": "",
    },

    # Uyghur Perso-Arabic → Latin (ULY)
    "uig_Arab_to_Latn": {
        "ا": "a", "ە": "e", "ب": "b", "پ": "p", "ت": "t", "ج": "j",
        "چ": "ch", "خ": "x", "د": "d", "ر": "r", "ز": "z", "ژ": "zh",
        "س": "s", "ش": "sh", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
        "گ": "g", "ڭ": "ng", "ل": "l", "م": "m", "ن": "n", "ھ": "h",
        "و": "o", "ۇ": "u", "ۆ": "ö", "ۈ": "ü", "ۋ": "w", "ې": "é",
        "ى": "i", "ي": "y",
    },

    # Crimean Tatar Cyrillic → Latin
    "crh_Cyrl_to_Latn": {
        "гъ": "ğ", "Гъ": "Ğ", "дж": "c", "Дж": "C", "къ": "q", "Къ": "Q",
        "нъ": "ñ", "Нъ": "Ñ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ж": "j", "Ж": "J", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "h", "Х": "H", "ц": "ts", "Ц": "Ts", "ч": "ç", "Ч": "Ç",
        "ш": "ş", "Ш": "Ş", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },
}
