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
        if (
            self.lang == "tuk"
            and self.source == Script.CYRILLIC
            and self.target == Script.LATIN
        ):
            return self._transliterate_tuk_cyrl_to_latn(text)
        if (
            self.lang == "uzb"
            and self.source == Script.CYRILLIC
            and self.target == Script.LATIN
        ):
            return self._transliterate_uzb_cyrl_to_latn(text)
        if (
            self.lang == "uzb"
            and self.source == Script.LATIN
            and self.target == Script.CYRILLIC
        ):
            text = self._normalize_uzb_apostrophes(text)
        if (
            self.lang == "tuk"
            and self.source == Script.LATIN
            and self.target == Script.CYRILLIC
        ):
            return self._transliterate_tuk_latn_to_cyrl(text)
        if (
            self.lang == "uig"
            and self.source == Script.LATIN
            and self.target == Script.PERSO_ARABIC
        ):
            return self._transliterate_uig_latn_to_arab(text)

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

    def _transliterate_uzb_cyrl_to_latn(self, text: str) -> str:
        """Uzbek Cyrillic -> Latin with context-sensitive ``е`` rules."""
        result: list[str] = []
        i = 0
        prev_vowelish = set("аАеЕёЁиИйЙоОуУўЎыЫэЭюЮяЯьЬъЪ")
        while i < len(text):
            matched = False
            for length in range(min(4, len(text) - i), 1, -1):
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
            if matched:
                continue

            ch = text[i]
            if ch in {"е", "Е"}:
                prev = text[i - 1] if i > 0 else ""
                word_initial = i == 0 or not text[i - 1].isalpha()
                if word_initial or prev in prev_vowelish:
                    result.append("ye" if ch == "е" else "Ye")
                else:
                    result.append("e" if ch == "е" else "E")
                i += 1
                continue

            if ch in self._forward_map:
                result.append(self._forward_map[ch])
            elif ch.lower() in self._forward_map:
                mapped = self._forward_map[ch.lower()]
                if ch[0].isupper():
                    mapped = (
                        mapped[0].upper() + mapped[1:] if len(mapped) > 1 else mapped.upper()
                    )
                result.append(mapped)
            else:
                result.append(ch)
            i += 1
        return "".join(result)

    def _transliterate_tuk_cyrl_to_latn(self, text: str) -> str:
        """Turkmen Cyrillic -> Latin with context-sensitive ``е`` rules.

        Rules implemented:
        1. Word-initial ``е`` -> ``ýe``.
        2. If ``е`` is preceded by ``и/ө/ү/ә`` (or uppercase variants),
           map ``е`` as ``ýe``.
        3. ``ъ`` before ``е`` yields ``ýe`` on ``е``; ``ъ`` and ``ь`` are
           otherwise dropped.
        """
        result: list[str] = []
        i = 0
        while i < len(text):
            # Greedy longest-match for multi-character mappings first.
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                chunk = text[i : i + length]
                if length > 1 and chunk in self._forward_map:
                    result.append(self._forward_map[chunk])
                    i += length
                    matched = True
                    break
                if length > 1 and chunk.lower() in self._forward_map:
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
            if matched:
                continue

            ch = text[i]

            # Drop hard/soft signs by default.
            if ch in {"ъ", "Ъ", "ь", "Ь"}:
                i += 1
                continue

            # Context-sensitive handling for Cyrillic e.
            if ch in {"е", "Е"}:
                prev = text[i - 1] if i > 0 else ""
                word_initial = i == 0 or not text[i - 1].isalpha()
                needs_y = (
                    word_initial
                    or prev in {"и", "И", "ө", "Ө", "ү", "Ү", "ә", "Ә", "ъ", "Ъ"}
                )
                if needs_y:
                    result.append("ýe" if ch == "е" else "Ýe")
                else:
                    result.append("e" if ch == "е" else "E")
                i += 1
                continue

            if ch in self._forward_map:
                result.append(self._forward_map[ch])
            elif ch.lower() in self._forward_map:
                mapped = self._forward_map[ch.lower()]
                if ch[0].isupper():
                    mapped = (
                        mapped[0].upper() + mapped[1:] if len(mapped) > 1 else mapped.upper()
                    )
                result.append(mapped)
            else:
                result.append(ch)
            i += 1

        return "".join(result)

    def _transliterate_tuk_latn_to_cyrl(self, text: str) -> str:
        """Turkmen Latin -> Cyrillic with contextual ``ýe`` -> ``е``."""
        result: list[str] = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i : i + 2] in {"ýe", "Ýe", "ÝE", "ýE"}:
                chunk = text[i : i + 2]
                result.append("е" if chunk in {"ýe", "ýE"} else "Е")
                i += 2
                continue

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
                            mapped[0].upper() + mapped[1:] if len(mapped) > 1 else mapped.upper()
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
    def _normalize_uzb_apostrophes(text: str) -> str:
        """Normalize common Uzbek apostrophe variants to ASCII apostrophe."""
        return (
            text.replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u02bb", "'")
            .replace("\u02bc", "'")
            .replace("\u02b9", "'")
            .replace("`", "'")
        )

    def _transliterate_uig_latn_to_arab(self, text: str) -> str:
        """Uyghur Latin -> Arabic with initial-hamza vowel handling."""
        result: list[str] = []
        i = 0
        initial_vowel_map = {
            "a": "ئا",
            "e": "ئە",
            "o": "ئو",
            "u": "ئۇ",
            "ö": "ئۆ",
            "ü": "ئۈ",
            "é": "ئې",
            "i": "ئى",
            "A": "ئا",
            "E": "ئە",
            "O": "ئو",
            "U": "ئۇ",
            "Ö": "ئۆ",
            "Ü": "ئۈ",
            "É": "ئې",
            "I": "ئى",
        }
        while i < len(text):
            word_initial = i == 0 or not text[i - 1].isalpha()
            if word_initial and text[i : i + 1] in initial_vowel_map:
                result.append(initial_vowel_map[text[i : i + 1]])
                i += 1
                continue

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
        "ئا": "a", "ئە": "e", "ئو": "o", "ئۇ": "u", "ئۆ": "ö", "ئۈ": "ü",
        "ئې": "é", "ئى": "i",
        "ا": "a", "ە": "e", "ب": "b", "پ": "p", "ت": "t", "ج": "j",
        "چ": "ch", "خ": "x", "د": "d", "ر": "r", "ز": "z", "ژ": "zh",
        "س": "s", "ش": "sh", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
        "گ": "g", "ڭ": "ng", "ل": "l", "م": "m", "ن": "n", "ھ": "h",
        "و": "o", "ۇ": "u", "ۆ": "ö", "ۈ": "ü", "ۋ": "w", "ې": "é",
        "ى": "i", "ي": "y", "ئ": "",
    },

    # Turkmen Cyrillic → Latin (1993 official Latin alphabet)
    "tuk_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "şç", "Щ": "Şç",
        # Turkmen-specific Cyrillic letters
        "ә": "ä", "Ә": "Ä", "җ": "j", "Җ": "J", "ң": "ň", "Ң": "Ň",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "ýo", "Ё": "Ýo", "ж": "ž", "Ж": "Ž", "з": "z", "З": "Z",
        "и": "i", "И": "I", "й": "ý", "Й": "Ý", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "h", "Х": "H", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "y", "Ы": "Y",
        "э": "e", "Э": "E", "ю": "ýu", "Ю": "Ýu", "я": "ýa", "Я": "Ýa",
        "ъ": "", "ь": "",
    },

    # Turkmen Latin → Cyrillic
    "tuk_Latn_to_Cyrl": {
        # Multi-char sequences first (greedy match)
        "şç": "щ", "Şç": "Щ",
        "ýo": "ё", "Ýo": "Ё", "ÝO": "Ё",
        "ýu": "ю", "Ýu": "Ю", "ÝU": "Ю",
        "ýa": "я", "Ýa": "Я", "ÝA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Turkmen-specific Latin letters
        "ä": "ә", "Ä": "Ә", "ň": "ң", "Ň": "Ң",
        "ö": "ө", "Ö": "Ө", "ü": "ү", "Ü": "Ү",
        "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "ž": "ж", "Ž": "Ж", "ý": "й", "Ý": "Й",
        # Standard Latin → Cyrillic (note: j → җ, w → в)
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "х", "H": "Х", "i": "и", "I": "И", "j": "җ", "J": "Җ",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "r": "р", "R": "Р", "s": "с", "S": "С", "t": "т", "T": "Т",
        "u": "у", "U": "У", "w": "в", "W": "В", "y": "ы", "Y": "Ы",
        "z": "з", "Z": "З",
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

    # Crimean Tatar Latin → Cyrillic
    "crh_Latn_to_Cyrl": {
        # Digraphs first (greedy match)
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Special Latin letters → Cyrillic digraphs
        "ğ": "гъ", "Ğ": "Гъ", "q": "къ", "Q": "Къ", "ñ": "нъ", "Ñ": "Нъ",
        "c": "дж", "C": "Дж",
        # Single characters
        "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш", "İ": "И", "i": "и",
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "х", "H": "Х", "j": "ж", "J": "Ж", "k": "к", "K": "К",
        "l": "л", "L": "Л", "m": "м", "M": "М", "n": "н", "N": "Н",
        "o": "о", "O": "О", "p": "п", "P": "П", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "v": "в", "V": "В", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Uzbek Latin → Cyrillic (reverse of 1995 official alphabet)
    "uzb_Latn_to_Cyrl": {
        # Digraphs/trigraphs first (greedy match)
        "sh": "ш", "Sh": "Ш", "SH": "Ш",
        "ch": "ч", "Ch": "Ч", "CH": "Ч",
        "ng": "нг", "Ng": "Нг", "NG": "НГ",
        "g'": "ғ", "G'": "Ғ", "o'": "ў", "O'": "Ў",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Single characters
        "'": "ъ",
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "ҳ", "H": "Ҳ", "i": "и", "I": "И", "j": "ж", "J": "Ж",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "q": "қ", "Q": "Қ", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "v": "в", "V": "В",
        "x": "х", "X": "Х", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Uyghur Latin (ULY) → Perso-Arabic
    "uig_Latn_to_Arab": {
        # Digraphs first (greedy match)
        "ch": "چ", "Ch": "چ", "CH": "چ",
        "sh": "ش", "Sh": "ش", "SH": "ش",
        "zh": "ژ", "Zh": "ژ", "ZH": "ژ",
        "gh": "غ", "Gh": "غ", "GH": "غ",
        "ng": "ڭ", "Ng": "ڭ", "NG": "ڭ",
        # Single characters
        "a": "ا", "e": "ە", "b": "ب", "p": "پ", "t": "ت", "j": "ج",
        "x": "خ", "d": "د", "r": "ر", "z": "ز", "s": "س", "f": "ف",
        "q": "ق", "k": "ك", "g": "گ", "l": "ل", "m": "م", "n": "ن",
        "h": "ھ", "o": "و", "u": "ۇ", "ö": "ۆ", "ü": "ۈ", "w": "ۋ",
        "é": "ې", "i": "ى", "y": "ي",
        # Capitals map to same Arabic letters (Arabic has no case)
        "A": "ا", "E": "ە", "B": "ب", "P": "پ", "T": "ت", "J": "ج",
        "X": "خ", "D": "د", "R": "ر", "Z": "ز", "S": "س", "F": "ف",
        "Q": "ق", "K": "ك", "G": "گ", "L": "ل", "M": "م", "N": "ن",
        "H": "ھ", "O": "و", "U": "ۇ", "Ö": "ۆ", "Ü": "ۈ", "W": "ۋ",
        "É": "ې", "I": "ى", "Y": "ي",
    },

    # Azerbaijani Cyrillic → Latin (1991 official Latin alphabet)
    "aze_Cyrl_to_Latn": {
        # Digraphs for Russian loanwords
        "щ": "şç", "Щ": "Şç",
        # Azerbaijani-specific Cyrillic letters
        "ә": "ə", "Ә": "Ə", "ғ": "ğ", "Ғ": "Ğ", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "ҹ": "c", "Ҹ": "C", "ҝ": "g", "Ҝ": "G",
        "һ": "h", "Һ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "q", "Г": "Q", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "İ", "й": "y", "Й": "Y", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Azerbaijani Latin → Cyrillic
    "aze_Latn_to_Cyrl": {
        # Digraphs first (greedy match)
        "şç": "щ", "Şç": "Щ",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Azerbaijani-specific Latin letters
        "ə": "ә", "Ə": "Ә", "ğ": "ғ", "Ğ": "Ғ", "ö": "ө", "Ö": "Ө",
        "ü": "ү", "Ü": "Ү", "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "İ": "И", "ı": "ы",
        # Standard Latin → Cyrillic (note: g → ҝ, q → г, c → ҹ)
        "a": "а", "A": "А", "b": "б", "B": "Б", "c": "ҹ", "C": "Ҹ",
        "d": "д", "D": "Д", "e": "е", "E": "Е", "f": "ф", "F": "Ф",
        "g": "ҝ", "G": "Ҝ", "h": "һ", "H": "Һ", "i": "и", "I": "Ы",
        "j": "ж", "J": "Ж", "k": "к", "K": "К", "l": "л", "L": "Л",
        "m": "м", "M": "М", "n": "н", "N": "Н", "o": "о", "O": "О",
        "p": "п", "P": "П", "q": "г", "Q": "Г", "r": "р", "R": "Р",
        "s": "с", "S": "С", "t": "т", "T": "Т", "u": "у", "U": "У",
        "v": "в", "V": "В", "x": "х", "X": "Х", "y": "й", "Y": "Й",
        "z": "з", "Z": "З",
    },

    # Tatar Cyrillic → Latin (Zamanälif)
    "tat_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "şç", "Щ": "Şç",
        # Tatar-specific Cyrillic letters
        "ә": "ä", "Ә": "Ä", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "җ": "c", "Җ": "C", "ң": "ñ", "Ң": "Ñ", "һ": "h", "Һ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "İ", "й": "y", "Й": "Y", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ц": "ts", "Ц": "Ts",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Tatar Latin (Zamanälif) → Cyrillic
    "tat_Latn_to_Cyrl": {
        # Digraphs first (greedy match)
        "şç": "щ", "Şç": "Щ",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Tatar-specific Latin letters
        "ä": "ә", "Ä": "Ә", "ö": "ө", "Ö": "Ө", "ü": "ү", "Ü": "Ү",
        "ñ": "ң", "Ñ": "Ң", "ç": "ч", "Ç": "Ч", "ş": "ш", "Ş": "Ш",
        "İ": "И", "ı": "ы",
        # Standard Latin → Cyrillic (note: c → җ, w → в)
        "a": "а", "A": "А", "b": "б", "B": "Б", "c": "җ", "C": "Җ",
        "d": "д", "D": "Д", "e": "е", "E": "Е", "f": "ф", "F": "Ф",
        "g": "г", "G": "Г", "h": "һ", "H": "Һ", "i": "и", "I": "Ы",
        "j": "ж", "J": "Ж", "k": "к", "K": "К", "l": "л", "L": "Л",
        "m": "м", "M": "М", "n": "н", "N": "Н", "o": "о", "O": "О",
        "p": "п", "P": "П", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "w": "в", "W": "В",
        "x": "х", "X": "Х", "y": "й", "Y": "Й", "z": "з", "Z": "З",
    },

    # Karakalpak Cyrillic → Latin (2016 Latin alphabet)
    "kaa_Cyrl_to_Latn": {
        # Digraphs for loanwords
        "щ": "shch", "Щ": "Shch",
        # Karakalpak-specific Cyrillic letters
        "ә": "á", "Ә": "Á", "ғ": "ǵ", "Ғ": "Ǵ", "қ": "q", "Қ": "Q",
        "ң": "ń", "Ң": "Ń", "ө": "ó", "Ө": "Ó", "ү": "ú", "Ү": "Ú",
        "ў": "w", "Ў": "W", "ҳ": "h", "Ҳ": "H",
        # Standard Cyrillic
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "и": "i", "И": "I", "й": "y", "Й": "Y", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ц": "ts", "Ц": "Ts", "ч": "ch", "Ч": "Ch",
        "ш": "sh", "Ш": "Sh", "ы": "í", "Ы": "Í",
        "э": "e", "Э": "E", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ъ": "", "ь": "",
    },

    # Karakalpak Latin → Cyrillic
    "kaa_Latn_to_Cyrl": {
        # Multi-char sequences first (greedy match)
        "shch": "щ", "Shch": "Щ",
        "sh": "ш", "Sh": "Ш", "SH": "Ш",
        "ch": "ч", "Ch": "Ч", "CH": "Ч",
        "yo": "ё", "Yo": "Ё", "YO": "Ё",
        "yu": "ю", "Yu": "Ю", "YU": "Ю",
        "ya": "я", "Ya": "Я", "YA": "Я",
        "ts": "ц", "Ts": "Ц", "TS": "Ц",
        # Karakalpak-specific Latin letters
        "á": "ә", "Á": "Ә", "ǵ": "ғ", "Ǵ": "Ғ", "ń": "ң", "Ń": "Ң",
        "ó": "ө", "Ó": "Ө", "ú": "ү", "Ú": "Ү", "í": "ы", "Í": "Ы",
        # Standard Latin → Cyrillic
        "a": "а", "A": "А", "b": "б", "B": "Б", "d": "д", "D": "Д",
        "e": "е", "E": "Е", "f": "ф", "F": "Ф", "g": "г", "G": "Г",
        "h": "ҳ", "H": "Ҳ", "i": "и", "I": "И", "j": "ж", "J": "Ж",
        "k": "к", "K": "К", "l": "л", "L": "Л", "m": "м", "M": "М",
        "n": "н", "N": "Н", "o": "о", "O": "О", "p": "п", "P": "П",
        "q": "қ", "Q": "Қ", "r": "р", "R": "Р", "s": "с", "S": "С",
        "t": "т", "T": "Т", "u": "у", "U": "У", "v": "в", "V": "В",
        "w": "ў", "W": "Ў", "x": "х", "X": "Х", "y": "й", "Y": "Й",
        "z": "з", "Z": "З",
    },

    # Ottoman Turkish Latin (academic transcription) → Perso-Arabic
    # Note: This is a simplified mapping. Ottoman script has many
    # Arabic/Persian letters that share Latin equivalents; this uses
    # the most common correspondence for each letter.
    "ota_Latn_to_Arab": {
        # Digraphs
        "ch": "چ", "Ch": "چ", "CH": "چ",
        "sh": "ش", "Sh": "ش", "SH": "ش",
        # Single characters
        "a": "ا", "A": "ا", "b": "ب", "B": "ب", "c": "ج", "C": "ج",
        "ç": "چ", "Ç": "چ", "d": "د", "D": "د", "e": "ه", "E": "ه",
        "f": "ف", "F": "ف", "g": "گ", "G": "گ", "ğ": "غ", "Ğ": "غ",
        "h": "ح", "H": "ح", "ı": "ی", "i": "ی", "I": "ی", "İ": "ی",
        "j": "ژ", "J": "ژ", "k": "ك", "K": "ك", "l": "ل", "L": "ل",
        "m": "م", "M": "م", "n": "ن", "N": "ن", "o": "و", "O": "و",
        "ö": "و", "Ö": "و", "p": "پ", "P": "پ", "r": "ر", "R": "ر",
        "s": "س", "S": "س", "ş": "ش", "Ş": "ش", "t": "ت", "T": "ت",
        "u": "و", "U": "و", "ü": "و", "Ü": "و", "v": "و", "V": "و",
        "y": "ی", "Y": "ی", "z": "ز", "Z": "ز",
    },

    # Old Turkic Runic (Orkhon) → Latin transliteration
    # Based on standard Turkological conventions for Orkhon inscriptions.
    # Unicode block U+10C00–U+10C4F.
    "otk_Orkh_to_Latn": {
        # Vowels
        "\U00010C00": "a",   # ORKHON A
        "\U00010C01": "a",   # YENISEI A
        "\U00010C02": "ä",   # YENISEI AE
        "\U00010C03": "ı",   # ORKHON I
        "\U00010C04": "ı",   # YENISEI I
        "\U00010C05": "e",   # YENISEI E
        "\U00010C06": "o",   # ORKHON O (also u)
        "\U00010C07": "ö",   # ORKHON OE (also ü)
        "\U00010C08": "ö",   # YENISEI OE
        # Consonants with back/front vowel variants
        "\U00010C09": "b",   # ORKHON AB (b with back vowels)
        "\U00010C0A": "b",   # YENISEI AB
        "\U00010C0B": "b",   # ORKHON AEB (b with front vowels)
        "\U00010C0C": "b",   # YENISEI AEB
        "\U00010C0D": "g",   # ORKHON AG (g with back vowels)
        "\U00010C0E": "g",   # YENISEI AG
        "\U00010C0F": "g",   # ORKHON AEG (g with front vowels)
        "\U00010C10": "g",   # YENISEI AEG
        "\U00010C11": "d",   # ORKHON AD (d with back vowels)
        "\U00010C12": "d",   # YENISEI AD
        "\U00010C13": "d",   # ORKHON AED (d with front vowels)
        "\U00010C14": "z",   # ORKHON AEZ
        "\U00010C15": "y",   # ORKHON AY (y with back vowels)
        "\U00010C16": "y",   # YENISEI AY
        "\U00010C17": "y",   # ORKHON AEY (y with front vowels)
        "\U00010C18": "y",   # YENISEI AEY
        "\U00010C19": "k",   # ORKHON AEK
        "\U00010C1A": "k",   # YENISEI AEK
        "\U00010C1B": "q",   # ORKHON AQ
        "\U00010C1C": "q",   # YENISEI AQ
        "\U00010C1D": "q",   # ORKHON IQ
        "\U00010C1E": "q",   # YENISEI IQ
        "\U00010C1F": "q",   # ORKHON OQ
        "\U00010C20": "q",   # YENISEI OQ
        "\U00010C21": "l",   # ORKHON AL
        "\U00010C22": "l",   # YENISEI AL
        "\U00010C23": "l",   # ORKHON AEL
        "\U00010C24": "l",   # YENISEI AEL
        "\U00010C25": "m",   # ORKHON EM
        "\U00010C26": "n",   # ORKHON AN
        "\U00010C27": "n",   # ORKHON AEN
        "\U00010C28": "n",   # YENISEI AEN
        "\U00010C29": "ŋ",   # ORKHON ENG (also ñ)
        "\U00010C2A": "ŋ",   # YENISEI AENG
        "\U00010C2B": "p",   # ORKHON EP
        "\U00010C2C": "p",   # YENISEI EP
        "\U00010C2D": "r",   # ORKHON AR
        "\U00010C2E": "r",   # YENISEI AR
        "\U00010C2F": "r",   # ORKHON AER
        "\U00010C30": "s",   # ORKHON AS
        "\U00010C31": "s",   # ORKHON AES
        "\U00010C32": "t",   # ORKHON AT (t with back vowels)
        "\U00010C33": "t",   # YENISEI AT
        "\U00010C34": "t",   # ORKHON AET (t with front vowels)
        "\U00010C35": "t",   # YENISEI AET
        "\U00010C36": "lt",  # ORKHON ALT/AEL
        "\U00010C37": "lt",  # YENISEI ALT
        "\U00010C38": "sh",  # ORKHON ASH
        "\U00010C39": "z",   # ORKHON AZ
        "\U00010C3A": "z",   # YENISEI AZ
        "\U00010C3B": "nt",  # ORKHON ANT
        "\U00010C3C": "nch", # ORKHON ANCH
        "\U00010C3D": "ch",  # ORKHON ICH
        "\U00010C3E": "ch",  # YENISEI ICH
        "\U00010C3F": "ch",  # ORKHON ECH
        "\U00010C40": "bash", # ORKHON BASH (head mark / punctuation)
        "\U00010C48": ":",   # OLD TURKIC WORD SEPARATOR
    },
}
