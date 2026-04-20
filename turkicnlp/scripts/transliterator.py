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
        if self.lang == "uig":
            src, tgt = self.source, self.target
            if src == Script.PERSO_ARABIC and tgt == Script.LATIN:
                return self._transliterate_uig_arab_to_latn(text)
            if src == Script.PERSO_ARABIC and tgt == Script.CYRILLIC:
                return self._transliterate_uig_arab_to_cyrl(text)
            if src == Script.CYRILLIC and tgt == Script.PERSO_ARABIC:
                return self._transliterate_uig_cyrl_to_arab(text)
            if src == Script.CYRILLIC and tgt == Script.LATIN:
                return self._transliterate_uig_cyrl_to_latn(text)
            if src == Script.LATIN and tgt == Script.CYRILLIC:
                return self._transliterate_uig_latn_to_cyrl(text)
            if src == Script.PERSO_ARABIC and tgt == Script.COMMON_TURKIC:
                return self._transliterate_uig_arab_to_cts(text)
            if src == Script.COMMON_TURKIC and tgt == Script.PERSO_ARABIC:
                return self._transliterate_uig_cts_to_arab(text)
            if src == Script.LATIN and tgt == Script.COMMON_TURKIC:
                return self._transliterate_uig_latn_to_cts(text)
            if src == Script.COMMON_TURKIC and tgt == Script.LATIN:
                return self._transliterate_uig_cts_to_latn(text)

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
        """Uyghur Latin (ULS) → Arabic via CTS pivot with full ئ insertion.

        Follows UMSC ULS2UAS logic: ULS→CTS chain, insert ئ before any vowel
        not preceded by a consonant, replace CTS→UAS, strip apostrophes, revise.
        This handles both word-initial vowels AND vowel-after-vowel sequences
        (e.g. 'radio' → رادىئو) and apostrophe syllable markers (e.g. "inik'ana").
        """
        import regex as re
        text = text.lower()
        text = self._uig_uls_to_cts(text)
        consonants = "bptcçxdrzjsşfñlmhvyqkgnğ"
        text = re.sub(
            rf"(?<=[^{consonants}]|^)[aeéiouöü]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    # ------------------------------------------------------------------
    # Uyghur multi-script helpers
    # Conversion tables and chain logic adapted from:
    # https://github.com/neouyghur/ScriptConverter4Uyghur (Apache-2.0)
    # ------------------------------------------------------------------

    # Positional tables: UAS / UCS / CTS share the same index positions.
    _UIG_UAS: list[str] = [
        "ا", "ە", "ب", "پ", "ت", "ج", "چ", "خ", "د", "ر",
        "ز", "ژ", "س", "ش", "ف", "ڭ", "لا", "ل", "م", "ھ",
        "و", "ۇ", "ۆ", "ۈ", "ۋ", "ې", "ى", "ي", "ق", "ك",
        "گ", "ن", "غ",
    ]
    _UIG_CTS: list[str] = [
        "a", "e", "b", "p", "t", "c", "ç", "x", "d", "r",
        "z", "j", "s", "ş", "f", "ñ", "la", "l", "m", "h",
        "o", "u", "ö", "ü", "v", "é", "i", "y", "q", "k",
        "g", "n", "ğ",
    ]
    _UIG_UCS: list[str] = [
        "а", "ә", "б", "п", "т", "җ", "ч", "х", "д", "р",
        "з", "ж", "с", "ш", "ф", "ң", "ла", "л", "м", "һ",
        "о", "у", "ө", "ү", "в", "е", "и", "й", "қ", "к",
        "г", "н", "ғ",
    ]
    _UIG_CTS_CONSONANTS: str = "bptcçxdrzjsşfñlmhvyqkgnğ"

    @staticmethod
    def _uig_replace(text: str, src: list[str], tgt: list[str]) -> str:
        for s, t in zip(src, tgt):
            text = text.replace(s, t)
        return text

    @staticmethod
    def _uig_revise_cts(text: str) -> str:
        """Remove word-initial ئ; remove post-vowel ئ; replace remaining ئ with apostrophe."""
        import regex as re
        text = re.sub(r"(?<=[^aeuoöübptcçxdrzjsşfñlmhvéiyqkgnğ]|^)\u0626", "", text)
        text = re.sub(r"(([aeéiouöü])\u0626)", lambda m: m.group()[0], text)
        text = text.replace("\u0626", "'")
        return text

    @staticmethod
    def _uig_revise_cts_keep_apos(text: str) -> str:
        """Remove word-initial ئ; keep post-vowel ئ as apostrophe (for Latin/Cyrillic output)."""
        import regex as re
        text = re.sub(r"(?<=[^aeuoöübptcçxdrzjsşfñlmhvéiyqkgnğ]|^)\u0626", "", text)
        # Unlike _uig_revise_cts, we do NOT strip vowel+ئ here — it becomes an apostrophe
        text = text.replace("\u0626", "'")
        return text

    @staticmethod
    def _uig_revise_uas(text: str) -> str:
        """Fix consecutive Arabic vowels by inserting ئ between them."""
        import regex as re
        return re.sub(
            r"(^|-|\s|[اەېىوۇۆۈ])([اەېىوۇۆۈ])",
            lambda m: m.group(1) + "ئ" + m.group(2),
            text,
        )

    @staticmethod
    def _uig_uls_to_cts(text: str) -> str:
        """ULS (Latin) → CTS chain."""
        return (
            text.replace("j", "c")
            .replace("ng", "ñ")
            .replace("n'g", "ng")
            .replace("'ng", "ñ")
            .replace("ch", "ç")
            .replace("zh", "j")
            .replace("sh", "ş")
            .replace("'gh", "ğ")
            .replace("gh", "ğ")
            .replace("w", "v")
        )

    @staticmethod
    def _uig_cts_to_uls(text: str) -> str:
        """CTS → ULS (Latin) chain."""
        return (
            text.replace("ng", "n'g")
            .replace("sh", "s'h")
            .replace("ch", "c'h")
            .replace("zh", "z'h")
            .replace("gh", "g'h")
            .replace("nğ", "n'gh")
            .replace("ñ", "ng")
            .replace("j", "zh")
            .replace("c", "j")
            .replace("ç", "ch")
            .replace("ş", "sh")
            .replace("ğ", "gh")
            .replace("v", "w")
        )

    def _transliterate_uig_arab_to_cts(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        return self._uig_revise_cts(text)

    def _transliterate_uig_arab_to_latn(self, text: str) -> str:
        """Arab → CTS (preserving inter-vowel ئ as apostrophe) → ULS chain."""
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        text = self._uig_revise_cts_keep_apos(text)
        return self._uig_cts_to_uls(text.lower())

    def _transliterate_uig_cts_to_arab(self, text: str) -> str:
        import regex as re
        vowels = "aeéiouöü"
        text = re.sub(
            rf"(?<=[^{self._UIG_CTS_CONSONANTS}]|^)[{vowels}]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    def _transliterate_uig_arab_to_cyrl(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UAS, self._UIG_CTS)
        text = self._uig_revise_cts_keep_apos(text)
        text = text.replace("ya", "я").replace("yu", "ю")
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UCS)
        return text

    def _transliterate_uig_cyrl_to_arab(self, text: str) -> str:
        import regex as re
        text = self._uig_replace(text, self._UIG_UCS, self._UIG_CTS)
        text = text.replace("я", "ya").replace("ю", "yu")
        vowels = "aeéiouöü"
        text = re.sub(
            rf"(?<=[^{self._UIG_CTS_CONSONANTS}]|^)[{vowels}]",
            lambda m: "\u0626" + m.group(),
            text,
        )
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UAS)
        text = text.replace("'", "")
        return self._uig_revise_uas(text)

    def _transliterate_uig_cyrl_to_latn(self, text: str) -> str:
        text = self._uig_replace(text, self._UIG_UCS, self._UIG_CTS)
        text = text.replace("я", "ya").replace("ю", "yu")
        return self._uig_cts_to_uls(text)

    def _transliterate_uig_latn_to_cyrl(self, text: str) -> str:
        text = text.lower()
        text = self._uig_uls_to_cts(text)
        text = text.replace("ya", "я").replace("yu", "ю")
        text = self._uig_replace(text, self._UIG_CTS, self._UIG_UCS)
        return text

    def _transliterate_uig_latn_to_cts(self, text: str) -> str:
        return self._uig_uls_to_cts(text.lower())

    def _transliterate_uig_cts_to_latn(self, text: str) -> str:
        return self._uig_cts_to_uls(text.lower())

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

    # ---------------------------------------------------------------------------
    # Common Turkic Script (CTS) tables
    # Based on the Common Turkic Alphabet: https://en.wikipedia.org/wiki/Common_Turkic_alphabet
    # ---------------------------------------------------------------------------

    # Turkish Latin → CTS (near-identical; Turkish already uses CTS-compatible letters)
    "tur_Latn_to_CTS": {
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "d": "d", "D": "D", "e": "e", "E": "E",
        "f": "f", "F": "F", "g": "g", "G": "G", "ğ": "ğ", "Ğ": "Ğ",
        "h": "h", "H": "H", "ı": "ı", "I": "I", "i": "i", "İ": "İ",
        "j": "j", "J": "J", "k": "k", "K": "K", "l": "l", "L": "L",
        "m": "m", "M": "M", "n": "n", "N": "N", "o": "o", "O": "O",
        "ö": "ö", "Ö": "Ö", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "ş": "ş", "Ş": "Ş", "t": "t", "T": "T",
        "u": "u", "U": "U", "ü": "ü", "Ü": "Ü", "v": "v", "V": "V",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Azerbaijani Latin → CTS (ə → ä is the only non-trivial mapping)
    "aze_Latn_to_CTS": {
        "ə": "ä", "Ə": "Ä",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "d": "d", "D": "D", "e": "e", "E": "E",
        "f": "f", "F": "F", "g": "g", "G": "G", "ğ": "ğ", "Ğ": "Ğ",
        "h": "h", "H": "H", "x": "x", "X": "X", "ı": "ı", "I": "I",
        "i": "i", "İ": "İ", "j": "j", "J": "J", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "ö": "ö", "Ö": "Ö", "p": "p", "P": "P",
        "q": "q", "Q": "Q", "r": "r", "R": "R", "s": "s", "S": "S",
        "ş": "ş", "Ş": "Ş", "t": "t", "T": "T", "u": "u", "U": "U",
        "ü": "ü", "Ü": "Ü", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Azerbaijani Cyrillic → CTS
    "aze_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ц": "ts", "Ц": "Ts",
        "ə": "ä", "Ə": "Ä",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "ҹ": "c", "Ҹ": "C", "ҝ": "g", "Ҝ": "G",
        "һ": "h", "Һ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "q", "Г": "Q", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ж": "c", "Ж": "C", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # South Azerbaijani (Perso-Arabic) → CTS
    "azb_Arab_to_CTS": {
        # Vowel letters
        "آ": "a", "ا": "a", "ع": "", "ه": "h",
        "و": "v", "ی": "y", "ي": "y",
        # Consonants
        "ب": "b", "پ": "p", "ت": "t", "ث": "s",
        "ج": "c", "چ": "ç", "ح": "h", "خ": "x",
        "د": "d", "ذ": "z", "ر": "r", "ز": "z",
        "ژ": "j", "س": "s", "ش": "ş", "ص": "s",
        "ض": "z", "ط": "t", "ظ": "z", "غ": "ğ",
        "ف": "f", "ق": "q", "ک": "k", "ك": "k",
        "گ": "g", "ل": "l", "م": "m", "ن": "n",
        "ء": "",
    },

    # Kazakh Cyrillic → CTS
    "kaz_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç", "ж": "c", "Ж": "C",
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ұ": "u", "Ұ": "U",
        "ү": "ü", "Ү": "Ü", "і": "i", "І": "İ", "һ": "h", "Һ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "э": "e", "Э": "E", "ы": "ı", "Ы": "I", "ъ": "", "ь": "",
    },

    # Kazakh Latin (2021) → CTS
    "kaz_Latn_to_CTS": {
        "ä": "ä", "Ä": "Ä", "ğ": "ğ", "Ğ": "Ğ", "q": "q", "Q": "Q",
        "ñ": "ñ", "Ñ": "Ñ", "ö": "ö", "Ö": "Ö", "ū": "u", "Ū": "U",
        "ü": "ü", "Ü": "Ü", "ı": "ı",
        "sh": "ş", "Sh": "Ş", "SH": "Ş", "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "İ": "y",             # Kazakh 2021: İ = /j/ (Cyrillic й) → CTS y
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "x", "H": "X",   # Kazakh h = /x/ (velar fricative) → CTS x
        "i": "i", "I": "İ", "j": "c", "J": "C",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V",
        "y": "ı", "Y": "I",   # Kazakh y = /ɯ/ (back unrounded) → CTS ı
        "z": "z", "Z": "Z",
    },

    # Uzbek Latin (1995) → CTS
    "uzb_Latn_to_CTS": {
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "ng": "ñ", "Ng": "Ñ", "NG": "Ñ",
        "g'": "ğ", "G'": "Ğ",
        "o'": "ö", "O'": "Ö",
        "yo": "yo", "Yo": "Yo",
        "yu": "yu", "Yu": "Yu",
        "ya": "ya", "Ya": "Ya",
        "'": "",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "I": "İ", "j": "c", "J": "C",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "q": "q", "Q": "Q", "r": "r", "R": "R", "s": "s", "S": "S",
        "t": "t", "T": "T", "u": "u", "U": "U", "v": "v", "V": "V",
        "x": "x", "X": "X", "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Uzbek Cyrillic → CTS
    "uzb_Cyrl_to_CTS": {
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "нг": "ñ", "Нг": "Ñ",
        "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ҳ": "h", "Ҳ": "H", "ў": "ö", "Ў": "Ö",
        "ж": "c", "Ж": "C",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ц": "ts", "Ц": "Ts",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Kyrgyz Cyrillic → CTS
    "kir_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç", "ж": "c", "Ж": "C",
        "ц": "ts", "Ц": "Ts",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Turkmen Latin → CTS
    "tuk_Latn_to_CTS": {
        "şç": "şç", "Şç": "Şç",
        "ts": "ts", "Ts": "Ts",
        "ä": "ä", "Ä": "Ä", "ň": "ñ", "Ň": "Ñ",
        "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "ž": "j", "Ž": "J",
        "ý": "y", "Ý": "Y",   # Turkmen ý = /j/ → CTS y
        "y": "ı", "Y": "I",   # Turkmen y = /ɯ/ → CTS ı
        "j": "c", "J": "C",
        "w": "v", "W": "V",   # Turkmen w = /w/ → CTS V slot (Turkmen has no /v/)
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "I": "İ", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "z": "z", "Z": "Z",
    },

    # Turkmen Cyrillic → CTS
    "tuk_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ё": "yo", "Ё": "Yo", "ю": "ýu", "Ю": "Ýu", "я": "ýa", "Я": "Ýa",
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ң": "ñ", "Ң": "Ñ",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ч": "ç", "Ч": "Ç", "ш": "ş", "Ш": "Ş",
        "ж": "j", "Ж": "J", "з": "z", "З": "Z",
        "й": "y", "Й": "Y", "в": "v", "В": "V",   # Turkmen в = /w/ → CTS V slot
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "и": "i", "И": "İ",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "h", "Х": "H",
        "ы": "ı", "Ы": "I",   # Turkmen ы = /ɯ/ → CTS ı
        "э": "e", "Э": "E", "ъ": "", "ь": "",
        "х": "h", "Х": "H", "җ": "c", "Җ": "C",
    },

    # Tatar Cyrillic → CTS
    "tat_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J",   # Tatar ж = /ʒ/
        "җ": "c", "Җ": "C",   # Tatar Ж̧ = /dʒ/
        "ц": "ts", "Ц": "Ts",
        "ә": "ä", "Ә": "Ä", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ң": "ñ", "Ң": "Ñ", "һ": "h", "Һ": "H",
        "в": "w", "В": "W",   # Tatar в = /w/
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "ё": "yo", "Ё": "Yo",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Tatar Latin (Zamanälif) → CTS
    "tat_Latn_to_CTS": {
        "şç": "şç", "Şç": "Şç",
        "ts": "ts", "Ts": "Ts",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ñ": "ñ", "Ñ": "Ñ", "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "c": "c", "C": "C",   # Zamanälif c = /dʒ/
        "j": "j", "J": "J",   # Zamanälif j = /ʒ/
        "w": "w", "W": "W",   # Tatar w = /w/
        "İ": "İ", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "i": "i", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "x": "x", "X": "X", "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Uyghur Cyrillic (UCS) → CTS
    "uig_Cyrl_to_CTS": {
        "ла": "la", "Ла": "La",   # multi-char first
        "я": "ya", "Я": "Ya", "ю": "yu", "Ю": "Yu",
        "а": "a", "А": "A", "ә": "e", "Ә": "E",
        "б": "b", "Б": "B", "п": "p", "П": "P",
        "т": "t", "Т": "T", "җ": "c", "Җ": "C",
        "ч": "ç", "Ч": "Ç", "х": "x", "Х": "X",
        "д": "d", "Д": "D", "р": "r", "Р": "R",
        "з": "z", "З": "Z", "ж": "j", "Ж": "J",
        "с": "s", "С": "S", "ш": "ş", "Ш": "Ş",
        "ф": "f", "Ф": "F", "ң": "ñ", "Ң": "Ñ",
        "л": "l", "Л": "L", "м": "m", "М": "M",
        "һ": "h", "Һ": "H", "о": "o", "О": "O",
        "у": "u", "У": "U", "ө": "ö", "Ө": "Ö",
        "ү": "ü", "Ү": "Ü", "в": "v", "В": "V",
        "е": "é", "Е": "É", "и": "i", "И": "I",
        "й": "y", "Й": "Y", "қ": "q", "Қ": "Q",
        "к": "k", "К": "K", "г": "g", "Г": "G",
        "н": "n", "Н": "N", "ғ": "ğ", "Ғ": "Ğ",
    },

    # Uyghur CTS → Cyrillic (UCS)
    "uig_CTS_to_Cyrl": {
        "la": "ла", "La": "Ла",   # multi-char first
        "ya": "я", "Ya": "Я", "yu": "ю", "Yu": "Ю",
        "a": "а", "A": "А", "e": "ә", "E": "Ә",
        "b": "б", "B": "Б", "p": "п", "P": "П",
        "t": "т", "T": "Т", "c": "җ", "C": "Җ",
        "ç": "ч", "Ç": "Ч", "x": "х", "X": "Х",
        "d": "д", "D": "Д", "r": "р", "R": "Р",
        "z": "з", "Z": "З", "j": "ж", "J": "Ж",
        "s": "с", "S": "С", "ş": "ш", "Ş": "Ш",
        "f": "ф", "F": "Ф", "ñ": "ң", "Ñ": "Ң",
        "l": "л", "L": "Л", "m": "м", "M": "М",
        "h": "һ", "H": "Һ", "o": "о", "O": "О",
        "u": "у", "U": "У", "ö": "ө", "Ö": "Ө",
        "ü": "ү", "Ü": "Ү", "v": "в", "V": "В",
        "é": "е", "É": "Е", "i": "и", "I": "И",
        "y": "й", "Y": "Й", "q": "қ", "Q": "Қ",
        "k": "к", "K": "К", "g": "г", "G": "Г",
        "n": "н", "N": "Н", "ğ": "ғ", "Ğ": "Ғ",
    },

    # Uyghur: placeholder entries for pairs handled by special methods
    "uig_Arab_to_CTS": {},
    "uig_CTS_to_Arab": {},
    "uig_Arab_to_Cyrl": {},
    "uig_Cyrl_to_Arab": {},
    "uig_Latn_to_Cyrl": {},
    "uig_Cyrl_to_Latn": {},
    "uig_Latn_to_CTS": {},
    "uig_CTS_to_Latn": {},

    # Bashkir Cyrillic → CTS
    "bak_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J",   # Bashkir ж = /ʒ/
        "ц": "ts", "Ц": "Ts",
        "ä": "ä", "Ä": "Ä",   # Bashkir ä (a with breve in some fonts)
        "ă": "ä", "Ă": "Ä",   # alternative Bashkir reduced a
        "ö": "ö", "Ö": "Ö",
        "ü": "ü", "Ü": "Ü",
        "ҙ": "đ", "Ҙ": "Đ",   # Bashkir /ð/ → CTS Ź (written Đ in Bashkir CTS)
        "ҫ": "ŧ", "Ҫ": "Ŧ",   # Bashkir /θ/ → CTS Ś (written Ŧ in Bashkir CTS)
        "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ő": "ö", "Ő": "Ö",
        "ҡ": "q", "Ҡ": "Q",   # Bashkir q
        "в": "v", "В": "V",   # Bashkir В = /v/ (has both V and W per CTA)
        "а": "a", "А": "A", "б": "b", "Б": "B", "г": "g", "Г": "G",
        "д": "d", "Д": "D", "е": "e", "Е": "E", "ё": "yo", "Ё": "Yo",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
        "ү": "ü", "Ү": "Ü",
    },

    # Crimean Tatar Latin → CTS (already very close)
    "crh_Latn_to_CTS": {
        "ts": "ts", "Ts": "Ts",
        "ğ": "ğ", "Ğ": "Ğ", "q": "q", "Q": "Q",
        "ñ": "ñ", "Ñ": "Ñ", "c": "c", "C": "C",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "h": "h", "H": "H", "j": "j", "J": "J", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "ö": "ö", "Ö": "Ö", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "ü": "ü", "Ü": "Ü", "v": "v", "V": "V",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Crimean Tatar Cyrillic → CTS
    "crh_Cyrl_to_CTS": {
        "гъ": "ğ", "Гъ": "Ğ", "дж": "c", "Дж": "C",
        "къ": "q", "Къ": "Q", "нъ": "ñ", "Нъ": "Ñ",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "j", "Ж": "J", "ц": "ts", "Ц": "Ts",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "h", "Х": "H",
        "э": "e", "Э": "E", "ы": "ı", "Ы": "I", "ъ": "", "ь": "",
        "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
    },

    # Chuvash Cyrillic → CTS
    "chv_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ă": "ä", "Ă": "Ä",   # Chuvash reduced a → ä
        "ĕ": "e", "Ĕ": "E",   # Chuvash reduced e → e
        "ҫ": "ş", "Ҫ": "Ş",   # Chuvash palatal sibilant → ş
        "ӳ": "ü", "Ӳ": "Ü",   # Chuvash ü
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Sakha (Yakut) Cyrillic → CTS
    "sah_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ҕ": "ğ", "Ҕ": "Ğ",   # Sakha voiced uvular fricative
        "ҥ": "ñ", "Ҥ": "Ñ",   # Sakha ng
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "ё": "yo", "Ё": "Yo", "з": "z", "З": "Z", "и": "i", "И": "İ",
        "й": "y", "Й": "Y", "к": "k", "К": "K", "л": "l", "Л": "L",
        "м": "m", "М": "M", "н": "n", "Н": "N", "о": "o", "О": "O",
        "п": "p", "П": "P", "р": "r", "Р": "R", "с": "s", "С": "S",
        "т": "t", "Т": "T", "у": "u", "У": "U", "ф": "f", "Ф": "F",
        "х": "x", "Х": "X", "ы": "ı", "Ы": "I", "э": "e", "Э": "E",
        "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya", "ъ": "", "ь": "",
    },

    # Karakalpak Latin → CTS
    "kaa_Latn_to_CTS": {
        "shch": "şç", "Shch": "Şç",
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "ts": "ts", "Ts": "Ts",
        "á": "ä", "Á": "Ä", "ǵ": "ğ", "Ǵ": "Ğ",
        "ń": "ñ", "Ń": "Ñ", "ó": "ö", "Ó": "Ö",
        "ú": "ü", "Ú": "Ü", "í": "ı", "Í": "I",
        "q": "q", "Q": "Q", "h": "h", "H": "H",
        "a": "a", "A": "A", "b": "b", "B": "B", "d": "d", "D": "D",
        "e": "e", "E": "E", "f": "f", "F": "F", "g": "g", "G": "G",
        "i": "i", "I": "İ", "j": "c", "J": "C", "k": "k", "K": "K",
        "l": "l", "L": "L", "m": "m", "M": "M", "n": "n", "N": "N",
        "o": "o", "O": "O", "p": "p", "P": "P", "r": "r", "R": "R",
        "s": "s", "S": "S", "t": "t", "T": "T", "u": "u", "U": "U",
        "v": "v", "V": "V", "w": "w", "W": "W", "x": "x", "X": "X",
        "y": "y", "Y": "Y", "z": "z", "Z": "Z",
    },

    # Karakalpak Cyrillic → CTS
    "kaa_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ә": "ä", "Ә": "Ä", "ғ": "ğ", "Ғ": "Ğ", "қ": "q", "Қ": "Q",
        "ң": "ñ", "Ң": "Ñ", "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ў": "w", "Ў": "W", "ҳ": "h", "Ҳ": "H",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Gagauz Latin → CTS
    "gag_Latn_to_CTS": {
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ş": "ş", "Ş": "Ş", "ç": "ç", "Ç": "Ç",
        # Gagauz has no Ğ per CTA
        "ţ": "ts", "Ţ": "Ts",   # Gagauz Ţ = /ts/ → CTS Ț
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "â": "a", "Â": "A",   # circumflex variants in Gagauz
        "î": "i", "Î": "İ",
        "û": "u", "Û": "U",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Nogai Cyrillic → CTS
    "nog_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",   # Nogai has W not V
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Kumyk Cyrillic → CTS
    "kum_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "w", "В": "W",   # Kumyk has W not V
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Karachay-Balkar Cyrillic → CTS
    "krc_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ң": "ñ", "Ң": "Ñ", "ğ": "ğ", "Ğ": "Ğ", "қ": "q", "Қ": "Q",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Altai Cyrillic → CTS
    "alt_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "дж": "c", "Дж": "C",   # Altai /dʒ/ digraph
        "ж": "j", "Ж": "J",     # Altai ж = /ʒ/
        "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ӓ": "ä", "Ӓ": "Ä",   # Altai ä
        "ӧ": "ö", "Ӧ": "Ö",   # Altai ö
        "ӱ": "ü", "Ӱ": "Ü",   # Altai ü
        "ҥ": "ñ", "Ҥ": "Ñ",   # Altai ng
        "й": "y", "Й": "Y",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "к": "k", "К": "K",
        "л": "l", "Л": "L", "м": "m", "М": "M", "н": "n", "Н": "N",
        "о": "o", "О": "O", "п": "p", "П": "P", "р": "r", "Р": "R",
        "с": "s", "С": "S", "т": "t", "Т": "T", "у": "u", "У": "U",
        "ф": "f", "Ф": "F", "х": "x", "Х": "X", "ы": "ı", "Ы": "I",
        "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Tuvan Cyrillic → CTS
    "tyv_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ө": "ö", "Ө": "Ö", "ү": "ü", "Ү": "Ü",
        "ң": "ñ", "Ң": "Ñ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Khakas Cyrillic → CTS
    "kjh_Cyrl_to_CTS": {
        "щ": "şç", "Щ": "Şç",
        "ш": "ş", "Ш": "Ş", "ч": "ç", "Ч": "Ç",
        "ж": "c", "Ж": "C", "ц": "ts", "Ц": "Ts",
        "ё": "yo", "Ё": "Yo", "ю": "yu", "Ю": "Yu", "я": "ya", "Я": "Ya",
        "ӧ": "ö", "Ӧ": "Ö",   # Khakas ö
        "ӱ": "ü", "Ӱ": "Ü",   # Khakas ü
        "ң": "ñ", "Ң": "Ñ",
        "ғ": "ğ", "Ғ": "Ğ",
        "а": "a", "А": "A", "б": "b", "Б": "B", "в": "v", "В": "V",
        "г": "g", "Г": "G", "д": "d", "Д": "D", "е": "e", "Е": "E",
        "з": "z", "З": "Z", "и": "i", "И": "İ", "й": "y", "Й": "Y",
        "к": "k", "К": "K", "л": "l", "Л": "L", "м": "m", "М": "M",
        "н": "n", "Н": "N", "о": "o", "О": "O", "п": "p", "П": "P",
        "р": "r", "Р": "R", "с": "s", "С": "S", "т": "t", "Т": "T",
        "у": "u", "У": "U", "ф": "f", "Ф": "F", "х": "x", "Х": "X",
        "ы": "ı", "Ы": "I", "э": "e", "Э": "E", "ъ": "", "ь": "",
    },

    # Ottoman Turkish Latin (academic) → CTS
    "ota_Latn_to_CTS": {
        "ch": "ç", "Ch": "Ç", "CH": "Ç",
        "sh": "ş", "Sh": "Ş", "SH": "Ş",
        "gh": "ğ", "Gh": "Ğ", "GH": "Ğ",
        "ç": "ç", "Ç": "Ç", "ş": "ş", "Ş": "Ş",
        "ğ": "ğ", "Ğ": "Ğ", "ö": "ö", "Ö": "Ö",
        "ü": "ü", "Ü": "Ü", "ı": "ı", "İ": "İ", "i": "i",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },

    # Khalaj Latin → CTS (klj; Khalaj is an archaic Turkic language)
    "klj_Latn_to_CTS": {
        "ä": "ä", "Ä": "Ä", "ö": "ö", "Ö": "Ö", "ü": "ü", "Ü": "Ü",
        "ş": "ş", "Ş": "Ş", "ç": "ç", "Ç": "Ç", "ğ": "ğ", "Ğ": "Ğ",
        "ñ": "ñ", "Ñ": "Ñ", "q": "q", "Q": "Q", "x": "x", "X": "X",
        "İ": "İ", "i": "i", "ı": "ı", "I": "I",
        "a": "a", "A": "A", "b": "b", "B": "B", "c": "c", "C": "C",
        "d": "d", "D": "D", "e": "e", "E": "E", "f": "f", "F": "F",
        "g": "g", "G": "G", "h": "h", "H": "H", "j": "j", "J": "J",
        "k": "k", "K": "K", "l": "l", "L": "L", "m": "m", "M": "M",
        "n": "n", "N": "N", "o": "o", "O": "O", "p": "p", "P": "P",
        "r": "r", "R": "R", "s": "s", "S": "S", "t": "t", "T": "T",
        "u": "u", "U": "U", "v": "v", "V": "V", "y": "y", "Y": "Y",
        "z": "z", "Z": "Z",
    },
}
