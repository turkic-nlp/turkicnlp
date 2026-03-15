"""MorphemeTokenizer — hybrid neural + FST morpheme segmentation for Turkic languages.

Segments inflected Turkic words into surface morphemes:

    >>> tok = MorphemeTokenizer(lang="kaz")
    >>> tok.load()
    >>> tok.segment("үйлеріңіздегілерден")
    ['үй', 'лер', 'іңіз', 'де', 'гі', 'лер', 'ден']

Architecture:
    1. **Neural morph model** (primary) → lemma, UPOS, UD features
    2. **HFST Apertium FST** (secondary) → fine-grained Apertium tags including
       derivational morphology (<subst>, <caus>, etc.)
    3. **Suffix allomorph tables** + phonological rules → surface morpheme boundaries

The neural model handles OOV words and provides broad coverage (21 languages).
The HFST backend enriches the analysis with derivational tags that UD features
don't capture, enabling finer segmentation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from turkicnlp.models.document import Document

logger = logging.getLogger(__name__)


def _turkic_lower(text: str) -> str:
    """Lowercase with Turkic I/İ awareness.

    Python's ``str.lower()`` maps ``I`` → ``i``, but in Turkish/Azerbaijani
    ``I`` → ``ı`` and ``İ`` → ``i``.  This function handles both correctly.
    """
    result = []
    for ch in text:
        if ch == "I":
            result.append("ı")
        elif ch == "\u0130":  # İ
            result.append("i")
        else:
            result.append(ch.lower())
    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Phonological classification
# ═══════════════════════════════════════════════════════════════════════════

# --- Cyrillic character classes (Kazakh, Kyrgyz, Tatar, Bashkir, etc.) ---

_CY_BACK_V = set("аоұыуАОҰЫУӑӐ")
_CY_FRONT_V = set("еөүіэЕӨҮІЭĕĔӳӲ")
_CY_VOWELS = _CY_BACK_V | _CY_FRONT_V
# Liquid sonorants — trigger л-initial suffix allomorphs
_CY_SONORANTS = set("рлйРЛЙ")
# Nasals — trigger д-initial (or н-initial for ablative)
_CY_NASALS = set("мнңМНҢ")
# Voiced obstruents
_CY_VOICED_OBS = set("бвгғджзБВГҒДЖЗ")
# Voiceless obstruents — trigger т-initial / қ-initial
_CY_VOICELESS = set("пфкқтсшщцчхПФКҚТСШЩЦЧХ")

# --- Latin character classes (Turkish, Azerbaijani, Turkmen, Uzbek, etc.) ---

_LA_BACK_V = set("aıouAIOUāĀ")
_LA_FRONT_V = set("eiöüEİÖÜəƏ")
_LA_VOWELS = _LA_BACK_V | _LA_FRONT_V
_LA_SONORANTS = set("rlyRLY")
_LA_NASALS = set("mnMNñÑňŇŋŊ")
_LA_VOICED_OBS = set("bdgğjvzBDGĞJVZ")
_LA_VOICELESS = set("çfhkpstşÇFHKPSTŞ")

# --- Arabic script character classes (Uyghur) ---
# Minimal set — Arabic script segmentation is less reliable
_AR_BACK_V = set("اوۇ")
_AR_FRONT_V = set("ەېۆۈ")
_AR_VOWELS = _AR_BACK_V | _AR_FRONT_V


class _Phonology:
    """Phonological context for suffix allomorph selection."""

    __slots__ = ("back_vowels", "front_vowels", "vowels",
                 "sonorants", "nasals", "voiced", "voiceless")

    def __init__(self, script: str):
        if script == "Cyrl":
            self.back_vowels = _CY_BACK_V
            self.front_vowels = _CY_FRONT_V
            self.vowels = _CY_VOWELS
            self.sonorants = _CY_SONORANTS
            self.nasals = _CY_NASALS
            self.voiced = _CY_VOICED_OBS | _CY_NASALS
            self.voiceless = _CY_VOICELESS
        elif script == "Latn":
            self.back_vowels = _LA_BACK_V
            self.front_vowels = _LA_FRONT_V
            self.vowels = _LA_VOWELS
            self.sonorants = _LA_SONORANTS
            self.nasals = _LA_NASALS
            self.voiced = _LA_VOICED_OBS | _LA_NASALS
            self.voiceless = _LA_VOICELESS
        else:
            # Fallback to Cyrillic rules
            self.back_vowels = _CY_BACK_V
            self.front_vowels = _CY_FRONT_V
            self.vowels = _CY_VOWELS
            self.sonorants = _CY_SONORANTS
            self.nasals = _CY_NASALS
            self.voiced = _CY_VOICED_OBS | _CY_NASALS
            self.voiceless = _CY_VOICELESS


def _get_harmony(text: str, phon: _Phonology) -> str:
    """Determine vowel harmony (back/front) from the last vowel in *text*."""
    for ch in reversed(text):
        if ch in phon.back_vowels:
            return "B"
        if ch in phon.front_vowels:
            return "F"
    return "B"  # default: back


def _get_harmony4(text: str, phon: _Phonology) -> str:
    """4-way harmony: BU (back-unrounded), FU (front-unrounded),
    BR (back-rounded), FR (front-rounded).

    Used by Turkish for certain suffixes (accusative, possessive, etc.).
    """
    _ROUND_CY = set("оөұүОӨҰҮ")
    _ROUND_LA = set("oöuüOÖUÜ")
    for ch in reversed(text):
        if ch in phon.back_vowels:
            if ch in _ROUND_CY or ch in _ROUND_LA:
                return "BR"
            return "BU"
        if ch in phon.front_vowels:
            if ch in _ROUND_CY or ch in _ROUND_LA:
                return "FR"
            return "FU"
    return "BU"


def _char_class(ch: str, phon: _Phonology) -> str:
    """Classify the final character: V(owel), S(onorant), N(asal), D(voiced), T(voiceless)."""
    if ch in phon.vowels:
        return "V"
    if ch in phon.sonorants:
        return "S"
    if ch in phon.nasals:
        return "N"
    if ch in phon.voiced:
        return "D"
    if ch in phon.voiceless:
        return "T"
    return "T"  # default: treat unknown as voiceless


# ═══════════════════════════════════════════════════════════════════════════
# Suffix allomorph tables — loaded from morpheme_rules.json
# ═══════════════════════════════════════════════════════════════════════════
#
# All language-specific suffix tables live in turkicnlp/resources/morpheme_rules.json.
# They are loaded once at module import time and converted to the internal format:
#   tag -> list of (condition_chars, forms_tuple)
# condition_chars: string of character classes that trigger this allomorph.
#   V = vowel, S = sonorant, N = nasal, D = voiced obstruent, T = voiceless
# forms_tuple: (back, front) for 2-way or (BU, FU, BR, FR) for 4-way harmony.
# First matching condition wins.
#
# To regenerate morpheme_rules.json from Wiktionary + manual tables, run:
#   PYTHONPATH=. python scripts/extract_suffix_rules.py

def _2way(conditions: list[tuple[str, str, str]]) -> list[tuple[str, tuple]]:
    """Helper: 2-way harmony entries."""
    return [(c, (b, f)) for c, b, f in conditions]


def _4way(conditions: list[tuple[str, str, str, str, str]]) -> list[tuple[str, tuple]]:
    """Helper: 4-way harmony entries."""
    return [(c, (bu, fu, br, fr)) for c, bu, fu, br, fr in conditions]


_MORPHEME_RULES_PATH = Path(__file__).resolve().parent.parent / "resources" / "morpheme_rules.json"
_JSON_RULES_CACHE: dict | None = None


def _load_json_rules() -> dict:
    """Load morpheme rules from JSON file (cached)."""
    global _JSON_RULES_CACHE
    if _JSON_RULES_CACHE is not None:
        return _JSON_RULES_CACHE
    if _MORPHEME_RULES_PATH.exists():
        with open(_MORPHEME_RULES_PATH, encoding="utf-8") as f:
            _JSON_RULES_CACHE = json.load(f)
        logger.debug("Loaded morpheme rules from %s", _MORPHEME_RULES_PATH)
    else:
        _JSON_RULES_CACHE = {"languages": {}}
        logger.warning("No morpheme_rules.json found at %s", _MORPHEME_RULES_PATH)
    return _JSON_RULES_CACHE


def _json_conditions_to_entries(conditions: list[dict]) -> list[tuple[str, tuple]]:
    """Convert JSON condition format to the internal (context, forms_tuple) format.

    JSON format: [{"context": "VSND", "forms": {"B": "da", "F": "de"}}]
    Internal:    [("VSND", ("da", "de"))]  or  [("VSND", ("ı", "i", "u", "ü"))]
    """
    entries = []
    for cond in conditions:
        ctx = cond["context"]
        forms = cond["forms"]
        if "BU" in forms:
            # 4-way harmony
            t = (forms.get("BU", ""), forms.get("FU", ""),
                 forms.get("BR", ""), forms.get("FR", ""))
        else:
            # 2-way harmony
            t = (forms.get("B", ""), forms.get("F", ""))
        entries.append((ctx, t))
    return entries


def _build_suffix_map_from_json() -> dict[str, tuple[dict, str]]:
    """Build the (suffix_table, script) map for all languages from JSON."""
    rules = _load_json_rules()
    result = {}

    for lang_code, lang_data in rules.get("languages", {}).items():
        script = lang_data.get("script", "Latn")
        suffix_table: dict[str, list[tuple[str, tuple]]] = {}

        for section in ("noun_suffixes", "verb_suffixes"):
            for key, rule_data in lang_data.get(section, {}).items():
                conditions = rule_data.get("conditions", [])
                if conditions:
                    entries = _json_conditions_to_entries(conditions)
                    if entries:
                        suffix_table[key] = entries

        result[lang_code] = (suffix_table, script)

    return result


# Build the suffix map at module load time
_LANG_SUFFIX_MAP: dict[str, tuple[dict, str]] = _build_suffix_map_from_json()


# Consonant alternation rules — also stored in JSON but keep accessible
def _build_stem_alternations() -> dict[str, dict[str, str]]:
    """Extract stem alternation rules from JSON."""
    rules = _load_json_rules()
    result = {}
    for lang_code, lang_data in rules.get("languages", {}).items():
        alt = lang_data.get("stem_alternations", {})
        if alt:
            result[lang_code] = alt
    return result


_STEM_ALTERNATIONS: dict[str, dict[str, str]] = _build_stem_alternations()


# Provide access to individual language tables for test imports
def _get_suffix_table(lang: str) -> dict:
    """Get the suffix table for a language."""
    table, _ = _LANG_SUFFIX_MAP.get(lang, ({}, "Latn"))
    return table


_KAZ_SUFFIXES = _get_suffix_table("kaz")
_TUR_SUFFIXES = _get_suffix_table("tur")

# ═══════════════════════════════════════════════════════════════════════════
# Apertium tag → suffix key mapping
# ═══════════════════════════════════════════════════════════════════════════

# Apertium tags that correspond to morpheme slots (ordered by suffix position)
_APT_TAG_TO_KEY: dict[str, str] = {
    # Number
    "pl": "pl",
    # Possessive
    "px1sg": "px1sg", "px2sg": "px2sg", "px3sg": "px3sg", "px3sp": "px3sp",
    "px1pl": "px1pl", "px2pl": "px2pl", "px3pl": "px3pl",
    "px2sg_frm": "px2sg_frm",
    # Case
    "nom": "nom", "acc": "acc", "dat": "dat", "loc": "loc",
    "abl": "abl", "gen": "gen", "ins": "ins", "equ": "equ",
    # Derivation
    "subst": "subst", "attr": "attr",
    "ly": "ly", "siz": "siz", "lyk": "lyk", "shi": "shi",
    # Verb voice
    "pass": "pass", "caus": "caus", "coop": "coop",
    # Negation
    "neg": "neg",
    # Tense / mood
    "past": "past", "pres": "pres", "fut": "fut", "aor": "aor",
    "evid": "evid", "imp": "imp", "opt": "opt", "cond": "cond",
    "neces": "neces",
    # Person-number agreement (verb)
    "p1": "p1sg", "p2": "p2sg", "p3": "p3sg",
    # Converbs / participles
    "gna_perf": "gna_perf", "gna_impf": "gna_impf",
    "prc_perf": "prc_perf", "prc_impf": "prc_impf", "prc_fut": "prc_fut",
    # Non-finite
    "inf": "inf", "ger": "ger", "ger_past": "ger_past",
    # Copula / question
    "cop": "cop", "qst": "qst",
}

# Tags to skip (POS tags, features that don't correspond to a surface morpheme)
_APT_SKIP_TAGS = {
    "n", "v", "adj", "adv", "np", "prn", "det", "post", "cnjcoo", "cnjsub",
    "cnjadv", "part", "ij", "abbr", "num", "vaux",
    "sg",  # singular is usually zero-marked
    "nom",  # nominative is zero-marked
    "ind",  # indicative mood — no surface morpheme
}


# ═══════════════════════════════════════════════════════════════════════════
# UD feature → suffix key mapping (for neural-only fallback)
# ═══════════════════════════════════════════════════════════════════════════

# UD features that correspond to morpheme slots.
# Order matters: this defines the canonical suffix ordering for reconstruction.
# Each entry is (required_features, suffix_key).
# required_features is a frozenset — ALL features must be present in feat_set.
_UD_NOUN_SUFFIX_ORDER: list[tuple[frozenset[str], str]] = [
    (frozenset({"Number=Plur"}), "pl"),
    # Possessive — each requires multiple features ([psor] variants)
    (frozenset({"Number[psor]=Sing", "Person[psor]=1"}), "px1sg"),
    (frozenset({"Number[psor]=Sing", "Person[psor]=2"}), "px2sg"),
    (frozenset({"Number[psor]=Sing", "Person[psor]=3"}), "px3sg"),
    (frozenset({"Number[psor]=Plur", "Person[psor]=1"}), "px1pl"),
    (frozenset({"Number[psor]=Plur", "Person[psor]=2"}), "px2pl"),
    (frozenset({"Number[psor]=Plur", "Person[psor]=3"}), "px3pl"),
    # Possessive — neural model sometimes uses plain Person= for nouns
    # Try plural possessor first (longer form matches first)
    (frozenset({"Person=1", "Poss=Yes"}), "px1pl"),
    (frozenset({"Person=2", "Poss=Yes"}), "px2pl"),
    (frozenset({"Person=3", "Poss=Yes"}), "px3pl"),
    (frozenset({"Person=1", "Poss=Yes"}), "px1sg"),
    (frozenset({"Person=2", "Poss=Yes"}), "px2sg"),
    (frozenset({"Person=3", "Poss=Yes"}), "px3sg"),
    # Possessive — neural model sometimes uses just Person= (no Poss=Yes)
    (frozenset({"Person=1", "Number=Sing"}), "px1sg"),
    (frozenset({"Person=2", "Number=Sing"}), "px2sg"),
    (frozenset({"Person=3", "Number=Sing"}), "px3sg"),
    # Case
    (frozenset({"Case=Acc"}), "acc"),
    (frozenset({"Case=Dat"}), "dat"),
    (frozenset({"Case=Loc"}), "loc"),
    (frozenset({"Case=Abl"}), "abl"),
    (frozenset({"Case=Gen"}), "gen"),
    (frozenset({"Case=Ins"}), "ins"),
    (frozenset({"Case=Equ"}), "equ"),
]

_UD_VERB_SUFFIX_ORDER: list[tuple[frozenset[str], str]] = [
    (frozenset({"Voice=Pass"}), "pass"),
    (frozenset({"Voice=Cau"}), "caus"),
    (frozenset({"Voice=Rcp"}), "coop"),
    # Compound neg+tense entries (must precede individual neg/tense)
    (frozenset({"Polarity=Neg", "Tense=Pres"}), "neg_pres"),
    (frozenset({"Polarity=Neg", "Tense=Aor"}), "neg_aor"),
    (frozenset({"Polarity=Neg", "Aspect=Prog"}), "neg_prog"),
    (frozenset({"Polarity=Neg"}), "neg"),
    (frozenset({"VerbForm=Conv"}), "gna_perf"),
    (frozenset({"VerbForm=Part"}), "prc_perf"),
    (frozenset({"VerbForm=Inf"}), "inf"),
    (frozenset({"VerbForm=Ger"}), "ger"),
    (frozenset({"Aspect=Prog"}), "prog"),
    (frozenset({"Tense=Past"}), "past"),
    (frozenset({"Tense=Pres"}), "pres"),
    (frozenset({"Tense=Fut"}), "fut"),
    (frozenset({"Tense=Aor"}), "aor"),
    (frozenset({"Aspect=Perf"}), "evid"),
    (frozenset({"Evidentiality=Fh"}), "evid"),
    (frozenset({"Mood=Imp"}), "imp"),
    (frozenset({"Mood=Opt"}), "opt"),
    (frozenset({"Mood=Cnd"}), "cond"),
    (frozenset({"Mood=Nec"}), "neces"),
    (frozenset({"Person=1", "Number=Plur"}), "p1pl"),
    (frozenset({"Person=2", "Number=Plur"}), "p2pl"),
    (frozenset({"Person=3", "Number=Plur"}), "p3pl"),
    (frozenset({"Person=1", "Number=Sing"}), "p1sg"),
    (frozenset({"Person=2", "Number=Sing"}), "p2sg"),
    (frozenset({"Person=3"}), "p3sg"),  # fallback for 3rd person
    (frozenset({"Person=1"}), "p1sg"),  # fallback when Number missing
    (frozenset({"Person=2"}), "p2sg"),  # fallback when Number missing
    # Plural after participle (substantivized: оқығандар)
    (frozenset({"Number=Plur"}), "pl"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Consonant alternation rules (stem-final devoicing reversal)
# ═══════════════════════════════════════════════════════════════════════════

_STEM_ALTERNATIONS: dict[str, dict[str, str]] = {
    "tur": {"p": "b", "ç": "c", "t": "d", "k": "ğ"},
    "aze": {"p": "b", "ç": "c", "t": "d", "k": "y"},
    "tuk": {"p": "b", "ç": "c", "t": "d", "k": "g"},
    "crh": {"p": "b", "ç": "c", "t": "d", "k": "g"},
    "gag": {"p": "b", "ç": "c", "t": "d", "k": "g"},
    "kaz": {"п": "б", "к": "г", "қ": "ғ"},
    "kir": {"п": "б", "к": "г"},
    "tat": {"п": "б", "к": "г"},
    "bak": {"п": "б", "к": "г"},
    "uzb": {"p": "b", "k": "g"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Segmentation result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Morpheme:
    """A single morpheme in a segmented word."""
    surface: str
    label: str  # e.g., "STEM", "PLUR", "ACC", "DAT", "POSS.1SG", etc.


@dataclass
class SegmentationResult:
    """Result of morpheme segmentation for a single word."""
    word: str
    morphemes: list[Morpheme]
    source: str  # "hfst", "neural", "greedy"

    @property
    def segments(self) -> list[str]:
        """Surface forms only."""
        return [m.surface for m in self.morphemes]

    @property
    def labeled(self) -> list[tuple[str, str]]:
        """(surface, label) pairs."""
        return [(m.surface, m.label) for m in self.morphemes]


# Pretty labels for Apertium tags
_TAG_LABELS: dict[str, str] = {
    "pl": "PLUR", "px1sg": "POSS.1SG", "px2sg": "POSS.2SG",
    "px2sg_frm": "POSS.2SG.FRM", "px3sg": "POSS.3SG", "px3sp": "POSS.3SG",
    "px1pl": "POSS.1PL", "px2pl": "POSS.2PL", "px3pl": "POSS.3PL",
    "acc": "ACC", "dat": "DAT", "loc": "LOC", "abl": "ABL",
    "gen": "GEN", "ins": "INS", "equ": "EQU",
    "subst": "SUBST", "attr": "ATTR", "ly": "ADJ", "siz": "PRIV",
    "lyk": "NMLZ", "shi": "AGT",
    "neg": "NEG", "pass": "PASS", "caus": "CAUS", "coop": "RECIP",
    "neg_aor": "NEG.AOR", "neg_pres": "NEG.PRS", "neg_prog": "NEG.PROG",
    "prog": "PROG",
    "past": "PST", "pres": "PRS", "fut": "FUT", "aor": "AOR",
    "evid": "EVID", "imp": "IMP", "opt": "OPT", "cond": "COND",
    "neces": "NEC",
    "prc_pres": "PTCP.PRS",
    "p1sg": "1SG", "p2sg": "2SG", "p3sg": "3SG",
    "p1pl": "1PL", "p2pl": "2PL", "p3pl": "3PL",
    "gna_perf": "CVB.PRF", "gna_impf": "CVB.IPFV",
    "prc_perf": "PTCP.PST", "prc_impf": "PTCP.IPFV", "prc_fut": "PTCP.FUT",
    "inf": "INF", "ger": "GER", "ger_past": "GER.PST",
    "cop": "COP", "qst": "Q",
}

# Tuvan analytic person pronouns (space-separated after verb form)
_TUVAN_PERSON: dict[str, str] = {
    "мен": "1SG", "сен": "2SG", "ол": "3SG",
    "бис": "1PL", "силер": "2PL", "олар": "3PL",
}


def _normalize_labels(
    morphemes: list[Morpheme],
    upos: str,
    feats: str,
) -> list[Morpheme]:
    """Post-process morpheme labels for accuracy.

    Fixes:
    - Noun possessive labels → verb person labels (POSS.1SG → 1SG)
    - PLUR → 3PL after tense markers in verbs
    - Fused tense+person: PST → PST.3SG when 3sg is implied
    - EVID → PTCP.PST when VerbForm=Part in features
    - ``?`` labels after apostrophe → proper case labels from features
    """
    if len(morphemes) <= 1:
        return morphemes

    feat_set = set(feats.split("|")) if feats and feats != "_" else set()
    is_verb = upos in ("VERB", "AUX")

    result = list(morphemes)

    if is_verb:
        _N2V = {
            "POSS.1SG": "1SG", "POSS.2SG": "2SG", "POSS.3SG": "3SG",
            "POSS.1PL": "1PL", "POSS.2PL": "2PL", "POSS.3PL": "3PL",
            "POSS.2SG.FRM": "2SG.FRM",
        }
        has_tense = False
        has_neg = False
        has_compound_neg_tense = False  # NEG.AOR, NEG.PRS, etc.
        for i, m in enumerate(result):
            if m.label == "STEM":
                continue
            if m.label in ("NEG", "NEG.AOR", "NEG.PRS", "NEG.PROG"):
                has_neg = True
            if m.label in ("NEG.AOR", "NEG.PRS", "NEG.PROG"):
                has_compound_neg_tense = True
            if m.label in (
                "PST", "PRS", "AOR", "FUT", "EVID", "PROG",
                "NEG.AOR", "NEG.PRS", "NEG.PROG",
                "PTCP.PST", "PTCP.PRS", "PTCP.FUT",
                "CVB.PRF", "CVB.IPFV", "GER", "GER.PST",
            ):
                has_tense = True
            if m.label in _N2V:
                result[i] = Morpheme(surface=m.surface, label=_N2V[m.label])
            elif m.label == "PLUR" and has_tense:
                result[i] = Morpheme(surface=m.surface, label="3PL")
            elif m.label == "ACC":
                if has_compound_neg_tense:
                    # After NEG.AOR/NEG.PRS, ACC-shaped suffix is person 3SG
                    result[i] = Morpheme(surface=m.surface, label="3SG")
                elif has_neg and not has_tense:
                    # After NEG alone (no tense yet), ACC is PST
                    result[i] = Morpheme(surface=m.surface, label="PST")
                    has_tense = True
                elif has_tense:
                    result[i] = Morpheme(surface=m.surface, label="3SG")
            elif m.label == "INS" and (has_tense or has_neg):
                # INS after tense/neg marker in verb context is likely 1SG
                result[i] = Morpheme(surface=m.surface, label="1SG")

        # Fused tense+3SG: when features indicate 3sg and no separate person suffix
        has_person_suffix = any(
            m.label in ("1SG", "2SG", "3SG", "1PL", "2PL", "3PL")
            for m in result if m.label != "STEM"
        )
        if not has_person_suffix:
            last = result[-1]
            if "Person=3" in feat_set and "Number=Plur" not in feat_set:
                if last.label in ("PST", "PRS", "AOR"):
                    result[-1] = Morpheme(surface=last.surface, label=f"{last.label}.3SG")
            elif "Number=Sing" in feat_set and "Person=3" not in feat_set:
                if last.label in ("PST", "PRS", "AOR"):
                    result[-1] = Morpheme(surface=last.surface, label=f"{last.label}.3SG")

        # EVID → PTCP.PST heuristics
        if "VerbForm=Part" in feat_set:
            for i, m in enumerate(result):
                if m.label in ("EVID", "EVID.3SG"):
                    result[i] = Morpheme(surface=m.surface, label="PTCP.PST")
        else:
            for i, m in enumerate(result):
                if m.label in ("EVID", "EVID.3SG"):
                    if i + 1 < len(result):
                        # EVID followed by person/number suffix → PTCP.PST
                        next_label = result[i + 1].label
                        if next_label in ("PLUR", "3PL", "1SG", "2SG", "1PL", "2PL"):
                            result[i] = Morpheme(surface=m.surface, label="PTCP.PST")
                            if next_label == "3PL":
                                result[i + 1] = Morpheme(
                                    surface=result[i + 1].surface, label="PLUR"
                                )
                    elif m.label in ("EVID", "EVID.3SG"):
                        # Standalone EVID at end with no person suffix
                        # → likely PTCP.PST (participle used attributively/nominally)
                        result[i] = Morpheme(surface=m.surface, label="PTCP.PST")

    # Apostrophe suffix labels: identify from UD features
    for i, m in enumerate(result):
        if m.label == "?" and m.surface and m.surface[0] in ("'", "\u2019"):
            for feat, label in (
                ("Case=Dat", "DAT"), ("Case=Loc", "LOC"), ("Case=Abl", "ABL"),
                ("Case=Gen", "GEN"), ("Case=Acc", "ACC"), ("Case=Ins", "INS"),
            ):
                if feat in feat_set:
                    result[i] = Morpheme(surface=m.surface, label=label)
                    break

    # Fix greedy label misassignments using UD features
    # Case label at the end of a noun should match the UD Case feature
    if not is_verb and len(result) > 1:
        last = result[-1]
        _CASE_LABELS = {"ACC", "DAT", "LOC", "ABL", "GEN", "INS", "EQU",
                        "POSS.1SG", "POSS.2SG", "POSS.3SG",
                        "POSS.1PL", "POSS.2PL", "POSS.3PL", "COND"}
        if last.label in _CASE_LABELS:
            # Check if a different case is indicated by features
            for feat, label in (
                ("Case=Gen", "GEN"), ("Case=Dat", "DAT"), ("Case=Loc", "LOC"),
                ("Case=Abl", "ABL"), ("Case=Acc", "ACC"), ("Case=Ins", "INS"),
            ):
                if feat in feat_set and last.label != label:
                    # The greedy matched the wrong label — fix it
                    result[-1] = Morpheme(surface=last.surface, label=label)
                    break

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Core segmentation algorithm
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_allomorph(
    tag: str,
    preceding_text: str,
    suffix_table: dict,
    phon: _Phonology,
) -> Optional[str]:
    """Look up the correct allomorph for a suffix tag given phonological context."""
    entries = suffix_table.get(tag)
    if not entries:
        return None

    last_char = preceding_text[-1] if preceding_text else ""
    cc = _char_class(last_char, phon)

    for condition_str, forms in entries:
        if cc in condition_str:
            if len(forms) == 2:
                # 2-way harmony
                h = _get_harmony(preceding_text, phon)
                return forms[0] if h == "B" else forms[1]
            elif len(forms) == 4:
                # 4-way harmony
                h4 = _get_harmony4(preceding_text, phon)
                idx = {"BU": 0, "FU": 1, "BR": 2, "FR": 3}[h4]
                return forms[idx]
    return None


def _best_matching_allomorph(
    key: str,
    remaining: str,
    preceding: str,
    suffix_table: dict,
    phon: _Phonology,
) -> Optional[str]:
    """Find the longest matching allomorph for a suffix key against remaining text.

    Unlike ``_resolve_allomorph`` which returns the first phonologically-correct
    allomorph, this tries ALL condition entries and returns the longest form that
    actually matches the start of *remaining*.  This handles cases where a suffix
    has both short (м) and long (мын) allomorphs for the same phonological context.
    """
    entries = suffix_table.get(key)
    if not entries:
        return None

    last_char = preceding[-1] if preceding else ""
    cc = _char_class(last_char, phon)
    remaining_lower = _turkic_lower(remaining)

    candidates: list[str] = []
    for condition_str, forms in entries:
        if cc not in condition_str:
            continue
        if len(forms) == 2:
            h = _get_harmony(preceding, phon)
            form = forms[0] if h == "B" else forms[1]
        elif len(forms) == 4:
            h4 = _get_harmony4(preceding, phon)
            idx = {"BU": 0, "FU": 1, "BR": 2, "FR": 3}[h4]
            form = forms[idx]
        else:
            continue
        if form and remaining_lower.startswith(_turkic_lower(form)):
            candidates.append(remaining[:len(form)])

    if candidates:
        return max(candidates, key=len)
    return None


def _hfst_pos_to_upos(pos: str) -> str:
    """Map Apertium POS tag to UD UPOS."""
    return {"v": "VERB", "vaux": "AUX", "n": "NOUN", "np": "PROPN",
            "adj": "ADJ", "adv": "ADV", "prn": "PRON"}.get(pos, "")


def _strip_infinitive(lemma: str, lang: str) -> str:
    """Strip infinitive suffix from a verb lemma to get the bare stem.

    Turkic infinitive endings by branch:
    - Oghuz (tur, aze, tuk, gag): -mak/-mek, -maq/-mək, -maa/-mää
    - Karluk (uzb): -moq
    - Kipchak (kaz, kir, tat, bak, kum, crh, nog, krc): -у/-ю/-ү, -уу/-үү, -ыу/-еү
    - Siberian (sah, kjh, tyv, alt): -ар/-ер/-ыр/-ир (sometimes)
    - Chuvash: -ма/-ме
    """
    ll = _turkic_lower(lemma)
    # Try longest endings first
    _INF_SUFFIXES = [
        "maq", "mək", "mak", "mek", "moq",  # Oghuz, Karluk (Latin)
        "maa", "mää",  # Gagauz
        "макъ", "мекъ",  # Kumyk
        "уу", "үү", "ыу", "еү",  # Kyrgyz, Bashkir long infinitives
    ]
    _INF_SUFFIXES_SHORT = [
        "у", "ү", "ю",  # Kazakh, Tatar, etc.
    ]
    for suf in _INF_SUFFIXES:
        if ll.endswith(suf) and len(ll) > len(suf):
            return lemma[:len(lemma) - len(suf)]
    for suf in _INF_SUFFIXES_SHORT:
        if ll.endswith(suf) and len(ll) > len(suf) + 1:
            return lemma[:len(lemma) - len(suf)]
    return lemma


def _find_stem_boundary(
    surface: str, lemma: str, lang: str, upos: str = "",
) -> int:
    """Find where the stem ends in the surface form.

    Returns the character index where suffixes begin.  Handles:
    0. Apostrophe boundary (Turkish proper nouns)
    1. Verb lemma normalization (strip infinitive)
    2. Exact prefix match (most common)
    3. Consonant alternation at stem boundary (kitap→kitab-)
    4. Fallback: longest common prefix
    """
    # 0. Apostrophe boundary (Turkish proper nouns: İstanbul'da)
    apos_pos = surface.find("'")
    if apos_pos == -1:
        apos_pos = surface.find("\u2019")  # right single quote
    if apos_pos > 0:
        return apos_pos

    # 1. For verbs, strip infinitive ending to get bare stem
    if upos in ("VERB", "AUX"):
        lemma = _strip_infinitive(lemma, lang)

    surface_lower = _turkic_lower(surface)
    lemma_lower = _turkic_lower(lemma)

    # 1b. Handle negative verb lemmas: neural model sometimes returns the
    # negative form as the lemma (e.g., жазбай, білме, bilmir).
    # Try stripping known negative morphemes if the positive root matches the surface.
    if upos in ("VERB", "AUX"):
        _NEG_ENDINGS = [
            # Sorted longest first to avoid partial matches
            "маст", "мест", "meýar", "meýär",  # Chuvash, Turkmen neg.pres
            "бай", "бей", "пай", "пей",  # Kipchak neg.aor
            "бас", "бес", "пас", "пес",  # Altai neg.aor
            "май", "мей", "мәй", "мый",  # Kipchak neg variants
            "may", "mey",  # Latin neg
            "маз", "мез", "maz", "məz",  # Oghuz neg.aor
            "мір", "мир", "mir", "mır",  # Azerbaijani neg.prog
            "meer",  # Gagauz neg.pres
            "ма", "ме", "мо", "мө",  # Cyrillic neg
            "ma", "me", "mo", "mö", "mə",  # Latin neg
        ]
        for neg in _NEG_ENDINGS:
            if (lemma_lower.endswith(neg) and len(lemma_lower) > len(neg)
                    and surface_lower.startswith(lemma_lower[:len(lemma_lower) - len(neg)])):
                lemma = lemma[:len(lemma) - len(neg)]
                lemma_lower = _turkic_lower(lemma)
                break

    # 2. Exact prefix match
    if surface_lower.startswith(lemma_lower):
        stem_len = len(lemma)
        # 2b. Kipchak vowel stem extension: when infinitive -у/-ү is stripped,
        # the underlying root vowel (ы/і) may appear in the surface form.
        # E.g., оқу→оқ but surface оқы-ған has root оқы-.
        if (upos in ("VERB", "AUX")
                and stem_len < len(surface) - 1
                and lemma_lower
                and lemma_lower[-1] not in (_CY_VOWELS | _LA_VOWELS)):
            next_ch = surface_lower[stem_len]
            # Only extend for epenthetic linking vowels (ы/і), not full vowels (а/е)
            if next_ch in ("ы", "і", "ı"):
                after_ch = surface_lower[stem_len + 1] if stem_len + 1 < len(surface) else ""
                if after_ch and after_ch not in (_CY_VOWELS | _LA_VOWELS):
                    stem_len += 1
        return stem_len

    # 3. Try with consonant alternation
    alternations = _STEM_ALTERNATIONS.get(lang, {})
    if lemma_lower and lemma_lower[-1] in alternations:
        voiced_stem = lemma[:-1] + alternations[lemma_lower[-1]]
        if surface_lower.startswith(_turkic_lower(voiced_stem)):
            return len(voiced_stem)

    # 4. Longest common prefix
    lcp = 0
    for i, (a, b) in enumerate(zip(surface_lower, lemma_lower)):
        if a == b:
            lcp = i + 1
        else:
            break

    # Ensure at least 1 character as stem
    return max(lcp, 1)


def _segment_by_tags(
    surface: str,
    stem_end: int,
    tags: list[str],
    suffix_table: dict,
    phon: _Phonology,
) -> list[Morpheme]:
    """Segment suffix chain using an ordered list of Apertium tags.

    Each tag is matched to its surface allomorph greedily.
    """
    stem = surface[:stem_end]
    remaining = surface[stem_end:]
    morphemes = [Morpheme(surface=stem, label="STEM")]
    preceding = stem

    for tag in tags:
        if not remaining:
            break

        key = _APT_TAG_TO_KEY.get(tag)
        if not key or tag in _APT_SKIP_TAGS:
            continue

        expected = _resolve_allomorph(key, preceding, suffix_table, phon)
        if expected and _turkic_lower(remaining).startswith(_turkic_lower(expected)):
            actual = remaining[:len(expected)]
            morphemes.append(Morpheme(
                surface=actual,
                label=_TAG_LABELS.get(key, key.upper()),
            ))
            preceding = preceding + actual
            remaining = remaining[len(expected):]
        else:
            # Try direct greedy match: look for any suffix from the table
            # that matches the start of remaining
            matched = _greedy_match_one(remaining, preceding, key, suffix_table, phon)
            if matched:
                morphemes.append(Morpheme(
                    surface=matched,
                    label=_TAG_LABELS.get(key, key.upper()),
                ))
                preceding = preceding + matched
                remaining = remaining[len(matched):]

    # If there's leftover suffix material, add as unknown morphemes
    if remaining:
        morphemes.extend(_greedy_segment_remainder(remaining, preceding, suffix_table, phon))

    return morphemes


def _greedy_match_one(
    remaining: str,
    preceding: str,
    target_key: str,
    suffix_table: dict,
    phon: _Phonology,
) -> Optional[str]:
    """Try to match a specific suffix slot against the remaining string.

    Tests all allomorphs for the given key and returns the longest match.
    """
    entries = suffix_table.get(target_key)
    if not entries:
        return None

    candidates = []
    for _, forms in entries:
        for f in forms:
            if f and _turkic_lower(remaining).startswith(_turkic_lower(f)):
                candidates.append(remaining[:len(f)])

    if candidates:
        return max(candidates, key=len)
    return None


def _greedy_segment_remainder(
    remaining: str,
    preceding: str,
    suffix_table: dict,
    phon: _Phonology,
) -> list[Morpheme]:
    """Greedily segment leftover suffix material using all known allomorphs."""
    morphemes: list[Morpheme] = []

    # Minimum suffix length to prevent single-char over-segmentation
    # (e.g., -мын split into м+ы+н). Allow 1-char only if it's the final morpheme
    # or if the key is a known tense marker (pres: а/е/a).
    MIN_SUFFIX_LEN = 2

    # Known 1-char person suffixes that are valid as word-final morphemes
    _VALID_1CHAR = set("мнкрпmntk")

    # Keys that are allowed to have 1-char matches even with leftover
    _ALLOW_1CHAR_KEYS = {"pres", "aor", "dat", "loc"}

    # Handle apostrophe prefix: strip it and re-add to matched surface
    apos_prefix = ""
    if remaining and remaining[0] in ("'", "\u2019"):
        apos_prefix = remaining[0]
        remaining = remaining[1:]

    while remaining:
        best_match = ""
        best_key = ""

        for key, entries in suffix_table.items():
            for _, forms in entries:
                for f in forms:
                    if not f:
                        continue
                    f_len = len(f)
                    leftover_len = len(remaining) - f_len
                    # Skip if match is too short (unless it consumes everything
                    # or key is a known short-form suffix like pres)
                    if f_len < MIN_SUFFIX_LEN and leftover_len > 0:
                        if key not in _ALLOW_1CHAR_KEYS:
                            continue
                    # Skip if match would leave a 1-char remainder
                    # UNLESS that char is a known word-final person suffix
                    if leftover_len == 1:
                        leftover_ch = _turkic_lower(remaining[-1])
                        if leftover_ch not in _VALID_1CHAR:
                            continue
                    if (_turkic_lower(remaining).startswith(_turkic_lower(f))
                            and f_len > len(best_match)):
                        best_match = remaining[:f_len]
                        best_key = key

        if best_match:
            surface_form = apos_prefix + best_match
            apos_prefix = ""  # only prepend to first match
            morphemes.append(Morpheme(
                surface=surface_form,
                label=_TAG_LABELS.get(best_key, best_key.upper()),
            ))
            preceding = preceding + best_match
            remaining = remaining[len(best_match):]
        else:
            # Can't match — treat rest as a single unknown morpheme
            surface_form = apos_prefix + remaining
            apos_prefix = ""
            morphemes.append(Morpheme(surface=surface_form, label="?"))
            break

    return morphemes


def _segment_by_ud_features(
    surface: str,
    stem_end: int,
    upos: str,
    feats: str,
    suffix_table: dict,
    phon: _Phonology,
) -> list[Morpheme]:
    """Segment using UD features (neural model output).

    Converts UD features to an ordered suffix sequence, then uses
    the same allomorph matching logic.
    """
    stem = surface[:stem_end]
    remaining = surface[stem_end:]
    morphemes = [Morpheme(surface=stem, label="STEM")]
    preceding = stem

    if not feats or feats == "_":
        if remaining:
            morphemes.append(Morpheme(surface=remaining, label="?"))
        return morphemes

    feat_set = set(feats.split("|"))

    # Handle apostrophe prefix (Turkish proper nouns: İstanbul'da)
    apos_prefix = ""
    if remaining and remaining[0] in ("'", "\u2019"):
        apos_prefix = remaining[0]
        remaining = remaining[1:]

    # Choose suffix order based on POS
    if upos in ("VERB", "AUX"):
        order = _UD_VERB_SUFFIX_ORDER
    else:
        order = _UD_NOUN_SUFFIX_ORDER

    consumed_keys: set[str] = set()

    for required_feats, key in order:
        if not remaining:
            break
        if key in consumed_keys:
            continue
        # All features in the required set must be present
        if required_feats <= feat_set:
            # Use _best_matching_allomorph for longest match
            matched = _best_matching_allomorph(
                key, remaining, preceding, suffix_table, phon,
            )
            if matched:
                surface_form = apos_prefix + matched
                apos_prefix = ""  # only prepend to first match
                morphemes.append(Morpheme(
                    surface=surface_form,
                    label=_TAG_LABELS.get(key, key.upper()),
                ))
                preceding = preceding + matched
                remaining = remaining[len(matched):]
                consumed_keys.add(key)
                # Mark compound suffix features as consumed to prevent
                # double-matching (e.g., neg_aor consumes both neg and aor)
                if key.startswith("neg_"):
                    consumed_keys.update(("neg", key.split("_", 1)[1]))

    # Greedy fallback for unmatched remainder
    if remaining:
        remaining_with_apos = apos_prefix + remaining
        morphemes.extend(_greedy_segment_remainder(remaining_with_apos, preceding, suffix_table, phon))

    return morphemes


# ═══════════════════════════════════════════════════════════════════════════
# MorphemeTokenizer — public API
# ═══════════════════════════════════════════════════════════════════════════


class MorphemeTokenizer:
    """Hybrid neural + FST morpheme tokenizer for Turkic languages.

    Uses the neural morph model (Glot500 backbone) as the primary analyzer
    for broad coverage (21 languages), and Apertium HFST as a secondary
    backend for fine-grained derivational morphology.

    Usage::

        tok = MorphemeTokenizer(lang="kaz")
        tok.load()
        result = tok.segment("үйлеріңіздегілерден")
        print(result.segments)
        # ['үй', 'лер', 'іңіз', 'де', 'гі', 'лер', 'ден']
        print(result.labeled)
        # [('үй', 'STEM'), ('лер', 'PLUR'), ('іңіз', 'POSS.2SG.FRM'), ...]
    """

    SUPPORTED_LANGS = list(_LANG_SUFFIX_MAP.keys())

    def __init__(
        self,
        lang: str,
        use_gpu: bool = False,
        script: Optional[str] = None,
    ) -> None:
        if lang not in _LANG_SUFFIX_MAP:
            raise ValueError(
                f"Unsupported language '{lang}'. "
                f"Supported: {', '.join(sorted(_LANG_SUFFIX_MAP.keys()))}"
            )
        self.lang = lang
        self.use_gpu = use_gpu
        self._suffix_table, default_script = _LANG_SUFFIX_MAP[lang]
        self._script = script or default_script
        self._phon = _Phonology(self._script)

        self._neural_available = False
        self._hfst_analyzer = None  # ApertiumMorphProcessor instance
        self._hfst_available = False
        self._loaded = False

    def load(self) -> None:
        """Load backends: neural morph model (primary) + HFST (secondary)."""
        # 1. Try loading neural morph model
        try:
            from turkicnlp.processors.multilingual_morph_backend import (
                _MorphAnalyzerManager,
            )
            _MorphAnalyzerManager.ensure_downloaded()
            _MorphAnalyzerManager.get_model(self.use_gpu)
            self._neural_available = True
            logger.info("MorphemeTokenizer: neural morph model loaded")
        except Exception as e:
            logger.warning("MorphemeTokenizer: neural morph model unavailable: %s", e)

        # 2. Try loading HFST analyzer
        try:
            from turkicnlp.processors.morphology import ApertiumMorphProcessor
            from turkicnlp.resources.registry import ModelRegistry

            proc = ApertiumMorphProcessor(lang=self.lang)
            base = ModelRegistry.default_dir()

            # Try to find the HFST model path
            from turkicnlp.scripts import get_script_config
            try:
                sc = get_script_config(self.lang)
                apt_script = sc.apertium_script.value if sc.apertium_script else self._script
            except (ValueError, AttributeError):
                apt_script = self._script

            model_dir = base / self.lang / apt_script / "morph" / "apertium"
            if model_dir.exists() and list(model_dir.glob("*.automorf.hfst")):
                proc.load(model_dir)
                self._hfst_analyzer = proc
                self._hfst_available = True
                logger.info("MorphemeTokenizer: HFST analyzer loaded for %s", self.lang)
        except Exception as e:
            logger.debug("MorphemeTokenizer: HFST unavailable for %s: %s", self.lang, e)

        if not self._neural_available and not self._hfst_available:
            raise RuntimeError(
                f"MorphemeTokenizer: no backend available for '{self.lang}'. "
                "Install turkicnlp[transformers] for neural model, or "
                "turkicnlp[hfst] and run turkicnlp.download() for HFST."
            )

        self._loaded = True

    def segment(self, word: str, labels: bool = False) -> SegmentationResult:
        """Segment a single word into morphemes.

        Args:
            word: A single inflected word form.
            labels: Ignored (labels are always in the result).

        Returns:
            SegmentationResult with morphemes, segments, and labeled output.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .segment()")

        # Handle Tuvan/Sakha analytic verb forms (space-separated person pronouns)
        if " " in word:
            parts = word.split()
            first_result = self.segment(parts[0])
            for part in parts[1:]:
                label = _TUVAN_PERSON.get(part, "?")
                first_result.morphemes.append(Morpheme(surface=part, label=label))
            first_result.word = word
            return first_result

        # Punctuation / single character → return as-is
        if len(word) <= 1 or not any(c.isalpha() for c in word):
            return SegmentationResult(
                word=word,
                morphemes=[Morpheme(surface=word, label="STEM")],
                source="skip",
            )

        # --- Strategy 1: HFST analysis (fine-grained tags) ---
        hfst_result = self._try_hfst(word)

        # --- Strategy 2: Neural analysis ---
        neural_result = self._try_neural(word)

        # Track UPOS and features for label normalization
        _upos = ""
        _feats = "_"
        if neural_result:
            _upos = neural_result.get("upos", "")
            _feats = neural_result.get("feats", "_")
        elif hfst_result:
            _upos = _hfst_pos_to_upos(hfst_result.get("pos", ""))

        # --- Hybrid merge ---
        # Prefer HFST tags for segmentation (finer granularity),
        # but use neural lemma (better OOV handling)
        if hfst_result and neural_result:
            # Use neural lemma, HFST tags
            lemma = neural_result["lemma"]
            upos = neural_result.get("upos", "")
            tags = hfst_result["tags"]
            stem_end = _find_stem_boundary(word, lemma, self.lang, upos)
            morphemes = _segment_by_tags(
                word, stem_end, tags, self._suffix_table, self._phon,
            )
            result = SegmentationResult(word=word, morphemes=morphemes, source="hybrid")
            result.morphemes = _normalize_labels(result.morphemes, _upos, _feats)
            return result

        if hfst_result:
            lemma = hfst_result["lemma"]
            tags = hfst_result["tags"]
            # Infer upos from HFST POS tag
            hfst_upos = _hfst_pos_to_upos(hfst_result.get("pos", ""))
            stem_end = _find_stem_boundary(word, lemma, self.lang, hfst_upos)
            morphemes = _segment_by_tags(
                word, stem_end, tags, self._suffix_table, self._phon,
            )
            result = SegmentationResult(word=word, morphemes=morphemes, source="hfst")
            result.morphemes = _normalize_labels(result.morphemes, _upos, _feats)
            return result

        if neural_result:
            lemma = neural_result["lemma"]
            upos = neural_result["upos"]
            feats = neural_result["feats"]
            stem_end = _find_stem_boundary(word, lemma, self.lang, upos)
            morphemes = _segment_by_ud_features(
                word, stem_end, upos, feats, self._suffix_table, self._phon,
            )
            # If neural says NOUN but lemma looks like a verb or segmentation
            # has unknowns, retry as VERB (neg-lemma stripping often helps)
            if upos in ("NOUN", "ADJ") and len(morphemes) > 1:
                n_unknown = sum(1 for m in morphemes if m.label == "?")
                ll = _turkic_lower(lemma)
                _VERB_LEMMA_ENDS = (
                    "mak", "mek", "maq", "mək", "moq",
                    "maa", "mää",
                )
                is_verb_lemma = any(ll.endswith(s) for s in _VERB_LEMMA_ENDS)
                feat_set = set(feats.split("|")) if feats and feats != "_" else set()
                has_person = any(f.startswith("Person=") for f in feat_set)
                if n_unknown > 0 or is_verb_lemma:
                    stem_end_v = _find_stem_boundary(word, lemma, self.lang, "VERB")
                    morphemes_v = _segment_by_ud_features(
                        word, stem_end_v, "VERB", feats,
                        self._suffix_table, self._phon,
                    )
                    n_unk_v = sum(1 for m in morphemes_v if m.label == "?")
                    if n_unk_v < n_unknown:
                        morphemes = morphemes_v
                        _upos = "VERB"
                    elif n_unk_v == n_unknown and is_verb_lemma:
                        morphemes = morphemes_v
                        _upos = "VERB"
            result = SegmentationResult(word=word, morphemes=morphemes, source="neural")
            result.morphemes = _normalize_labels(result.morphemes, _upos, _feats)
            return result

        # Fallback: greedy segmentation with no analysis
        morphemes = [Morpheme(surface=word, label="STEM")]
        return SegmentationResult(word=word, morphemes=morphemes, source="none")

    def segment_text(self, text: str) -> list[SegmentationResult]:
        """Segment all words in a text string.

        Tokenizes with the regex tokenizer, then segments each word.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .segment_text()")

        from turkicnlp.processors.tokenizer import RegexTokenizer
        from turkicnlp.models.document import Document as Doc

        doc = Doc(text=text, lang=self.lang)
        tokenizer = RegexTokenizer(lang=self.lang)
        tokenizer.load("")
        doc = tokenizer.process(doc)

        results = []
        for sentence in doc.sentences:
            for word in sentence.words:
                results.append(self.segment(word.text))

        return results

    def process(self, doc: Document) -> Document:
        """Add morpheme segmentation to each word in a Document.

        Stores result as ``word._morphemes`` (list of Morpheme).
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .process()")

        for sentence in doc.sentences:
            for word in sentence.words:
                result = self.segment(word.text)
                word._morphemes = result.morphemes  # type: ignore[attr-defined]
                word._morpheme_segments = result.segments  # type: ignore[attr-defined]

        doc._processor_log.append(f"morpheme_tokenizer:{self.lang}")
        return doc

    # --- Private helpers ---

    def _try_hfst(self, word: str) -> Optional[dict]:
        """Try HFST analysis, return best reading as {lemma, tags} or None."""
        if not self._hfst_available or self._hfst_analyzer is None:
            return None

        try:
            readings = self._hfst_analyzer._analyze(word)
            if not readings:
                return None

            best = self._hfst_analyzer._disambiguate(readings)
            lemma = best.get("lemma", "")
            all_tags = [best.get("pos", "")] + best.get("feats", [])
            # Filter to only morpheme-bearing tags
            morph_tags = [t for t in all_tags if t not in _APT_SKIP_TAGS and t]
            return {"lemma": lemma, "tags": morph_tags, "pos": best.get("pos", "")}
        except Exception as e:
            logger.debug("HFST analysis failed for '%s': %s", word, e)
            return None

    def _try_neural(self, word: str) -> Optional[dict]:
        """Try neural morph analysis, return {lemma, upos, feats} or None."""
        if not self._neural_available:
            return None

        try:
            from turkicnlp.processors.multilingual_morph_backend import (
                _MorphAnalyzerManager,
            )
            from turkicnlp.processors.multilingual_morph_model import (
                resolve_morph_lang,
                tokenize_words,
                encode_chars,
                apply_edit_script,
                UPOS_TAGS,
                UD_MORPH_FEATS,
                LANG_ID_MAP,
            )
            from turkicnlp.processors.multilingual_morph_backend import (
                FEAT_THRESHOLD,
            )
            import torch

            model, device, edit_vocab = _MorphAnalyzerManager.get_model(self.use_gpu)
            short_code, script = resolve_morph_lang(self.lang)
            lang_id = LANG_ID_MAP[short_code]

            words = [word]
            input_ids, attention_mask, word_starts, word_lengths = tokenize_words(
                model.tokenizer, [words], device,
            )
            lang_ids = torch.tensor([lang_id], dtype=torch.long, device=device)
            char_ids = encode_chars([words], max_word_len=1).to(device)

            with torch.no_grad():
                pos_logits, feat_logits, edit_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_starts=word_starts,
                    word_lengths=word_lengths,
                    lang_ids=lang_ids,
                    script=script,
                    char_ids=char_ids,
                )

            pos_id = pos_logits[0, 0].argmax(-1).item()
            upos = UPOS_TAGS[pos_id]

            feat_probs = torch.sigmoid(feat_logits[0, 0])
            active = (feat_probs > FEAT_THRESHOLD).nonzero(as_tuple=True)[0]
            if len(active) == 0:
                feats = "_"
            else:
                feats = "|".join(sorted(
                    UD_MORPH_FEATS[idx.item()] for idx in active
                ))

            edit_id = edit_logits[0, 0].argmax(-1).item()
            edit_str = edit_vocab.decode(edit_id)
            if edit_str == "<unk>":
                lemma = word.lower()
            else:
                lemma = apply_edit_script(word, edit_str)

            return {"lemma": lemma, "upos": upos, "feats": feats}
        except Exception as e:
            logger.debug("Neural analysis failed for '%s': %s", word, e)
            return None

    def __repr__(self) -> str:
        backends = []
        if self._neural_available:
            backends.append("neural")
        if self._hfst_available:
            backends.append("hfst")
        return (
            f"MorphemeTokenizer(lang={self.lang!r}, script={self._script!r}, "
            f"backends=[{', '.join(backends)}])"
        )
