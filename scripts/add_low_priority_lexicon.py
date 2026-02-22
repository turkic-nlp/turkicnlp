"""
Add low-priority lexicon entry types to all 21 Turkic language lexicons.

Types added:
  interjection    → INTJ | feats=_        (yes / no / okay response words)
  numeral_cardinal → NUM | NumType=Card   (cardinal numerals 1–10)

Locations updated (both must stay in sync):
  PKG_DIR   = turkicnlp/resources/lexicons/
  STAGE_DIR = resources/grammar_sources/lexicons/

Notes:
  - "no" response words are deliberately omitted from interjection where they already
    appear in negation_particle to avoid dual-UPOS lookup noise.
  - Numeral forms are nominative/citation form only; inflected case forms are
    handled by the FST.
  - Empty forms list → entry is silently skipped (language gap).
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

PKG_DIR   = ROOT / "turkicnlp" / "resources" / "lexicons"
STAGE_DIR = ROOT.parent / "resources" / "grammar_sources" / "lexicons"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Interjection data: yes / okay response words.
# "no" forms omitted when already present in negation_particle for that language.
INTERJECTIONS: dict[str, list[str]] = {
    "tur": ["evet", "tamam", "hayır"],
    "aze": ["bəli", "hə", "tamam", "xeyr"],
    "tuk": ["hawa", "bolýar"],
    "uzb": ["ha", "xo'p", "mayli"],
    "crh": ["evet", "tamam"],
    "gag": ["evet", "tamam"],
    "kaa": ["äwä", "maqqul"],
    "kaz": ["иә", "ия", "жарайды", "мақұл"],
    "kir": ["ооба", "ийе", "болду"],
    "tat": ["әйе", "ярый"],
    "bak": ["эйе", "яраша"],
    "chv": ["ă", "лайăх"],
    "sah": ["сөп"],
    "alt": ["ии", "болды"],
    "kum": ["ха", "яхшы"],
    "nog": ["ийе", "ярай"],
    "krc": ["ха", "яхшы"],
    "kjh": ["хоосха", "ии"],
    "tyv": ["ийе"],
    "uig": ["ھە", "بولىدۇ"],
    "azb": ["بله", "تامام"],
}

# Cardinal numerals 1–10 in citation form, correct script per language.
# Characters used per Cyrillic language:
#   kaz/kir: ü=ү(U+04AF), ö=ө(U+04E9), standard Cyrillic
#   tat: ü=ү(U+04AF), ö=ø=ө(U+04E9), ə=ä=ә(U+04D9), ĵ=җ(U+0497)
#   bak: ü=ү(U+04AF), ö=ő=ö-bar=ö̈=ő Cyrillic ø→(U+04E9), h=ħ=(U+04BB), ź=ҙ(U+0499), ğ=ғ(U+0493)
#   chv: ĕ(U+0115), ă(U+0103), ç(U+00E7)  — all Latin mixed with Cyrillic
#   sah: ü=ү(U+04AF), ö=ő=ö→ø=(U+04E9), ɣ=ҕ(U+0495)
#   alt: ü=ү(U+04AF), ö=ö(Latin U+00F6, same convention as kjh)
#   kum: ü=уь digraph, ö=ё
#   nog: ü=уь digraph, ö=ё
#   krc: ü=ю, ö=ё
#   kjh: ü=ÿ(U+00FF Latin), ö=ö(U+00F6 Latin), i=і(U+0456 Cyrillic І)
#   tyv: ü=ү(U+04AF), ö=ő=ø=ö→ő Cyrillic ø=(U+04E9)
#   uig: Arabic script — standard Uyghur Unicode vowels
#   azb: Arabic/Perso-Arabic script
NUMERALS: dict[str, list[str]] = {
    # ── Latin script ──────────────────────────────────────────────────────────
    "tur": ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"],
    "aze": ["bir", "iki", "üç", "dörd", "beş", "altı", "yeddi", "səkkiz", "doqquz", "on"],
    "tuk": ["bir", "iki", "üç", "dört", "bäş", "alty", "ýedi", "sekiz", "dokuz", "on"],
    "uzb": ["bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz", "o'n"],
    "crh": ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"],
    "gag": ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"],
    "kaa": ["bir", "eki", "üsh", "tört", "bes", "altı", "jeti", "segiz", "toğız", "on"],
    # ── Cyrillic script ───────────────────────────────────────────────────────
    "kaz": ["бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он"],
    "kir": ["бир", "эки", "үч", "төрт", "беш", "алты", "жети", "сегиз", "тогуз", "он"],
    # tat: ö→ө(U+04E9), ü→ү(U+04AF), ĵ→җ(U+0497)
    "tat": ["бер", "ике", "өч", "дүрт", "биш", "алты", "җиде", "сигез", "тугыз", "ун"],
    # bak: ö→ø→ö (U+04E9), ü→ü (U+04AF), ħ→ħ (U+04BB), ź→ź (U+0499), ğ→ğ (U+0493)
    "bak": ["бер", "ике", "өс", "дүрт", "биш", "алты", "ете", "һигеҙ", "туғыҙ", "ун"],
    # chv: ĕ(U+0115), ă(U+0103), ç(U+00E7)
    "chv": ["пĕр", "иккĕ", "виç", "тăват", "пилĕк", "улт", "çиççĕ", "саккăр", "тăхăр", "вун"],
    # sah: ü→ü (U+04AF), ö→ő (U+04E9), ɣ→ɣ (U+0495)
    "sah": ["биир", "икки", "үс", "түөрт", "биэс", "алта", "сэттэ", "аҕыс", "тоҕус", "уон"],
    # alt: ü→ü (U+04AF) like мүмкин, ö→ö (Latin U+00F6) like kjh convention
    "alt": ["бир", "эки", "үч", "тöрт", "беш", "алты", "йети", "сегис", "тогус", "он"],
    # kum: ü→уь digraph, ö→ё
    "kum": ["бир", "эки", "уьч", "дёрт", "беш", "алты", "йетти", "сегиз", "тогъуз", "он"],
    # nog: ü→уь digraph, ö→ё
    "nog": ["бир", "эки", "уьш", "дёрт", "бес", "алты", "йети", "сегиз", "тогыз", "он"],
    # krc: ü→ю, ö→ё
    "krc": ["бир", "эки", "юч", "дёрт", "беш", "алты", "жети", "сегиз", "тогъуз", "он"],
    # kjh: ü→ÿ(U+00FF), ö→ö(U+00F6), і(U+0456)
    "kjh": ["пір", "ікі", "ÿс", "тöрт", "піс", "алты", "четі", "сегіс", "тоғыс", "он"],
    # tyv: ü→ü (U+04AF), ö→ő (U+04E9)
    "tyv": ["бир", "ийи", "үш", "дөрт", "беш", "алды", "чеди", "сес", "тос", "он"],
    # ── Arabic script ─────────────────────────────────────────────────────────
    # uig: standard Uyghur Arabic Unicode vowels
    "uig": ["بىر", "ئىككى", "ئۈچ", "تۆت", "بەش", "ئالتە", "يەتتە", "سەككىز", "توققۇز", "ئون"],
    # azb: Perso-Arabic; ö→ؤ(U+0624), /e/→ئ(U+0626)
    "azb": ["بیر", "ایکی", "اوچ", "دؤرد", "بئش", "آلتی", "یئددی", "سکیز", "دوققوز", "اون"],
}

NEW_ENTRIES: dict[str, list[dict]] = {}
for lang in INTERJECTIONS:
    entries = []
    if INTERJECTIONS.get(lang):
        entries.append({
            "type": "interjection",
            "upos": "INTJ",
            "feats": "_",
            "lemma_strategy": "lower",
            "forms": INTERJECTIONS[lang],
        })
    if NUMERALS.get(lang):
        entries.append({
            "type": "numeral_cardinal",
            "upos": "NUM",
            "feats": "NumType=Card",
            "lemma_strategy": "lower",
            "forms": NUMERALS[lang],
        })
    NEW_ENTRIES[lang] = entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_file(path: pathlib.Path, lang: str) -> list[str]:
    """Append new entry types to a lexicon JSON, skipping existing types."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    existing_types = {e["type"] for e in data["entries"]}
    added = []
    for entry in NEW_ENTRIES.get(lang, []):
        if not entry["forms"]:
            continue
        if entry["type"] in existing_types:
            continue
        data["entries"].append(entry)
        added.append(entry["type"])

    if added:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")  # trailing newline

    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_LANGS = sorted(INTERJECTIONS.keys())

for lang in ALL_LANGS:
    print(f"── {lang} ──")
    for directory in (PKG_DIR, STAGE_DIR):
        path = directory / f"{lang}.json"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        added = update_file(path, lang)
        label = "pkg  " if directory == PKG_DIR else "stage"
        if added:
            print(f"  [{label}] added {added}")
        else:
            print(f"  [{label}] already up to date")

print("\nDone.")
