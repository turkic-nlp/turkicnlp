#!/usr/bin/env python3
"""Add three high-priority lexicon entry types to all 21 Turkic language lexicon JSON files.

Types added:
  - pronoun_reflexive  → PRON | Reflex=Yes
  - adverb_degree      → ADV  | feats=_
  - auxiliary_evidential → AUX | Evident=Nfh
"""

import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "turkicnlp" / "resources" / "lexicons"
STAGE_DIR = Path(__file__).resolve().parents[2] / "resources" / "grammar_sources" / "lexicons"

# Each value: list of (type, upos, feats, forms)
NEW_ENTRIES: dict[str, list[tuple[str, str, str, list[str]]]] = {
    # ── Oghuz (Latin script) ───────────────────────────────────────────────
    "tur": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["kendi", "kendim", "kendin", "kendisi", "kendimiz", "kendiniz", "kendileri"]),
        ("adverb_degree", "ADV", "_",
         ["çok", "az", "daha", "en", "fazla", "biraz", "epey"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["imiş"]),
    ],
    "aze": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["öz", "özüm", "özün", "özü", "özümüz", "özünüz", "özləri"]),
        ("adverb_degree", "ADV", "_",
         ["çox", "az", "daha", "ən", "biraz"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["imiş"]),
    ],
    "tuk": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["öz", "özüm", "özüň", "özi", "özümiz", "özüňiz", "özleri"]),
        ("adverb_degree", "ADV", "_",
         ["köp", "gaty", "az", "has", "iň"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["eken", "imiş"]),
    ],
    "uzb": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["o'z", "o'zim", "o'zing", "o'zi", "o'zimiz", "o'zingiz", "o'zlari"]),
        ("adverb_degree", "ADV", "_",
         ["juda", "o'ta", "ko'p", "oz", "ko'proq", "eng"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["ekan"]),
    ],
    "crh": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["öz", "özüm", "özüñ", "özü", "özümüz", "özüñüz", "özleri"]),
        ("adverb_degree", "ADV", "_",
         ["çok", "az", "daha", "en", "fazla", "biraz"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["imiş"]),
    ],
    "gag": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["kendi", "kendim", "kendin", "kendisi", "kendimiz", "kendiniz", "kendileri"]),
        ("adverb_degree", "ADV", "_",
         ["çok", "az", "daha", "en", "fazla", "biraz"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["imiş"]),
    ],
    "kaa": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["öz", "özim", "özing", "özi", "özimiz", "özingiz", "özleri"]),
        ("adverb_degree", "ADV", "_",
         ["köp", "juda", "az", "kem", "köprek", "eŋ"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["eken", "ekan"]),
    ],
    # ── Kipchak (Cyrillic script) ─────────────────────────────────────────
    "kaz": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["өз", "өзім", "өзің", "өзі", "өзіміз", "өзіңіз", "өздері"]),
        ("adverb_degree", "ADV", "_",
         ["өте", "тым", "көп", "аз", "ең", "біраз"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["екен"]),
    ],
    "kir": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["өз", "өзүм", "өзүң", "өзү", "өзүбүз", "өзүңүз", "өздөрү"]),
        ("adverb_degree", "ADV", "_",
         ["өтө", "абдан", "көп", "аз", "эң", "бираз"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["экен", "имиш"]),
    ],
    "tat": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["үз", "үзем", "үзең", "үзе", "үземез", "үзегез", "үзләре"]),
        ("adverb_degree", "ADV", "_",
         ["бик", "бигрәк", "күп", "аз", "иң"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["имеш"]),
    ],
    "bak": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["үҙ", "үҙем", "үҙең", "үҙе", "үҙебеҙ", "үҙегеҙ", "үҙҙәре"]),
        ("adverb_degree", "ADV", "_",
         ["бик", "бигерәк", "күп", "аз", "иң"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["имеш"]),
    ],
    "kum": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["оьз", "оьзюм", "оьзюнг", "оьзю", "оьзюбюз", "оьзлери"]),
        ("adverb_degree", "ADV", "_",
         ["бек", "кёп", "аз", "эң", "артыкъ"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["имиш"]),
    ],
    "nog": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["оьз", "оьзим", "оьзинг", "оьзи", "оьзимиз", "оьзлери"]),
        ("adverb_degree", "ADV", "_",
         ["бек", "коьп", "аз", "иң", "артык"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["имис"]),
    ],
    "krc": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["кеси", "кесим", "кесинг", "кеси", "кесибиз", "кесилери"]),
        ("adverb_degree", "ADV", "_",
         ["бек", "кёп", "аз", "иң", "артыкъ"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["имиш"]),
    ],
    # ── Siberian Turkic (Cyrillic) ─────────────────────────────────────────
    "chv": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["хăй", "хăйĕм", "хăйĕ", "хăймăр", "хăйсем"]),
        ("adverb_degree", "ADV", "_",
         ["питĕ", "нумай", "сахал", "ытла"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["пулнă"]),
    ],
    "sah": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["бэйэ", "бэйэм", "бэйэҥ", "бэйэтэ", "бэйэбит", "бэйэлэрэ"]),
        ("adverb_degree", "ADV", "_",
         ["нааш", "элбэх", "аҕыйах"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["эбит"]),
    ],
    "alt": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["бой", "бойым", "бойыҥ", "бойы", "бойыбыс", "бойлоры"]),
        ("adverb_degree", "ADV", "_",
         ["бек", "кебер", "аз", "эң"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["эмиш"]),
    ],
    # Khakas uses ö (U+00F6) from Latin extensions, per its orthography
    "kjh": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["пос", "позым", "позың", "позы", "позыбыс", "позлары"]),
        ("adverb_degree", "ADV", "_",
         ["нарын", "кöп", "öте", "аз", "эҥ"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["иміс"]),
    ],
    "tyv": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["боду", "бодум", "бодуң", "боду", "бодувус", "бодулар"]),
        ("adverb_degree", "ADV", "_",
         ["нарын", "хөй", "эвээш", "эң"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["иймит"]),
    ],
    # ── Arabic script ──────────────────────────────────────────────────────
    "uig": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["ئۆز", "ئۆزۈم", "ئۆزۈڭ", "ئۆزى", "ئۆزىمىز", "ئۆزۈڭىز", "ئۆزلىرى"]),
        ("adverb_degree", "ADV", "_",
         ["ناھايىتى", "كۆپ", "بەك", "ئاز", "تېخىمۇ", "ئەڭ"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["ئىكەن"]),
    ],
    "azb": [
        ("pronoun_reflexive", "PRON", "Reflex=Yes",
         ["اوز", "اوزوم", "اوزون", "اوزو", "اوزوموز", "اوزونوز", "اوزلری"]),
        ("adverb_degree", "ADV", "_",
         ["چوخ", "آز", "داها", "ان", "بیراز"]),
        ("auxiliary_evidential", "AUX", "Evident=Nfh",
         ["ایمیش"]),
    ],
}

KNOWN_TYPES = {e[0] for entries in NEW_ENTRIES.values() for e in entries}


def build_entry(type_: str, upos: str, feats: str, forms: list[str]) -> dict:
    return {
        "type": type_,
        "upos": upos,
        "feats": feats,
        "lemma_strategy": "lower",
        "forms": forms,
    }


def update_file(path: Path, lang: str) -> bool:
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict] = data.get("entries", [])

    # Collect types already present
    existing_types = {e.get("type") for e in entries}

    added = []
    for type_, upos, feats, forms in NEW_ENTRIES[lang]:
        if type_ in existing_types:
            print(f"  already has '{type_}' — skipping")
        else:
            entries.append(build_entry(type_, upos, feats, forms))
            added.append(type_)

    if not added:
        print(f"  no changes for {lang}")
        return False

    data["entries"] = entries
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  added {added}")
    return True


def main() -> None:
    for lang in sorted(NEW_ENTRIES):
        print(f"\n── {lang} ──")
        pkg_path = PKG_DIR / f"{lang}.json"
        stage_path = STAGE_DIR / f"{lang}.json"

        update_file(pkg_path, lang)
        update_file(stage_path, lang)

    print("\nDone.")


if __name__ == "__main__":
    main()
