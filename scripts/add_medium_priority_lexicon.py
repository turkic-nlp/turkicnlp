#!/usr/bin/env python3
"""Add medium-priority lexicon entry types to all 21 Turkic language lexicon JSON files.

Types added:
  - adverb_interrogative  → ADV  | PronType=Int    (where/when/how/why)
  - determiner_universal  → DET  | PronType=Tot    (all/every/each)
  - determiner_indefinite → DET  | PronType=Ind    (some/several)
  - determiner_negative   → DET  | PronType=Neg    (no/none as pre-nominal)
  - adverb_modal          → ADV  | feats=_         (maybe/certainly/only/still)
  - particle_additive     → PART | PartType=Add    (also/even — standalone forms)
  - temporal_adverb       → ADV  | feats=_         (today/yesterday/now/tomorrow)
  - pronoun_reciprocal    → PRON | PronType=Rcp    (each other)
"""

import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "turkicnlp" / "resources" / "lexicons"
STAGE_DIR = Path(__file__).resolve().parents[2] / "resources" / "grammar_sources" / "lexicons"

# Each value: list of (type, upos, feats, forms)
# Empty forms list → entry skipped (language-specific gaps)
NEW_ENTRIES: dict[str, list[tuple[str, str, str, list[str]]]] = {
    # ── Oghuz (Latin script) ─────────────────────────────────────────────
    "tur": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["nerede", "nereye", "nereden", "nasıl", "neden", "niçin", "kaç"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["tüm", "bütün", "hep", "hepsi", "her"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["bazı", "birkaç", "birçok", "kimi"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hiç", "hiçbir"]),
        ("adverb_modal", "ADV", "_",
         ["belki", "tabii", "elbette", "hâlâ", "henüz", "artık", "zaten",
          "sadece", "yalnızca", "özellikle"]),
        ("particle_additive", "PART", "PartType=Add",
         ["bile", "dahi"]),
        ("temporal_adverb", "ADV", "_",
         ["bugün", "dün", "şimdi", "yarın", "sabah", "akşam"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["birbirine", "birbirini", "birbirinden", "birbirleri"]),
    ],
    "aze": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["harada", "haraya", "haradan", "necə", "niyə", "neçə"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["bütün", "hamısı", "hər", "hamı"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["bəzi"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["heç"]),
        ("adverb_modal", "ADV", "_",
         ["bəlkə", "əlbəttə", "hələ", "artıq", "yalnız", "xüsusilə"]),
        ("particle_additive", "PART", "PartType=Add",
         ["belə", "hətta"]),
        ("temporal_adverb", "ADV", "_",
         ["dünən", "indi", "sabah", "axşam"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["bir-birinə", "bir-birini", "bir-birindən"]),
    ],
    "tuk": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["nirede", "nireye", "nireden", "nähili", "nämüçin", "näçe", "haçan"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["ähli", "bütin", "her"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["käbir", "birnäçe"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hiç"]),
        ("adverb_modal", "ADV", "_",
         ["belki", "elbetde", "entek", "eýýäm", "diňe", "aýratyn"]),
        ("particle_additive", "PART", "PartType=Add",
         ["hatda", "hem"]),
        ("temporal_adverb", "ADV", "_",
         ["düýn", "häzir", "ertir", "irden"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["biri-birine", "biri-birini"]),
    ],
    "uzb": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["qayerda", "qayerga", "qayerdan", "qanday", "nega", "qancha", "necha", "qachon"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["hamma", "barcha", "butun", "har"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["ba'zi", "biroz", "ayrim"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hech"]),
        ("adverb_modal", "ADV", "_",
         ["balki", "albatta", "hali", "faqat", "ayniqsa"]),
        ("particle_additive", "PART", "PartType=Add",
         ["ham", "hatto"]),
        ("temporal_adverb", "ADV", "_",
         ["bugun", "kecha", "hozir", "ertaga"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["bir-biriga", "bir-birini", "bir-biridan"]),
    ],
    "crh": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["nerede", "nereye", "nereden", "nasıl", "neden", "kaç"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["bütün", "hep", "her"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["bazı", "birkaç"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hiç", "hiçbir"]),
        ("adverb_modal", "ADV", "_",
         ["belki", "tabii", "elbette", "hâlâ", "sadece"]),
        ("particle_additive", "PART", "PartType=Add",
         ["bile", "dahi"]),
        ("temporal_adverb", "ADV", "_",
         ["bugün", "dün", "şimdi", "yarın", "sabah"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["bir-birine", "bir-birini"]),
    ],
    "gag": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["nerede", "nereye", "nereden", "nasıl", "neden", "kaç"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["bütün", "hep", "her"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["bazı", "birkaç"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hiç", "hiçbir"]),
        ("adverb_modal", "ADV", "_",
         ["belki", "elbette", "sadece", "hâlâ"]),
        ("particle_additive", "PART", "PartType=Add",
         ["bile", "dahi"]),
        ("temporal_adverb", "ADV", "_",
         ["büün", "dün", "şindi", "yarın", "sabah"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["biri-birinä", "biri-birini"]),
    ],
    "kaa": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["qayerda", "qayerge", "qayerden", "qanday", "nege", "qansha", "qaçan"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["barlıq", "hämme", "har"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["käybir", "birneshe"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["hiç", "heç"]),
        ("adverb_modal", "ADV", "_",
         ["balki", "ärine", "enshi", "tek"]),
        ("particle_additive", "PART", "PartType=Add",
         ["ham", "hatta"]),
        ("temporal_adverb", "ADV", "_",
         ["bügün", "keşege", "häzir", "erteng"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["bir-birine", "bir-birini"]),
    ],
    # ── Kipchak (Cyrillic script) ─────────────────────────────────────────
    "kaz": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["қайда", "қайға", "қашан", "қалай", "неге", "қанша"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлық", "бүкіл", "бәрі", "әр"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кейбір", "бірнеше"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["ешбір", "еш"]),
        ("adverb_modal", "ADV", "_",
         ["бәлкім", "мүмкін", "әрине", "әлі", "тек", "ғана", "әсіресе"]),
        ("particle_additive", "PART", "PartType=Add",
         ["тіпті"]),
        ("temporal_adverb", "ADV", "_",
         ["бүгін", "кеше", "қазір", "ертең"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бір-біріне", "бір-бірін"]),
    ],
    "kir": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["кайда", "кайга", "кайдан", "качан", "кандай", "канча"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["бардык", "баары", "ар"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["айрым", "кээ"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["эч"]),
        ("adverb_modal", "ADV", "_",
         ["балким", "мүмкүн", "гана", "эле"]),
        ("particle_additive", "PART", "PartType=Add",
         ["дагы", "деле"]),
        ("temporal_adverb", "ADV", "_",
         ["бүгүн", "кечээ", "азыр", "эртең"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бири-бирине", "бири-бирин"]),
    ],
    "tat": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["кайда", "кайдан", "кайчан", "нинди", "ничек", "ничә"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлык", "бөтен", "бары"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кайбер", "берничә"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["бернинди"]),
        ("adverb_modal", "ADV", "_",
         ["бәлки", "мөгаен", "тик", "инде"]),
        ("particle_additive", "PART", "PartType=Add",
         ["хәтта"]),
        ("temporal_adverb", "ADV", "_",
         ["бүген", "кичә", "хәзер", "иртәгә", "иртән"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бер-береңне", "бер-береңә"]),
    ],
    "bak": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["ҡайҙа", "ҡайҙан", "ҡасан", "нисек", "ниса"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлык", "бары"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["ҡайһы", "берничә"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["бернинсе"]),
        ("adverb_modal", "ADV", "_",
         ["бәлки", "мөгаен", "тик", "инде"]),
        ("particle_additive", "PART", "PartType=Add",
         ["хатта"]),
        ("temporal_adverb", "ADV", "_",
         ["бөгөн", "кисә", "хәҙер", "иртәгә"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бер-береһен", "бер-береһенә"]),
    ],
    # ── Siberian Turkic (Cyrillic) ─────────────────────────────────────────
    "chv": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["ăçта", "ăçтан", "хăçан", "мĕнле", "миçе"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["пĕтĕм", "пурĕ"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["нихăш", "пĕр"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["никам", "нимĕн"]),
        ("adverb_modal", "ADV", "_",
         ["çапах", "чăнах", "кăна", "анçах"]),
        ("particle_additive", "PART", "PartType=Add",
         []),       # Chuvash additives are clitics — no standalone tokens
        ("temporal_adverb", "ADV", "_",
         ["паян", "ĕнер", "халь", "ыран"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["пĕр-пĕрне"]),
    ],
    "sah": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["хайа", "хаhан", "хайдах", "хас"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барыта", "бары"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["сорох"]),
        ("determiner_negative", "DET", "PronType=Neg",
         []),       # Sakha negative det forms uncertain
        ("adverb_modal", "ADV", "_",
         ["эрэ", "ордук"]),
        ("particle_additive", "PART", "PartType=Add",
         []),
        ("temporal_adverb", "ADV", "_",
         ["бүгүн", "сарсын", "билигин"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         []),       # Sakha reciprocal forms uncertain
    ],
    "alt": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["кайда", "кайдан", "качан", "канай", "неде"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["бары", "бастыра"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кезик"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["эч"]),
        ("adverb_modal", "ADV", "_",
         ["мүмкин", "чын", "тек"]),
        ("particle_additive", "PART", "PartType=Add",
         []),
        ("temporal_adverb", "ADV", "_",
         ["эрте", "азыр"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бой-бойун", "бой-бойына"]),
    ],
    # Khakas uses ö (U+00F6) and ç (U+00E7) from Latin within Cyrillic text,
    # and ÿ (U+00FF) for /ü/.
    "kjh": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["хайда", "хайдаң", "хаçан", "неше"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["пöрi", "чыла"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["нимес"]),    # нимес = some/certain (different from negation use)
        ("determiner_negative", "DET", "PronType=Neg",
         []),
        ("adverb_modal", "ADV", "_",
         ["белки", "тек", "чылап"]),
        ("particle_additive", "PART", "PartType=Add",
         []),
        ("temporal_adverb", "ADV", "_",
         ["пÿгÿн", "ирте"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         []),
    ],
    "kum": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["қайда", "қайдан", "ничик", "нечун", "нече"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлыкъ", "битев"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кесек"]),
        ("determiner_negative", "DET", "PronType=Neg",
         []),
        ("adverb_modal", "ADV", "_",
         ["тек", "бусагъат"]),
        ("particle_additive", "PART", "PartType=Add",
         ["хатта"]),
        ("temporal_adverb", "ADV", "_",
         ["бюгюн", "тюнегюн", "хазир", "ярын"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бир-бирине", "бир-бирин"]),
    ],
    "nog": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["кайда", "кайдан", "качан", "кайтип", "нешше"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлык", "битев"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кесек"]),
        ("determiner_negative", "DET", "PronType=Neg",
         []),
        ("adverb_modal", "ADV", "_",
         ["тек", "балки"]),
        ("particle_additive", "PART", "PartType=Add",
         ["хатта"]),
        ("temporal_adverb", "ADV", "_",
         ["буьгуьн", "кеше", "казыр", "ярын"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бир-биринге", "бир-бирин"]),
    ],
    "krc": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["хайда", "хайдан", "хачан", "нечик", "нече"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["барлыкъ", "битеу"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["кесек"]),
        ("determiner_negative", "DET", "PronType=Neg",
         []),
        ("adverb_modal", "ADV", "_",
         ["белки", "тек"]),
        ("particle_additive", "PART", "PartType=Add",
         ["хатта"]),
        ("temporal_adverb", "ADV", "_",
         ["бюгюн", "шинди", "тамбла"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бир-бирине", "бир-бирин"]),
    ],
    "tyv": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["кайда", "кайга", "кайдан", "качан", "кандыг", "канчаа"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["шупту"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["чамдык"]),
        ("determiner_negative", "DET", "PronType=Neg",
         []),
        ("adverb_modal", "ADV", "_",
         ["магаадыр", "чүгле", "тек"]),
        ("particle_additive", "PART", "PartType=Add",
         []),
        ("temporal_adverb", "ADV", "_",
         ["амдыы", "даарта"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["бот-боттарын", "бот-боттарынга"]),
    ],
    # ── Arabic script ─────────────────────────────────────────────────────
    "uig": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["قەيەردە", "قەيەرگە", "قەيەردىن", "قاچان", "قانداق", "نېمىشقا", "قانچە"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["ھەممە", "بارلىق", "ھەر"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["بەزى"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["ھېچ", "ھېچبىر"]),
        ("adverb_modal", "ADV", "_",
         ["بەلكىم", "مۇمكىن", "پەقەت", "ئالاھىدە"]),
        ("particle_additive", "PART", "PartType=Add",
         ["ھەتتا", "يەنە"]),
        ("temporal_adverb", "ADV", "_",
         ["بۈگۈن", "كېچە", "ھازىر", "ئەتە"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["بىر-بىرىگە", "بىر-بىرىنى"]),
    ],
    "azb": [
        ("adverb_interrogative", "ADV", "PronType=Int",
         ["هاراده", "هاراوا", "نئجه", "نیه", "نئچه"]),
        ("determiner_universal", "DET", "PronType=Tot",
         ["همه", "بوتون", "هر"]),
        ("determiner_indefinite", "DET", "PronType=Ind",
         ["بعضی"]),
        ("determiner_negative", "DET", "PronType=Neg",
         ["هئچ"]),
        ("adverb_modal", "ADV", "_",
         ["بئلکه", "یالنیز", "خوصوصیله"]),
        ("particle_additive", "PART", "PartType=Add",
         ["حتی", "بئله"]),
        ("temporal_adverb", "ADV", "_",
         ["بوگون", "دونن", "ایندی", "ساباح"]),
        ("pronoun_reciprocal", "PRON", "PronType=Rcp",
         ["بیر-بیرینه", "بیر-بیرینی"]),
    ],
}


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
    existing_types = {e.get("type") for e in entries}

    added = []
    for type_, upos, feats, forms in NEW_ENTRIES[lang]:
        if not forms:
            continue          # language-specific gap — skip silently
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
        update_file(PKG_DIR / f"{lang}.json", lang)
        update_file(STAGE_DIR / f"{lang}.json", lang)
    print("\nDone.")


if __name__ == "__main__":
    main()
