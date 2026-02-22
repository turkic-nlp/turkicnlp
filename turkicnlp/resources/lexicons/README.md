# Closed-Class Lexicons for Turkic Languages

This directory contains manually curated closed-class word lexicons for 21 Turkic languages.
They are used by `ApertiumMorphProcessor` to improve morphological analysis quality.

## Purpose

Apertium HFST finite-state transducers (FSTs) fail in two systematic ways for closed-class words:

1. **Missing lemmas** — common function words absent from the FST dictionary return no analysis,
   resulting in `UPOS=X`.
2. **Disambiguation failure** — the FST finds multiple readings and the wrong one wins (e.g. degree
   adverb `çok` has an adjectival reading that can beat the adverbial one; Kazakh `өз` overlaps
   with a noun stem).

The lexicons fix both: they serve as a fallback when the FST returns nothing
(`_fallback_for_unknown`), and as a `+2` scoring signal during multi-reading disambiguation
(`context_score` in `_disambiguate`).

## Files

One JSON file per language, named by ISO 639-3 code:

| File | Language | Script |
|------|----------|--------|
| `alt.json` | Altai | Cyrillic |
| `azb.json` | South Azerbaijani | Arabic |
| `aze.json` | Azerbaijani | Latin |
| `bak.json` | Bashkir | Cyrillic |
| `chv.json` | Chuvash | Cyrillic |
| `crh.json` | Crimean Tatar | Latin |
| `gag.json` | Gagauz | Latin |
| `kaa.json` | Karakalpak | Latin |
| `kaz.json` | Kazakh | Cyrillic |
| `kir.json` | Kyrgyz | Cyrillic |
| `kjh.json` | Khakas | Cyrillic |
| `krc.json` | Karachay-Balkar | Cyrillic |
| `kum.json` | Kumyk | Cyrillic |
| `nog.json` | Nogai | Cyrillic |
| `sah.json` | Sakha (Yakut) | Cyrillic |
| `tat.json` | Tatar | Cyrillic |
| `tuk.json` | Turkmen | Latin |
| `tur.json` | Turkish | Latin |
| `tyv.json` | Tuvan | Cyrillic |
| `uig.json` | Uyghur | Arabic |
| `uzb.json` | Uzbek | Latin |

## Schema

```json
{
  "entries": [
    {
      "type":           "string  — entry type identifier (see table below)",
      "upos":           "string  — Universal Dependencies UPOS tag",
      "feats":          "string  — UD morphological features string, or \"_\"",
      "lemma_strategy": "string  — how to derive lemma: \"lower\" (lowercase surface form)",
      "forms":          ["list", "of", "surface", "forms"]
    }
  ]
}
```

**Conventions:**
- All forms are lowercase, in the correct script for the language.
- Only single-token surface forms are included; cliticized or multi-token constructions are excluded.
- An entry type is omitted from a file entirely if no reliable standalone forms exist for that language.
- `feats: "_"` means no morphological features beyond UPOS are assigned.

## Entry Types

### Base types (all 21 languages)

| Type | UPOS | UD Feats | What it covers |
|------|------|----------|----------------|
| `pronoun_personal` | PRON | `PronType=Prs` | I / you / he / we / you-pl / they |
| `pronoun_demonstrative` | PRON | `PronType=Dem` | this / that / these / those |
| `pronoun_interrogative` | PRON | `PronType=Int` | who / what (pronoun uses) |
| `question_particle` | PART | `PartType=Int` | polar question marker (Turkish *mi/mı/mu/mü*, Kazakh *ма/ме/ба/бе*) |
| `postposition` | ADP | `AdpType=Post` | postpositions (for, about, until, according to, …) |
| `conjunction_coord` | CCONJ | `_` | and / or / but / however |
| `conjunction_sub` | SCONJ | `_` | because / if / therefore / while |
| `negation_particle` | PART | `Polarity=Neg` | standalone negation word (*değil*, *емес*, *يوق*) |

### Extended types — high priority (all 21 languages, added 2026-02-22)

| Type | UPOS | UD Feats | What it covers |
|------|------|----------|----------------|
| `pronoun_reflexive` | PRON | `Reflex=Yes` | self / own pronoun series (*kendi*, *öz*, *өз*, *бэйэ*, *ئۆز*) |
| `adverb_degree` | ADV | `_` | degree / intensifier adverbs: very, little, more, most (*çok/az/daha/en*, *өте/аз/ең*) |
| `auxiliary_evidential` | AUX | `Evident=Nfh` | reported / evidential copula (*imiş*, *екен*, *эбит*, *ئىكەن*) |

### Extended types — medium priority (all 21 languages, added 2026-02-22)

| Type | UPOS | UD Feats | What it covers | Language gaps |
|------|------|----------|----------------|---------------|
| `adverb_interrogative` | ADV | `PronType=Int` | where / when / how / why (adverb uses) | — |
| `determiner_universal` | DET | `PronType=Tot` | all / every / each (*tüm/her*, *барлық/бүкіл*, *шупту*) | — |
| `determiner_indefinite` | DET | `PronType=Ind` | some / several / a few (*bazı/birkaç*, *кейбір/бірнеше*) | — |
| `determiner_negative` | DET | `PronType=Neg` | no / none as pre-nominal (*hiçbir*, *ешбір*) | chv, sah, kjh, tyv, azb — encliticize or use verbal negation |
| `adverb_modal` | ADV | `_` | maybe / certainly / only / still (*belki/sadece*, *бәлкім/тек*) | — |
| `particle_additive` | PART | `PartType=Add` | also / even as standalone tokens (*bile/dahi*, *да/де*, *тіпті*) | chv, sah — additive is always a suffix |
| `temporal_adverb` | ADV | `_` | today / yesterday / now / tomorrow (*bugün/dün*, *бүгін/кеше*) | — |
| `pronoun_reciprocal` | PRON | `PronType=Rcp` | each other as a single hyphenated token (*birbirini*, *бір-бірін*) | sah, kjh, tyv — reciprocal constructions are multi-token |

### Extended types — low priority (all 21 languages, added 2026-02-22)

| Type | UPOS | UD Feats | What it covers | Notes |
|------|------|----------|----------------|-------|
| `interjection` | INTJ | `_` | yes / okay response words (*evet/tamam*, *иә/жарайды*, *ھە/بولىدۇ*) | "no" forms omitted where they overlap with `negation_particle` |
| `numeral_cardinal` | NUM | `NumType=Card` | cardinal numerals 1–10 in citation form (*bir…on*, *бір…он*, *بىر…ئون*) | Nominative only; FST handles inflected case forms |

**`interjection` forms per language** (yes / okay; "no" omitted if already in `negation_particle`):

| Lang | Forms |
|------|-------|
| tur | evet, tamam, hayır |
| aze | bəli, hə, tamam, xeyr |
| tuk | hawa, bolýar |
| uzb | ha, xo'p, mayli |
| crh | evet, tamam |
| gag | evet, tamam |
| kaa | äwä, maqqul |
| kaz | иә, ия, жарайды, мақұл |
| kir | ооба, ийе, болду |
| tat | әйе, ярый |
| bak | эйе, яраша |
| chv | ă, лайăх |
| sah | сөп |
| alt | ии, болды |
| kum | ха, яхшы |
| nog | ийе, ярай |
| krc | ха, яхшы |
| kjh | хоосха, ии |
| tyv | ийе |
| uig | ھە, بولىدۇ |
| azb | بله, تامام |

**`numeral_cardinal` forms per language** (1 through 10):

| Lang | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|---|
| tur | bir | iki | üç | dört | beş | altı | yedi | sekiz | dokuz | on |
| aze | bir | iki | üç | dörd | beş | altı | yeddi | səkkiz | doqquz | on |
| tuk | bir | iki | üç | dört | bäş | alty | ýedi | sekiz | dokuz | on |
| uzb | bir | ikki | uch | to'rt | besh | olti | yetti | sakkiz | to'qqiz | o'n |
| crh | bir | iki | üç | dört | beş | altı | yedi | sekiz | dokuz | on |
| gag | bir | iki | üç | dört | beş | altı | yedi | sekiz | dokuz | on |
| kaa | bir | eki | üsh | tört | bes | altı | jeti | segiz | toğız | on |
| kaz | бір | екі | үш | төрт | бес | алты | жеті | сегіз | тоғыз | он |
| kir | бир | эки | үч | төрт | беш | алты | жети | сегиз | тогуз | он |
| tat | бер | ике | өч | дүрт | биш | алты | җиде | сигез | тугыз | ун |
| bak | бер | ике | өс | дүрт | биш | алты | ете | һигеҙ | туғыҙ | ун |
| chv | пĕр | иккĕ | виç | тăват | пилĕк | улт | çиççĕ | саккăр | тăхăр | вун |
| sah | биир | икки | үс | түөрт | биэс | алта | сэттэ | аҕыс | тоҕус | уон |
| alt | бир | эки | үч | тöрт | беш | алты | йети | сегис | тогус | он |
| kum | бир | эки | уьч | дёрт | беш | алты | йетти | сегиз | тогъуз | он |
| nog | бир | эки | уьш | дёрт | бес | алты | йети | сегиз | тогыз | он |
| krc | бир | эки | юч | дёрт | беш | алты | жети | сегиз | тогъуз | он |
| kjh | пір | ікі | ÿс | тöрт | піс | алты | четі | сегіс | тоғыс | он |
| tyv | бир | ийи | үш | дөрт | беш | алды | чеди | сес | тос | он |
| uig | بىر | ئىككى | ئۈچ | تۆت | بەش | ئالتە | يەتتە | سەككىز | توققۇز | ئون |
| azb | بیر | ایکی | اوچ | دؤرد | بئش | آلتی | یئددی | سکیز | دوققوز | اون |

## How the Lexicon Is Used

```
ApertiumMorphProcessor.load()
  └─ reads resources/lexicons/<lang>.json into self._lexicon
       dict[form → list[(upos, feats)]]

Per token:
  1. FST lookup → zero readings?
       └─ _fallback_for_unknown(): lexicon hit → return lexicon UPOS/feats
                                   no hit    → return X/_
  2. FST lookup → multiple readings?
       └─ _disambiguate(): context_score() adds +2 to any reading whose
          UD UPOS matches the lexicon entry for that surface form
```

The `_lookup_lexicon(text)` helper checks both the native-script form and, when transliteration
is active, the FST-script equivalent, so cross-script pipelines benefit automatically.

## Design Constraints

- **Closed classes only.** Nouns, verbs, and adjectives are open-class — they are excluded.
  The FST handles them; the lexicon does not duplicate that work.
- **Nominative / citation forms only** for pronouns and determiners. Inflected case forms are
  handled by the FST suffix machinery.
- **No multi-token entries.** Constructions like Kazakh *бір-біріне* (reciprocal dative) are
  included as a single hyphenated token only when they are conventionally written without spaces.
- **Script-faithful.** Each file uses only the standard script for that language. No ASCII
  transliterations or mixed-script forms.
- **Forms must be lowercase.** The processor lowercases the surface form before lookup.

## Adding or Updating Entries

1. Edit the JSON file(s) in this directory **and** the staging copy at
   `resources/grammar_sources/lexicons/<lang>.json` (kept as source of truth outside the package).
2. Keep `feats` consistent with Universal Dependencies v2 feature inventory.
3. Run `pytest turkicnlp/tests/test_morphology.py` to verify no regressions.
4. If adding a new `type` value with a new UPOS/feats combination, check that
   `_normalize_ud_feats_for_upos` in `turkicnlp/processors/morphology.py` allows the feature
   prefix for that UPOS (e.g. `PronType=` is allowed for ADV after the 2026-02-22 fix).

Scripts used for bulk additions:
- `scripts/add_high_priority_lexicon.py` — added `pronoun_reflexive`, `adverb_degree`, `auxiliary_evidential`
- `scripts/add_medium_priority_lexicon.py` — added the eight medium-priority types above
- `scripts/add_low_priority_lexicon.py` — added `interjection`, `numeral_cardinal`
