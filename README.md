<p align="center">
  <img src="https://sherzod-hakimov.github.io/images/cover.png" alt="TurkicNLP — Six Branches of Turkic Language Family" width="200">
</p>

<h1 align="center">TurkicNLP</h1>

<p align="center">
  <strong>NLP toolkit for 20+ Turkic languages</strong> — a pip-installable Python library inspired by <a href="https://stanfordnlp.github.io/stanza/">Stanza</a>, with adaptations for the low-resource, morphologically rich Turkic language family.
</p>

<p align="center">
  Maintained by <a href="https://sherzod-hakimov.github.io/">Sherzod Hakimov</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg" alt="Python 3.9 | 3.10 | 3.11 | 3.12"></a>
  <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status: Pre-Alpha">
  <img src="https://img.shields.io/badge/languages-24_Turkic-green.svg" alt="24 Turkic Languages">
  <a href="https://github.com/turkic-nlp/turkicnlp/actions/workflows/test-installation.yml"><img src="https://github.com/turkic-nlp/turkicnlp/actions/workflows/test-installation.yml/badge.svg" alt="Package Installation Tests"></a>
</p>

## Citation

If you use TurkicNLP in your research, please cite:

```bibtex
@misc{hakimov2026turkicnlpnlptoolkit,
      title={TurkicNLP: An NLP Toolkit for Turkic Languages}, 
      author={Sherzod Hakimov},
      year={2026},
      eprint={2602.19174},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.19174}, 
}
```

## Arxiv preprint
[PDF](https://arxiv.org/pdf/2602.19174)

## Features

- **24 Turkic languages** from Turkish to Sakha, Kazakh to Uyghur
- **Script-aware from the ground up** — Latin, Cyrillic, Perso-Arabic, Old Turkic Runic
- **Automatic script detection** and bidirectional transliteration
- **[Apertium FST morphology](https://wiki.apertium.org/wiki/Turkic_languages)** for ~20 Turkic languages via Python-native `hfst` bindings (no system install)
- **Stanza/UD integration** — pretrained tokenization, POS tagging, lemmatization, dependency parsing, and NER via [Stanza](https://stanfordnlp.github.io/stanza/) models trained on [Universal Dependencies](https://universaldependencies.org/) treebanks
- **NLLB embeddings + translation backend** — sentence/document vectors and MT via [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Multiple backends** — choose between rule-based, Apertium FST, or Stanza neural backends per processor
- **License isolation** — library is Apache-2.0; Apertium GPL-3.0 data downloaded separately
- **Stanza-compatible API** — `Pipeline`, `Document`, `Sentence`, `Word`

## Installation

**Requirements:** Python 3.9, 3.10, 3.11, or 3.12

```bash
pip install turkicnlp                    # core — tokenization, rule-based processing, CoNLL-U I/O
pip install "turkicnlp[hfst]"           # + Apertium FST morphology (Linux and macOS only)
pip install "turkicnlp[stanza]"         # + Stanza neural models (tokenize, POS, lemma, depparse, NER)
pip install "turkicnlp[translation]"    # + NLLB embeddings and machine translation
pip install "turkicnlp[all]"            # everything above (Linux and macOS only)
pip install "turkicnlp[dev]"            # development tools (pytest, black, ruff, mypy)
```

### Platform compatibility

Installation tests run nightly across all combinations of OS, Python version, and install extra (see [CI workflow](https://github.com/turkic-nlp/turkicnlp/actions/workflows/test-installation.yml)).

| Extra | Ubuntu 22.04 / 24.04 | macOS 14 / 15 | Windows 2025 |
|---|---|---|---|
| base | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 |
| `[hfst]` | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 | ❌ not available |
| `[stanza]` | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 |
| `[translation]` | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 |
| `[all]` | ✅ 3.9 – 3.12 | ✅ 3.9 – 3.12 | ❌ not available |

> **Windows users:** the `hfst` Python package has no published wheels for Python 3.7 or later on Windows — this is an upstream limitation with no current workaround. All features except Apertium FST morphology work normally on Windows; use `turkicnlp[stanza]` or `turkicnlp[translation]` instead. If you need Apertium FST morphology on Windows, the recommended approach is [Windows Subsystem for Linux (WSL)](https://wiki.apertium.org/wiki/Apertium_on_Windows), where `hfst` installs normally.

## Quick Start

```python
import turkicnlp

# Download models for a language
turkicnlp.download("kaz")

# Build a pipeline
nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "pos", "lemma", "ner", "depparse"])

# Process text
doc = nlp("Мен мектепке бардым")

# Access annotations
for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text}\t{word.lemma}\t{word.upos}\t{word.feats}")

# Export to CoNLL-U
print(doc.to_conllu())
```

### Embeddings (NLLB)

```python
import math
import turkicnlp

turkicnlp.download("tur", processors=["embeddings"])
nlp = turkicnlp.Pipeline("tur", processors=["embeddings"])

doc1 = nlp("Bugün hava çok güzel ve parkta yürüyüş yaptım.")
doc2 = nlp("Parkta yürüyüş yapmak bugün çok keyifliydi çünkü hava güzeldi.")

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

print(len(doc1.embedding), len(doc2.embedding))
print(f"cosine = {cosine_similarity(doc1.embedding, doc2.embedding):.4f}")
print(doc1._processor_log)  # ['embeddings:nllb']
```

### Machine Translation (NLLB)

```python
import turkicnlp

# Downloads once into ~/.turkicnlp/models/huggingface/facebook--nllb-200-distilled-600M
turkicnlp.download("tur", processors=["translate"])

nlp = turkicnlp.Pipeline(
    "tur",
    processors=["translate"],
    translate_tgt_lang="eng",
)

doc = nlp("Bugün hava çok güzel ve parkta yürüyüş yaptım.")
print(doc.translation)
print(doc._processor_log)  # ['translate:nllb']
```

`translate_tgt_lang` accepts either ISO-639-3 (`"eng"`, `"tuk"`, `"kaz"`) or explicit [Flores-200 codes](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200) (`"eng_Latn"`, `"kaz_Cyrl"`).

### Using the Stanza Backend

```python
from turkicnlp.processors.stanza_backend import (
    StanzaTokenizer, StanzaPOSTagger, StanzaLemmatizer, StanzaNERProcessor, StanzaDepParser
)
from turkicnlp.models.document import Document

# Models are downloaded automatically on first use
doc = Document(text="Merhaba dünya.", lang="tur")

for Proc in [StanzaTokenizer, StanzaPOSTagger, StanzaLemmatizer, StanzaNERProcessor, StanzaDepParser]:
    proc = Proc(lang="tur")
    proc.load()
    doc = proc.process(doc)

for word in doc.words:
    print(f"{word.text:12} {word.upos:6} {word.lemma:12} head={word.head} {word.deprel}")

# Export to CoNLL-U
print(doc.to_conllu())
```

### Mixed Backends

```python
from turkicnlp.processors.tokenizer import RegexTokenizer
from turkicnlp.processors.stanza_backend import StanzaPOSTagger, StanzaNERProcessor, StanzaDepParser
from turkicnlp.models.document import Document

doc = Document(text="Мен мектепке бардым.", lang="kaz")

# Rule-based tokenizer + Stanza POS/parsing (pretokenized mode)
tokenizer = RegexTokenizer(lang="kaz")
tokenizer.load()
doc = tokenizer.process(doc)

pos = StanzaPOSTagger(lang="kaz")
pos.load()
doc = pos.process(doc)

ner = StanzaNERProcessor(lang="kaz")
ner.load()
doc = ner.process(doc)

parser = StanzaDepParser(lang="kaz")
parser.load()
doc = parser.process(doc)
```

### Multi-Script Support

```python
# Kazakh — auto-detects Cyrillic vs Latin
doc = nlp("Мен мектепке бардым")    # Cyrillic
doc = nlp("Men mektepke bardym")     # Latin

# Explicit script selection
nlp_cyrl = turkicnlp.Pipeline("kaz", script="Cyrl")
nlp_latn = turkicnlp.Pipeline("kaz", script="Latn")

# Transliteration bridge — run Cyrillic model on Latin input
nlp = turkicnlp.Pipeline("kaz", script="Latn", transliterate_to="Cyrl")
```

### Uyghur (Perso-Arabic)

```python
nlp_ug = turkicnlp.Pipeline("uig", script="Arab")
doc = nlp_ug("مەن مەكتەپكە باردىم")
```

### Transliteration

The `Transliterator` class converts text between scripts for any supported language pair:

```python
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

# Kazakh Cyrillic → Latin (2021 official alphabet)
t = Transliterator("kaz", Script.CYRILLIC, Script.LATIN)
print(t.transliterate("Қазақстан Республикасы"))
# → Qazaqstan Respublıkasy

# Uzbek Latin → Cyrillic
t = Transliterator("uzb", Script.LATIN, Script.CYRILLIC)
print(t.transliterate("O'zbekiston Respublikasi"))
# → Ўзбекистон Республикаси

# Uyghur Perso-Arabic → Latin (ULY)
t = Transliterator("uig", Script.PERSO_ARABIC, Script.LATIN)
print(t.transliterate("مەكتەپ"))
# → mektep

# Azerbaijani Latin → Cyrillic
t = Transliterator("aze", Script.LATIN, Script.CYRILLIC)
print(t.transliterate("Azərbaycan"))
# → Азәрбайҹан

# Turkmen Latin → Cyrillic
t = Transliterator("tuk", Script.LATIN, Script.CYRILLIC)
print(t.transliterate("Türkmenistan"))
# → Түркменистан

# Tatar Cyrillic → Latin (Zamanälif)
t = Transliterator("tat", Script.CYRILLIC, Script.LATIN)
print(t.transliterate("Татарстан Республикасы"))
# → Tatarstan Respublikası
```

#### Old Turkic Runic Script

TurkicNLP supports transliteration of [Old Turkic runic inscriptions](https://en.wikipedia.org/wiki/Old_Turkic_script) (Orkhon-Yenisei script, Unicode block U+10C00–U+10C4F) to Latin:

```python
from turkicnlp.scripts import Script
from turkicnlp.scripts.transliterator import Transliterator

t = Transliterator("otk", Script.OLD_TURKIC_RUNIC, Script.LATIN)

# Individual runic characters
print(t.transliterate("\U00010C34\U00010C07\U00010C2F\U00010C19"))
# → törk  (Türk)

# The transliterator maps each runic character to its standard
# Turkological Latin equivalent, handling both Orkhon and Yenisei
# variant forms (e.g., separate glyphs for consonants with
# back vs. front vowel contexts).
```

## Supported Languages and Components

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Turkic_Languages_distribution_map.png" alt="Distribution map of Turkic languages" width="700">
  <br>
  <em>Geographic distribution of Turkic languages (source: <a href="https://commons.wikimedia.org/wiki/File:Turkic_Languages_distribution_map.png">Wikimedia Commons</a>)</em>
</p>

The table below shows all supported languages with their available scripts and processor status.

**Backend legend:**
- **rule** — Rule-based (regex tokenizer, abbreviation lists)
- **Apertium** — Finite-state transducers via [Apertium](https://apertium.org/) + `hfst` ([GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html), downloaded separately)
- **Stanza/UD** — Neural models from [Stanza](https://stanfordnlp.github.io/stanza/) trained on [Universal Dependencies](https://universaldependencies.org/) treebanks ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0))
- **NLLB** — Shared [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) backend for embeddings and machine translation

**Status legend:** 

- ✅  Available
- 🔧 Planned 
- — Not available (yet)

### Oghuz Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Turkish](https://en.wikipedia.org/wiki/Turkish_language) | `tur` | Latn | ✅ rule, ✅ Stanza/UD | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza | ✅ NLLB | ✅ NLLB |
| [Azerbaijani](https://en.wikipedia.org/wiki/Azerbaijani_language) | `aze` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 | 🔧 | 🔧 | — | ✅ NLLB | ✅ NLLB |
| [Iranian Azerbaijani](https://en.wikipedia.org/wiki/South_Azerbaijani_language) | `azb` | Arab | 🔧 rule_arabic | — | — | — | — | — | ✅ NLLB | ✅ NLLB |
| [Turkmen](https://en.wikipedia.org/wiki/Turkmen_language) | `tuk` | Latn, Cyrl | ✅ rule | ✅ Apertium (beta) | 🔧 | 🔧 | 🔧 | — | ✅ NLLB | ✅ NLLB |
| [Gagauz](https://en.wikipedia.org/wiki/Gagauz_language) | `gag` | Latn | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |

### Kipchak Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) | `kaz` | Cyrl, Latn | ✅ rule, ✅ Stanza/UD | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza | ✅ NLLB | ✅ NLLB |
| [Kyrgyz](https://en.wikipedia.org/wiki/Kyrgyz_language) | `kir` | Cyrl | ✅ rule | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | — | ✅ NLLB | ✅ NLLB |
| [Tatar](https://en.wikipedia.org/wiki/Tatar_language) | `tat` | Cyrl, Latn | ✅ rule | ✅ Apertium | 🔧 | 🔧 | 🔧 | — | ✅ NLLB | ✅ NLLB |
| [Bashkir](https://en.wikipedia.org/wiki/Bashkir_language) | `bak` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | ✅ NLLB | ✅ NLLB |
| [Crimean Tatar](https://en.wikipedia.org/wiki/Crimean_Tatar_language) | `crh` | Latn, Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | ✅ NLLB | ✅ NLLB |
| [Karakalpak](https://en.wikipedia.org/wiki/Karakalpak_language) | `kaa` | Latn, Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Nogai](https://en.wikipedia.org/wiki/Nogai_language) | `nog` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Kumyk](https://en.wikipedia.org/wiki/Kumyk_language) | `kum` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Karachay-Balkar](https://en.wikipedia.org/wiki/Karachay-Balkar_language) | `krc` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |

### Karluk Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Uzbek](https://en.wikipedia.org/wiki/Uzbek_language) | `uzb` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 | 🔧 | 🔧 | — | ✅ NLLB | ✅ NLLB |
| [Uyghur](https://en.wikipedia.org/wiki/Uyghur_language) | `uig` | Arab, Latn | 🔧 rule_arabic, ✅ Stanza/UD | ✅ Apertium (beta) | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | — | ✅ NLLB | ✅ NLLB |

### Siberian Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Sakha (Yakut)](https://en.wikipedia.org/wiki/Sakha_language) | `sah` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Altai](https://en.wikipedia.org/wiki/Altai_language) | `alt` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Tuvan](https://en.wikipedia.org/wiki/Tuvan_language) | `tyv` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |
| [Khakas](https://en.wikipedia.org/wiki/Khakas_language) | `kjh` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — | — |

### Oghur Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Chuvash](https://en.wikipedia.org/wiki/Chuvash_language) | `chv` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | — | — |

### Arghu Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Khalaj](https://en.wikipedia.org/wiki/Khalaj_language) | `klj` | Arab | - | - | — | — | — | — | — | — |

### Historical Languages

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Embeddings | Translation |
|---|---|---|---|---|---|---|---|---|---|---|
| [Ottoman Turkish](https://en.wikipedia.org/wiki/Ottoman_Turkish_language) | `ota` | Arab, Latn | - | — | - | - | - | — | — | — |
| [Old Turkish](https://en.wikipedia.org/wiki/Old_Turkic_language) | `otk` | Orkh, Latn | 🔧 rule | — | — | — | — | — | — | — |

### Stanza/UD Model Details

The Stanza backend provides neural models trained on [Universal Dependencies](https://universaldependencies.org/) treebanks. The table below lists the UD treebanks and NER datasets powering each language, along with the available Stanza processors.

| Language | Stanza Code | UD Treebank(s) | Stanza Processors | NER Dataset |
|---|---|---|---|---|
| Turkish | `tr` | IMST (default), BOUN, FrameNet, KeNet, ATIS, Penn, Tourism | tokenize, mwt, pos, lemma, depparse, ner | Starlang NER |
| Kazakh | `kk` | KTB | tokenize, mwt, pos, lemma, depparse, ner | KazNERD |
| Uyghur | `ug` | UDT | tokenize, pos, lemma, depparse | — |
| Kyrgyz | `ky` | KTMU | tokenize, pos, lemma, depparse | — |
| Ottoman Turkish | `ota` | BOUN | tokenize, mwt, pos, lemma, depparse | — |

### Transliteration Support

Bidirectional script conversion is available for all multi-script languages. The transliterator uses a greedy longest-match algorithm with per-language mapping tables.

| Language | Direction | Scripts | Standard |
|---|---|---|---|
| Kazakh | ↔ Bidirectional | Cyrillic ↔ Latin | 2021 official Latin alphabet |
| Uzbek | ↔ Bidirectional | Cyrillic ↔ Latin | 1995 official Latin alphabet |
| Azerbaijani | ↔ Bidirectional | Cyrillic ↔ Latin | 1991 official Latin alphabet |
| Tatar | ↔ Bidirectional | Cyrillic ↔ Latin | Zamanälif |
| Turkmen | ↔ Bidirectional | Cyrillic ↔ Latin | 1993 official Latin alphabet |
| Karakalpak | ↔ Bidirectional | Cyrillic ↔ Latin | 2016 Latin alphabet |
| Crimean Tatar | ↔ Bidirectional | Cyrillic ↔ Latin | Standard Crimean Tatar Latin |
| Uyghur | ↔ Bidirectional | Perso-Arabic ↔ Latin | Uyghur Latin Yéziqi (ULY) |
| Ottoman Turkish | → One-way | Latin → Perso-Arabic | Academic transcription |
| Old Turkic | → One-way | Runic → Latin | Turkological convention |

### Apertium FST Quality Levels

| Level | Description | Languages |
|---|---|---|
| **Production** | >90% coverage on news text | Turkish, Kazakh, Tatar |
| **Stable** | Good coverage, actively maintained | Azerbaijani, Kyrgyz, Uzbek |
| **Beta** | Reasonable coverage, some gaps | Turkmen, Bashkir, Uyghur, Crimean Tatar, Chuvash |
| **Prototype** | Limited coverage, experimental | Gagauz, Sakha, Karakalpak, Nogai, Kumyk, Karachay-Balkar, Altai, Tuvan, Khakas |

### Model Catalog and Apertium Downloads

TurkicNLP uses a model catalog to define download sources per language/script/processor. The catalog lives in:

- `turkicnlp/resources/catalog.json` (packaged default)
- Remote override: `ModelRegistry.CATALOG_URL` (or `TURKICNLP_CATALOG_URL`)

For each language, the catalog stores the Apertium source repo and the expected FST script. When `turkicnlp.download()` is called, it reads the catalog and downloads precompiled `.hfst` binaries from the `url` fields. If a language has no URL configured, download will fail with a clear error until the catalog is populated with hosted binaries (for example, a `turkic-nlp/apertium-data` releases repository).

#### Download folder
All models and resources are downloaded into this folder: `~/.turkicnlp`.


## Architecture

TurkicNLP follows Stanza's modular pipeline design:

```
Pipeline("tur", processors=["tokenize", "morph", "pos", "ner", "depparse"])
    │
    ▼
  Document ─── text: "Ben okula vardım"
    │
    ├── script_detect    → script = "Latn"
    ├── tokenize         → sentences, tokens, words
    ├── morph (Apertium) → lemma, pos, feats (via HFST)
    ├── pos (neural)     → refined UPOS, XPOS, feats
    ├── ner (neural)     → BIO tags and entity spans
    └── depparse         → head, deprel
    │
    ▼
  Document ─── annotated with all layers
```

```
Pipeline("azb", processors=["embeddings", "translate"], translate_tgt_lang="eng")
    │
    ▼
  Document ─── text: "من کتاب اوخویورام"
    │
    ├── script_detect          → script = "Arab"
    ├── embeddings (NLLB)      → sentence/document vectors
    └── translate (NLLB)       → sentence/document translation
           (src resolved from FLORES map: azb -> azb_Arab,
            tgt resolved from ISO-3: eng -> eng_Latn)
    │
    ▼
  Document ─── annotated with all layers
```

### Key Abstractions

- **Document** → Sentence → Token → Word hierarchy (maps to CoNLL-U)
- **Processor** ABC with `PROVIDES`, `REQUIRES`, `NAME` class attributes
- **Pipeline** orchestrator with dependency resolution and script-aware model loading
- **ProcessorRegistry** for pluggable backends (rule, Apertium, Stanza, NLLB)
- **ModelRegistry** with remote catalog and local caching at `~/.turkicnlp/models/`
- **NLLB FLORES language map** for ISO-3 to NLLB code resolution in translation (e.g. `tuk` -> `tuk_Latn`)

### Model Storage Layout

```
~/.turkicnlp/models/
├── kaz/
│   ├── Cyrl/
│   │   ├── tokenize/rule/
│   │   ├── morph/apertium/    ← GPL-3.0 (downloaded separately)
│   │   │   ├── kaz.automorf.hfst
│   │   │   ├── LICENSE
│   │   │   └── metadata.json
│   │   ├── pos/neural/
│   │   └── depparse/neural/
│   └── Latn/
│       └── tokenize/rule/
├── tur/
│   └── Latn/
│       └── ...
├── huggingface/
│   └── facebook--nllb-200-distilled-600M/
│       ├── config.json
│       ├── model.safetensors (or pytorch_model.bin)
│       ├── tokenizer.json
│       └── ...
└── catalog.json

# Stanza models are managed by Stanza at ~/stanza_resources/
```

Notes:
- NLLB embeddings and translation use a shared Hugging Face model under `~/.turkicnlp/models/huggingface/`.
- The NLLB model is downloaded once and reused across supported Turkic languages.
- Unlike Apertium/Stanza components, NLLB artifacts are not duplicated per language/script directory.

## License

- **Library code**: [Apache License 2.0](LICENSE)
- **Stanza models**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — managed by Stanza's own download mechanism
- **Apertium FST data**: [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) — downloaded separately at runtime, never bundled in the pip package
- **NLLB-200 model weights/tokenizer**: [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — downloaded from Hugging Face at runtime and reused from `~/.turkicnlp/models/huggingface/` (non-commercial license terms apply)

## Development

```bash
git clone https://github.com/turkic-nlp/turkicnlp.git
cd turkicnlp
pip install -e ".[dev]"
pytest
```

## Contributing

Contributions are welcome, especially:

- **New language support** — tag mappings, abbreviation lists, test data
- **Neural model training** — POS taggers, parsers, NER models
- **Apertium FST improvements** — better coverage for prototype-level languages
- **Other** -  any other aspect that you want

Create issues, Pull Requests etc.



## Acknowledgements

TurkicNLP builds on the work of many researchers and communities. We gratefully acknowledge the following:

### Stanza

[Stanza](https://stanfordnlp.github.io/stanza/) provides the pretrained neural models for tokenization, POS tagging, lemmatization, dependency parsing, and NER.

> Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning. 2020. *Stanza: A Python Natural Language Processing Toolkit for Many Human Languages*. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations. [[paper]](https://aclanthology.org/2020.acl-demos.14/)

### Universal Dependencies Treebanks

The Stanza models are trained on [Universal Dependencies](https://universaldependencies.org/) treebanks created by the following teams:

**Turkish (UD_Turkish-IMST)**
> Umut Sulubacak, Memduh Gokirmak, Francis Tyers, Cagri Coltekin, Joakim Nivre, and Gulsen Cebiroglu Eryigit. *Universal Dependencies for Turkish*. COLING 2016. [[paper]](https://aclanthology.org/C16-1325/)

**Turkish (UD_Turkish-BOUN)**
> Utku Turk, Furkan Atmaca, Saziye Betul Ozates, Gozde Berk, Seyyit Talha Bedir, Abdullatif Koksal, Balkiz Ozturk Basaran, Tunga Gungor, and Arzucan Ozgur. *Resources for Turkish Dependency Parsing: Introducing the BOUN Treebank and the BoAT Annotation Tool*. Language Resources and Evaluation 56(1), 2022. [[paper]](https://doi.org/10.1007/s10579-021-09558-0)

**Turkish (UD_Turkish-FrameNet, KeNet, ATIS, Penn, Tourism)**
> Busra Marsan, Neslihan Kara, Merve Ozcelik, Bilge Nas Arican, Neslihan Cesur, Asli Kuzgun, Ezgi Saniyar, Oguzhan Kuyrukcu, and Olcay Taner Yildiz. [Starlang Software](https://starlangyazilim.com/) and [Ozyegin University](https://www.ozyegin.edu.tr/). These treebanks cover diverse domains including FrameNet frames, WordNet examples, airline travel, Penn Treebank translations, and tourism reviews.

**Kazakh (UD_Kazakh-KTB)**
> Aibek Makazhanov, Jonathan North Washington, and Francis Tyers. *Towards a Free/Open-source Universal-dependency Treebank for Kazakh*. TurkLang 2015. [[paper]](https://universaldependencies.org/treebanks/kk_ktb/)

**Uyghur (UD_Uyghur-UDT)**
> Marhaba Eli (Xinjiang University), Daniel Zeman (Charles University), and Francis Tyers. [[treebank]](https://universaldependencies.org/treebanks/ug_udt/)

**Kyrgyz (UD_Kyrgyz-KTMU)**
> Ibrahim Benli. [[treebank]](https://universaldependencies.org/treebanks/ky_ktmu/)

**Ottoman Turkish (UD_Ottoman_Turkish-BOUN)**
> Saziye Betul Ozates, Tarik Emre Tiras, Efe Eren Genc, and Esma Fatima Bilgin Tasdemir. *Dependency Annotation of Ottoman Turkish with Multilingual BERT*. LAW-XVIII, 2024. [[paper]](https://aclanthology.org/2024.law-1.18)

### NER Datasets

**Turkish NER (Starlang)**
> B. Ertopcu, A. B. Kanburoglu, O. Topsakal, O. Acikgoz, A. T. Gurkan, B. Ozenc, I. Cam, B. Avar, G. Ercan, and O. T. Yildiz. *A New Approach for Named Entity Recognition*. UBMK 2017. [[paper]](https://doi.org/10.1109/UBMK.2017.8093439)

**Kazakh NER (KazNERD)**
> Rustem Yeshpanov, Yerbolat Khassanov, and Huseyin Atakan Varol (ISSAI, Nazarbayev University). *KazNERD: Kazakh Named Entity Recognition Dataset*. LREC 2022. [[paper]](https://aclanthology.org/2022.lrec-1.44)

### NLLB Embeddings & Machine Translation

TurkicNLP embeddings backend uses encoder pooling on:

> [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)

Reference:
> NLLB Team, Marta R. Costa-jussà, et al. 2022. *No Language Left Behind: Scaling Human-Centered Machine Translation*. [[paper]](https://arxiv.org/abs/2207.04672)

### Other Organisations

- [Apertium](https://apertium.org/) — morphological transducers covering 20+ Turkic languages
- [SIGTURK](https://sigturk.github.io/) — ACL Special Interest Group on Turkic Languages
- [ISSAI](https://issai.nu.edu.kz/) — Institute of Smart Systems and Artificial Intelligence, Nazarbayev University, for Kazakh NLP resources
- [Universal Dependencies](https://universaldependencies.org/) — the framework and community behind Turkic treebanks
- [Turkic Interlingua](https://github.com/turkic-interlingua) — resources for machine translation for Turkic languages
- [Turkic UD](https://github.com/ud-turkic) - group working on harmonizing Turkic UD treebanks