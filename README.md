<p align="center">
  <img src="images/cover.png" alt="TurkicNLP — Language Family Tree" width="300">
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
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status: Pre-Alpha">
  <img src="https://img.shields.io/badge/languages-24_Turkic-green.svg" alt="24 Turkic Languages">
</p>

## Features

- **24 Turkic languages** from Turkish to Sakha, Kazakh to Uyghur
- **Script-aware from the ground up** — Latin, Cyrillic, Perso-Arabic, Old Turkic Runic
- **Automatic script detection** and bidirectional transliteration
- **Apertium FST morphology** for ~20 languages via Python-native `hfst` bindings (no system install)
- **Stanza/UD integration** — pretrained tokenization, POS tagging, lemmatization, dependency parsing, and NER via [Stanza](https://stanfordnlp.github.io/stanza/) models trained on [Universal Dependencies](https://universaldependencies.org/) treebanks
- **Multiple backends** — choose between rule-based, Apertium FST, or Stanza neural backends per processor
- **License isolation** — library is Apache-2.0; Apertium GPL-3.0 data downloaded separately
- **Stanza-compatible API** — `Pipeline`, `Document`, `Sentence`, `Word`

## Installation

```bash
pip install turkicnlp
```

With optional dependencies:

```bash
pip install turkicnlp[hfst]          # Apertium FST support
pip install turkicnlp[stanza]        # Stanza/UD neural models
pip install turkicnlp[torch]         # PyTorch neural model support
pip install turkicnlp[all]           # Everything
pip install turkicnlp[dev]           # Development tools
```

## Quick Start

```python
import turkicnlp

# Download models for a language
turkicnlp.download("kaz")

# Build a pipeline
nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "pos", "lemma", "depparse"])

# Process text
doc = nlp("Мен мектепке бардым")

# Access annotations
for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text}\t{word.lemma}\t{word.upos}\t{word.feats}")

# Export to CoNLL-U
print(doc.to_conllu())
```

### Using the Stanza Backend

```python
from turkicnlp.processors.stanza_backend import (
    StanzaTokenizer, StanzaPOSTagger, StanzaLemmatizer, StanzaDepParser
)
from turkicnlp.models.document import Document

# Models are downloaded automatically on first use
doc = Document(text="Merhaba dünya.", lang="tur")

for Proc in [StanzaTokenizer, StanzaPOSTagger, StanzaLemmatizer, StanzaDepParser]:
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
from turkicnlp.processors.stanza_backend import StanzaPOSTagger, StanzaDepParser
from turkicnlp.models.document import Document

doc = Document(text="Мен мектепке бардым.", lang="kaz")

# Rule-based tokenizer + Stanza POS/parsing (pretokenized mode)
tokenizer = RegexTokenizer(lang="kaz")
tokenizer.load()
doc = tokenizer.process(doc)

pos = StanzaPOSTagger(lang="kaz")
pos.load()
doc = pos.process(doc)

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

## Supported Languages and Components

<p align="center">
  <img src="images/Turkic_Languages_distribution_map.png" alt="Distribution map of Turkic languages" width="700">
  <br>
  <em>Geographic distribution of Turkic languages (source: <a href="https://commons.wikimedia.org/wiki/File:Turkic_Languages_distribution_map.png">Wikimedia Commons</a>)</em>
</p>

The table below shows all supported languages with their available scripts and processor status.

**Backend legend:**
- **rule** — Rule-based (regex tokenizer, abbreviation lists)
- **Apertium** — Finite-state transducers via [Apertium](https://apertium.org/) + `hfst` ([GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html), downloaded separately)
- **Stanza/UD** — Neural models from [Stanza](https://stanfordnlp.github.io/stanza/) trained on [Universal Dependencies](https://universaldependencies.org/) treebanks ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0))

**Status legend:** ✅ = Available | 🔧 = Planned | — = Not applicable

### Oghuz Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Turkish](https://en.wikipedia.org/wiki/Turkish_language) | `tur` | Latn | ✅ rule, ✅ Stanza/UD | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza |
| [Azerbaijani](https://en.wikipedia.org/wiki/Azerbaijani_language) | `aze` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 | 🔧 | 🔧 | — |
| [Iranian Azerbaijani](https://en.wikipedia.org/wiki/South_Azerbaijani_language) | `azb` | Arab | 🔧 rule_arabic | — | — | — | — | — |
| [Turkmen](https://en.wikipedia.org/wiki/Turkmen_language) | `tuk` | Latn | ✅ rule | ✅ Apertium (beta) | 🔧 | 🔧 | — | — |
| [Gagauz](https://en.wikipedia.org/wiki/Gagauz_language) | `gag` | Latn | ✅ rule | ✅ Apertium (proto) | — | — | — | — |

### Kipchak Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) | `kaz` | Cyrl, Latn | ✅ rule, ✅ Stanza/UD | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza |
| [Kyrgyz](https://en.wikipedia.org/wiki/Kyrgyz_language) | `kir` | Cyrl | ✅ rule, ✅ Stanza/UD | ✅ Apertium | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | — |
| [Tatar](https://en.wikipedia.org/wiki/Tatar_language) | `tat` | Cyrl, Latn | ✅ rule | ✅ Apertium | 🔧 | 🔧 | — | — |
| [Bashkir](https://en.wikipedia.org/wiki/Bashkir_language) | `bak` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — |
| [Crimean Tatar](https://en.wikipedia.org/wiki/Crimean_Tatar_language) | `crh` | Latn, Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — |
| [Karakalpak](https://en.wikipedia.org/wiki/Karakalpak_language) | `kaa` | Latn, Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Nogai](https://en.wikipedia.org/wiki/Nogai_language) | `nog` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Kumyk](https://en.wikipedia.org/wiki/Kumyk_language) | `kum` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Karachay-Balkar](https://en.wikipedia.org/wiki/Karachay-Balkar_language) | `krc` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |

### Karluk Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Uzbek](https://en.wikipedia.org/wiki/Uzbek_language) | `uzb` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 | 🔧 | 🔧 | — |
| [Uyghur](https://en.wikipedia.org/wiki/Uyghur_language) | `uig` | Arab, Latn | 🔧 rule_arabic, ✅ Stanza/UD | ✅ Apertium (beta) | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | — |

### Siberian Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Sakha (Yakut)](https://en.wikipedia.org/wiki/Sakha_language) | `sah` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Altai](https://en.wikipedia.org/wiki/Altai_language) | `alt` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Tuvan](https://en.wikipedia.org/wiki/Tuvan_language) | `tyv` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |
| [Khakas](https://en.wikipedia.org/wiki/Khakas_language) | `kjh` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — |

### Oghur Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Chuvash](https://en.wikipedia.org/wiki/Chuvash_language) | `chv` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — |

### Historical Languages

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER |
|---|---|---|---|---|---|---|---|---|
| [Ottoman Turkish](https://en.wikipedia.org/wiki/Ottoman_Turkish_language) | `ota` | Arab, Latn | ✅ Stanza/UD | — | ✅ Stanza/UD | ✅ Stanza/UD | ✅ Stanza/UD | — |
| [Old Turkish](https://en.wikipedia.org/wiki/Old_Turkic_language) | `otk` | Orkh, Latn | 🔧 rule | — | — | — | — | — |

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

| Language Pair | Direction | Status |
|---|---|---|
| Kazakh Cyrillic ↔ Latin | Bidirectional | ✅ |
| Uzbek Cyrillic → Latin | One-way | ✅ |
| Uyghur Arabic → Latin (ULY) | One-way | ✅ |
| Crimean Tatar Cyrillic → Latin | One-way | ✅ |
| Azerbaijani Cyrillic ↔ Latin | Bidirectional | 🔧 |
| Tatar Cyrillic ↔ Latin | Bidirectional | 🔧 |

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

## Architecture

TurkicNLP follows Stanza's modular pipeline design:

```
Pipeline("tur", processors=["tokenize", "morph", "pos", "depparse"])
    │
    ▼
  Document ─── text: "Ben okula vardım"
    │
    ├── script_detect    → script = "Latn"
    ├── tokenize         → sentences, tokens, words
    ├── morph (Apertium) → lemma, pos, feats (via HFST)
    ├── pos (neural)     → refined UPOS, XPOS, feats
    └── depparse         → head, deprel
    │
    ▼
  Document ─── annotated with all layers
```


```
Pipeline("kaz", processors=["tokenize", "morph", "pos", "depparse"])
    │
    ▼
  Document ─── text: "Мен мектепке бардым"
    │
    ├── script_detect    → script = "Cyrl"
    ├── tokenize         → sentences, tokens, words
    ├── morph (Apertium) → lemma, pos, feats (via HFST)
    ├── pos (neural)     → refined UPOS, XPOS, feats
    └── depparse         → head, deprel
    │
    ▼
  Document ─── annotated with all layers
```

### Key Abstractions

- **Document** → Sentence → Token → Word hierarchy (maps to CoNLL-U)
- **Processor** ABC with `PROVIDES`, `REQUIRES`, `NAME` class attributes
- **Pipeline** orchestrator with dependency resolution and script-aware model loading
- **ProcessorRegistry** for pluggable backends (Apertium, Stanza, neural, rule-based)
- **ModelRegistry** with remote catalog and local caching at `~/.turkicnlp/models/`

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
└── catalog.json

# Stanza models are managed by Stanza at ~/stanza_resources/
```

## License

- **Library code**: [Apache License 2.0](LICENSE)
- **Stanza models**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — managed by Stanza's own download mechanism
- **Apertium FST data**: [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html) — downloaded separately at runtime, never bundled in the pip package

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

## Citation

If you use TurkicNLP in your research, please cite:

```bibtex
@software{turkicnlp,
  title = {TurkicNLP: NLP Toolkit for Turkic Languages},
  author = {Sherzod Hakimov},
  year = {2026},
  url = {https://github.com/turkic-nlp/turkicnlp},
  license = {Apache-2.0},
}
```

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

### Other

- [Apertium](https://apertium.org/) — morphological transducers covering 20+ Turkic languages
- [SIGTURK](https://sigturk.com/) — ACL Special Interest Group on Turkic Languages
- [ISSAI](https://issai.nu.edu.kz/) — Institute of Smart Systems and Artificial Intelligence, Nazarbayev University, for Kazakh NLP resources
- [Universal Dependencies](https://universaldependencies.org/) — the framework and community behind Turkic treebanks
- [Turkic Interlingua](https://github.com/turkic-interlingua) — resources for machine translation for Turkic languages
