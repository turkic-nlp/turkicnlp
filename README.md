<p align="center">
  <img src="images/cover.png" alt="TurkicNLP — Language Family Tree" width="300">
</p>

<h1 align="center">TurkicNLP</h1>

<p align="center">
  <strong>NLP toolkit for 20+ Turkic languages</strong> — a pip-installable Python library inspired by <a href="https://stanfordnlp.github.io/stanza/">Stanza</a>, with adaptations for the low-resource, morphologically rich Turkic language family.
</p>

<p align="center">
  Developed by <a href="https://sherzod-hakimov.github.io/">Sherzod Hakimov</a>
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
- **Neural processors** — POS tagging, dependency parsing, NER, sentiment analysis
- **License isolation** — library is Apache-2.0; Apertium GPL-3.0 data downloaded separately
- **Stanza-compatible API** — `Pipeline`, `Document`, `Sentence`, `Word`

## Installation

```bash
pip install turkicnlp
```

With optional dependencies:

```bash
pip install turkicnlp[hfst]          # Apertium FST support
pip install turkicnlp[torch]         # Neural model support
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

The table below shows all supported languages with their available scripts and processor status. Components marked with the Apertium FST backend are available via GPL-3.0 licensed data downloaded separately.

**Legend:** ✅ = Implemented | 🔧 = Planned | — = Not applicable

### Oghuz Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Turkish](https://en.wikipedia.org/wiki/Turkish_language) | `tur` | Latn | ✅ rule, 🔧 neural | ✅ Apertium | 🔧 neural | 🔧 neural | 🔧 neural | 🔧 neural | 🔧 neural |
| [Azerbaijani](https://en.wikipedia.org/wiki/Azerbaijani_language) | `aze` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 neural | 🔧 neural | 🔧 neural | 🔧 neural | — |
| [Iranian Azerbaijani](https://en.wikipedia.org/wiki/South_Azerbaijani_language) | `azb` | Arab | 🔧 rule_arabic | — | — | — | — | — | — |
| [Turkmen](https://en.wikipedia.org/wiki/Turkmen_language) | `tuk` | Latn | ✅ rule | ✅ Apertium (beta) | 🔧 neural | 🔧 neural | — | — | — |
| [Gagauz](https://en.wikipedia.org/wiki/Gagauz_language) | `gag` | Latn | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |

### Kipchak Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Kazakh](https://en.wikipedia.org/wiki/Kazakh_language) | `kaz` | Cyrl, Latn | ✅ rule, 🔧 neural | ✅ Apertium | 🔧 neural | 🔧 neural | 🔧 neural | 🔧 neural | 🔧 neural |
| [Kyrgyz](https://en.wikipedia.org/wiki/Kyrgyz_language) | `kir` | Cyrl | ✅ rule | ✅ Apertium | 🔧 neural | 🔧 neural | — | — | — |
| [Tatar](https://en.wikipedia.org/wiki/Tatar_language) | `tat` | Cyrl, Latn | ✅ rule | ✅ Apertium | 🔧 neural | 🔧 neural | — | — | — |
| [Bashkir](https://en.wikipedia.org/wiki/Bashkir_language) | `bak` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | — |
| [Crimean Tatar](https://en.wikipedia.org/wiki/Crimean_Tatar_language) | `crh` | Latn, Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | — |
| [Karakalpak](https://en.wikipedia.org/wiki/Karakalpak_language) | `kaa` | Latn, Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Nogai](https://en.wikipedia.org/wiki/Nogai_language) | `nog` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Kumyk](https://en.wikipedia.org/wiki/Kumyk_language) | `kum` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Karachay-Balkar](https://en.wikipedia.org/wiki/Karachay-Balkar_language) | `krc` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |

### Karluk Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Uzbek](https://en.wikipedia.org/wiki/Uzbek_language) | `uzb` | Latn, Cyrl | ✅ rule | ✅ Apertium | 🔧 neural | 🔧 neural | 🔧 neural | — | — |
| [Uyghur](https://en.wikipedia.org/wiki/Uyghur_language) | `uig` | Arab, Latn | 🔧 rule_arabic, ✅ rule (Latn) | ✅ Apertium (beta) | — | — | — | — | — |

### Siberian Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Sakha (Yakut)](https://en.wikipedia.org/wiki/Sakha_language) | `sah` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Altai](https://en.wikipedia.org/wiki/Altai_language) | `alt` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Tuvan](https://en.wikipedia.org/wiki/Tuvan_language) | `tyv` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |
| [Khakas](https://en.wikipedia.org/wiki/Khakas_language) | `kjh` | Cyrl | ✅ rule | ✅ Apertium (proto) | — | — | — | — | — |

### Oghur Branch

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Chuvash](https://en.wikipedia.org/wiki/Chuvash_language) | `chv` | Cyrl | ✅ rule | ✅ Apertium (beta) | — | — | — | — | — |

### Historical Languages

| Language | Code | Script(s) | Tokenize | Morph (FST) | POS | Lemma | DepParse | NER | Sentiment |
|---|---|---|---|---|---|---|---|---|---|
| [Ottoman Turkish](https://en.wikipedia.org/wiki/Ottoman_Turkish_language) | `ota` | Arab, Latn | 🔧 rule_arabic | — | — | — | — | — | — |
| [Old Turkish](https://en.wikipedia.org/wiki/Old_Turkic_language) | `otk` | Orkh, Latn | 🔧 rule | — | — | — | — | — | — |

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
- **ProcessorRegistry** for pluggable backends (Apertium, neural, rule-based)
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
```

## License

- **Library code**: [Apache License 2.0](LICENSE)
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

- [Stanza](https://stanfordnlp.github.io/stanza/) — for the architectural inspiration
- [Apertium](https://apertium.org/) — for morphological transducers covering 20 Turkic languages
- [SIGTURK](https://sigturk.com/) — ACL Special Interest Group on Turkic Languages
- [ISSAI](https://issai.nu.edu.kz/) — for Kazakh NLP resources
- [Universal Dependencies](https://universaldependencies.org/) — for Turkic treebanks
- [Turkic Interlingua](https://github.com/turkic-interlingua) - Resources for Machine Translation for Turkic Languages
