# Contributing to TurkicNLP

Thank you for helping TurkicNLP cover more Turkic languages. We welcome code, linguistic knowledge, test data, models,
documentation and bug reports alike. You do not need to be a programmer to help: native-speaker checks of test data
and analyser output are some of the most valuable contributions.

- Ideas and planned work: [ROADMAP.md](ROADMAP.md)
- Questions and discussion: [Discord](https://discord.gg/CeVTbGpmMQ) or [GitHub issues](https://github.com/turkic-nlp/turkicnlp/issues)
- Tutorials: [turkic-nlp-code-samples](https://github.com/turkic-nlp/turkic-nlp-code-samples)

## Contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Adding or improving a language](#adding-or-improving-a-language)
- [Adding a processor or backend](#adding-a-processor-or-backend)
- [Contributing models and data](#contributing-models-and-data)
- [Tests](#tests)
- [Code style](#code-style)
- [Pull requests](#pull-requests)
- [Licensing](#licensing)

## Ways to contribute

| You know… | You can… |
|---|---|
| a Turkic language | check gold segmentations and analyser output, extend closed-class lexicons, transliteration tables, abbreviation lists and suffix tables, report wrong analyses |
| Python | fix bugs, implement roadmap items, add backends, improve tests and documentation |
| ML / NLP | train and evaluate models (NER, parsers, tokenizers, ASR), add benchmarks |
| data | point us to (or release) corpora, treebanks, dictionaries and speech data with clear licences |

Good first issues: items marked **S** in [ROADMAP.md](ROADMAP.md), wrong analyses you find for your language, and
missing examples in the README or the code samples.

### Reporting a bug or a wrong analysis

Please include:

- the TurkicNLP version (`python -c "import turkicnlp; print(turkicnlp.__version__)"`), Python version and OS
- which environment you use (text or speech + text, see the [README](README.md#installation))
- a minimal snippet, the input text, the output you got and the output you expected

For linguistic errors, also give a gloss or translation and, if possible, a reference (grammar, dictionary, treebank).

## Development setup

```bash
git clone https://github.com/turkic-nlp/turkicnlp.git
cd turkicnlp

# Text environment (all text components)
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[all,dev]"

# Optional: speech + text environment (separate, see README → Installation)
python3.10 -m venv venv-asr
source venv-asr/bin/activate
pip install -e ".[all,asr,dev]"
```

To reproduce the exact tested versions, install `requirements/lock-text.txt` or `requirements/lock-speech.txt`
first and then `pip install -e .`.

Models are downloaded on first use into `~/.turkicnlp/models/` (Stanza models into `~/stanza_resources/`,
Omnilingual ASR checkpoints into `~/.cache/fairseq2/assets/`).

## Project layout

```
turkicnlp/
├── pipeline.py              Pipeline, processor order, default processors
├── asr.py                   SpeechRecognizer (Omnilingual ASR)
├── language_id.py           LanguageDetection (GlotLID)
├── models/document.py       Document → Sentence → Token → Word, CoNLL-U export
├── processors/              one module per processor family (tokenizer, morphology, stanza_backend,
│                            multilingual_*, morpheme_tokenizer, sp_tokenizer, translate, embeddings, …)
├── scripts/                 Script enum, per-language ScriptConfig, detector, transliterator
├── resources/
│   ├── catalog.json         which processors/backends exist per language and script, download URLs
│   ├── registry.py          ProcessorRegistry (backends) and ModelRegistry (paths, catalog)
│   ├── downloader.py        turkicnlp.download()
│   ├── tag_mappings/        Apertium → UD tag mapping, one module per language
│   ├── lexicons/            closed-class lexicons used by the Apertium processor
│   ├── mwt_rules/           multi-word token rules
│   └── morpheme_rules.json  suffix allomorph tables for the MorphemeTokenizer
├── tools/                   corpus annotation and other utilities
└── tests/                   pytest suite (runs without model downloads)
```

## Adding or improving a language

A new language usually needs the following pieces, roughly in this order. Each can be its own pull request.

1. **Script configuration.** Add a `ScriptConfig` entry in `turkicnlp/scripts/__init__.py`: the available scripts,
   the primary script and the Apertium script.
2. **Transliteration.** Add mapping tables to `TRANSLITERATION_TABLES` in `turkicnlp/scripts/transliterator.py`, including the Common Turkic
   Alphabet direction. Add round-trip tests to `tests/test_transliterator.py`.
3. **Tokenization.** Add abbreviations (in `processors/tokenizer.py`) and multi-word token rules
   (`resources/mwt_rules/<lang>.json`) where they apply.
4. **Catalog entry.** Add the language to `resources/catalog.json` with one block per script and processor. Each
   processor lists its backends and a `default`.
5. **Morphology.**
   - If an Apertium analyser exists, publish the compiled FST through the `apertium-data` releases and add its URL
     to the catalog.
   - Add a tag mapper in `resources/tag_mappings/<lang>.py`. Subclass `CommonTurkicTagMapper` and override only
     what differs, then return it from `load_tag_map()` in `resources/tag_mappings/__init__.py`.
   - Add a closed-class lexicon `resources/lexicons/<lang>.json` (see `resources/lexicons/README.md` for the schema).
6. **Morpheme segmentation.** Add the suffix allomorph table to `resources/morpheme_rules.json`, plus test words for
   each category to the morph-segmentation test set.
7. **Neural models and ASR.**
   - Zero-shot support in the multilingual Glot500 models needs a proxy-language entry
     (`processors/multilingual_morph_model.py`).
   - ASR needs an Omnilingual language code in `asr.py` (`OMNIASR_LANG_CODES`).
8. **README.** Add the language to the support tables.

When you change an Apertium tag mapping, check the actual analyser output first. For example, `<ifi>` is the -DI past
in every Turkic analyser, not an evidential. Run the language's sentences through the pipeline before and after the
change.

## Adding a processor or backend

1. Subclass `turkicnlp.processors.base.Processor`:

   ```python
   from turkicnlp.processors.base import Processor

   class MyNERProcessor(Processor):
       NAME = "ner"               # processor name used in Pipeline(processors=[...])
       PROVIDES = ["ner"]         # annotation layers it writes
       REQUIRES = ["tokenize"]    # layers that must exist before it runs

       def load(self, model_path: str) -> None:
           ...                    # load weights; import heavy dependencies here, not at module level
           self._loaded = True

       def process(self, doc):
           ...                    # annotate doc in place
           doc._processor_log.append("ner:my_backend")
           return doc
   ```

2. Register the backend in `_register_builtins()` in `resources/registry.py`:
   `ProcessorRegistry.register("ner", "my_backend", MyNERProcessor)`. Guard optional dependencies with
   `try/except ImportError` so the core package still imports.
3. If it is a new processor name, add it to `PROCESSOR_ORDER` in `pipeline.py`. If it downloads a large model or
   needs an optional extra, also add it to `OPTIONAL_PROCESSORS` so `Pipeline(lang)` does not load it by default.
4. Add catalog entries for the languages it supports, and an optional extra in `pyproject.toml` if it needs new
   dependencies.
5. Configuration reaches the processor as `Pipeline(..., <processor>_<key>=value)`. For example,
   `translate_tgt_lang="eng"` arrives as `self.config["tgt_lang"]`.
6. Add tests with the model mocked (see `tests/test_translate.py` and `tests/test_asr.py`), a README section with a
   runnable example, and ideally a notebook in the code-samples repository.

## Contributing models and data

- **Hosting.** Host model weights in a GitHub release (as the custom Stanza and Glot500 models are) or on the Hugging
  Face Hub under the `turkicnlp` organisation, with names like `turkicnlp/<lang>-<task>`. Never commit weights to
  this repository. The catalog points to the hosted file.
- **Model cards.** Every model needs a card covering:
  - training data and licence
  - languages and scripts
  - evaluation scores on a public test set
  - known limitations
- **Datasets.** Datasets need a dataset card with the source, licence, preprocessing and splits. For scraped text,
  check the source's terms before releasing it.
- **Evaluation.** Evaluation scripts for the paper live in the companion repository (`coling-2027-code-base`). Please
  report new results in the same format (dated output folders, per language and category).

## Tests

```bash
python -m pytest                       # use "python -m" so the pytest of the active environment runs
python -m pytest turkicnlp/tests/test_morphology.py -k fst   # a subset
python -m pytest --cov=turkicnlp
```

- The suite runs without downloading models: neural backends and the ASR model are mocked. Keep it that way, and mark
  tests that need real models with `@pytest.mark.slow`.
- Run the suite in both environments if you touch shared code (`pipeline.py`, `models/`, `processors/base.py`) or ASR.
- Add a regression test for every bug you fix.

## Code style

- Python ≥ 3.9 syntax with `from __future__ import annotations`, type hints on public functions, and Google-style
  docstrings.
- Format and lint before committing:

  ```bash
  black turkicnlp
  ruff check turkicnlp
  mypy turkicnlp            # optional, not yet enforced in CI
  ```

  The line length is 100.
- Import heavy libraries (torch, transformers, stanza, hfst, omnilingual_asr) lazily inside `load()`, so
  `import turkicnlp` stays fast and works with the core install.
- Keep language data (tables, lexicons, rules) in `resources/` as JSON or per-language modules, not hard-coded in
  processors.

## Pull requests

1. Open or comment on an issue first for anything larger than a small fix, so we can agree on the approach.
2. Create a branch from `main` (`fix/krc-accent-lookup`, `feat/ner-glot500`, `lang/salar`).
3. Keep each pull request focused. Include tests, update the README for user-visible changes, and update
   [ROADMAP.md](ROADMAP.md) if you complete or add an item.
4. Describe what changed, how you tested it, and, for linguistic changes, before/after output for a few sentences.
5. A maintainer reviews the pull request. CI must pass (installation tests across OS × Python × extras).

## Licensing

- By contributing code you agree that it is released under the [Apache License 2.0](LICENSE).
- Apertium data is GPL-3.0 and is **downloaded at runtime, never bundled**. Do not copy GPL data or code into this
  repository.
- Only add data and models whose licence allows redistribution. Name the licence in the catalog entry
  (`"license": ...`) and in the model or dataset card.
- Cite the original authors of the resources you wrap, in the README Acknowledgements section.
