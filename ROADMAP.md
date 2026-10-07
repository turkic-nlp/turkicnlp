# TurkicNLP Roadmap

This page collects the planned work and the open ideas for TurkicNLP in one place. It replaces the separate planning
notes that used to live in this repository and in related project folders (see [Where these items come from](#where-these-items-come-from)).

Want to work on something? Comment on the matching GitHub issue (or open one), then follow [CONTRIBUTING.md](CONTRIBUTING.md).
Items are grouped by area; inside each group they are roughly ordered by value for effort.

**Effort:** **S** = days, **M** = weeks, **L** = months. **Status:** `[ ]` open · `[~]` in progress / partly done · `[x]` done.

---

## Contents

1. [Where things stand](#1-where-things-stand)
2. [Known issues](#2-known-issues)
3. [Next release](#3-next-release)
4. [Morphology, lemmas and lexicons](#4-morphology-lemmas-and-lexicons)
5. [Tokenization for language models](#5-tokenization-for-language-models)
6. [Syntax and semantics](#6-syntax-and-semantics)
7. [Speech](#7-speech)
8. [Translation and embeddings](#8-translation-and-embeddings)
9. [Text normalization and spelling](#9-text-normalization-and-spelling)
10. [Datasets and the Hugging Face Hub](#10-datasets-and-the-hugging-face-hub)
11. [Evaluation and benchmarks](#11-evaluation-and-benchmarks)
12. [Tooling, API and documentation](#12-tooling-api-and-documentation)
13. [GenAI-era components](#13-genai-era-components)
14. [External resources worth wrapping](#14-external-resources-worth-wrapping)
15. [Related projects that feed the toolkit](#15-related-projects-that-feed-the-toolkit)
16. [Language priorities](#16-language-priorities)
17. [Community and outreach](#17-community-and-outreach)

---

## 1. Where things stand

| Area | Status |
|---|---|
| Languages | 24 (Oghuz, Kipchak, Karluk, Siberian, Oghur, Arghu, historical) |
| Scripts | Latin, Cyrillic, Perso-Arabic, Old Turkic runic; Common Turkic Alphabet (CTS) for 23 languages; automatic script detection |
| Tokenization | Rule-based (all), Stanza (tur, kaz, kir, uig + custom aze, uzb, tuk, tat, bak), rule-based multi-word token expansion |
| Morphology | Apertium HFST for 20 languages; Glot500 neural morph analyzer + lemmatizer for 23 languages |
| Morpheme segmentation | Hybrid Apertium + neural `MorphemeTokenizer` (87% exact segmentation on a 483-word, 20-language test set) |
| Subword tokenization | `sp_tokenize`: SentencePiece with morpheme pre-segmentation (models not yet trained/released) |
| POS / dependencies | Stanza (official + custom) and multilingual Glot500 parser (10 trained + 5 zero-shot languages) |
| NER | Stanza, Turkish and Kazakh only |
| Language ID | GlotLID, restricted to Turkic labels by default |
| Translation / embeddings | NLLB-200 (600M), 11 languages |
| Speech | Omnilingual ASR (`SpeechRecognizer`), 20 languages, separate environment |
| Placeholders | `sentiment`, and the `neural` backends of `tokenize`, `pos`, `lemma`, `depparse`, `ner` are registered stubs |

## 2. Known issues

- [ ] **Neural fallback of the morpheme tokenizer is weak.** It misplaces stem boundaries in verbs and almost never splits derivational suffixes (12% exact on voice/derivation). **M**
- [ ] **Low Apertium coverage** for Azerbaijani (common verbs such as *getdi*, *gəldi* are missing), Khakas, Altai, Karakalpak and Karachay-Balkar. Contribute upstream to Apertium; meanwhile use a lexicon fallback (see 4.1). **M–L**
- [ ] **Disambiguation errors** in the Apertium processor:
  - Temporal adverbs are read as verbs: bak *кисә*, uzb *kecha* (kaz *кеше* is handled by the lexicon).
  - Noun readings win over verb readings: tur/gag *yarın* → `NOUN|Case=Gen`, gag *bekler* → noun plural.
  - Proper nouns are tagged wrongly: bak *Беҙ* → PROPN instead of PRON, *Өс* → PROPN instead of NUM.
  - Kyrgyz *балалар* gets only a verb reading. **S–M**
- [ ] **Morpheme-segmentation numbers are optimistic.** The test set was also used during development, and gold data for 9 languages is unverified. Add a held-out split and native-speaker checks (see 11). **M**
- [ ] **SentencePiece boundary marker.** The default morpheme boundary character in `sp_tokenize` can collide with SentencePiece's `▁`. The current default is kept for compatibility, so pick a safe default before the first model release. **S**
- [ ] **Two environments are needed.** Speech recognition pins torch 2.8 / numpy < 2 / transformers 4.x (through fairseq2), so it cannot share an environment with transformers 5.x. Revisit when `omnilingual-asr` relaxes its pins. **S**
- [ ] **Python version range.** The text environment with transformers 5.x needs Python ≥ 3.10. The core package still installs on 3.9, so decide whether to drop 3.9. **S**

Fixed recently:

- [x] **Apertium transducer file selection** was arbitrary when a release ships several files. For example, Karachay-Balkar loaded `krc@Seegmiller` (a romanisation), so every Cyrillic word came out `X`; `kaz@Arab` and `uzb_guesser` were also possible picks.
- [x] **Stress accents** blocked FST lookup (krc *барды́*).
- [x] **`<ifi>` (the -DI past) was mapped to `Evident=Nfh`.** It now maps to `Tense=Past` (Turkish: `Evident=Fh`), and Turkish `<past>` (-mIş) maps to `Evident=Nfh|Tense=Past`.
- [x] **Unmapped Apertium tags:**
  - converbs (`gna_*`, `prc_*`, `cvb`) → `VerbForm=Conv`
  - participles (`gpr_*`) → `VerbForm=Part`
  - verbal nouns (`ger_*`) → `VerbForm=Vnoun`
  - question clitic (`qst`, `encl`) → `PART|PartType=Int`
  - copula (`cop`) → `AUX`
- [x] **NLLB HTML entities.** Outputs contained `&apos;` and similar entities; they are now unescaped.
- [x] **`Pipeline(lang)` with no processors** loaded every catalog entry, including NLLB, ASR, both morphology analyzers and embeddings. It now loads the core set; optional processors must be requested.
- [x] Khalaj (`klj`) script config; `transformers` extra.

## 3. Next release

- [ ] Regression run of the Apertium processor on the challenge sentences after the tag-mapping fixes. Diff the old and new outputs, and keep that diff script in `scripts/` for future runs. **S**
- [ ] Disambiguation adjacency rules: postposition after a case-marked noun, copula after a noun/adjective, and a PROPN boost for capitalised known names. **S**
- [ ] Old Turkic (`otk`): either add morphology/POS (e.g. from the Old Turkic UD treebank) or state clearly in the docs that `otk` is transliteration + tokenization only. **S**
- [ ] Remove or hide the stub backends (`sentiment`, `neural` tokenize/pos/lemma/depparse/ner) until they are implemented, so users cannot select them by accident. **S**
- [ ] `CHANGELOG.md` and tagged releases on PyPI with release notes. **S**
- [ ] ASR in CI (mocked tests already exist; add a job with the speech lock file). **S**

## 4. Morphology, lemmas and lexicons

### 4.1 Wiktionary lookup lemmatizer — **S–M**
- [ ] Build per-language form → (lemma, UPOS, feats) indices from the kaikki/Wiktextract dump. The dump has 118K entries and 4.8M inflected forms across 20 languages.
- [ ] Host the indices like the Apertium FSTs, with catalog entries and `download()` support. Use a compressed trie/FSA for Turkish and Azerbaijani.
- [ ] Wire them in as a `lemma` backend `wiktionary` and as the fallback in `ApertiumMorphProcessor._fallback_for_unknown()`. Use the order FST → Wiktionary → neural.
- [ ] Use attested (lemma, POS) pairs as a signal in `_disambiguate()`.

### 4.2 Closed-class lexicons — **S**
- [ ] Extend `resources/lexicons/<lang>.json` from Wiktionary: pronouns, postpositions, particles, numerals and interjections. This reduces `X` tags.
- [ ] Add lexicons for the three languages without one (`klj`, `ota`, `otk`).

### 4.3 FST generation (inflection) — **S**
- [ ] Expose the downloaded `*.autogen.hfst` generators: `inflect("кітап", ["n", "pl", "px1sg", "dat"]) → "кітаптарыма"`. Useful for data augmentation, paradigm tables and spelling suggestions.

### 4.4 Better neural morphology — **L**
- [ ] Train a stronger neural segmenter/lemmatizer on Wiktionary + UniMorph paradigms (character-level BiLSTM for CPU use, or ByT5). It becomes tier 3 behind FST and lookup.
- [ ] Teach the neural fallback derivational suffixes (causative, passive, *-lIk*, *-CI*, *-sIz*, *-lI*).

### 4.5 Upstream Apertium work — **M–L**
- [ ] Report or add the missing lexemes for kjh, krc, alt, aze and kaa (the X-tag analysis lists the most frequent misses).
- [ ] Consider GiellaLT `lang-kjh` (Khakas FST + speller) and MorAz (Azerbaijani FST) as alternative analyzers.

## 5. Tokenization for language models

- [ ] **Train and release SentencePiece models.** Annotate large corpora with `tools/annotate_corpus.py` and add a quality-threshold check, then train monolingual models plus a shared `sp_turkic_64k`. **M–L**
- [ ] **Benchmark them** against XLM-R, LLaMA 3, Qwen, OLMo and mBERT. Measure fertility on FLORES-200, morpheme-boundary alignment and cross-language parity (cf. the [tilde tokenizer benchmark](https://tilde-nlp.github.io/tokenizer-bench.html)). **M**
- [ ] **Export as HF tokenizers.** Use `PreTrainedTokenizerFast`, with model cards and one Hub repo per language. **S**
- [ ] **CTS-based tokenizers.** Converting Cyrillic/Arabic text to the Common Turkic Alphabet already cuts fertility by 25–50% for 13 languages. Release a CTS-normalising tokenizer wrapper. **S**

## 6. Syntax and semantics

- [ ] **Multilingual NER beyond tur/kaz.** Put a head on the Glot500 backbone, trained on WikiANN (silver) plus gold sets: KazNERD, KyrgyzNER, Uzbek NER, the Uyghur NER dataset and Turkish Starlang. Target languages are az, kk, ky, uz, ug, tt, ba, tk, cv and tr. **M**
- [ ] **Stanza models from silver UD data.** Retrain aze/tat/bak from `generated-ud-data` with the same recipe as the uzb/tuk parsers, and report gold-test scores where they exist. **M**
- [ ] **Turkmen UD treebank.** Once `UD_Turkmen-TUD` is accepted into UD, switch the Turkmen parser to it (see 15). **M**
- [ ] **Unified multi-task parser.** Build one checkpoint for POS + dependencies across 8+ languages, frozen backbone with script adapters (plan in `train-unified-models`). Compare NLLB-encoder vs Glot500 backbones. **L**
- [ ] **Cross-lingual transfer framework.** Do Turkish → low-resource transfer, annotation projection through word alignment (awesome-align / SimAlign over parallel Turkic corpora), and LLM pseudo-labelling. **L**
- [ ] **Semantic role labelling** via transfer from Turkish PropBank/FrameNet. **L**
- [ ] Coreference resolution, relation/event extraction. **L**

## 7. Speech

- [ ] **ASR evaluation.** Report WER/CER per language on FLEURS and Common Voice for the Omnilingual models (300M/1B/3B, CTC vs LLM). **M**
- [ ] **Lightweight / offline ASR backend.** Fine-tune Whisper (or Qwen3-ASR) with LoRA on Mozilla Data Collective / Common Voice data and export to sherpa-onnx INT8. Auto-download from the HF Hub and run on CPU, Windows, Android and Raspberry Pi. It would cover Tuvan, which Omnilingual lacks. **L**
- [ ] **More ASR backends.** IS2AI Söyle (Whisper ONNX, 11 languages incl. Turkmen) and TurkicASR (ESPnet, 10 languages). **M**
- [ ] **Timestamps.** Forced alignment (e.g. Qwen3-ForcedAligner or CTC alignment) for word-level timing. **M**
- [ ] **Text-to-speech.** Wrap IS2AI TurkicTTS (10 languages, reuses transliteration to a shared alphabet) with Meta MMS-TTS as fallback, and add a `synthesize()` API symmetric to `SpeechRecognizer`. **M**
- [ ] **Speech corpora loaders.** KSC2, TSC, USC, TatSC, TurkmenSpeech, FLEURS, Common Voice, chuvash_voice (see 10). **S–M**

## 8. Translation and embeddings

- [ ] **More MT backends** for languages NLLB lacks or translates poorly:
  - Apertium RBMT pairs (tur–gag, tur–crh, kaz–tat, kaz–kaa, kum, tyv)
  - MADLAD-400 (az, ba, crh, kk, ky, tk, tr, tt, ug, uz)
  - ISSAI Tilmash (Kazakh)
  - Dilmash (Karakalpak)
  - NLLB-1.3B as a quality option

  **M**
- [ ] **MT quality estimation** (COMETKiwi) and a post-editing friendly output (scores per sentence). **M**
- [ ] **Script bridging for MT.** Transliterate to the script NLLB was trained on before translating, then back. **S**
- [ ] **Alternative sentence encoders** as `embeddings` backends: LaBSE and LASER-3 per-language encoders (azb, azj, bak, crh, kaz, kir, tat, tuk, tur, uig, uzn). NLLB mean-pooling is not trained for similarity. **S**
- [ ] **Retrieval-tuned embedder** fine-tuned on Turkic parallel/paraphrase pairs (TIL, OPUS, X-WMT). **M**
- [ ] **Word vectors:** fastText crawl vectors, Kuriyozov's aligned Turkic embeddings, ConceptNet Numberbatch. **S**

## 9. Text normalization and spelling

- [ ] **Spell checker.** Use `hfst-ospell` over the existing Apertium FSTs (~20 languages) as a fast baseline. The Glot500 seq2seq model from `train-spell-checker` comes later as a neural backend, after regenerating its corpus at scale and reporting real per-language scores. **M**
- [ ] **Normalization:**
  - Turkish deasciifier (Zemberek/VNLP style)
  - Kazakh noisy-text normalizer (KazNLP)
  - Uzbek apostrophe/okina normalisation
  - Russian–Turkic code-switch detection

  **M**
- [ ] **Mixed-script detection** at token level (e.g. Kazakh Cyrillic with Latin words). **S**
- [ ] **Uyghur multi-script converter coverage** (ULS/UAS/CTS/UCS/Yengi Yeziq) and Kazakh 1929/1940 historical alphabets. **S**
- [ ] **OCR** for Turkic scripts (UyghurOCR; Ottoman Arabic script). **L**

## 10. Datasets and the Hugging Face Hub

- [ ] **`turkicnlp.datasets` loaders** with one schema (CoNLL-U fields + `lang`, `script`, `split`, `license`). Cover:
  - UD Turkic treebanks + the parallel UD sets (ud-turkic, Akhundjanova et al. 2025)
  - TueCL test sets
  - FLORES+, TIL-MT, X-WMT, KazParC
  - WikiANN, KazNERD
  - TUMLU, Kardeş-NLU, SIB-200
  - Common Voice, FLEURS
  - the project's own news corpora, silver UD treebanks and spell-check corpus (license review needed for scraped news)

  **M**
- [ ] **HF Hub organisation conventions:**
  - Naming: `turkicnlp/<lang>-<task>` for datasets, `turkicnlp/<lang>-<model>` for models.
  - Every repo gets a dataset/model card with citation and license.
  - Add contribution guidelines for uploads. **S**
- [ ] **First dataset uploads:** Turkish dependency treebank, Azerbaijani POS, Kazakh NER, Uzbek morphology, Uyghur NER/POS, and a cross-script (Latin/Cyrillic/Arabic) parallel corpus. **M**
- [ ] **Model mirror on the Hub.** Mirror the Glot500 checkpoints and custom Stanza models (today they live in GitHub releases) with model cards. **S**

## 11. Evaluation and benchmarks

- [ ] **`turkicnlp.evaluate` + CLI.** Turn the paper scripts into package features:
  - UPOS/UFeats/LAS on UD test sets
  - morph-segmentation exact/boundary F1
  - transliteration round-trip CER
  - LID accuracy
  - tokenizer fertility
  - chrF/COMET for MT
  - WER/CER for ASR

  **M**
- [ ] **Benchmark tables per language and processor.** Publish them in the documentation and regenerate them per release. **S** once the harness exists.
- [ ] **Baselines:** UDPipe 2, TurkishDelightNLP and Zemberek for tur, LaBSE/LASER for embeddings, FLORES chrF for MT, cross-script retrieval on MIRACL. **M**
- [ ] **Held-out morph-segmentation test set** plus native-speaker verification for the 9 low-confidence languages:
  - gag, nog, kum, krc
  - sah, alt, tyv, kjh, chv

  **M**
- [ ] **Leaderboard / lm-evaluation-harness tasks** for Turkic (TUMLU, Kardeş-NLU, morphology probes). **L**
- [ ] **Profiling:** throughput and memory per processor on CPU/GPU/MPS. **S**

## 12. Tooling, API and documentation

- [ ] **Command-line interface:**
  - `turkicnlp annotate input.txt --lang kaz --processors tokenize,morph -o out.conllu`
  - `turkicnlp translate`, `turkicnlp transcribe`, `turkicnlp download`

  **M**
- [ ] **Optional REST server** (FastAPI + Docker image) for annotation, translation and ASR. **M**
- [ ] **Documentation site** (mkdocs-material + mkdocstrings):
  - API reference for processor names, `*_backend` values and config keys
  - a single support matrix
  - model cards

  **M**
- [ ] **Batch / streaming processing** of large corpora with multiprocessing and GPU batching for neural processors. **M**
- [ ] **spaCy component wrapper** and a Stanza-compatible `Document` export, for interoperability. **S**
- [ ] **Type hints and mypy** in CI; ruff + black pre-commit. **S**
- [ ] **Notebooks:** add `sp_tokenize`, GlotLID, NLLB-as-pipeline and, once shipped, the spell checker and the Wiktionary lemmatizer to `turkic-nlp-code-samples`. Fix the duplicated notebook numbers and the transliteration guide examples there. **S**

## 13. GenAI-era components

These come from the strategic component review. They are larger projects that build on the toolkit rather than live inside it.

- [ ] **RAG building blocks:**
  - morphology-aware chunking at sentence boundaries
  - lemma-normalised indexing with Apertium
  - Turkic retrieval pairs generated from parallel corpora
  - a cross-encoder reranker
  - a BEIR-style Turkic retrieval benchmark

  **L**
- [ ] **Synthetic data pipeline:**
  - LLM generation with perplexity (kenlm) and MinHash filtering
  - round-trip consistency checks
  - teacher → student silver labelling for NER/POS/sentiment

  **L**
- [ ] **Instruction-tuning data:** translate and adapt Alpaca/OpenHermes-style data with NLLB + LLM post-editing, plus community-collected native instructions and preference pairs. **L**
- [ ] **Sentiment and hate-speech classifiers.**
  - Sentiment: KazSAnDRA (kk), BERTurk sentiment (tr) and Uzbek sentiment, plus one multilingual head for zero-shot transfer.
  - Hate speech: culturally specific taxonomies, with Kazakh/Russian code-switching test sets. **M–L**
- [ ] **Question answering, summarization, dialogue, and domain LMs** (legal, medical). Mostly evaluation sets and adapters; the toolkit supplies preprocessing. **L**

## 14. External resources worth wrapping

| Resource | What it adds | Languages |
|---|---|---|
| Zemberek, TRmorph, VNLP | Turkish morphology, deasciifier, spelling | tr |
| MorAz | Azerbaijani FST | az |
| GiellaLT lang-kjh | Khakas FST + speller | kjh |
| THUUyMorph | Uyghur morphology | ug |
| UniMorph | Inflection paradigms | ~15 |
| KeNet, UzWordNet | Wordnets | tr, uz |
| BERTurk, KazBERT/Kaz-RoBERTa, UzBERT/BERTbek, KyrgyzBERT, aLLMA | Monolingual encoders as feature extractors | tr, kk, uz, ky, az |
| KazNERD, KyrgyzNER, Uzbek NER, uyghur_ner_dataset, HisTR | NER training data | kk, ky, uz, ug, ota |
| KazSAnDRA, Uzbek ABSA | Sentiment data | kk, uz |
| KazQAD | QA data | kk |
| TIL corpus, X-WMT, KazParC, Dilmash, chuvash_parallel | Parallel data | 22+ |
| LASER-3, LaBSE, MADLAD-400, Tilmash, OPUS-MT | Encoders and MT | various |
| TurkicTTS, KazakhTTS2, Söyle, TurkicASR | Speech | 10–11 |
| TSC, KSC2, USC, TatSC, TurkmenSpeech, FLEURS, Common Voice | Speech corpora | various |
| TUMLU, Kardeş-NLU, SIB-200, Mukayese | Benchmarks | various |
| ITU NLP resources ([ddi.itu.edu.tr](https://ddi.itu.edu.tr/en/toolsandresources)) | Turkish tools and data | tr |
| OpenLID-v2 | Second LID model to compare with GlotLID | all |

Check licences and current versions before wrapping. Several survey notes predate these resources' latest releases.

## 15. Related projects that feed the toolkit

| Project | State | What the toolkit gets |
|---|---|---|
| `apertium-data` | CI compiles FSTs and publishes release zips + catalog snippets | Morphology downloads (in use) |
| `generated-ud-data` | Silver UD treebanks for az/tk/tt/ba (LLM translation of Turkish treebanks + auto-annotation) | Custom Stanza models (in use); retraining (6) |
| `generated-ud-data/tuk` | `UD_Turkmen-TUD` submission: splits done, ~5,900 validation errors left, features/aux/deprels to register in the UD validator; next UD release deadline 1 Nov | Gold-ish Turkmen treebank and parser |
| `generate-ud-annotations` | LLM translate-and-annotate CLI | Silver data for more languages |
| `train-unified-models` | Design + skeleton code for a frozen-backbone multi-task parser | Unified parser backend (6) |
| `train-unified-models/Wiktionary` | Kaikki dump analysed: 118K entries, 4.8M forms, 20 languages | Lookup lemmatizer, lexicons, neural morph data (4) |
| `train-spell-checker` | Synthetic-error corpus (19.5K records, 13 languages, 49% Turkish), training/eval scripts, no real results yet | Spell checker backend (9) |
| `news-scraper` | Config-driven scraper with per-region SQLite and JSONL output. Open source checks: Kazakh (6 of 9 outlets blocked), Kyrgyz (Super-Info, Maidan), South Azerbaijani (Araz, Oyannews, TRT), Tuvan (only shyn.ru, 60–70% Tuvan, needs LID filtering) | Monolingual corpora for SP training, LID tests and datasets (10) |
| `dictionaries` | Word-list source survey for 10 languages; Turkmen (enedilim) scraped | `lexicon` lookup / `is_word()` / frequency API |
| `parallel` | ud-turkic parallel treebanks (TueCL, Little Prince) for 9 languages | Evaluation data (11) |
| `turkic-nlp-book` | 14 chapter drafts; preface, three content gaps and the companion code layout pending | Hands-on chapters should use the toolkit (MorphemeTokenizer, `sp_tokenize`, transliterator, ASR) |
| `turkic-nlp-code-samples` | Notebooks per language and component, incl. ASR | Tutorials (12) |

## 16. Language priorities

- **Tier 1:** Uzbek, Kyrgyz, Azerbaijani, Turkmen. These have large speaker communities and growing institutional interest, and transfer well from Turkish.
- **Tier 2:** Crimean Tatar, Karakalpak, Tatar, Uyghur. These are endangered or diaspora communities, and some resources already exist.
- **Tier 3:** Kumyk, Karachay-Balkar, Nogai, Tuvan, Khakas, Gagauz, Bashkir, Chuvash, Sakha, Altai. These are the smallest by resources and gain most from cross-lingual transfer.
- **Language gaps by component:**
  - ASR: Tuvan, Khalaj, Ottoman and Old Turkic.
  - Dependency parsing: Gagauz, Altai, Tuvan, Khakas, Chuvash, Khalaj.
  - Translation: Gagauz, Karakalpak, Nogai, Kumyk, Karachay-Balkar, the Siberian languages and Chuvash.
- **Candidates for new languages:**
  - Salar, Shor, Urum, Karaim and Dolgan (no TurkicNLP support yet).
  - Fuyu Kyrgyz and Western Yugur (very low resource).

## 17. Community and outreach

- [ ] Project website with news, demos and a resource catalogue (in the style of [slaih.sk](https://slaih.sk/)). **M**
- [ ] Data sharing platform / catalogue following CLARIN/ELRA practice (in the style of [language-data-space.eu](https://language-data-space.eu/)). **L**
- [ ] Shared tasks and workshops with [SIGTURK](https://sigturk.github.io/); annotation partnerships with universities in Tashkent, Almaty, Bishkek, Baku and Ashgabat. **M**
- [ ] Contribute data back to Universal Dependencies, Common Voice and the HF Hub. **ongoing**
- [ ] Funding applications (EU Horizon, national science foundations) framed as digital inclusion and preservation. **ongoing**

---

## Where these items come from

This roadmap consolidates and replaces:

- `TODOs.md`
- `TODO_Vital_Components.md` (20-component strategic review)
- `TODOs_ASR.md`
- `TODOs_HF_Integration.md`
- `morph_extensions.md` (lexicon and disambiguation work log)
- `Perplexity_Research_Resources.md` (resource survey)

It also summarises the planning notes of the related project folders:

- `morpheme_aware_tokenizer_plan.md`
- `train-unified-models`, `train-spell-checker`, `news-scraper`, `generated-ud-data`, `dictionaries`
- the paper notes on additional experiments and on text/speech technology for Turkic languages
- the code-samples audit
