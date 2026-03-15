"""
Self-contained inference-only TurkicMorphAnalyzer model for the turkicnlp toolkit.

Minimal copy of the morph model architecture from train-unified-models,
with all configuration values hard-coded for the Glot500 Phase 2 morph checkpoint.
No external config files are needed.

Architecture:
  Glot500 Backbone (frozen) -> Script Adapter -> Lang Embedding
  -> Shared BiLSTM -> POSHead + MorphFeatureHead + LemmaHead (edit-script)
"""

from __future__ import annotations

import unicodedata

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Hard-coded constants (Glot500 morph Phase 2 checkpoint)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 768
ADAPTER_BOTTLENECK = 128
LANG_EMBED_DIM = 64
BILSTM_HIDDEN = 400
BILSTM_LAYERS = 2
DROPOUT = 0.33
CHAR_VOCAB_SIZE = 512
CHAR_EMBED_DIM = 64
CHAR_FILTERS = 128

BACKBONE_NAME = "cis-lmu/glot500-base"

# Reuse shared model components from the parser model
from turkicnlp.processors.multilingual_model import (
    UPOS_TAGS,
    NUM_UPOS,
    LANG_TO_SCRIPT as _BASE_LANG_TO_SCRIPT,
    LANG_ID_MAP as _BASE_LANG_ID_MAP,
    ISO3_TO_SHORT as _BASE_ISO3_TO_SHORT,
    tokenize_words,
    ScriptAdapter,
    LangEmbedding,
    SharedBiLSTM,
    POSHead,
)

# ---------------------------------------------------------------------------
# Morph-specific language maps (21 languages)
#
# The morph model was trained on 10 base UD languages plus 10 additional
# languages from UniMorph/Wiktionary data. The additional languages share
# language embeddings with their closest trained relative (proxy).
# Karakalpak is supported via zero-shot transfer through Uzbek proxy.
# ---------------------------------------------------------------------------

# Base 10 languages (direct training with UD treebanks)
LANG_TO_SCRIPT = {
    **_BASE_LANG_TO_SCRIPT,
    # Additional languages trained via UniMorph/Wiktionary
    "crh": "Latn",   # Crimean Tatar
    "kjh": "Cyrl",   # Khakas
    "sah": "Cyrl",   # Sakha/Yakut
    "tyv": "Cyrl",   # Tuvan
    "chv": "Cyrl",   # Chuvash
    "gag": "Latn",   # Gagauz
    "kum": "Cyrl",   # Kumyk
    "alt": "Cyrl",   # Southern Altai
    "atv": "Cyrl",   # Northern Altai
    "klj": "Latn",   # Khalaj
    # Zero-shot via proxy embedding
    "kaa": "Latn",   # Karakalpak → Uzbek proxy
}

LANG_ID_MAP = {
    **_BASE_LANG_ID_MAP,
    # Proxy embeddings from closest related trained language
    "crh": _BASE_LANG_ID_MAP["tr"],    # Crimean Tatar → Turkish
    "kjh": _BASE_LANG_ID_MAP["ba"],    # Khakas → Bashkir
    "sah": _BASE_LANG_ID_MAP["ba"],    # Sakha → Bashkir
    "tyv": _BASE_LANG_ID_MAP["ba"],    # Tuvan → Bashkir
    "chv": _BASE_LANG_ID_MAP["ba"],    # Chuvash → Bashkir
    "gag": _BASE_LANG_ID_MAP["tr"],    # Gagauz → Turkish
    "kum": _BASE_LANG_ID_MAP["tt"],    # Kumyk → Tatar
    "alt": _BASE_LANG_ID_MAP["ba"],    # S. Altai → Bashkir
    "atv": _BASE_LANG_ID_MAP["ba"],    # N. Altai → Bashkir
    "klj": _BASE_LANG_ID_MAP["tr"],    # Khalaj → Turkish
    "kaa": _BASE_LANG_ID_MAP["uz"],    # Karakalpak → Uzbek
}
NUM_LANGS = len(_BASE_LANG_ID_MAP)  # Embedding table size = 10 (base languages)

# ISO 639-3 → short code for all 21 morph-supported languages
ISO3_TO_SHORT = {
    **_BASE_ISO3_TO_SHORT,
    "crh": "crh",
    "kjh": "kjh",
    "sah": "sah",
    "tyv": "tyv",
    "chv": "chv",
    "gag": "gag",
    "kum": "kum",
    "alt": "alt",
    "kaa": "kaa",
    "atv": "atv",
    "klj": "klj",
}

SUPPORTED_LANGS = set(ISO3_TO_SHORT.keys())


def resolve_morph_lang(iso3: str) -> tuple[str, str]:
    """Resolve ISO 639-3 code to (short_code, script) for the morph model.

    The morph model supports 21 languages: 10 base + 10 trained with proxy
    embeddings + Karakalpak via zero-shot Uzbek proxy.

    Returns:
        (short_code, script) tuple for model routing.

    Raises:
        ValueError: If language is not supported.
    """
    if iso3 in ISO3_TO_SHORT:
        short = ISO3_TO_SHORT[iso3]
        return short, LANG_TO_SCRIPT[short]
    raise ValueError(
        f"Language '{iso3}' is not supported by the multilingual morph model. "
        f"Supported: {sorted(SUPPORTED_LANGS)}"
    )

# ---------------------------------------------------------------------------
# UD morphological feature vocabulary (must match training order exactly)
# ---------------------------------------------------------------------------

UD_MORPH_FEATS = [
    "Abbr=Yes",
    "Aspect=Hab", "Aspect=Imp", "Aspect=Perf", "Aspect=Prog", "Aspect=Prosp",
    "Case=Abl", "Case=Acc", "Case=Dat", "Case=Equ", "Case=Gen",
    "Case=Ins", "Case=Loc", "Case=Nom",
    "Definite=Def", "Definite=Ind",
    "Degree=Cmp", "Degree=Sup",
    "Evident=Fh", "Evident=Nfh",
    "Mood=Cnd", "Mood=Des", "Mood=Gen", "Mood=GenPot", "Mood=Imp",
    "Mood=Ind", "Mood=Nec", "Mood=Opt", "Mood=Pot",
    "Number=Plur", "Number=Sing",
    "Number[psor]=Plur", "Number[psor]=Sing",
    "NumType=Card", "NumType=Dist", "NumType=Ord",
    "Person=1", "Person=2", "Person=3",
    "Person[psor]=1", "Person[psor]=2", "Person[psor]=3",
    "Polarity=Neg", "Polarity=Pos",
    "Polite=Form", "Polite=Infm",
    "Poss=Yes",
    "PronType=Dem", "PronType=Ind", "PronType=Int", "PronType=Prs",
    "PronType=Rcp", "PronType=Tot",
    "Reflex=Yes",
    "Tense=Fut", "Tense=Past", "Tense=Pqp", "Tense=Pres",
    "VerbForm=Conv", "VerbForm=Fin", "VerbForm=Ger", "VerbForm=Inf",
    "VerbForm=Part", "VerbForm=Vnoun",
    "Voice=Cau", "Voice=CauPass", "Voice=Pass", "Voice=Rcp", "Voice=Rfl",
]

NUM_MORPH_FEATS = len(UD_MORPH_FEATS)

# ---------------------------------------------------------------------------
# Edit-script helpers for lemmatization
# ---------------------------------------------------------------------------


def _normalize_lower(text: str) -> str:
    """Lowercase + NFC normalize to handle Turkish İ/I and composed chars."""
    return unicodedata.normalize("NFC", text.lower())


def apply_edit_script(word: str, script: str) -> str:
    """Apply an edit script to a word to produce a lemma.

    Edit scripts encode suffix removal + addition:
      - "0" = identity (word == lemma)
      - "-3+mek" = remove 3 chars from end, append "mek"
    """
    base = _normalize_lower(word)

    if script == "0":
        return base

    parts = script.split("+", 1)
    remove = abs(int(parts[0]))  # "-3" → 3
    add = parts[1] if len(parts) > 1 else ""

    if remove == 0:
        return base + add
    return base[:-remove] + add


class EditScriptVocab:
    """Vocabulary of edit scripts loaded from checkpoint."""

    def __init__(self) -> None:
        self.script_to_id: dict[str, int] = {"<unk>": 0, "0": 1}
        self.id_to_script: dict[int, str] = {0: "<unk>", 1: "0"}

    def __len__(self) -> int:
        return len(self.script_to_id)

    def decode(self, idx: int) -> str:
        return self.id_to_script.get(idx, "<unk>")

    def load_state_dict(self, state: dict) -> None:
        self.script_to_id = state["script_to_id"]
        self.id_to_script = {v: k for k, v in self.script_to_id.items()}


def encode_chars(
    words_batch: list[list[str]],
    max_word_len: int,
    max_char_len: int = 32,
    vocab_size: int = CHAR_VOCAB_SIZE,
) -> torch.Tensor:
    """Encode words as character ID tensors.

    Maps each character to its Unicode codepoint mod vocab_size.
    Reserves 0 for padding.

    Returns:
        [B, max_word_len, max_char_len] tensor of character IDs.
    """
    B = len(words_batch)
    char_ids = torch.zeros(B, max_word_len, max_char_len, dtype=torch.long)

    for i, words in enumerate(words_batch):
        for j, word in enumerate(words):
            if j >= max_word_len:
                break
            for k, ch in enumerate(word[:max_char_len]):
                char_ids[i, j, k] = (ord(ch) % (vocab_size - 1)) + 1  # 0 = padding

    return char_ids


# ---------------------------------------------------------------------------
# Model components (morph-specific heads)
# ---------------------------------------------------------------------------


class MorphFeatureHead(nn.Module):
    """Multi-label classifier for UD morphological features.

    Predicts a binary vector over all possible feature=value pairs.
    Uses sigmoid (not softmax) because multiple features are active per word.
    """

    def __init__(self, input_dim: int, num_feats: int, dropout: float = 0.33):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CharCNN(nn.Module):
    """Character-level CNN to extract surface form features for lemmatization."""

    def __init__(
        self,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        char_embed_dim: int = CHAR_EMBED_DIM,
        num_filters: int = CHAR_FILTERS,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
    ):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k, padding=0)
            for k in kernel_sizes
        ])
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.char_embed(char_ids)  # [N, C, embed]
        x = x.transpose(1, 2)  # [N, embed, C]

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))  # [N, filters, C-k+1]
            c = c.max(dim=2).values  # [N, filters]
            conv_outs.append(c)

        return torch.cat(conv_outs, dim=1)  # [N, output_dim]


class LemmaHead(nn.Module):
    """Edit-script classifier for lemmatization.

    Combines BiLSTM contextual features with character-level surface form
    features to classify each word into an edit script class.
    """

    def __init__(
        self,
        input_dim: int,
        num_scripts: int,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        char_embed_dim: int = CHAR_EMBED_DIM,
        char_filters: int = CHAR_FILTERS,
        dropout: float = 0.33,
    ):
        super().__init__()
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            num_filters=char_filters,
        )
        combined_dim = input_dim + self.char_cnn.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, num_scripts),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        char_ids: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = hidden.shape
        max_chars = char_ids.size(2)

        flat_chars = char_ids.reshape(B * T, max_chars)
        char_feats = self.char_cnn(flat_chars)  # [B*T, char_output_dim]
        char_feats = char_feats.reshape(B, T, -1)  # [B, T, char_output_dim]

        combined = torch.cat([hidden, char_feats], dim=-1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Full morph model
# ---------------------------------------------------------------------------


class TurkicMorphAnalyzer(nn.Module):
    """Unified model for morphological analysis and lemmatization across Turkic languages.

    Loads the Glot500 backbone from HuggingFace and assembles adapters,
    language embeddings, BiLSTM, and morph task heads.
    """

    def __init__(self, hf_cache_dir: str, num_edit_scripts: int):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        import io, sys, logging as _logging
        _loggers = [
            _logging.getLogger(name)
            for name in ("transformers.modeling_utils", "transformers.configuration_utils")
        ]
        _prev_levels = [lg.level for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(_logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(
            BACKBONE_NAME, cache_dir=hf_cache_dir
        )
        # Suppress "not sharded" stderr spam from safetensors
        import os
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr = os.dup(2)
        os.dup2(_devnull, 2)
        try:
            self.backbone = AutoModel.from_pretrained(
                BACKBONE_NAME, cache_dir=hf_cache_dir
            )
        finally:
            os.dup2(_saved_stderr, 2)
            os.close(_saved_stderr)
            os.close(_devnull)

        for lg, lvl in zip(_loggers, _prev_levels):
            lg.setLevel(lvl)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleDict({
            "Latn": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Cyrl": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Arab": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
        })

        self.lang_embed = LangEmbedding(NUM_LANGS, LANG_EMBED_DIM, HIDDEN_DIM)

        bilstm_out = BILSTM_HIDDEN * 2
        self.bilstm = SharedBiLSTM(HIDDEN_DIM, BILSTM_HIDDEN, BILSTM_LAYERS, DROPOUT)

        # Task heads
        self.pos_head = POSHead(bilstm_out, NUM_UPOS)
        self.morph_head = MorphFeatureHead(bilstm_out, NUM_MORPH_FEATS, DROPOUT)
        self.lemma_head = LemmaHead(
            input_dim=bilstm_out,
            num_scripts=num_edit_scripts,
            char_vocab_size=CHAR_VOCAB_SIZE,
            char_embed_dim=CHAR_EMBED_DIM,
            char_filters=CHAR_FILTERS,
            dropout=DROPOUT,
        )

    def _align_subwords_to_words(
        self, hidden: torch.Tensor, word_starts: torch.Tensor,
    ) -> torch.Tensor:
        """First-subword pooling: gather hidden state at each word's first subword."""
        indices = word_starts.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        return torch.gather(hidden, 1, indices)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        word_lengths: torch.Tensor,
        lang_ids: torch.Tensor,
        script: str,
        char_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        Returns:
            pos_logits: [B, T_words, NUM_UPOS]
            feat_logits: [B, T_words, NUM_MORPH_FEATS]
            edit_logits: [B, T_words, num_edit_scripts]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self._align_subwords_to_words(hidden, word_starts)
        hidden = self.adapters[script](hidden)
        hidden = self.lang_embed(hidden, lang_ids)

        lstm_out = self.bilstm(hidden, word_lengths)

        pos_logits = self.pos_head(lstm_out)
        feat_logits = self.morph_head(lstm_out)

        # Word mask for lemma head (all True for valid words)
        word_mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0)
        word_mask = word_mask < word_lengths.unsqueeze(1)

        edit_logits = self.lemma_head(lstm_out, char_ids, word_mask)

        return pos_logits, feat_logits, edit_logits
