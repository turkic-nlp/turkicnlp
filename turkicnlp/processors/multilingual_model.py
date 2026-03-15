"""
Self-contained inference-only TurkicParser model for the turkicnlp toolkit.

This is a minimal copy of the model architecture from train-unified-models,
with all configuration values hard-coded for the Glot500 Phase 1 checkpoint.
No external config files are needed.

Architecture:
  Glot500 Backbone (frozen) -> Script Adapter -> Lang Embedding
  -> Shared BiLSTM -> POS Head + Biaffine Dep Parser
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Hard-coded constants (Glot500 Phase 1 checkpoint)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 768
ADAPTER_BOTTLENECK = 128
LANG_EMBED_DIM = 64
BILSTM_HIDDEN = 400
BILSTM_LAYERS = 2
DROPOUT = 0.33
ARC_MLP_DIM = 500
LABEL_MLP_DIM = 100

BACKBONE_NAME = "cis-lmu/glot500-base"

# UD UPOS tagset (17 tags) — order must match training
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
NUM_UPOS = len(UPOS_TAGS)

# UD universal dependency relations (37 base relations) — order must match training
DEPREL_TAGS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc",
    "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith",
    "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl",
    "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp",
]
NUM_DEPRELS = len(DEPREL_TAGS)

# Short code -> script mapping (used by the model internally)
LANG_TO_SCRIPT = {
    "tr": "Latn", "az": "Latn", "uz": "Latn", "tm": "Latn",
    "kk": "Cyrl", "ky": "Cyrl", "ba": "Cyrl", "tt": "Cyrl",
    "ug": "Arab", "ota": "Arab",
}

# Short code -> integer ID for language embedding lookup
LANG_ID_MAP = {
    "tr": 0, "az": 1, "uz": 2, "tm": 3, "kk": 4,
    "ky": 5, "ba": 6, "tt": 7, "ug": 8, "ota": 9,
}
NUM_LANGS = len(LANG_ID_MAP)

# ISO 639-3 (turkicnlp) -> short code (model)
ISO3_TO_SHORT = {
    "tur": "tr", "aze": "az", "uzb": "uz", "tuk": "tm", "kaz": "kk",
    "kir": "ky", "bak": "ba", "tat": "tt", "uig": "ug", "ota": "ota",
}

# Zero-shot language configuration: ISO 639-3 -> proxy info
ZEROSHOT_LANGS = {
    "kaa": {"name": "Karakalpak", "script": "Latn", "proxy_lang": "uz"},
    "kum": {"name": "Kumyk",      "script": "Cyrl", "proxy_lang": "tt"},
    "sah": {"name": "Sakha",      "script": "Cyrl", "proxy_lang": "ba"},
}

# All supported languages (trained + zero-shot)
SUPPORTED_LANGS = set(ISO3_TO_SHORT.keys()) | set(ZEROSHOT_LANGS.keys())


def resolve_lang(iso3: str) -> tuple[str, str]:
    """Resolve ISO 639-3 code to (short_code, script) for the model.

    For trained languages, returns the direct mapping.
    For zero-shot languages, returns the proxy language's short code and
    the zero-shot language's own script.

    Returns:
        (short_code, script) tuple for model routing.

    Raises:
        ValueError: If language is not supported.
    """
    if iso3 in ISO3_TO_SHORT:
        short = ISO3_TO_SHORT[iso3]
        return short, LANG_TO_SCRIPT[short]
    if iso3 in ZEROSHOT_LANGS:
        cfg = ZEROSHOT_LANGS[iso3]
        proxy_short = cfg["proxy_lang"]
        return proxy_short, cfg["script"]
    raise ValueError(
        f"Language '{iso3}' is not supported by the multilingual Glot500 model. "
        f"Supported: {sorted(SUPPORTED_LANGS)}"
    )


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class ScriptAdapter(nn.Module):
    """Bottleneck adapter: down -> GELU -> up + residual + LayerNorm."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return self.norm(residual + x)


class LangEmbedding(nn.Module):
    """Per-language embedding concatenated with hidden states, projected back."""

    def __init__(self, num_langs: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_langs, embed_dim)
        self.proj = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        lang_vecs = self.embed(lang_ids)  # [B, embed_dim]
        lang_vecs = lang_vecs.unsqueeze(1).expand(-1, hidden.size(1), -1)
        concat = torch.cat([hidden, lang_vecs], dim=-1)
        return self.norm(self.proj(concat))


class SharedBiLSTM(nn.Module):
    """Multi-layer BiLSTM with dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.clamp(min=1, max=x.size(1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return self.dropout(output)


class POSHead(nn.Module):
    """Linear classifier for Universal POS tagging."""

    def __init__(self, input_dim: int, num_tags: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BiaffineScorer(nn.Module):
    """Bilinear scorer for arc and label prediction."""

    def __init__(self, in1_dim: int, in2_dim: int, out_dim: int = 1,
                 bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_dim = out_dim
        self.weight = nn.Parameter(
            torch.zeros(out_dim, in1_dim + int(bias_x), in2_dim + int(bias_y))
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        if self.out_dim == 1:
            return s.squeeze(1)
        else:
            return s.permute(0, 2, 3, 1)


class BiaffineParser(nn.Module):
    """Deep Biaffine Attention parser (Dozat & Manning, 2017)."""

    def __init__(self, input_dim: int, arc_mlp_dim: int, label_mlp_dim: int,
                 num_labels: int, dropout: float):
        super().__init__()
        self.arc_mlp_head = self._mlp(input_dim, arc_mlp_dim, dropout)
        self.arc_mlp_dep = self._mlp(input_dim, arc_mlp_dim, dropout)
        self.lbl_mlp_head = self._mlp(input_dim, label_mlp_dim, dropout)
        self.lbl_mlp_dep = self._mlp(input_dim, label_mlp_dim, dropout)
        self.arc_scorer = BiaffineScorer(arc_mlp_dim, arc_mlp_dim, out_dim=1)
        self.label_scorer = BiaffineScorer(label_mlp_dim, label_mlp_dim, out_dim=num_labels)

    def _mlp(self, in_dim: int, out_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        arc_h = self.arc_mlp_head(x)
        arc_d = self.arc_mlp_dep(x)
        lbl_h = self.lbl_mlp_head(x)
        lbl_d = self.lbl_mlp_dep(x)
        arc_logits = self.arc_scorer(arc_d, arc_h)
        label_logits = self.label_scorer(lbl_d, lbl_h)
        return arc_logits, label_logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TurkicParser(nn.Module):
    """Unified model for POS tagging and dependency parsing across Turkic languages.

    Loads the Glot500 backbone from HuggingFace and assembles adapters,
    language embeddings, BiLSTM, and task heads with hard-coded architecture.
    """

    def __init__(self, hf_cache_dir: str):
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
        # Freeze backbone entirely (inference only)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleDict({
            "Latn": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Cyrl": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
            "Arab": ScriptAdapter(HIDDEN_DIM, ADAPTER_BOTTLENECK),
        })

        self.lang_embed = LangEmbedding(NUM_LANGS, LANG_EMBED_DIM, HIDDEN_DIM)

        self.root_embedding = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))

        bilstm_out = BILSTM_HIDDEN * 2
        self.bilstm = SharedBiLSTM(HIDDEN_DIM, BILSTM_HIDDEN, BILSTM_LAYERS, DROPOUT)
        self.pos_head = POSHead(bilstm_out, NUM_UPOS)
        self.dep_head = BiaffineParser(
            bilstm_out, ARC_MLP_DIM, LABEL_MLP_DIM, NUM_DEPRELS, DROPOUT,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for inference.

        Returns:
            pos_logits: [B, T_words, NUM_UPOS]
            arc_logits: [B, T_words, T_words+1]
            label_logits: [B, T_words, T_words+1, NUM_DEPRELS]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self._align_subwords_to_words(hidden, word_starts)
        hidden = self.adapters[script](hidden)
        hidden = self.lang_embed(hidden, lang_ids)

        B = hidden.size(0)
        root = self.root_embedding.expand(B, -1, -1)
        hidden_with_root = torch.cat([root, hidden], dim=1)
        lengths_with_root = word_lengths + 1

        lstm_out = self.bilstm(hidden_with_root, lengths_with_root)

        pos_logits = self.pos_head(lstm_out[:, 1:, :])
        arc_logits, label_logits = self.dep_head(lstm_out)
        arc_logits = arc_logits[:, 1:, :]
        label_logits = label_logits[:, 1:, :]

        return pos_logits, arc_logits, label_logits


def tokenize_words(
    tokenizer,
    words_list: list[list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize pre-split word lists and compute subword-to-word alignment.

    Args:
        tokenizer: HuggingFace tokenizer.
        words_list: List of word lists (one per sentence).
        device: Target device for tensors.

    Returns:
        (input_ids, attention_mask, word_starts, word_lengths) tensors.
    """
    encodings = tokenizer(
        words_list,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    max_word_len = max(len(ws) for ws in words_list)
    word_starts = []
    for i in range(len(words_list)):
        word_ids = encodings.word_ids(batch_index=i)
        starts = []
        seen: set[int] = set()
        for subword_idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen:
                starts.append(subword_idx)
                seen.add(word_id)
        while len(starts) < max_word_len:
            starts.append(0)
        word_starts.append(starts)

    word_starts_t = torch.tensor(word_starts, dtype=torch.long, device=device)
    word_lengths = torch.tensor(
        [len(ws) for ws in words_list], dtype=torch.long, device=device
    )

    return (
        encodings["input_ids"].to(device),
        encodings["attention_mask"].to(device),
        word_starts_t,
        word_lengths,
    )
