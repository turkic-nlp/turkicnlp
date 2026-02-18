"""Tests for NLLB embeddings processor."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from turkicnlp.models.document import Document, Sentence
from turkicnlp.processors.embeddings import NLLBEmbeddingsProcessor


class _FakeTensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self.value, axis=dim))

    def sum(self, dim):
        return _FakeTensor(np.sum(self.value, axis=dim))

    def clamp(self, min=0):
        return _FakeTensor(np.clip(self.value, min, None))

    def tolist(self):
        return self.value.tolist()

    def __mul__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self.value * other.value)
        return _FakeTensor(self.value * other)

    def __truediv__(self, other):
        if isinstance(other, _FakeTensor):
            return _FakeTensor(self.value / other.value)
        return _FakeTensor(self.value / other)


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()

    class nn:  # noqa: N801
        class functional:  # noqa: N801
            @staticmethod
            def normalize(t, p=2, dim=1):
                norms = np.linalg.norm(t.value, ord=p, axis=dim, keepdims=True)
                norms[norms == 0] = 1.0
                return _FakeTensor(t.value / norms)


class _FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        max_len = max(len(t.split()) for t in texts)
        input_ids = []
        attention_mask = []
        for text in texts:
            n = len(text.split())
            ids = [1] * n + [0] * (max_len - n)
            mask = [1] * n + [0] * (max_len - n)
            input_ids.append(ids)
            attention_mask.append(mask)
        return {
            "input_ids": _FakeTensor(input_ids),
            "attention_mask": _FakeTensor(attention_mask),
        }


class _FakeEncoder:
    def __call__(self, input_ids, attention_mask, return_dict=True):
        batch, seq = input_ids.value.shape
        hidden = np.zeros((batch, seq, 2), dtype=float)
        for b in range(batch):
            hidden[b, :, 0] = b + 1
            hidden[b, :, 1] = np.arange(1, seq + 1, dtype=float)
        return SimpleNamespace(last_hidden_state=_FakeTensor(hidden))


class _FakeModel:
    def get_encoder(self):
        return _FakeEncoder()


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeTokenizer()


class _FakeAutoModelForSeq2SeqLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeModel()


def test_embeddings_processor_with_sentences(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForSeq2SeqLM=_FakeAutoModelForSeq2SeqLM,
        ),
    )

    proc = NLLBEmbeddingsProcessor(
        lang="tur",
        config={"src_lang": "tur_Latn", "normalize": False},
    )
    proc.load("/tmp/fake-models")

    doc = Document(
        text="Merhaba dünya.",
        lang="tur",
        sentences=[Sentence(text="Merhaba"), Sentence(text="dünya bugün")],
    )
    out = proc.process(doc)

    assert out.sentences[0].embedding == [1.0, 1.0]
    assert out.sentences[1].embedding == [2.0, 1.5]
    assert out.embedding is not None
    assert "embeddings:nllb" in out._processor_log


def test_embeddings_processor_without_sentences(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForSeq2SeqLM=_FakeAutoModelForSeq2SeqLM,
        ),
    )

    proc = NLLBEmbeddingsProcessor(
        lang="kaz",
        config={"src_lang": "kaz_Cyrl", "normalize": True},
    )
    proc.load("/tmp/fake-models")

    doc = Document(text="Мен мектепке бардым.", lang="kaz")
    out = proc.process(doc)

    assert len(out.sentences) == 1
    assert out.sentences[0].embedding is not None
    assert out.embedding is not None
