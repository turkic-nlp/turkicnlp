"""Tests for NLLB translation processor."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from turkicnlp.models.document import Document, Sentence
from turkicnlp.processors.translate import NLLBTranslateProcessor


class _FakeTensor:
    def __init__(self, value):
        self.value = value


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        return {
            "input_ids": _FakeTensor([[1, 2]] * len(texts)),
            "attention_mask": _FakeTensor([[1, 1]] * len(texts)),
        }

    def convert_tokens_to_ids(self, token):
        if token == "eng_Latn":
            return 17
        return -1

    def batch_decode(self, generated, skip_special_tokens=True):
        return [f"translated-{i}" for i, _ in enumerate(generated, start=1)]


class _FakeModel:
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        forced_bos_token_id=None,
        num_beams=None,
        max_new_tokens=None,
    ):
        return [[100, 1] for _ in input_ids.value]


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeTokenizer()


class _FakeAutoModelForSeq2SeqLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeModel()


def test_translate_processor_with_sentences(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForSeq2SeqLM=_FakeAutoModelForSeq2SeqLM,
        ),
    )

    proc = NLLBTranslateProcessor(lang="tur", config={"tgt_lang": "eng"})
    proc.load("/tmp/fake-models")

    doc = Document(
        text="Merhaba dünya.",
        lang="tur",
        sentences=[Sentence(text="Merhaba"), Sentence(text="dünya")],
    )
    out = proc.process(doc)

    assert out.sentences[0].translation == "translated-1"
    assert out.sentences[1].translation == "translated-2"
    assert out.translation == "translated-1\ntranslated-2"
    assert "translate:nllb" in out._processor_log


def test_translate_processor_without_sentences(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForSeq2SeqLM=_FakeAutoModelForSeq2SeqLM,
        ),
    )

    proc = NLLBTranslateProcessor(lang="kaz", config={"tgt_lang": "eng"})
    proc.load("/tmp/fake-models")

    doc = Document(text="Мен мектепке бардым.", lang="kaz")
    out = proc.process(doc)

    assert len(out.sentences) == 1
    assert out.sentences[0].translation == "translated-1"
    assert out.translation == "translated-1"


def test_translate_processor_unsupported_target(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForSeq2SeqLM=_FakeAutoModelForSeq2SeqLM,
        ),
    )

    proc = NLLBTranslateProcessor(lang="tur", config={"tgt_lang": "zzz"})
    try:
        proc.load("/tmp/fake-models")
    except ValueError as exc:
        assert "Unsupported translate_tgt_lang='zzz'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported target language")
