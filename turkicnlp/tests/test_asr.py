"""Tests for the Omnilingual ASR wrapper (no model download: the backend is faked)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import turkicnlp.asr as asr
from turkicnlp.asr import (
    OMNIASR_LANG_CODES,
    SpeechRecognizer,
    list_asr_languages,
    resolve_asr_language,
    split_audio,
)

# Subset of omnilingual_asr...lang_ids.supported_langs relevant to the tests
_SUPPORTED = {
    "tur_Latn", "aze_Latn", "aze_Cyrl", "aze_Arab", "tuk_Latn", "tuk_Arab", "gag_Latn",
    "gag_Cyrl", "kaz_Cyrl", "kir_Cyrl", "tat_Cyrl", "bak_Cyrl", "crh_Cyrl", "kaa_Cyrl",
    "nog_Cyrl", "kum_Cyrl", "krc_Cyrl", "uzn_Latn", "uzb_Latn", "uzb_Cyrl", "uig_Arab",
    "uig_Cyrl", "sah_Cyrl", "alt_Cyrl", "kjh_Cyrl", "chv_Cyrl", "eng_Latn", "rus_Cyrl",
}


class _FakeASRPipeline:
    instances: list["_FakeASRPipeline"] = []

    def __init__(self, model_card, *, model=None, tokenizer=None, device=None, dtype=None):
        self.model_card = model_card if model_card is not None else model.card
        self.preloaded = model is not None
        self.device = device
        self.dtype = dtype
        self.calls: list[dict] = []
        self.outputs: list[str] | None = None
        _FakeASRPipeline.instances.append(self)

    def transcribe(self, inp, lang=None, batch_size=2):
        assert all(isinstance(x, dict) and {"waveform", "sample_rate"} <= set(x) for x in inp)
        self.calls.append({"inp": inp, "lang": list(lang), "batch_size": batch_size})
        if self.outputs is not None:
            out, self.outputs = self.outputs[: len(inp)], self.outputs[len(inp):]
            return out
        return [f"seg{i}" for i in range(len(inp))]


class _FakeTorch(ModuleType):
    device = staticmethod(lambda name: f"device:{name}")
    float32 = "torch.float32"
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"
    cuda = SimpleNamespace(is_available=lambda: False)


@pytest.fixture
def fake_omniasr(monkeypatch):
    """Install fake ``omnilingual_asr`` and ``torch`` modules."""
    root = ModuleType("omnilingual_asr")
    lang_ids = ModuleType("omnilingual_asr.models.wav2vec2_llama.lang_ids")
    lang_ids.supported_langs = sorted(_SUPPORTED)
    pipe_mod = ModuleType("omnilingual_asr.models.inference.pipeline")
    pipe_mod.ASRInferencePipeline = _FakeASRPipeline
    load_calls = []
    models_hub = ModuleType("fairseq2.models.hub")
    models_hub.load_model = lambda card, **kw: (
        load_calls.append((card, kw)) or SimpleNamespace(card=card)
    )
    tok_hub = ModuleType("fairseq2.data.tokenizers.hub")
    tok_hub.load_tokenizer = lambda card: SimpleNamespace(card=card)
    lang_ids.load_calls = load_calls
    for name, mod in {
        "omnilingual_asr": root,
        "omnilingual_asr.models": ModuleType("omnilingual_asr.models"),
        "omnilingual_asr.models.wav2vec2_llama": ModuleType("x"),
        "omnilingual_asr.models.wav2vec2_llama.lang_ids": lang_ids,
        "omnilingual_asr.models.inference": ModuleType("y"),
        "omnilingual_asr.models.inference.pipeline": pipe_mod,
        "torch": _FakeTorch("torch"),
        "fairseq2": ModuleType("fairseq2"),
        "fairseq2.models": ModuleType("fairseq2.models"),
        "fairseq2.models.hub": models_hub,
        "fairseq2.data": ModuleType("fairseq2.data"),
        "fairseq2.data.tokenizers": ModuleType("fairseq2.data.tokenizers"),
        "fairseq2.data.tokenizers.hub": tok_hub,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(asr, "_SUPPORTED_LANGS_CACHE", None)
    monkeypatch.setattr(asr, "_PIPELINE_CACHE", {})
    _FakeASRPipeline.instances.clear()
    return lang_ids


def _tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# -- language resolution ----------------------------------------------------

def test_resolve_primary_scripts(fake_omniasr):
    assert resolve_asr_language("kaz") == ("kaz", "kaz_Cyrl", "Cyrl")
    assert resolve_asr_language("uzb") == ("uzb", "uzn_Latn", "Latn")
    assert resolve_asr_language("uzb", "Cyrl") == ("uzb", "uzb_Cyrl", "Cyrl")
    assert resolve_asr_language("azb") == ("azb", "aze_Arab", "Arab")
    # Model only has Cyrillic for Karakalpak and Crimean Tatar
    assert resolve_asr_language("kaa") == ("kaa", "kaa_Cyrl", "Cyrl")
    assert resolve_asr_language("crh") == ("crh", "crh_Cyrl", "Cyrl")


def test_resolve_explicit_omniasr_code(fake_omniasr):
    assert resolve_asr_language("uig_Cyrl") == ("uig", "uig_Cyrl", "Cyrl")


@pytest.mark.parametrize("code", ["eng", "rus", "deu", "eng_Latn", "rus_Cyrl"])
def test_non_turkic_languages_rejected(fake_omniasr, code):
    with pytest.raises(ValueError, match="not a"):
        resolve_asr_language(code)


@pytest.mark.parametrize("code", ["tyv", "klj", "ota", "otk"])
def test_turkic_languages_without_model_token(fake_omniasr, code):
    with pytest.raises(ValueError, match="not available"):
        resolve_asr_language(code)


def test_checks_model_supported_langs(fake_omniasr):
    # uzn_Latn missing from the model -> falls back to uzb_Latn
    fake_omniasr.supported_langs = sorted(_SUPPORTED - {"uzn_Latn"})
    assert resolve_asr_language("uzb")[1] == "uzb_Latn"
    # code missing entirely -> clear error
    asr._SUPPORTED_LANGS_CACHE = None
    fake_omniasr.supported_langs = sorted(_SUPPORTED - {"kjh_Cyrl"})
    with pytest.raises(ValueError, match="supported by the installed model"):
        resolve_asr_language("kjh")


def test_mapping_covers_twenty_languages_and_catalog():
    assert len(OMNIASR_LANG_CODES) == 20
    langs = list_asr_languages()
    assert langs["kaz"] == {"Cyrl": "kaz_Cyrl"}

    from turkicnlp.resources.registry import ModelRegistry

    catalog = ModelRegistry._load_packaged_catalog()
    for lang, scripts in OMNIASR_LANG_CODES.items():
        procs = [p for p in catalog[lang]["processors"].values() if "asr" in p]
        assert procs, lang
        code = procs[0]["asr"]["backends"]["omniasr"]["lang_code"]
        assert code in [c for cs in scripts.values() for c in cs]


def test_unknown_language_fails_at_construction():
    with pytest.raises(ValueError):
        SpeechRecognizer(lang="eng")


# -- audio segmentation -----------------------------------------------------

def test_split_short_audio_is_single_segment():
    assert split_audio(_tone(10), 16000, 30) == [(0, 160000)]


def test_split_long_audio_respects_limit_and_cuts_at_silence():
    sr = 16000
    wav = np.concatenate([_tone(27), np.zeros(sr, np.float32), _tone(40)])  # 68 s
    bounds = split_audio(wav, sr, max_seconds=30)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(wav)
    assert all(b[0] == a[1] for a, b in zip(bounds, bounds[1:]))
    assert all((e - s) / sr <= 30 for s, e in bounds)
    assert 27 * sr <= bounds[0][1] <= 28 * sr  # cut inside the silent second


# -- transcription ----------------------------------------------------------

def test_transcribe_single_and_batch(fake_omniasr):
    rec = SpeechRecognizer(lang="kaz", batch_size=4)
    audio = {"array": _tone(3), "sampling_rate": 16000}
    assert rec.transcribe(audio) == "seg0"
    pipe = _FakeASRPipeline.instances[0]
    assert pipe.model_card == "omniASR_LLM_1B"
    assert pipe.device == "cpu" and pipe.dtype == "torch.bfloat16"
    # low-memory loading: checkpoint is memory-mapped, model passed in preloaded
    assert pipe.preloaded
    assert fake_omniasr.load_calls == [
        ("omniASR_LLM_1B", {"device": "device:cpu", "dtype": "torch.bfloat16", "mmap": True})
    ]
    assert pipe.calls[-1]["lang"] == ["kaz_Cyrl"] and pipe.calls[-1]["batch_size"] == 4

    out = rec.transcribe([audio, audio], lang=["kaz", "uzb"], batch_size=2)
    assert out == ["seg0", "seg1"]
    assert pipe.calls[-1]["lang"] == ["kaz_Cyrl", "uzn_Latn"]


def test_lang_list_length_must_match(fake_omniasr):
    rec = SpeechRecognizer()
    audio = {"waveform": _tone(1), "sample_rate": 16000}
    with pytest.raises(ValueError, match="one entry per input"):
        rec.transcribe([audio, audio], lang=["kaz"])


def test_output_transliterated_to_toolkit_script(fake_omniasr):
    rec = SpeechRecognizer(lang="kaa").load()
    rec._pipeline.outputs = ["Мен мектепке бардым"]
    res = rec.transcribe({"waveform": _tone(2), "sample_rate": 16000}, return_details=True)
    assert res.model_lang == "kaa_Cyrl" and res.script == "Latn"
    assert res.model_text == "Мен мектепке бардым"
    assert res.text == "Men mektepke bardım"

    # Requested script overrides the default
    rec._pipeline.outputs = ["Мен мектепке бардым"]
    assert rec.transcribe({"waveform": _tone(2), "sample_rate": 16000}, script="Cyrl") == (
        "Мен мектепке бардым"
    )


def test_long_audio_is_segmented_and_joined(fake_omniasr):
    sr = 8000
    rec = SpeechRecognizer(lang="tur", max_segment_seconds=30)
    res = rec.transcribe(_tone(70, sr), sample_rate=sr, return_details=True)
    assert [s.text for s in res.segments] == ["seg0", "seg1", "seg2"]
    assert res.text == "seg0 seg1 seg2"
    assert res.segments[0].start == 0.0 and res.segments[-1].end == pytest.approx(70.0)
    call = _FakeASRPipeline.instances[0].calls[-1]
    assert call["lang"] == ["tur_Latn"] * 3
    assert all(len(x["waveform"]) <= 30 * sr for x in call["inp"])


def test_stereo_and_file_input(fake_omniasr, tmp_path):
    sf = pytest.importorskip("soundfile")
    stereo = np.stack([_tone(2), _tone(2)], axis=1)
    path = tmp_path / "speech.wav"
    sf.write(path, stereo, 16000)
    rec = SpeechRecognizer(lang="tat")
    assert rec.transcribe(str(path)) == "seg0"
    sent = _FakeASRPipeline.instances[0].calls[-1]["inp"][0]
    assert sent["waveform"].ndim == 1 and sent["sample_rate"] == 16000
    assert rec.transcribe(path.read_bytes()) == "seg0"


def test_model_shared_between_recognizers(fake_omniasr):
    a = SpeechRecognizer(lang="kaz").load()
    b = SpeechRecognizer(lang="tur").load()
    assert a._pipeline is b._pipeline
    assert len(_FakeASRPipeline.instances) == 1


def test_missing_package_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "omnilingual_asr", None)
    monkeypatch.setattr(asr, "_PIPELINE_CACHE", {})
    with pytest.raises(ImportError, match=r"turkicnlp\[asr\]"):
        SpeechRecognizer(lang="kaz").load()


# -- pipeline integration ---------------------------------------------------

def test_pipeline_from_audio(fake_omniasr):
    import turkicnlp

    nlp = turkicnlp.Pipeline("kaz", processors=["asr", "tokenize"], tokenize_backend="rule",
                             asr_model_card="omniASR_LLM_300M")
    rec = nlp._get_asr().load()
    rec._pipeline.outputs = ["Мен мектепке бардым."]
    doc = nlp.from_audio({"waveform": _tone(2), "sample_rate": 16000})
    assert doc.text == "Мен мектепке бардым."
    assert [w.text for w in doc.words] == ["Мен", "мектепке", "бардым", "."]
    assert doc.audio_segments == [{"start": 0.0, "end": 2.0, "text": "Мен мектепке бардым."}]
    assert doc._processor_log[0] == "asr:omniasr:omniASR_LLM_300M:kaz_Cyrl"
    assert rec._pipeline.model_card == "omniASR_LLM_300M"


def test_download_skips_asr_backend():
    from turkicnlp.resources.downloader import download

    download("kaz", processors=["asr"])  # nothing to fetch, must not raise


def test_missing_audio_file_has_clear_error(fake_omniasr):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        SpeechRecognizer(lang="kaz").transcribe("does_not_exist.wav")


def test_standard_loading_and_explicit_dtype(fake_omniasr):
    rec = SpeechRecognizer(lang="kaz", low_memory=False, dtype="float32").load()
    assert not rec._pipeline.preloaded and rec._pipeline.dtype == "torch.float32"
    assert fake_omniasr.load_calls == []
