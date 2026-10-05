"""
Automatic speech recognition (ASR) for Turkic languages.

Wraps Meta's Omnilingual ASR models (``omniASR_LLM_1B`` by default,
https://huggingface.co/facebook/omniASR-LLM-1B) behind the TurkicNLP
conventions:

* languages are given as TurkicNLP ISO 639-3 codes (``kaz``, ``uzb``, ...)
  and mapped to Omnilingual ``{code}_{script}`` language tokens;
* only the Turkic languages covered by TurkicNLP are accepted, and each is
  checked against the model's ``supported_langs`` list when it is used;
* transcripts are returned in the language's primary TurkicNLP script (or a
  requested one), using the toolkit's transliterator when the model writes a
  language in another script (e.g. Karakalpak and Crimean Tatar are produced
  in Cyrillic by the model and converted to Latin);
* audio longer than the model's 40-second limit is split into segments at
  low-energy points and the segment transcripts are joined.

Usage::

    import turkicnlp

    asr = turkicnlp.SpeechRecognizer(model_card="omniASR_LLM_1B")
    texts = asr.transcribe(["kaz_audio.wav", "uzb_audio.flac"],
                           lang=["kaz", "uzb"], batch_size=2)

    # Speech -> text -> analysis in one pipeline
    nlp = turkicnlp.Pipeline("kaz", processors=["asr", "tokenize", "pos"])
    doc = nlp.from_audio("kaz_audio.wav")

Requires ``pip install turkicnlp[asr]`` (Python 3.10-3.12, ``libsndfile``).
Model weights are downloaded by fairseq2 on first use and cached in
``~/.cache/fairseq2/assets/``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from turkicnlp.scripts import Script, get_script_config

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CARD = "omniASR_LLM_1B"
#: Omnilingual ASR rejects (non-streaming) inputs longer than this.
MAX_AUDIO_SECONDS = 40.0
#: Default maximum segment length used when splitting long audio.
DEFAULT_SEGMENT_SECONDS = 30.0

# ---------------------------------------------------------------------------
# TurkicNLP language -> Omnilingual ASR language tokens
# ---------------------------------------------------------------------------
# For each TurkicNLP language: TurkicNLP script -> candidate Omnilingual codes
# (first candidate present in ``supported_langs`` wins). The first script is
# used when the requested output script has no model code of its own; the
# output is then transliterated with the TurkicNLP transliterator.
OMNIASR_LANG_CODES: dict[str, dict[str, list[str]]] = {
    # Oghuz
    "tur": {"Latn": ["tur_Latn"]},
    "aze": {"Latn": ["aze_Latn"], "Cyrl": ["aze_Cyrl"]},
    "azb": {"Arab": ["aze_Arab"]},
    "tuk": {"Latn": ["tuk_Latn"]},
    "gag": {"Latn": ["gag_Latn"]},
    # Kipchak
    "kaz": {"Cyrl": ["kaz_Cyrl"]},
    "kir": {"Cyrl": ["kir_Cyrl"]},
    "tat": {"Cyrl": ["tat_Cyrl"]},
    "bak": {"Cyrl": ["bak_Cyrl"]},
    "crh": {"Cyrl": ["crh_Cyrl"]},
    "kaa": {"Cyrl": ["kaa_Cyrl"]},
    "nog": {"Cyrl": ["nog_Cyrl"]},
    "kum": {"Cyrl": ["kum_Cyrl"]},
    "krc": {"Cyrl": ["krc_Cyrl"]},
    # Karluk
    "uzb": {"Latn": ["uzn_Latn", "uzb_Latn"], "Cyrl": ["uzb_Cyrl"]},
    "uig": {"Arab": ["uig_Arab"], "Cyrl": ["uig_Cyrl"]},
    # Siberian
    "sah": {"Cyrl": ["sah_Cyrl"]},
    "alt": {"Cyrl": ["alt_Cyrl"]},
    "kjh": {"Cyrl": ["kjh_Cyrl"]},
    # Oghur
    "chv": {"Cyrl": ["chv_Cyrl"]},
}

#: TurkicNLP languages without an Omnilingual ASR language token.
UNSUPPORTED_ASR_LANGS: dict[str, str] = {
    "tyv": "Tuvan",
    "klj": "Khalaj",
    "ota": "Ottoman Turkish",
    "otk": "Old Turkic",
}

# Reverse map: Omnilingual code -> (TurkicNLP language, TurkicNLP script)
_CODE_TO_LANG: dict[str, tuple[str, str]] = {
    code: (lang, script)
    for lang, scripts in OMNIASR_LANG_CODES.items()
    for script, codes in scripts.items()
    for code in codes
}

_SUPPORTED_LANGS_CACHE: Optional[frozenset[str]] = None
_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}

AudioInput = Union[str, Path, bytes, dict, Any]


def _require_omniasr() -> None:
    try:
        import omnilingual_asr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Speech recognition requires the `omnilingual-asr` package "
            "(Python 3.10-3.12 and libsndfile). Install with: "
            "pip install turkicnlp[asr]"
        ) from exc


def omniasr_supported_langs() -> frozenset[str]:
    """Return the language tokens supported by the installed Omnilingual ASR.

    Reads ``omnilingual_asr.models.wav2vec2_llama.lang_ids.supported_langs``
    once and caches it.
    """
    global _SUPPORTED_LANGS_CACHE
    if _SUPPORTED_LANGS_CACHE is None:
        _require_omniasr()
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

        _SUPPORTED_LANGS_CACHE = frozenset(supported_langs)
    return _SUPPORTED_LANGS_CACHE


def list_asr_languages(check_model: bool = False) -> dict[str, dict[str, str]]:
    """List TurkicNLP languages with speech recognition support.

    Args:
        check_model: If ``True``, keep only codes present in the installed
            model's ``supported_langs`` (requires ``omnilingual-asr``).

    Returns:
        ``{lang: {script: omnilingual_code}}`` with the code that would be used.
    """
    supported = omniasr_supported_langs() if check_model else None
    result: dict[str, dict[str, str]] = {}
    for lang, scripts in OMNIASR_LANG_CODES.items():
        for script, codes in scripts.items():
            usable = [c for c in codes if supported is None or c in supported]
            if usable:
                result.setdefault(lang, {})[script] = usable[0]
    return result


@dataclass
class ASRSegment:
    """A transcribed stretch of audio (times in seconds)."""

    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    """Transcription of one audio input.

    Attributes:
        text: Full transcript in ``script``.
        lang: TurkicNLP ISO 639-3 language code.
        script: Script of ``text``.
        model_lang: Omnilingual language token used for decoding
            (``None`` if decoding was not language-conditioned).
        model_text: Transcript as produced by the model (before transliteration).
        segments: Per-segment transcripts with time offsets.
    """

    text: str
    lang: Optional[str]
    script: Optional[str]
    model_lang: Optional[str]
    model_text: str
    segments: list[ASRSegment] = field(default_factory=list)


@dataclass
class _LangPlan:
    lang: Optional[str]
    model_lang: Optional[str]
    model_script: Optional[str]
    out_script: Optional[str]
    transliterator: Any = None


def resolve_asr_language(
    lang: str,
    script: Optional[str] = None,
    check_model: bool = True,
) -> tuple[str, str, str]:
    """Map a TurkicNLP language to an Omnilingual ASR language token.

    Args:
        lang: TurkicNLP ISO 639-3 code (``kaz``) or an Omnilingual token of a
            covered Turkic language (``kaz_Cyrl``).
        script: Desired output script (``Latn``, ``Cyrl``, ``Arab``); defaults
            to the language's primary TurkicNLP script.
        check_model: Verify the token against the model's ``supported_langs``.

    Returns:
        ``(turkicnlp_lang, omnilingual_code, model_script)``.

    Raises:
        ValueError: If the language is not a Turkic language covered by
            TurkicNLP, or has no ASR support.
    """
    if "_" in lang:
        if lang not in _CODE_TO_LANG:
            raise ValueError(
                f"'{lang}' is not an Omnilingual code of a Turkic language covered "
                f"by TurkicNLP. Covered: {sorted(_CODE_TO_LANG)}"
            )
        base, code_script = _CODE_TO_LANG[lang]
        if check_model:
            supported = omniasr_supported_langs()
            if lang not in supported:
                raise ValueError(f"'{lang}' is not supported by the installed Omnilingual ASR model.")
        return base, lang, code_script

    if lang in UNSUPPORTED_ASR_LANGS:
        raise ValueError(
            f"Speech recognition is not available for {UNSUPPORTED_ASR_LANGS[lang]} "
            f"('{lang}'): Omnilingual ASR has no language token for it."
        )
    if lang not in OMNIASR_LANG_CODES:
        raise ValueError(
            f"'{lang}' is not a Turkic language covered by TurkicNLP speech recognition. "
            f"Supported: {sorted(OMNIASR_LANG_CODES)}"
        )

    scripts = OMNIASR_LANG_CODES[lang]
    target = script or str(get_script_config(lang).primary)
    order = ([target] if target in scripts else []) + [s for s in scripts if s != target]
    supported = omniasr_supported_langs() if check_model else None
    for scr in order:
        for code in scripts[scr]:
            if supported is None or code in supported:
                return lang, code, scr
    raise ValueError(
        f"None of the Omnilingual codes for '{lang}' "
        f"({[c for cs in scripts.values() for c in cs]}) are supported by the installed model."
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _to_mono_float(waveform: Any) -> Any:
    import numpy as np

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 2:
        # (frames, channels) as returned by soundfile, or (channels, frames)
        axis = 1 if arr.shape[0] >= arr.shape[1] else 0
        arr = arr.mean(axis=axis)
    elif arr.ndim != 1:
        raise ValueError(f"Expected a 1-D or 2-D waveform, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_audio(audio: AudioInput, sample_rate: Optional[int] = None) -> tuple[Any, int]:
    """Decode an audio input to a mono float32 waveform.

    Args:
        audio: File path, raw encoded bytes, a waveform array (needs
            ``sample_rate``), or a dict with ``waveform``/``array`` and
            ``sample_rate``/``sampling_rate`` keys (e.g. a Hugging Face
            ``datasets`` audio column).
        sample_rate: Sample rate of a raw waveform array.

    Returns:
        ``(waveform, sample_rate)``.
    """
    if isinstance(audio, dict):
        wav = audio.get("waveform", audio.get("array"))
        sr = audio.get("sample_rate", audio.get("sampling_rate"))
        if wav is None or sr is None:
            raise ValueError("Audio dicts need 'waveform' (or 'array') and 'sample_rate' keys.")
        return _to_mono_float(wav), int(sr)

    if isinstance(audio, (str, Path)) and not Path(audio).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    if isinstance(audio, (str, Path, bytes, bytearray)):
        source = io.BytesIO(bytes(audio)) if isinstance(audio, (bytes, bytearray)) else str(audio)
        try:
            import soundfile as sf

            data, sr = sf.read(source, dtype="float32", always_2d=False)
            return _to_mono_float(data), int(sr)
        except ImportError:
            pass
        try:
            import torchaudio

            if isinstance(source, io.BytesIO):
                source.seek(0)
            wav, sr = torchaudio.load(source)
            return _to_mono_float(wav.numpy()), int(sr)
        except ImportError as exc:
            raise ImportError(
                "Decoding audio files requires `soundfile` (pip install soundfile) "
                "or `torchaudio`."
            ) from exc

    # numpy array / torch tensor / list
    if sample_rate is None:
        raise ValueError("A raw waveform needs `sample_rate`.")
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return _to_mono_float(audio), int(sample_rate)


def split_audio(
    waveform: Any,
    sample_rate: int,
    max_seconds: float = DEFAULT_SEGMENT_SECONDS,
    search_seconds: float = 5.0,
    frame_seconds: float = 0.05,
) -> list[tuple[int, int]]:
    """Split a waveform into segments of at most ``max_seconds``.

    Each cut is placed at the quietest ``frame_seconds`` frame within the last
    ``search_seconds`` of the window, so words are rarely cut in half.

    Returns:
        List of ``(start_sample, end_sample)`` pairs covering the waveform.
    """
    import numpy as np

    n = len(waveform)
    max_len = int(max_seconds * sample_rate)
    if max_len <= 0:
        raise ValueError("max_seconds must be positive.")
    if n <= max_len:
        return [(0, n)]

    frame = max(1, int(frame_seconds * sample_rate))
    search = min(int(search_seconds * sample_rate), max_len // 2)
    bounds: list[tuple[int, int]] = []
    start = 0
    while n - start > max_len:
        win_lo = start + max_len - search
        win_hi = start + max_len
        region = np.asarray(waveform[win_lo:win_hi], dtype=np.float32)
        n_frames = max(1, len(region) // frame)
        energies = [
            float(np.mean(region[i * frame:(i + 1) * frame] ** 2)) for i in range(n_frames)
        ]
        best = int(np.argmin(energies))
        cut = win_lo + best * frame + frame // 2
        cut = min(max(cut, start + 1), win_hi)
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


# ---------------------------------------------------------------------------
# Speech recognizer
# ---------------------------------------------------------------------------

def _resolve_device_and_dtype(device: Optional[str], dtype: str) -> tuple[Any, Any, str]:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = str(device)
    if dtype == "auto":
        # Half precision halves the memory of the model (omniASR_LLM_1B: ~9 GB in
        # float32, ~4.5 GB in bfloat16). MPS handles float16 more reliably.
        dtype = "float16" if dev.startswith("mps") else "bfloat16"
    torch_dtype = getattr(torch, str(dtype), None)
    if torch_dtype is None:
        raise ValueError(f"Unknown dtype '{dtype}'. Use auto, float32, float16 or bfloat16.")
    return dev, torch_dtype, str(dtype)


def _get_pipeline(
    model_card: str, device: Optional[str], dtype: str, low_memory: bool = True
) -> Any:
    """Load (or reuse) an ``ASRInferencePipeline``; one model serves all languages.

    With ``low_memory`` the checkpoint is memory-mapped instead of read into
    RAM before the weights are copied into the model. ``ASRInferencePipeline``
    itself always reads the whole checkpoint (9 GB for ``omniASR_LLM_1B``), so
    together with the model the peak exceeds 16 GB; memory-mapping keeps the
    peak close to the size of the model in the chosen dtype.
    """
    _require_omniasr()
    dev, torch_dtype, dtype_name = _resolve_device_and_dtype(device, dtype)
    key = (model_card, dev, dtype_name)
    if key not in _PIPELINE_CACHE:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        logger.info("Loading Omnilingual ASR model %s on %s (%s)", model_card, dev, dtype_name)
        if low_memory:
            import torch
            from fairseq2.data.tokenizers.hub import load_tokenizer
            from fairseq2.models.hub import load_model

            model = load_model(
                model_card, device=torch.device(dev), dtype=torch_dtype, mmap=True
            )
            tokenizer = load_tokenizer(model_card)
            _PIPELINE_CACHE[key] = ASRInferencePipeline(
                None, model=model, tokenizer=tokenizer, device=dev, dtype=torch_dtype
            )
        else:
            _PIPELINE_CACHE[key] = ASRInferencePipeline(
                model_card=model_card, device=dev, dtype=torch_dtype
            )
    return _PIPELINE_CACHE[key]


class SpeechRecognizer:
    """Speech-to-text for Turkic languages with Omnilingual ASR.

    Args:
        lang: Default TurkicNLP language code (``kaz``, ``uzb``, ...). Can be
            overridden per call. ``None`` decodes without language
            conditioning (lower quality with LLM models).
        model_card: Omnilingual model card, e.g. ``omniASR_LLM_1B`` (default),
            ``omniASR_LLM_300M``, ``omniASR_LLM_7B`` or a CTC model
            (``omniASR_CTC_1B``; CTC models ignore the language).
        script: Output script (``Latn``, ``Cyrl``, ``Arab``); defaults to the
            language's primary TurkicNLP script.
        device: ``cuda``, ``cpu``, ... (default: CUDA if available).
        dtype: ``auto`` (bfloat16; float16 on Apple ``mps``), ``float32``,
            ``float16`` or ``bfloat16``. float32 doubles the memory use.
        batch_size: Default number of audio segments per forward pass.
        max_segment_seconds: Longer audio is split into segments of at most
            this length (must stay below the model limit of 40 s).
        check_model_langs: Validate language codes against the model's
            ``supported_langs`` list.
        low_memory: Memory-map the checkpoint while loading (recommended on
            machines with 16 GB RAM or less).
    """

    def __init__(
        self,
        lang: Optional[str] = None,
        model_card: str = DEFAULT_MODEL_CARD,
        script: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        batch_size: int = 2,
        max_segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
        check_model_langs: bool = True,
        low_memory: bool = True,
    ) -> None:
        if not 0 < max_segment_seconds <= MAX_AUDIO_SECONDS:
            raise ValueError(
                f"max_segment_seconds must be in (0, {MAX_AUDIO_SECONDS}] "
                "(Omnilingual ASR limit)."
            )
        self.lang = lang
        self.model_card = model_card
        self.script = script
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_segment_seconds = max_segment_seconds
        self.check_model_langs = check_model_langs
        self.low_memory = low_memory
        self._pipeline: Any = None
        if lang is not None:
            # Fail early on languages outside TurkicNLP's ASR coverage; the
            # check against the model's supported_langs happens at first use.
            resolve_asr_language(lang, script, check_model=False)

    # -- setup ------------------------------------------------------------
    def load(self) -> "SpeechRecognizer":
        """Load the model now instead of on the first call."""
        if self._pipeline is None:
            self._pipeline = _get_pipeline(
                self.model_card, self.device, self.dtype, self.low_memory
            )
        return self

    def _plan(self, lang: Optional[str], script: Optional[str]) -> _LangPlan:
        if lang is None:
            return _LangPlan(None, None, None, None)
        base, code, model_script = resolve_asr_language(
            lang, script, check_model=self.check_model_langs
        )
        out_script = script or (
            model_script if "_" in lang else str(get_script_config(base).primary)
        )
        translit = None
        if out_script != model_script:
            from turkicnlp.scripts.transliterator import Transliterator

            try:
                translit = Transliterator(base, Script(model_script), Script(out_script))
            except ValueError as exc:
                raise ValueError(
                    f"The model writes '{base}' in {model_script}, and TurkicNLP cannot "
                    f"transliterate {model_script}->{out_script} for it."
                ) from exc
        return _LangPlan(base, code, model_script, out_script, translit)

    # -- inference --------------------------------------------------------
    def transcribe(
        self,
        audio: Union[AudioInput, Sequence[AudioInput]],
        lang: Union[str, Sequence[Optional[str]], None] = None,
        *,
        script: Optional[str] = None,
        sample_rate: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_details: bool = False,
    ) -> Union[str, ASRResult, list[str], list[ASRResult]]:
        """Transcribe one or more audio inputs.

        Mirrors ``ASRInferencePipeline.transcribe`` but takes TurkicNLP
        language codes, accepts long audio, and returns text in TurkicNLP
        scripts.

        Args:
            audio: One input or a list of inputs (file paths, encoded bytes,
                waveform arrays with ``sample_rate``, or dicts with
                ``waveform``/``array`` and ``sample_rate``/``sampling_rate``).
            lang: Language code for all inputs or a list with one code per
                input. Defaults to the recognizer's ``lang``.
            script: Output script for this call.
            sample_rate: Sample rate for raw waveform arrays.
            batch_size: Segments per forward pass.
            return_details: Return :class:`ASRResult` objects with segments
                instead of plain strings.

        Returns:
            A string (or :class:`ASRResult`) for a single input, a list for a
            list of inputs.
        """
        single = not isinstance(audio, (list, tuple))
        inputs = [audio] if single else list(audio)
        if not inputs:
            return []

        if lang is None or isinstance(lang, str):
            langs: list[Optional[str]] = [lang if lang is not None else self.lang] * len(inputs)
        else:
            langs = list(lang)
            if len(langs) != len(inputs):
                raise ValueError(
                    f"`lang` must have one entry per input ({len(inputs)}), got {len(langs)}."
                )
        out_script = script if script is not None else self.script

        plans: dict[Optional[str], _LangPlan] = {}
        for code in langs:
            if code not in plans:
                plans[code] = self._plan(code, out_script)

        # Decode and segment every input
        seg_audio: list[dict] = []
        seg_langs: list[Optional[str]] = []
        owners: list[tuple[int, float, float]] = []
        for idx, (item, code) in enumerate(zip(inputs, langs)):
            wav, sr = load_audio(item, sample_rate)
            for s, e in split_audio(wav, sr, self.max_segment_seconds):
                seg_audio.append({"waveform": wav[s:e], "sample_rate": sr})
                seg_langs.append(plans[code].model_lang)
                owners.append((idx, s / sr, e / sr))

        self.load()
        texts = self._pipeline.transcribe(
            seg_audio, lang=seg_langs, batch_size=batch_size or self.batch_size
        )

        results: list[ASRResult] = []
        for idx, code in enumerate(langs):
            plan = plans[code]
            segs = []
            for (owner, start, end), raw in zip(owners, texts):
                if owner != idx:
                    continue
                raw = (raw or "").strip()
                text = plan.transliterator.transliterate(raw) if plan.transliterator else raw
                segs.append(ASRSegment(round(start, 3), round(end, 3), text))
            model_text = " ".join(
                (raw or "").strip()
                for (owner, _, _), raw in zip(owners, texts)
                if owner == idx and (raw or "").strip()
            )
            results.append(
                ASRResult(
                    text=" ".join(s.text for s in segs if s.text),
                    lang=plan.lang,
                    script=plan.out_script,
                    model_lang=plan.model_lang,
                    model_text=model_text,
                    segments=segs,
                )
            )

        if return_details:
            return results[0] if single else results
        plain = [r.text for r in results]
        return plain[0] if single else plain

    def __repr__(self) -> str:
        return (
            f"SpeechRecognizer(lang={self.lang!r}, model_card={self.model_card!r}, "
            f"loaded={self._pipeline is not None})"
        )


# Short alias
ASR = SpeechRecognizer
