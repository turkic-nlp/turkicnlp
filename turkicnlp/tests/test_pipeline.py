"""Tests for the Pipeline orchestrator."""

from __future__ import annotations

import pytest

from turkicnlp.pipeline import PROCESSOR_ORDER, Pipeline


class TestProcessorOrder:
    def test_canonical_order(self) -> None:
        assert "tokenize" in PROCESSOR_ORDER
        assert PROCESSOR_ORDER.index("tokenize") < PROCESSOR_ORDER.index("pos")
        assert PROCESSOR_ORDER.index("pos") < PROCESSOR_ORDER.index("depparse")

    def test_script_steps_in_order(self) -> None:
        assert "script_detect" in PROCESSOR_ORDER
        assert "transliterate" in PROCESSOR_ORDER
        assert "transliterate_back" in PROCESSOR_ORDER

    def test_embeddings_in_order(self) -> None:
        assert "embeddings" in PROCESSOR_ORDER
        assert PROCESSOR_ORDER.index("embeddings") < PROCESSOR_ORDER.index("sentiment")


class TestPipelineInit:
    def test_invalid_script_raises(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            Pipeline("tur", script="Cyrl")

    def test_valid_script(self) -> None:
        # Should not raise
        pipe = Pipeline("kaz", script="Cyrl")
        assert pipe.lang == "kaz"

    def test_resolve_noncanonical_processor(self) -> None:
        pipe = Pipeline("tur", processors=None)
        resolved = pipe._resolve_dependencies(["translate"])
        assert "translate" in resolved


class TestDefaultProcessors:
    def test_optional_processors_are_not_loaded_by_default(self) -> None:
        from turkicnlp.pipeline import OPTIONAL_PROCESSORS, default_processors

        catalog = {p: {} for p in [
            "morph", "tokenize", "pos", "lemma", "depparse", "ner",
            "embeddings", "translate", "morph_neural", "feats", "asr",
        ]}
        names = default_processors(catalog)
        assert not set(names) & OPTIONAL_PROCESSORS
        assert "morph" in names and "morph_neural" not in names
        assert "tokenize" in names and "depparse" in names

    def test_morph_neural_kept_without_apertium(self) -> None:
        from turkicnlp.pipeline import default_processors

        names = default_processors({"tokenize": {}, "morph_neural": {}, "translate": {}})
        assert names == ["tokenize", "morph_neural"]
