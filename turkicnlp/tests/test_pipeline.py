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
