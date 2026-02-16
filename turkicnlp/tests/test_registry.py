"""Tests for the processor and model registries."""

from __future__ import annotations

import pytest

from turkicnlp.resources.registry import ProcessorRegistry


class TestProcessorRegistry:
    def test_builtin_registrations(self) -> None:
        # These should be registered by _register_builtins()
        assert ProcessorRegistry.get("tokenize", "rule") is not None
        assert ProcessorRegistry.get("tokenize", "neural") is not None
        assert ProcessorRegistry.get("morph", "apertium") is not None
        assert ProcessorRegistry.get("pos", "neural") is not None
        assert ProcessorRegistry.get("depparse", "neural") is not None
        assert ProcessorRegistry.get("ner", "neural") is not None

    def test_unknown_processor(self) -> None:
        with pytest.raises(ValueError, match="Unknown processor"):
            ProcessorRegistry.get("nonexistent", "neural")

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            ProcessorRegistry.get("tokenize", "nonexistent")

    def test_get_any(self) -> None:
        cls = ProcessorRegistry.get_any("tokenize")
        assert cls is not None
