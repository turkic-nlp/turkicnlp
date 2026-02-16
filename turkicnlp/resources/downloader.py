"""
Model and FST data download manager for TurkicNLP.

Downloads neural model archives and Apertium FST transducers on demand.
Apertium data (GPL-3.0) is stored separately with license files alongside.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

from turkicnlp.resources.registry import ModelRegistry


def download(
    lang: str,
    processors: Optional[list[str]] = None,
    script: Optional[str] = None,
    model_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download models and FST data for a language.

    Apertium FST files are GPL-3.0 licensed and stored separately from
    the Apache-2.0 library code.

    Args:
        lang: ISO 639-3 language code.
        processors: Processors to download. ``None`` downloads all available.
        script: Script to download for (e.g. ``Cyrl``). ``None`` downloads all.
        model_dir: Override default model directory.
        force: Re-download even if already present.
    """
    raise NotImplementedError("download() not yet implemented.")


def list_languages() -> list[dict]:
    """List all supported languages with available processors and licenses.

    Returns:
        List of dicts with keys ``code``, ``name``, ``scripts``, ``processors``.
    """
    raise NotImplementedError("list_languages() not yet implemented.")


def list_processors(lang: str) -> dict[str, list[str]]:
    """List available processors and backends for a language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        Dict mapping processor names to lists of available backends.
    """
    raise NotImplementedError("list_processors() not yet implemented.")


def list_scripts(lang: str) -> dict:
    """Show available scripts and per-script processors for a language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        Dict with keys ``available``, ``primary``, ``processors``.
    """
    raise NotImplementedError("list_scripts() not yet implemented.")


def _download_apertium_fst(
    lang: str, script: str, proc_name: str, backend_info: dict, dest: Path
) -> None:
    """Download Apertium FST files (GPL-3.0) with license isolation."""
    raise NotImplementedError


def _download_neural_model(
    lang: str, script: str, proc_name: str, backend_info: dict, dest: Path
) -> None:
    """Download a neural model archive."""
    raise NotImplementedError


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    """Simple download progress indicator."""
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    print(f"\r  {min(percent, 100)}%", end="", flush=True)
    if percent >= 100:
        print()
