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
    catalog = ModelRegistry.load_catalog()
    if lang not in catalog:
        raise ValueError(
            f"Language '{lang}' not found in catalog. "
            f"Available: {list(catalog.keys())}"
        )

    lang_info = catalog[lang]
    processors_section = lang_info.get("processors", {})
    if not processors_section:
        raise ValueError(f"No processors listed for language '{lang}'.")

    scripts_to_download = []
    if script:
        if script not in processors_section:
            raise ValueError(
                f"Script '{script}' not found for language '{lang}'. "
                f"Available: {list(processors_section.keys())}"
            )
        scripts_to_download = [script]
    else:
        scripts_to_download = list(processors_section.keys())

    base_dir = Path(model_dir) if model_dir else ModelRegistry.default_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    for scr in scripts_to_download:
        proc_data = processors_section.get(scr, {})
        for proc_name, proc_info in proc_data.items():
            if processors and proc_name not in processors:
                continue
            if not isinstance(proc_info, dict):
                continue
            backends = proc_info.get("backends", {})
            for backend_name, backend_info in backends.items():
                if not isinstance(backend_info, dict):
                    continue
                dest = base_dir / lang / scr / proc_name / backend_name
                if dest.exists() and not force:
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                backend_type = backend_info.get("type")
                if backend_type == "apertium_fst":
                    _download_apertium_fst(lang, scr, proc_name, backend_info, dest)
                elif backend_type == "neural_model":
                    _download_neural_model(lang, scr, proc_name, backend_info, dest)
                else:
                    raise ValueError(
                        f"Unknown backend type '{backend_type}' for "
                        f"{lang}/{scr}/{proc_name}/{backend_name}"
                    )


def list_languages() -> list[dict]:
    """List all supported languages with available processors and licenses.

    Returns:
        List of dicts with keys ``code``, ``name``, ``scripts``, ``processors``.
    """
    catalog = ModelRegistry.load_catalog()
    result = []
    for code, info in sorted(catalog.items()):
        entry = {
            "code": code,
            "name": info.get("name", code),
            "scripts": info.get("scripts", {}).get("available", []),
            "processors": {},
        }
        for scr, procs in info.get("processors", {}).items():
            entry["processors"][scr] = {}
            for proc_name, proc_info in procs.items():
                if isinstance(proc_info, dict):
                    entry["processors"][scr][proc_name] = {
                        "backends": list(proc_info.get("backends", {}).keys()),
                        "default": proc_info.get("default"),
                    }
        result.append(entry)
    return result


def list_processors(lang: str) -> dict[str, list[str]]:
    """List available processors and backends for a language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        Dict mapping processor names to lists of available backends.
    """
    catalog = ModelRegistry.load_catalog()
    if lang not in catalog:
        raise ValueError(
            f"Language '{lang}' not found in catalog. "
            f"Available: {list(catalog.keys())}"
        )
    processors_section = catalog[lang].get("processors", {})
    result: dict[str, set[str]] = {}
    for procs in processors_section.values():
        for proc_name, proc_info in procs.items():
            if isinstance(proc_info, dict):
                backends = list(proc_info.get("backends", {}).keys())
                result.setdefault(proc_name, set()).update(backends)
    return {k: sorted(v) for k, v in result.items()}


def list_scripts(lang: str) -> dict:
    """Show available scripts and per-script processors for a language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        Dict with keys ``available``, ``primary``, ``processors``.
    """
    catalog = ModelRegistry.load_catalog()
    if lang not in catalog:
        raise ValueError(
            f"Language '{lang}' not found in catalog. "
            f"Available: {list(catalog.keys())}"
        )
    info = catalog[lang]
    return {
        "available": info.get("scripts", {}).get("available", []),
        "primary": info.get("scripts", {}).get("primary"),
        "processors": info.get("processors", {}),
    }


def _download_apertium_fst(
    lang: str, script: str, proc_name: str, backend_info: dict, dest: Path
) -> None:
    """Download Apertium FST files (GPL-3.0) with license isolation."""
    url = backend_info.get("url")
    if not url:
        raise FileNotFoundError(
            f"No download URL configured for {lang}/{script}/{proc_name}/apertium. "
            "Update the catalog with precompiled FST URLs."
        )

    archive_path = dest / "apertium_fst.zip"
    print(f"  ↓ Downloading Apertium FST for {lang}/{script} from {url}")
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook)

    expected_sha = backend_info.get("sha256")
    if expected_sha:
        actual_sha = _sha256(archive_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {archive_path}: expected {expected_sha}, got {actual_sha}"
            )

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest)

    try:
        archive_path.unlink()
    except FileNotFoundError:
        pass

    # Normalize layout if archive contains a nested dist/<lang>/ directory.
    nested = dest / "dist" / lang
    if nested.exists():
        for item in nested.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    continue
                target.unlink()
            if item.is_dir():
                continue
            item.replace(target)
        # Clean up empty directories
        try:
            (dest / "dist").rmdir()
        except OSError:
            pass

    metadata = {
        "lang": lang,
        "script": backend_info.get("script", script),
        "backend": "apertium",
        "source": backend_info.get("source"),
        "license": backend_info.get("license", "GPL-3.0-or-later"),
        "type": "apertium_fst",
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"  ✓ {lang}/{script}/{proc_name}/apertium — ready (GPL-3.0 licensed)")


def _download_neural_model(
    lang: str, script: str, proc_name: str, backend_info: dict, dest: Path
) -> None:
    """Download a neural model archive."""
    url = backend_info.get("url")
    if not url:
        raise FileNotFoundError(
            f"No download URL configured for {lang}/{script}/{proc_name}. "
            "Update the catalog with model URLs."
        )
    archive_path = dest / "model.zip"
    print(f"  ↓ Downloading model for {lang}/{script}/{proc_name} from {url}")
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook)

    expected_sha = backend_info.get("sha256")
    if expected_sha:
        actual_sha = _sha256(archive_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {archive_path}: expected {expected_sha}, got {actual_sha}"
            )

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest)

    try:
        archive_path.unlink()
    except FileNotFoundError:
        pass

    metadata = {
        "lang": lang,
        "script": script,
        "backend": backend_info.get("backend", "neural"),
        "source": backend_info.get("source"),
        "license": backend_info.get("license", "Apache-2.0"),
        "type": "neural_model",
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


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
