#!/usr/bin/env python3
"""Extract MWT rules from Universal Dependencies treebanks.

Downloads UD treebank zips for supported languages, scans CoNLL-U files,
collects MWT lines (e.g., "1-2\tgidiyorum\t_..."), and writes per-language
rules into resources/mwt_rules_extracted/{lang}.json.
"""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
import urllib.request

LANGUAGES: dict[str, str | None] = {
    "alt": None,
    "azb": None,
    "aze": None,
    "bak": None,
    "chv": None,
    "crh": None,
    "gag": None,
    "kaa": None,
    "kaz": "Kazakh",
    "kir": "Kyrgyz",
    "kjh": None,
    "krc": None,
    "kum": None,
    "nog": None,
    "ota": "Ottoman_Turkish",
    "otk": None,
    "sah": None,
    "tat": "Tatar",
    "tuk": "Turkmen",
    "tur": "Turkish",
    "tyv": None,
    "uig": "Uyghur",
    "uzb": "Uzbek",
}

UD_ALLZIP_URL = (
    "https://lindat.mff.cuni.cz/repository/server/api/core/items/"
    "b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/allzip?handleId=11234/1-6036"
)
UD_VERSION = "v2.17"
UD_ZIP_NAME = f"ud-{UD_VERSION}-allzip.zip"
UD_TGZ_NAME = f"ud-treebanks-{UD_VERSION}.tgz"


def _download_ud_allzip(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / UD_ZIP_NAME
    tgz_path = cache_dir / UD_TGZ_NAME

    if not zip_path.exists():
        print(f"Downloading UD {UD_VERSION} allzip...")
        print(UD_ALLZIP_URL)
        urllib.request.urlretrieve(UD_ALLZIP_URL, zip_path)

    if not tgz_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract(UD_TGZ_NAME, cache_dir)

    return tgz_path


def _extract_rules_from_conllu(data: str) -> Counter:
    rules: Counter = Counter()
    pending_surface = None
    pending_end = None
    pending_pieces: list[str] = []

    for line in data.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            continue
        tok_id = fields[0]

        if "-" in tok_id:
            try:
                start, end = tok_id.split("-", 1)
                pending_surface = fields[1]
                pending_end = int(end)
                pending_pieces = []
            except ValueError:
                pending_surface = None
                pending_end = None
                pending_pieces = []
            continue

        if "." in tok_id:
            continue

        try:
            wid = int(tok_id)
        except ValueError:
            continue

        if pending_surface and pending_end:
            pending_pieces.append(fields[1])
            if wid == pending_end:
                rules[(pending_surface, tuple(pending_pieces))] += 1
                pending_surface = None
                pending_end = None
                pending_pieces = []

    return rules


def _iter_conllu_files(tgz_path: Path, lang_prefix: str) -> Iterable[str]:
    prefix = f"ud-treebanks-{UD_VERSION}/UD_{lang_prefix}"
    with tarfile.open(tgz_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.startswith(prefix) and member.name.endswith(".conllu"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                yield f.read().decode("utf-8", errors="ignore")


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    cache_dir = root_dir / "data"
    out_dir = root_dir / "turkicnlp" / "resources" / "mwt_rules_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        tgz_path = _download_ud_allzip(cache_dir)
    except Exception as exc:
        print(f"[error] failed to download UD allzip: {exc}")
        return 1

    for lang, lang_prefix in LANGUAGES.items():
        if not lang_prefix:
            print(f"[skip] {lang}: no UD language mapping")
            continue

        rules = Counter()
        for conllu_text in _iter_conllu_files(tgz_path, lang_prefix):
            rules.update(_extract_rules_from_conllu(conllu_text))

        by_surface: dict[str, Counter] = {}
        for (surface, pieces), count in rules.items():
            by_surface.setdefault(surface, Counter())[pieces] += count

        rule_list = []
        for surface, splits in sorted(by_surface.items()):
            pieces, count = splits.most_common(1)[0]
            rule_list.append(
                {
                    "surface": surface,
                    "split": list(pieces),
                    "count": count,
                }
            )
        out_path = out_dir / f"{lang}.json"
        out_path.write_text(json.dumps({"rules": rule_list}, indent=2, ensure_ascii=True) + "\n")
        print(f"[ok] {lang}: {len(rule_list)} rules")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
