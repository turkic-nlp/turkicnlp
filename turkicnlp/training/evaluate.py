"""Evaluation utilities for TurkicNLP models."""

from __future__ import annotations

from typing import Optional


def evaluate_pos(
    gold_path: str, pred_path: str, lang: Optional[str] = None
) -> dict[str, float]:
    """Evaluate POS tagging accuracy.

    Args:
        gold_path: Path to gold-standard CoNLL-U file.
        pred_path: Path to predicted CoNLL-U file.
        lang: Optional language code for reporting.

    Returns:
        Dict with ``upos_acc``, ``feats_acc``, ``all_tags_acc``.
    """
    raise NotImplementedError("evaluate_pos() not yet implemented.")


def evaluate_depparse(
    gold_path: str, pred_path: str, lang: Optional[str] = None
) -> dict[str, float]:
    """Evaluate dependency parsing.

    Args:
        gold_path: Path to gold-standard CoNLL-U file.
        pred_path: Path to predicted CoNLL-U file.
        lang: Optional language code for reporting.

    Returns:
        Dict with ``uas``, ``las``, ``clas``.
    """
    raise NotImplementedError("evaluate_depparse() not yet implemented.")


def evaluate_ner(
    gold_path: str, pred_path: str, lang: Optional[str] = None
) -> dict[str, float]:
    """Evaluate NER performance.

    Args:
        gold_path: Path to gold-standard data.
        pred_path: Path to predicted data.
        lang: Optional language code for reporting.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``.
    """
    raise NotImplementedError("evaluate_ner() not yet implemented.")
