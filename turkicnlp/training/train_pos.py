"""Training script for the neural POS tagger."""

from __future__ import annotations


def train_pos_tagger(
    train_data: str,
    dev_data: str,
    output_dir: str,
    lang: str,
    script: str = "Latn",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Train a POS tagger on UD treebank data.

    Args:
        train_data: Path to training CoNLL-U file.
        dev_data: Path to development CoNLL-U file.
        output_dir: Directory to save the trained model.
        lang: ISO 639-3 language code.
        script: Script code.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
    """
    raise NotImplementedError("train_pos_tagger() not yet implemented.")
