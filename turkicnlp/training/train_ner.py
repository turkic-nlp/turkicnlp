"""Training script for the NER model."""

from __future__ import annotations


def train_ner(
    train_data: str,
    dev_data: str,
    output_dir: str,
    lang: str,
    script: str = "Latn",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Train an NER model on BIO-tagged data.

    Args:
        train_data: Path to training data (CoNLL-U or BIO format).
        dev_data: Path to development data.
        output_dir: Directory to save the trained model.
        lang: ISO 639-3 language code.
        script: Script code.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
    """
    raise NotImplementedError("train_ner() not yet implemented.")
