"""Training script for the neural tokenizer."""

from __future__ import annotations


def train_tokenizer(
    train_data: str,
    dev_data: str,
    output_dir: str,
    lang: str,
    script: str = "Latn",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    """Train a character-level BiLSTM tokenizer.

    Args:
        train_data: Path to training data (raw text or CoNLL-U).
        dev_data: Path to development data.
        output_dir: Directory to save the trained model.
        lang: ISO 639-3 language code.
        script: Script code.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
    """
    raise NotImplementedError("train_tokenizer() not yet implemented.")
