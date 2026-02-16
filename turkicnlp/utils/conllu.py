"""
CoNLL-U format parsing and writing utilities.

Provides functions to read CoNLL-U files into :class:`Document` objects
and write annotated documents back to CoNLL-U format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from turkicnlp.models.document import Document, Sentence, Token, Word


def parse_conllu(text: str) -> Document:
    """Parse a CoNLL-U formatted string into a :class:`Document`.

    Args:
        text: CoNLL-U formatted text.

    Returns:
        A :class:`Document` containing the parsed sentences.
    """
    raise NotImplementedError("parse_conllu() not yet implemented.")


def read_conllu(path: Union[str, Path]) -> list[Document]:
    """Read a CoNLL-U file and return a list of documents.

    Args:
        path: Path to a ``.conllu`` file.

    Returns:
        List of :class:`Document` objects.
    """
    raise NotImplementedError("read_conllu() not yet implemented.")


def write_conllu(doc: Document, path: Union[str, Path]) -> None:
    """Write a document to a CoNLL-U file.

    Args:
        doc: The document to export.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc.to_conllu())
