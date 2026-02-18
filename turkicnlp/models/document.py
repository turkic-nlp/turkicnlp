"""
Core data model for TurkicNLP.

Implements the Document → Sentence → Token → Word hierarchy,
mapping directly to CoNLL-U format. All processors read from
and write to this structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """A syntactic word — corresponds to one line in CoNLL-U.

    Attributes:
        id: Word index within the sentence (1-based).
        text: Surface form of the word.
        lemma: Base/dictionary form.
        upos: Universal POS tag (UD tagset).
        xpos: Language-specific POS tag.
        feats: Morphological features in UD format (e.g. ``Case=Dat|Number=Sing``).
        head: Head word ID (0 = root).
        deprel: Dependency relation to head.
        deps: Enhanced dependency graph.
        misc: Miscellaneous annotations (SpaceAfter, Script, etc.).
        ner: NER tag in BIO format (e.g. ``B-PER``, ``I-LOC``, ``O``).
        start_char: Character offset start in original text.
        end_char: Character offset end in original text.
        script: Script of this word if different from document default.
    """

    id: int
    text: str
    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: Optional[str] = None
    head: Optional[int] = None
    deprel: Optional[str] = None
    deps: Optional[str] = None
    misc: Optional[str] = None
    ner: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    script: Optional[str] = None

    _provenance: dict = field(default_factory=dict, repr=False)

    def to_conllu_line(self) -> str:
        """Format as a single CoNLL-U line."""
        misc = self.misc or "_"
        if self.script:
            if misc == "_":
                misc = f"Script={self.script}"
            else:
                misc += f"|Script={self.script}"

        fields = [
            str(self.id),
            self.text,
            self.lemma or "_",
            self.upos or "_",
            self.xpos or "_",
            self.feats or "_",
            str(self.head) if self.head is not None else "_",
            self.deprel or "_",
            self.deps or "_",
            misc,
        ]
        return "\t".join(fields)


@dataclass
class Token:
    """A raw token from text.

    Usually contains one :class:`Word`, but multi-word tokens (MWTs)
    contain multiple Words. For most Turkic languages, Tokens and Words
    are 1:1.

    Attributes:
        id: ``(i,)`` for simple tokens, ``(i, j)`` for MWT spanning words i–j.
        text: Surface text of the token.
        words: Constituent syntactic words.
        start_char: Character offset start.
        end_char: Character offset end.
    """

    id: tuple[int, ...]
    text: str
    words: list[Word] = field(default_factory=list)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @property
    def is_mwt(self) -> bool:
        """Return ``True`` if this is a multi-word token."""
        return len(self.id) == 2


@dataclass
class Span:
    """A contiguous span of text, used for NER entities, chunks, etc.

    Attributes:
        text: The span text.
        type: Entity type (``PER``, ``ORG``, ``LOC``, etc.).
        start_char: Character offset start.
        end_char: Character offset end.
        words: References to constituent :class:`Word` objects.
    """

    text: str
    type: str
    start_char: int
    end_char: int
    words: list[Word] = field(default_factory=list)


@dataclass
class Sentence:
    """A single sentence with all its annotations.

    Attributes:
        text: Raw sentence text.
        tokens: List of raw tokens (may include MWTs).
        words: List of syntactic words (after MWT expansion).
        entities: Named entity spans.
        sentiment: Sentence-level sentiment label.
        embedding: Sentence-level dense vector representation.
        translation: Sentence-level machine translation output.
    """

    text: str
    tokens: list[Token] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    entities: list[Span] = field(default_factory=list)
    sentiment: Optional[str] = None
    embedding: Optional[list[float]] = None
    translation: Optional[str] = None

    @property
    def dependencies(self) -> list[tuple[Optional[Word], str, Word]]:
        """Return dependency triples ``(head_word, deprel, dep_word)``."""
        deps: list[tuple[Optional[Word], str, Word]] = []
        for word in self.words:
            if word.head is not None and word.deprel is not None:
                head = self.words[word.head - 1] if word.head > 0 else None
                deps.append((head, word.deprel, word))
        return deps

    def to_conllu(self) -> str:
        """Export sentence as a CoNLL-U block."""
        lines = [f"# text = {self.text}"]
        for token in self.tokens:
            if token.is_mwt:
                lines.append(
                    f"{token.id[0]}-{token.id[1]}\t{token.text}" + "\t_" * 8
                )
            for word in token.words:
                lines.append(word.to_conllu_line())
        return "\n".join(lines)


@dataclass
class Document:
    """Top-level container for a processed text.

    Attributes:
        text: Original raw text.
        sentences: List of annotated sentences.
        script: Detected or specified script code (``Cyrl``, ``Latn``, ``Arab``).
        script_segments: For mixed-script docs: ``[(start, end, script), ...]``.
        lang: ISO 639-3 language code, set by :class:`Pipeline`.
        embedding: Document-level dense vector representation.
        translation: Document-level machine translation output.
    """

    text: str
    sentences: list[Sentence] = field(default_factory=list)
    _processor_log: list[str] = field(default_factory=list, repr=False)
    script: Optional[str] = None
    script_segments: Optional[list[tuple[int, int, str]]] = None
    lang: Optional[str] = None
    embedding: Optional[list[float]] = None
    translation: Optional[str] = None
    _original_text: Optional[str] = field(default=None, repr=False)

    @property
    def words(self) -> list[Word]:
        """Flat list of all words across sentences."""
        return [w for s in self.sentences for w in s.words]

    @property
    def entities(self) -> list[Span]:
        """All entities across sentences."""
        return [e for s in self.sentences for e in s.entities]

    def to_conllu(self) -> str:
        """Export entire document as CoNLL-U."""
        header: list[str] = []
        if self.lang:
            header.append(f"# lang = {self.lang}")
        if self.script:
            header.append(f"# script = {self.script}")
        body = "\n\n".join(s.to_conllu() for s in self.sentences)
        if header:
            return "\n".join(header) + "\n\n" + body + "\n\n"
        return body + "\n\n"

    def to_dict(self) -> list[list[dict]]:
        """Export as nested dicts for JSON serialization."""
        return [
            [
                {
                    "id": w.id,
                    "text": w.text,
                    "lemma": w.lemma,
                    "upos": w.upos,
                    "xpos": w.xpos,
                    "feats": w.feats,
                    "head": w.head,
                    "deprel": w.deprel,
                }
                for w in s.words
            ]
            for s in self.sentences
        ]
