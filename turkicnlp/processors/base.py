"""
Base processor interface for TurkicNLP.

All NLP components implement the :class:`Processor` abstract base class.
Each processor declares what it provides (e.g. ``pos``, ``lemma``), what
it requires (e.g. ``tokenize``), and processes a :class:`Document` in place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from turkicnlp.models.document import Document
    from turkicnlp.scripts import Script


class ProcessorRequirementsError(Exception):
    """Raised when a processor's dependencies are not met."""


class Processor(ABC):
    """Abstract base class for all NLP processors.

    Subclasses must set the class attributes :attr:`NAME`, :attr:`PROVIDES`,
    and :attr:`REQUIRES`, and implement :meth:`load` and :meth:`process`.

    Attributes:
        NAME: Processor name used in pipeline configuration.
        PROVIDES: Annotation layers this processor produces.
        REQUIRES: Annotation layers that must already exist on the Document.
        SUPPORTED_SCRIPTS: Scripts this processor can handle, or ``None``
            for script-agnostic processors.
    """

    NAME: str = ""
    PROVIDES: list[str] = []
    REQUIRES: list[str] = []
    SUPPORTED_SCRIPTS: Optional[list[Script]] = None

    def __init__(
        self,
        lang: str,
        script: Optional[Script] = None,
        config: Optional[dict] = None,
    ) -> None:
        """
        Args:
            lang: ISO 639-3 language code.
            script: The script this processor instance is configured for.
            config: Processor-specific configuration overrides.
        """
        self.lang = lang
        self.script = script
        self.config = config or {}
        self._loaded = False

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights, FST, or other resources from disk.

        Args:
            model_path: Path to the model directory or file.
        """
        ...

    @abstractmethod
    def process(self, doc: Document) -> Document:
        """Annotate the document in place and return it.

        The processor should only write to fields it is responsible for
        and never overwrite fields set by other processors.

        Args:
            doc: The document to annotate.

        Returns:
            The same document, annotated in place.
        """
        ...

    def check_requirements(self, doc: Document) -> None:
        """Verify that required annotations and script compatibility are present.

        Args:
            doc: The document to check.

        Raises:
            ProcessorRequirementsError: If requirements are not satisfied.
        """
        if "tokenize" in self.REQUIRES and not doc.sentences:
            raise ProcessorRequirementsError(
                f"Processor '{self.NAME}' requires tokenization, "
                f"but document has no sentences."
            )
        if "pos" in self.REQUIRES:
            if not doc.words or doc.words[0].upos is None:
                raise ProcessorRequirementsError(
                    f"Processor '{self.NAME}' requires POS tags."
                )

        if self.SUPPORTED_SCRIPTS is not None and doc.script:
            from turkicnlp.scripts import Script

            doc_script = Script(doc.script)
            if doc_script not in self.SUPPORTED_SCRIPTS:
                raise ProcessorRequirementsError(
                    f"Processor '{self.NAME}' supports scripts "
                    f"{[str(s) for s in self.SUPPORTED_SCRIPTS]}, "
                    f"but document script is '{doc.script}'."
                )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(lang={self.lang}, script={self.script})>"
