"""
TurkicNLP: NLP toolkit for 20+ Turkic languages.

A pip-installable Python NLP library inspired by Stanza's modular architecture,
with adaptations for the low-resource, morphologically rich Turkic language family.

Usage:
    import turkicnlp

    turkicnlp.download("kaz")
    nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "pos", "lemma", "depparse"])
    doc = nlp("Мен мектепке бардым")

    for sentence in doc.sentences:
        for word in sentence.words:
            print(word.text, word.lemma, word.upos, word.feats)

License:
    Apache-2.0 (library code)
    Apertium FST data (GPL-3.0) is downloaded separately at runtime.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("turkicnlp")
except PackageNotFoundError:
    __version__ = "unknown"

from turkicnlp.pipeline import Pipeline
from turkicnlp.resources.downloader import download, list_languages, list_processors

__all__ = ["Pipeline", "download", "list_languages", "list_processors"]
