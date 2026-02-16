"""Tests for the Document data model."""

from __future__ import annotations

from turkicnlp.models.document import Document, Sentence, Token, Word, Span


class TestWord:
    def test_create_word(self) -> None:
        word = Word(id=1, text="мектеп", lemma="мектеп", upos="NOUN")
        assert word.text == "мектеп"
        assert word.upos == "NOUN"

    def test_conllu_line(self) -> None:
        word = Word(id=1, text="мектепке", lemma="мектеп", upos="NOUN", feats="Case=Dat|Number=Sing")
        line = word.to_conllu_line()
        assert "мектепке" in line
        assert "NOUN" in line

    def test_word_with_script(self) -> None:
        word = Word(id=1, text="test", script="Latn")
        line = word.to_conllu_line()
        assert "Script=Latn" in line


class TestToken:
    def test_simple_token(self) -> None:
        word = Word(id=1, text="мен")
        token = Token(id=(1,), text="мен", words=[word])
        assert not token.is_mwt

    def test_mwt_token(self) -> None:
        token = Token(id=(1, 2), text="gidiyorum")
        assert token.is_mwt


class TestSentence:
    def test_dependencies(self) -> None:
        words = [
            Word(id=1, text="Мен", head=3, deprel="nsubj"),
            Word(id=2, text="мектепке", head=3, deprel="obl"),
            Word(id=3, text="бардым", head=0, deprel="root"),
        ]
        sent = Sentence(text="Мен мектепке бардым", words=words)
        deps = sent.dependencies
        assert len(deps) == 3

    def test_to_conllu(self) -> None:
        word = Word(id=1, text="Мен", lemma="мен", upos="PRON")
        token = Token(id=(1,), text="Мен", words=[word])
        sent = Sentence(text="Мен", tokens=[token], words=[word])
        conllu = sent.to_conllu()
        assert "# text = Мен" in conllu


class TestDocument:
    def test_empty_document(self) -> None:
        doc = Document(text="test")
        assert doc.words == []
        assert doc.entities == []

    def test_to_conllu_with_metadata(self) -> None:
        doc = Document(text="test", lang="kaz", script="Cyrl")
        conllu = doc.to_conllu()
        assert "# lang = kaz" in conllu
        assert "# script = Cyrl" in conllu

    def test_to_dict(self) -> None:
        word = Word(id=1, text="test", lemma="test", upos="NOUN")
        sent = Sentence(text="test", words=[word])
        doc = Document(text="test", sentences=[sent])
        d = doc.to_dict()
        assert len(d) == 1
        assert d[0][0]["text"] == "test"
