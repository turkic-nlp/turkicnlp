"""Tests for CoNLL-U parsing utilities."""

from __future__ import annotations

from turkicnlp.utils.conllu import parse_conllu


def test_parse_conllu_basic() -> None:
    text = (
        "# sent_id = 1\n"
        "# text = Мен мектепке бардым .\n"
        "1\tМен\tмен\tPRON\t_\tCase=Nom|Number=Sing\t3\tnsubj\t_\t_\n"
        "2\tмектепке\tмектеп\tNOUN\t_\tCase=Dat|Number=Sing\t3\tobl\t_\t_\n"
        "3\tбардым\tбар\tVERB\t_\tTense=Past|Person=1\t0\troot\t_\t_\n"
        "4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_\n"
        "\n"
    )
    doc = parse_conllu(text)
    assert len(doc.sentences) == 1
    sent = doc.sentences[0]
    assert sent.text == "Мен мектепке бардым ."
    assert [w.text for w in sent.words] == ["Мен", "мектепке", "бардым", "."]
    assert sent.words[0].lemma == "мен"
    assert sent.words[0].upos == "PRON"
    assert sent.words[1].feats == "Case=Dat|Number=Sing"
    assert sent.words[2].head == 0
    assert sent.words[2].deprel == "root"


def test_parse_conllu_mwt_and_script() -> None:
    text = (
        "# lang = tur\n"
        "# script = Latn\n"
        "# text = gidiyorum .\n"
        "1-2\tgidiyorum\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tgid\tgit\tVERB\t_\tTense=Pres\t0\troot\t_\t_\n"
        "2\tiyorum\t_\tAUX\t_\t_\t1\taux\t_\t_\n"
        "3\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\tScript=Latn\n"
        "\n"
    )
    doc = parse_conllu(text)
    assert doc.lang == "tur"
    assert doc.script == "Latn"
    assert len(doc.sentences) == 1
    sent = doc.sentences[0]
    assert len(sent.tokens) == 2
    assert sent.tokens[0].is_mwt
    assert sent.tokens[0].id == (1, 2)
    assert [w.text for w in sent.words] == ["gid", "iyorum", "."]
    assert sent.words[-1].script == "Latn"
