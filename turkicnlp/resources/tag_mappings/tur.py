"""Turkish Apertium → UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.base import TagMapper


class TurkishTagMapper(TagMapper):
    """Tag mapper for Turkish (apertium-tur)."""

    FEAT_MAP: dict[str, str] = {
        "nom": "Case=Nom",
        "gen": "Case=Gen",
        "dat": "Case=Dat",
        "acc": "Case=Acc",
        "abl": "Case=Abl",
        "loc": "Case=Loc",
        "ins": "Case=Ins",
        "sg": "Number=Sing",
        "pl": "Number=Plur",
        "p1": "Person=1",
        "p2": "Person=2",
        "p3": "Person=3",
        "past": "Tense=Past",
        "pres": "Tense=Pres",
        "fut": "Tense=Fut",
        "aor": "Tense=Aor",
        "ind": "Mood=Ind",
        "imp": "Mood=Imp",
        "opt": "Mood=Opt",
        "cond": "Mood=Cnd",
        "neces": "Mood=Nec",
    }
