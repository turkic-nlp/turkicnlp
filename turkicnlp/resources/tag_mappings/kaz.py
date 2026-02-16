"""Kazakh Apertium → UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.base import TagMapper


class KazakhTagMapper(TagMapper):
    """Tag mapper for Kazakh (apertium-kaz)."""

    FEAT_MAP: dict[str, str] = {
        # Case
        "nom": "Case=Nom",
        "gen": "Case=Gen",
        "dat": "Case=Dat",
        "acc": "Case=Acc",
        "abl": "Case=Abl",
        "loc": "Case=Loc",
        "ins": "Case=Ins",
        # Number
        "sg": "Number=Sing",
        "pl": "Number=Plur",
        # Person
        "p1": "Person=1",
        "p2": "Person=2",
        "p3": "Person=3",
        # Tense
        "past": "Tense=Past",
        "pres": "Tense=Pres",
        "fut": "Tense=Fut",
        "aor": "Tense=Aor",
        # Mood
        "ind": "Mood=Ind",
        "imp": "Mood=Imp",
        "opt": "Mood=Opt",
        "cond": "Mood=Cnd",
        # Possession
        "px1sg": "Number[psor]=Sing|Person[psor]=1",
        "px2sg": "Number[psor]=Sing|Person[psor]=2",
        "px3sp": "Number[psor]=Sing|Person[psor]=3",
    }
