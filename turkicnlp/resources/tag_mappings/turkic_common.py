"""Shared Apertium -> UD tag mapping for Turkic languages.

This mapper is used for languages that currently do not have a dedicated
language-specific mapping module yet. It captures common Turkic morphology
tags used across Apertium analyzers.
"""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.base import TagMapper


class CommonTurkicTagMapper(TagMapper):
    """Common Turkic mapper used as a strong default for multiple languages."""

    FEAT_MAP: dict[str, str] = {
        # Case
        "nom": "Case=Nom",
        "gen": "Case=Gen",
        "dat": "Case=Dat",
        "acc": "Case=Acc",
        "abl": "Case=Abl",
        "loc": "Case=Loc",
        "ins": "Case=Ins",
        "equ": "Case=Equ",
        # Number
        "sg": "Number=Sing",
        "pl": "Number=Plur",
        # Person
        "p1": "Person=1",
        "p2": "Person=2",
        "p3": "Person=3",
        # Clusivity / dual (rare but present in some tagsets)
        "du": "Number=Dual",
        "excl": "Clusivity=Excl",
        "incl": "Clusivity=Incl",
        # Possession
        "px1sg": "Number[psor]=Sing|Person[psor]=1",
        "px2sg": "Number[psor]=Sing|Person[psor]=2",
        "px3sg": "Number[psor]=Sing|Person[psor]=3",
        "px3sp": "Number[psor]=Sing|Person[psor]=3",
        "px1pl": "Number[psor]=Plur|Person[psor]=1",
        "px2pl": "Number[psor]=Plur|Person[psor]=2",
        "px3pl": "Number[psor]=Plur|Person[psor]=3",
        # Tense
        "past": "Tense=Past",
        "pres": "Tense=Pres",
        "fut": "Tense=Fut",
        "aor": "Tense=Aor",
        # Aspect
        "prog": "Aspect=Prog",
        "perf": "Aspect=Perf",
        # Mood
        "ind": "Mood=Ind",
        "imp": "Mood=Imp",
        "opt": "Mood=Opt",
        "cond": "Mood=Cnd",
        "neces": "Mood=Nec",
        # Verb form / polarity / voice
        "inf": "VerbForm=Inf",
        "ger": "VerbForm=Ger",
        "part": "VerbForm=Part",
        "neg": "Polarity=Neg",
        "pass": "Voice=Pass",
        "rcp": "Voice=Rcp",
        "rfl": "Voice=Rfl",
        # Degree
        "comp": "Degree=Cmp",
        "sup": "Degree=Sup",
        # Gender (rare for Turkic but present in some tagsets)
        "f": "Gender=Fem",
        "m": "Gender=Masc",
        "n": "Gender=Neut",
        # Pronouns / particles
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
        "qst": "PartType=Int",
        "int": "PronType=Int",
        "rel": "PronType=Rel",
        "ind": "PronType=Ind",
        "refl": "Reflex=Yes",
    }
