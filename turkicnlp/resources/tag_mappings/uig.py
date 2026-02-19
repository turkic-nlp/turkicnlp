"""Uyghur Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class UyghurTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Uyghur (apertium-uig)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        "ifi": "Evident=Nfh",
        "prog": "Aspect=Prog",
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
        "qst": "PartType=Int",
    }
