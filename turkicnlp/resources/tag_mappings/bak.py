"""Bashkir Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class BashkirTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Bashkir (apertium-bak)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        "evid": "Evident=Nfh",
        "cvb": "VerbForm=Conv",
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
    }
