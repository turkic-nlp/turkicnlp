"""Chuvash Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class ChuvashTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Chuvash (apertium-chv)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        "evid": "Evident=Nfh",
        "cvb": "VerbForm=Conv",
        "prl": "Case=Prol",
        "ter": "Case=Ter",
    }
