"""Sakha (Yakut) Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class SakhaTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Sakha (Yakut) (apertium-sah)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        "evid": "Evident=Nfh",
        "cvb": "VerbForm=Conv",
        "par": "Case=Par",
    }
