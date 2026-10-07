"""Gagauz Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class GagauzTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Gagauz (apertium-gag)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
        "qst": "PartType=Int",
    }
