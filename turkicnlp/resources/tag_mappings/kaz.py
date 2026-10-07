"""Kazakh Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class KazakhTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Kazakh (apertium-kaz)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        # Kazakh-specific additions commonly present in Apertium analyses.
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
    }
