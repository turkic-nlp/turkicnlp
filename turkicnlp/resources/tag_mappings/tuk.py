"""Turkmen Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class TurkmenTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Turkmen (apertium-tuk).

    This is an initial mapper based on Apertium Turkic tag conventions and
    Turkmen nominal/verbal morphology.
    """

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        # Turkmen tags observed in extraction from apertium-tuk outputs.
        "ifi": "Evident=Nfh",
        "prog": "Aspect=Prog",
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
        "qst": "PartType=Int",
    }
