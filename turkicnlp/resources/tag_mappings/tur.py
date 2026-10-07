"""Turkish Apertium -> UD tag mapping."""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.turkic_common import CommonTurkicTagMapper


class TurkishTagMapper(CommonTurkicTagMapper):
    """Tag mapper for Turkish (apertium-tur)."""

    FEAT_MAP: dict[str, str] = {
        **CommonTurkicTagMapper.FEAT_MAP,
        # Turkish-specific tags seen in Apertium streams.
        # apertium-tur: <ifi> = -DI (witnessed past), <past> = -mIş (reported past).
        "ifi": "Evident=Fh|Tense=Past",
        "past": "Evident=Nfh|Tense=Past",
        "pers": "PronType=Prs",
        "dem": "PronType=Dem",
        "qst": "PartType=Int",
    }
