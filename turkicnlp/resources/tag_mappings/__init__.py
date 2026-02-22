"""
Apertium → Universal Dependencies tag mapping system.

Each language has its own mapping module that translates Apertium
morphological tags to UD UPOS and feature strings.
"""

from __future__ import annotations

from turkicnlp.resources.tag_mappings.base import TagMapper


def load_tag_map(lang: str) -> TagMapper:
    """Load the tag mapper for a given language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        A :class:`TagMapper` instance for the language.

    Raises:
        ValueError: If no tag mapping exists for the language.
    """
    if lang == "kaz":
        from turkicnlp.resources.tag_mappings.kaz import KazakhTagMapper
        return KazakhTagMapper()
    elif lang == "tat":
        from turkicnlp.resources.tag_mappings.tat import TatarTagMapper
        return TatarTagMapper()
    elif lang == "tur":
        from turkicnlp.resources.tag_mappings.tur import TurkishTagMapper
        return TurkishTagMapper()
    elif lang == "tuk":
        from turkicnlp.resources.tag_mappings.tuk import TurkmenTagMapper
        return TurkmenTagMapper()
    elif lang == "alt":
        from turkicnlp.resources.tag_mappings.alt import AltaiTagMapper
        return AltaiTagMapper()
    elif lang == "aze":
        from turkicnlp.resources.tag_mappings.aze import AzerbaijaniTagMapper
        return AzerbaijaniTagMapper()
    elif lang == "azb":
        from turkicnlp.resources.tag_mappings.azb import SouthAzerbaijaniTagMapper
        return SouthAzerbaijaniTagMapper()
    elif lang == "bak":
        from turkicnlp.resources.tag_mappings.bak import BashkirTagMapper
        return BashkirTagMapper()
    elif lang == "chv":
        from turkicnlp.resources.tag_mappings.chv import ChuvashTagMapper
        return ChuvashTagMapper()
    elif lang == "crh":
        from turkicnlp.resources.tag_mappings.crh import CrimeanTatarTagMapper
        return CrimeanTatarTagMapper()
    elif lang == "gag":
        from turkicnlp.resources.tag_mappings.gag import GagauzTagMapper
        return GagauzTagMapper()
    elif lang == "kaa":
        from turkicnlp.resources.tag_mappings.kaa import KarakalpakTagMapper
        return KarakalpakTagMapper()
    elif lang == "kir":
        from turkicnlp.resources.tag_mappings.kir import KyrgyzTagMapper
        return KyrgyzTagMapper()
    elif lang == "kjh":
        from turkicnlp.resources.tag_mappings.kjh import KhakasTagMapper
        return KhakasTagMapper()
    elif lang == "krc":
        from turkicnlp.resources.tag_mappings.krc import KarachayBalkarTagMapper
        return KarachayBalkarTagMapper()
    elif lang == "kum":
        from turkicnlp.resources.tag_mappings.kum import KumykTagMapper
        return KumykTagMapper()
    elif lang == "nog":
        from turkicnlp.resources.tag_mappings.nog import NogaiTagMapper
        return NogaiTagMapper()
    elif lang == "sah":
        from turkicnlp.resources.tag_mappings.sah import SakhaTagMapper
        return SakhaTagMapper()
    elif lang == "tyv":
        from turkicnlp.resources.tag_mappings.tyv import TuvanTagMapper
        return TuvanTagMapper()
    elif lang == "uig":
        from turkicnlp.resources.tag_mappings.uig import UyghurTagMapper
        return UyghurTagMapper()
    elif lang == "uzb":
        from turkicnlp.resources.tag_mappings.uzb import UzbekTagMapper
        return UzbekTagMapper()
    else:
        return TagMapper()
