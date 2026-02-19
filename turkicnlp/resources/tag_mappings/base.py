"""
Base class for Apertium → UD tag mapping.
"""

from __future__ import annotations


# Default Apertium POS → UD UPOS mapping (shared across languages)
DEFAULT_POS_MAP: dict[str, str] = {
    "n": "NOUN",
    "np": "PROPN",
    "v": "VERB",
    "vaux": "AUX",
    "adj": "ADJ",
    "adv": "ADV",
    "prn": "PRON",
    "det": "DET",
    "post": "ADP",
    "cnjcoo": "CCONJ",
    "cnjsub": "SCONJ",
    "num": "NUM",
    "ij": "INTJ",
    "part": "PART",
    "punct": "PUNCT",
    "sym": "SYM",
}


class TagMapper:
    """Maps Apertium morphological tags to Universal Dependencies.

    Override in language-specific subclasses for custom mappings.

    Attributes:
        POS_MAP: Mapping from Apertium POS tags to UD UPOS.
        FEAT_MAP: Mapping from Apertium feature tags to UD feature strings.
    """

    POS_MAP: dict[str, str] = DEFAULT_POS_MAP
    FEAT_MAP: dict[str, str] = {}

    def to_ud_pos(self, apertium_pos: str) -> str:
        """Convert an Apertium POS tag to UD UPOS.

        Args:
            apertium_pos: Apertium POS tag (e.g. ``n``, ``v``, ``adj``).

        Returns:
            UD UPOS tag (e.g. ``NOUN``, ``VERB``, ``ADJ``).
        """
        return self.POS_MAP.get(apertium_pos, "X")

    def to_ud_feats(self, apertium_feats: list[str]) -> str:
        """Convert a list of Apertium feature tags to a UD feature string.

        Args:
            apertium_feats: List of Apertium tags (e.g. ``["dat", "sg"]``).

        Returns:
            UD-format feature string (e.g. ``Case=Dat|Number=Sing``).
        """
        ud_feats, _ = self.map_ud_feats(apertium_feats)
        return "|".join(sorted(ud_feats)) if ud_feats else "_"

    def map_ud_feats(self, apertium_feats: list[str]) -> tuple[list[str], list[str]]:
        """Map Apertium features to UD and report unknown tags.

        Args:
            apertium_feats: List of Apertium tags.

        Returns:
            A tuple ``(mapped_ud_feats, unknown_apertium_feats)``.
        """
        ud_feats: list[str] = []
        unknown: list[str] = []
        for feat in apertium_feats:
            mapped = self.FEAT_MAP.get(feat)
            if mapped is None:
                unknown.append(feat)
            else:
                ud_feats.append(mapped)
        return ud_feats, unknown
