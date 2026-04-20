"""
Script system for TurkicNLP.

Defines the :class:`Script` enum, :class:`ScriptConfig` per-language
configuration, and the canonical ``LANGUAGE_SCRIPTS`` registry.

Multi-script support is a first-class concern: models are keyed by
``lang/script/processor/backend/``, and the pipeline auto-detects script
or accepts it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Script(str, Enum):
    """Writing scripts used across Turkic languages.

    Values are ISO 15924 script codes (or short identifiers for non-ISO scripts).
    """

    LATIN = "Latn"
    CYRILLIC = "Cyrl"
    PERSO_ARABIC = "Arab"
    OLD_TURKIC_RUNIC = "Orkh"
    COMMON_TURKIC = "CTS"

    def __str__(self) -> str:
        return self.value


SCRIPT_NAMES: dict[Script, str] = {
    Script.LATIN: "Latin",
    Script.CYRILLIC: "Cyrillic",
    Script.PERSO_ARABIC: "Perso-Arabic",
    Script.OLD_TURKIC_RUNIC: "Old Turkic Runic",
    Script.COMMON_TURKIC: "Common Turkic",
}


@dataclass
class ScriptConfig:
    """Per-language script configuration.

    Attributes:
        available: All scripts this language uses.
        primary: Default script if not specified by the user.
        direction: Text direction for the primary script (``ltr`` or ``rtl``).
        can_transliterate: Set of (source, target) script pairs that have
            transliteration support.
        apertium_script: Which script the Apertium FST expects as input.
    """

    available: list[Script]
    primary: Script
    direction: str = "ltr"
    can_transliterate: Optional[set[tuple[Script, Script]]] = None
    apertium_script: Optional[Script] = None

    def __post_init__(self) -> None:
        if self.can_transliterate is None:
            self.can_transliterate = set()
        if self.apertium_script is None:
            self.apertium_script = self.primary


# ---------------------------------------------------------------------------
# Canonical script configuration for every supported language
# ---------------------------------------------------------------------------

LANGUAGE_SCRIPTS: dict[str, ScriptConfig] = {
    "tur": ScriptConfig(
        available=[Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.LATIN, Script.COMMON_TURKIC),
        },
    ),
    "aze": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.CYRILLIC, Script.LATIN),
            (Script.LATIN, Script.CYRILLIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
        apertium_script=Script.LATIN,
    ),
    "azb": ScriptConfig(
        available=[Script.PERSO_ARABIC, Script.COMMON_TURKIC],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
        can_transliterate={
            (Script.PERSO_ARABIC, Script.COMMON_TURKIC),
        },
    ),
    "kaz": ScriptConfig(
        available=[Script.CYRILLIC, Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.LATIN, Script.CYRILLIC),
            (Script.CYRILLIC, Script.LATIN),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
            (Script.LATIN, Script.COMMON_TURKIC),
        },
        apertium_script=Script.CYRILLIC,
    ),
    "uzb": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.CYRILLIC, Script.LATIN),
            (Script.LATIN, Script.CYRILLIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
        apertium_script=Script.LATIN,
    ),
    "kir": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "tuk": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.CYRILLIC, Script.LATIN),
            (Script.LATIN, Script.CYRILLIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
        apertium_script=Script.LATIN,
    ),
    "tat": ScriptConfig(
        available=[Script.CYRILLIC, Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.LATIN, Script.CYRILLIC),
            (Script.CYRILLIC, Script.LATIN),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
            (Script.LATIN, Script.COMMON_TURKIC),
        },
        apertium_script=Script.CYRILLIC,
    ),
    "uig": ScriptConfig(
        available=[Script.PERSO_ARABIC, Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
        can_transliterate={
            (Script.LATIN, Script.PERSO_ARABIC),
            (Script.PERSO_ARABIC, Script.LATIN),
            (Script.PERSO_ARABIC, Script.CYRILLIC),
            (Script.CYRILLIC, Script.PERSO_ARABIC),
            (Script.LATIN, Script.CYRILLIC),
            (Script.CYRILLIC, Script.LATIN),
            (Script.PERSO_ARABIC, Script.COMMON_TURKIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
            (Script.COMMON_TURKIC, Script.PERSO_ARABIC),
            (Script.COMMON_TURKIC, Script.LATIN),
            (Script.COMMON_TURKIC, Script.CYRILLIC),
        },
        apertium_script=Script.PERSO_ARABIC,
    ),
    "bak": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "crh": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.CYRILLIC, Script.LATIN),
            (Script.LATIN, Script.CYRILLIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
        apertium_script=Script.LATIN,
    ),
    "chv": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "sah": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "kaa": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.CYRILLIC, Script.LATIN),
            (Script.LATIN, Script.CYRILLIC),
            (Script.LATIN, Script.COMMON_TURKIC),
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "gag": ScriptConfig(
        available=[Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.LATIN, Script.COMMON_TURKIC),
        },
    ),
    "nog": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "kum": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "krc": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "alt": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "tyv": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "kjh": ScriptConfig(
        available=[Script.CYRILLIC, Script.COMMON_TURKIC],
        primary=Script.CYRILLIC,
        can_transliterate={
            (Script.CYRILLIC, Script.COMMON_TURKIC),
        },
    ),
    "ota": ScriptConfig(
        available=[Script.PERSO_ARABIC, Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
        can_transliterate={
            (Script.LATIN, Script.PERSO_ARABIC),
            (Script.LATIN, Script.COMMON_TURKIC),
        },
    ),
    "otk": ScriptConfig(
        available=[Script.OLD_TURKIC_RUNIC, Script.LATIN],
        primary=Script.LATIN,
        can_transliterate={
            (Script.OLD_TURKIC_RUNIC, Script.LATIN),
        },
    ),
    "klj": ScriptConfig(
        available=[Script.LATIN, Script.COMMON_TURKIC],
        primary=Script.LATIN,
        can_transliterate={
            (Script.LATIN, Script.COMMON_TURKIC),
        },
    ),
}


def get_script_config(lang: str) -> ScriptConfig:
    """Get script configuration for a language.

    Args:
        lang: ISO 639-3 language code.

    Returns:
        The :class:`ScriptConfig` for the language.

    Raises:
        ValueError: If the language is not in the registry.
    """
    if lang not in LANGUAGE_SCRIPTS:
        raise ValueError(
            f"No script configuration for language '{lang}'. "
            f"Known languages: {sorted(LANGUAGE_SCRIPTS.keys())}"
        )
    return LANGUAGE_SCRIPTS[lang]
