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

    Values are ISO 15924 script codes.
    """

    LATIN = "Latn"
    CYRILLIC = "Cyrl"
    PERSO_ARABIC = "Arab"
    OLD_TURKIC_RUNIC = "Orkh"

    def __str__(self) -> str:
        return self.value


SCRIPT_NAMES: dict[Script, str] = {
    Script.LATIN: "Latin",
    Script.CYRILLIC: "Cyrillic",
    Script.PERSO_ARABIC: "Perso-Arabic",
    Script.OLD_TURKIC_RUNIC: "Old Turkic Runic",
}


@dataclass
class ScriptConfig:
    """Per-language script configuration.

    Attributes:
        available: All scripts this language uses.
        primary: Default script if not specified by the user.
        direction: Text direction for the primary script (``ltr`` or ``rtl``).
        can_transliterate: Mapping of source→target script pairs that have
            transliteration support (e.g. ``{Script.LATIN: Script.CYRILLIC}``).
        apertium_script: Which script the Apertium FST expects as input.
    """

    available: list[Script]
    primary: Script
    direction: str = "ltr"
    can_transliterate: Optional[dict[Script, Script]] = None
    apertium_script: Optional[Script] = None

    def __post_init__(self) -> None:
        if self.can_transliterate is None:
            self.can_transliterate = {}
        if self.apertium_script is None:
            self.apertium_script = self.primary


# ---------------------------------------------------------------------------
# Canonical script configuration for every supported language
# ---------------------------------------------------------------------------

LANGUAGE_SCRIPTS: dict[str, ScriptConfig] = {
    "tur": ScriptConfig(
        available=[Script.LATIN],
        primary=Script.LATIN,
    ),
    "aze": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC],
        primary=Script.LATIN,
        can_transliterate={Script.CYRILLIC: Script.LATIN, Script.LATIN: Script.CYRILLIC},
        apertium_script=Script.LATIN,
    ),
    "azb": ScriptConfig(
        available=[Script.PERSO_ARABIC],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
    ),
    "kaz": ScriptConfig(
        available=[Script.CYRILLIC, Script.LATIN],
        primary=Script.CYRILLIC,
        can_transliterate={Script.LATIN: Script.CYRILLIC, Script.CYRILLIC: Script.LATIN},
        apertium_script=Script.CYRILLIC,
    ),
    "uzb": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC],
        primary=Script.LATIN,
        can_transliterate={Script.CYRILLIC: Script.LATIN, Script.LATIN: Script.CYRILLIC},
        apertium_script=Script.LATIN,
    ),
    "kir": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "tuk": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC],
        primary=Script.LATIN,
        can_transliterate={Script.CYRILLIC: Script.LATIN, Script.LATIN: Script.CYRILLIC},
        apertium_script=Script.LATIN,
    ),
    "tat": ScriptConfig(
        available=[Script.CYRILLIC, Script.LATIN],
        primary=Script.CYRILLIC,
        can_transliterate={Script.LATIN: Script.CYRILLIC, Script.CYRILLIC: Script.LATIN},
        apertium_script=Script.CYRILLIC,
    ),
    "uig": ScriptConfig(
        available=[Script.PERSO_ARABIC, Script.LATIN],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
        can_transliterate={Script.LATIN: Script.PERSO_ARABIC, Script.PERSO_ARABIC: Script.LATIN},
        apertium_script=Script.PERSO_ARABIC,
    ),
    "bak": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "crh": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC],
        primary=Script.LATIN,
        can_transliterate={Script.CYRILLIC: Script.LATIN, Script.LATIN: Script.CYRILLIC},
        apertium_script=Script.LATIN,
    ),
    "chv": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "sah": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "kaa": ScriptConfig(
        available=[Script.LATIN, Script.CYRILLIC],
        primary=Script.LATIN,
        can_transliterate={Script.CYRILLIC: Script.LATIN, Script.LATIN: Script.CYRILLIC},
    ),
    "gag": ScriptConfig(
        available=[Script.LATIN],
        primary=Script.LATIN,
    ),
    "nog": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "kum": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "krc": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "alt": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "tyv": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "kjh": ScriptConfig(
        available=[Script.CYRILLIC],
        primary=Script.CYRILLIC,
    ),
    "ota": ScriptConfig(
        available=[Script.PERSO_ARABIC, Script.LATIN],
        primary=Script.PERSO_ARABIC,
        direction="rtl",
        can_transliterate={Script.LATIN: Script.PERSO_ARABIC},
    ),
    "otk": ScriptConfig(
        available=[Script.OLD_TURKIC_RUNIC, Script.LATIN],
        primary=Script.LATIN,
        can_transliterate={Script.OLD_TURKIC_RUNIC: Script.LATIN},
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
