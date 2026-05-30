"""ISO 639-2/B (3-letter) to ISO 639-1 (2-letter) mapping.

Open Library returns 3-letter codes; the silver schema standardises on 2-letter to match the
UCSD training data. Only the most common codes are mapped; unmapped codes pass through unchanged
(downstream filters drop non-English anyway).
"""

from __future__ import annotations

ISO_639_2B_TO_1: dict[str, str] = {
    "eng": "en",
    "spa": "es",
    "fre": "fr",
    "fra": "fr",
    "ger": "de",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "jpn": "ja",
    "chi": "zh",
    "zho": "zh",
    "kor": "ko",
    "ara": "ar",
    "dut": "nl",
    "nld": "nl",
    "swe": "sv",
    "nor": "no",
    "dan": "da",
    "fin": "fi",
    "pol": "pl",
    "tur": "tr",
    "heb": "he",
    "hin": "hi",
    "ind": "id",
    "vie": "vi",
    "tha": "th",
    "ukr": "uk",
    "cze": "cs",
    "ces": "cs",
}


def to_iso_639_1(code: str | None) -> str | None:
    """Normalise a language code to ISO 639-1 (2-letter).

    Args:
        code: A 2-letter or 3-letter language code, or None.

    Returns:
        The 2-letter equivalent, or the input unchanged if not in the map.
    """
    if code is None:
        return None
    code = code.lower().strip()
    if len(code) == 2:
        return code
    return ISO_639_2B_TO_1.get(code, code)
