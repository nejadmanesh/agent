"""Text cleaning utilities tailored for Persian language content."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PERSIAN_CHAR_MAP = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ؤ": "و",
    "ئ": "ی",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
    "ة": "ه",
    "ۀ": "ه",
})

_ARABIC_DIGIT_MAP = {
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}

_DIACRITIC_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u06d6-\u06ed]")
_TATWEEL_RE = re.compile(r"\u0640")
_PUNCTUATION_RE = re.compile(r"\s*([،؛,.!?؟])\s*")
_ZERO_WIDTH_SPACES = re.compile(r"[\u200c\u200f]")


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration flags that control cleaning behaviour."""

    remove_diacritics: bool = True
    normalize_whitespace: bool = True
    convert_arabic_chars: bool = True
    standardize_digits: bool = True
    remove_tatweel: bool = True
    standardize_punctuation_spacing: bool = True
    strip_zero_width_spaces: bool = False


class PersianTextCleaner:
    """Clean and normalise Persian text for downstream NLP tasks."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Return a normalised representation of *text*.

        The steps performed follow the recommendations from common Persian NLP
        preprocessing pipelines: unifying characters shared with Arabic,
        removing diacritics and elongation glyphs, normalising whitespace and
        punctuation spacing, and ensuring digits use ASCII code points so that
        downstream tooling can parse numerical tokens consistently.
        """

        cleaned = text
        if self.config.convert_arabic_chars:
            cleaned = cleaned.translate(_PERSIAN_CHAR_MAP)
        if self.config.standardize_digits:
            cleaned = "".join(_ARABIC_DIGIT_MAP.get(ch, ch) for ch in cleaned)
        if self.config.remove_diacritics:
            cleaned = _DIACRITIC_RE.sub("", cleaned)
        if self.config.remove_tatweel:
            cleaned = _TATWEEL_RE.sub("", cleaned)
        if self.config.strip_zero_width_spaces:
            cleaned = _ZERO_WIDTH_SPACES.sub("", cleaned)
        if self.config.standardize_punctuation_spacing:
            cleaned = _standardise_punctuation_spacing(cleaned)
        if self.config.normalize_whitespace:
            cleaned = _normalise_whitespace(cleaned)
        return cleaned


def _standardise_punctuation_spacing(text: str) -> str:
    """Ensure punctuation is attached to the preceding token."""

    def _replacer(match: re.Match[str]) -> str:
        punctuation = match.group(1)
        return f"{punctuation} "

    text = _PUNCTUATION_RE.sub(_replacer, text)
    return text.strip()


def _normalise_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and trim surrounding spaces."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()
