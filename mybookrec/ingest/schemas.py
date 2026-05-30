"""Pydantic schemas for the ingestion pipeline.

`SilverBook` is the canonical, source-agnostic representation a book must conform to before
entering downstream feature pipelines. Both Open Library and Google Books adapters produce
`SilverBook` instances and the merge step deduplicates across them.

The bronze layer is intentionally untyped — it is exactly whatever the API returned, serialised
as JSON. Adapters parse out only the fields silver needs; bronze stays available for backfills if
the silver schema later grows.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str | None) -> str | None:
    """Remove HTML tags from a string (Google Books descriptions arrive with `<b>` etc.).

    Args:
        text: Raw text or None.

    Returns:
        Plain text with tags removed and whitespace collapsed, or None.
    """
    if text is None:
        return None
    return HTML_TAG_RE.sub("", text).strip()


class SilverBook(BaseModel):
    """Source-agnostic cleaned book record."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    book_id: str = Field(..., description="Synthetic global id, e.g. 'ol_OL5738148W' or 'gb_xyz'.")
    raw_id: str = Field(..., description="The source's native identifier.")
    source: str = Field(..., description="'openlibrary' or 'google_books'.")
    title: str
    description: str | None = None
    num_pages: int | None = Field(default=None, ge=1, le=20000)
    average_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    ratings_count: int | None = Field(default=None, ge=0)
    is_ebook: bool | None = None
    authors: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    language: str | None = None
    published_year: int | None = Field(default=None, ge=1000, le=2100)
    isbn_13: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def normalise_description(cls, value: object) -> object:
        """Unwrap Open Library `{type, value}` description form and strip HTML.

        Args:
            value: Raw description as it appears in either API.

        Returns:
            Cleaned description string, or None / pass-through for unsupported types.
        """
        if isinstance(value, dict) and "value" in value:
            value = value.get("value")
        if isinstance(value, str):
            return strip_html(value)
        return value

    @field_validator("isbn_13")
    @classmethod
    def validate_isbn13(cls, value: str | None) -> str | None:
        """Require a 13-character all-digit ISBN-13 or return None.

        Args:
            value: Candidate ISBN string.

        Returns:
            The string if exactly 13 digits, else None (we'd rather drop than mislabel).
        """
        if value is None:
            return None
        digits = re.sub(r"\D", "", value)
        return digits if len(digits) == 13 else None

    def dedup_key(self) -> str:
        """Cross-source dedup key: prefer ISBN-13, fall back to title|authors[0].

        Returns:
            A normalised key safe to use in a set.
        """
        if self.isbn_13:
            return f"isbn:{self.isbn_13}"
        first_author = self.authors[0].lower().strip() if self.authors else ""
        return f"ta:{self.title.lower().strip()}|{first_author}"
