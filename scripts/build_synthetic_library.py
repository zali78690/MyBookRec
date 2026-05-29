"""Build a synthetic goodreads_library_export.csv from a UCSD power user.

Picks a user with rich, varied ratings from books_with_interactions.parquet,
joins with book metadata, writes the Goodreads export format, and runs the
same transform the notebook does to produce my_books.csv.
"""
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw"
TRANSFORMED = DATA_DIR / "transformed"

# Hand-picked from a rich-user scan: 243 high / 42 low / 8 mid out of 293 —
# strongest low-rating signal among power users (real "I disliked this" data).
USER_ID = "096c015beb9e7c53bc3a48faeb80da8e"

# Pull this user's rated interactions joined with book metadata. We only need
# title + num_pages from the books side; author etc. aren't in the parquet,
# so those columns get left blank in the export — the downstream transform
# only reads "Book Id" and "My Rating" anyway.
user_books = (
    pl.scan_parquet(TRANSFORMED / "books_with_interactions.parquet")
    .filter(
        (pl.col("user_id") == USER_ID)
        & (pl.col("rating") > 0)
    )
    .select(
        pl.col("book_id").cast(pl.Int64, strict=False),
        pl.col("rating").cast(pl.Int64).alias("my_rating"),
        pl.col("title"),
        pl.col("num_pages"),
        pl.col("date_added"),
    )
    .unique(subset=["book_id"])
    .sort("date_added", descending=True)
    .collect()
)

# We have to resolve the prefix match because the user_id was truncated above.
# Fall back to a prefix match if the literal match is empty.
if user_books.height == 0:
    user_books = (
        pl.scan_parquet(TRANSFORMED / "books_with_interactions.parquet")
        .filter(
            pl.col("user_id").str.starts_with(USER_ID[:32])
            & (pl.col("rating") > 0)
        )
        .select(
            pl.col("book_id").cast(pl.Int64, strict=False),
            pl.col("rating").cast(pl.Int64).alias("my_rating"),
            pl.col("title"),
            pl.col("num_pages"),
            pl.col("date_added"),
        )
        .unique(subset=["book_id"])
        .sort("date_added", descending=True)
        .collect()
    )

print(f"Pulled {user_books.height} rated books for user {USER_ID[:12]}…")
print(user_books.group_by("my_rating").len().sort("my_rating"))

# Build the Goodreads library export DataFrame matching the real export schema.
# Columns the transform reads: "Book Id", "My Rating". Others kept as empty
# strings to mirror the official export shape.
export = user_books.select(
    pl.col("book_id").alias("Book Id"),
    pl.col("title").alias("Title"),
    pl.lit("").alias("Author"),
    pl.lit("").alias("Author l-f"),
    pl.lit("").alias("Additional Authors"),
    pl.lit('=""').alias("ISBN"),
    pl.lit('=""').alias("ISBN13"),
    pl.col("my_rating").alias("My Rating"),
    pl.lit("").alias("Publisher"),
    pl.lit("").alias("Binding"),
    pl.col("num_pages").alias("Number of Pages"),
    pl.lit(None, dtype=pl.Int64).alias("Year Published"),
    pl.lit(None, dtype=pl.Int64).alias("Original Publication Year"),
    pl.col("date_added").dt.strftime("%Y/%m/%d").alias("Date Read"),
    pl.col("date_added").dt.strftime("%Y/%m/%d").alias("Date Added"),
    pl.lit("").alias("Bookshelves"),
    pl.lit("").alias("Bookshelves with positions"),
    pl.lit("read").alias("Exclusive Shelf"),
    pl.lit("").alias("My Review"),
    pl.lit("").alias("Spoiler"),
    pl.lit("").alias("Private Notes"),
    pl.lit(1).alias("Read Count"),
    pl.lit(0).alias("Owned Copies"),
)

export_path = RAW / "goodreads_library_export.csv"
export.write_csv(export_path)
print(f"Wrote {export.height} rows -> {export_path}")

# Re-run the same Polars transform the notebook does for my_books.csv.
my_books_path = TRANSFORMED / "my_books.csv"
(
    pl.read_csv(export_path)
    .select(
        pl.col("Book Id").alias("book_id").cast(pl.Int64, strict=False),
        pl.col("My Rating").alias("my_rating").cast(pl.Float64, strict=False),
    )
    .filter(pl.col("my_rating") > 0)
    .write_csv(my_books_path)
)
print(f"Wrote my_books.csv -> {my_books_path}")


"""
Synthetic user taste profile — user 096c015beb9e7c53bc3a48faeb80da8e
====================================================================

Rating distribution: 227×5★, 16×4★, 8×3★, 15×2★, 27×1★ (293 total)

LIKES (4-5★ — 243 books):
  - YA fantasy & adventure series: Percy Jackson, Heroes of Olympus,
    The Lunar Chronicles (Cinder/Cress), Hunger Games (Catching Fire),
    Seven Realms, Court of Fives, Serpentine, Wilde Island Chronicles
  - Manga, especially shoujo / coming-of-age: Ao Haru Ride, Arisa,
    Kodocha, plus Avatar: The Last Airbender graphic novels
  - Classic children's & middle-grade: A Little Princess, Heidi-adjacent
    warm fiction, The Graveyard Book, Sisterhood of the Traveling Pants
  - LDS / religious texts: Book of Mormon, Doctrine & Covenants,
    Pearl of Great Price (suggests an LDS reader)
  - Issue-driven YA / historical: Sold, Sarah's Key, Unhooked

DISLIKES (1-2★ — 42 books):
  - Twilight saga — ALL four books + the box set rated 1★ (very strong
    negative signal; useful for the dislike-embedding feature)
  - Literary classics taught in school: Wuthering Heights, Lord of
    the Flies, The Grapes of Wrath
  - Adult / "harder" fantasy: Elantris, Furies of Calderon, Sword of
    Shannara, The Golden Compass, The Alchemyst, Shadow of the Wind
  - Some hyped contemporary YA: Divergent, Paper Towns, Grave Mercy,
    The Iron King, The False Prince
  - Verse novels: Who Killed Mr. Chippendale?, Things Left Unsaid

Pattern: prefers character-driven YA fantasy with female protagonists,
manga, and faith-based content. Avoids adult literary fiction, paranormal
romance, and denser epic fantasy. Strong, consistent dislike signal on
Twilight makes this a good test profile for the dislike embedding in the
user tower.
"""
