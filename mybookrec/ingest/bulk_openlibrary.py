r"""Stream an Open Library bulk dump into sharded bronze JSONL.

The OL monthly dumps live at https://openlibrary.org/developers/dumps. Each is a
gzipped TSV with one record per line:

    <type>\t<key>\t<revision>\t<last_modified>\t<json>

We're interested in *works* (the abstract concept of "Mistborn", not specific
editions). The works dump is ~15 GB uncompressed, ~30M records. This loader:

1. Optionally downloads the dump if a local path isn't given (HTTP streaming).
2. Streams it line-by-line — never holds more than one record in memory.
3. Parses the 5th tab-separated field as JSON.
4. Applies coarse filters: must have title + at least one subject + a description.
5. Writes filtered records to sharded JSONL files under
   `data/bronze/openlibrary_dump/<YYYY-MM>/works_<NNNNNN>.jsonl` so DVC tracks
   each shard independently and downstream pipelines can parallelise.

A `manifest.json` next to the shards records source URL, totals, and timestamps
so DVC + downstream silver builds can audit provenance.

Usage:
    # Use a local dump file
    .venv/bin/python -m mybookrec.ingest.bulk_openlibrary --input ol_dump_works_latest.txt.gz

    # Or stream from the OL host (slow but no local file needed)
    .venv/bin/python -m mybookrec.ingest.bulk_openlibrary \
        --url https://openlibrary.org/data/ol_dump_works_latest.txt.gz \
        --limit 50000  # for testing
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings

OL_DUMP_DEFAULT_URL = "https://openlibrary.org/data/ol_dump_works_latest.txt.gz"
SHARD_SIZE = 50_000


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace with input/url + limit + shard-size + dest-tag.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Local .txt.gz dump file (preferred if available).")
    parser.add_argument("--url", type=str, default=OL_DUMP_DEFAULT_URL, help="HTTP source for streaming.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N kept records (useful for testing).")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="Records per JSONL shard.")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Subdirectory name under bronze/openlibrary_dump/ (defaults to today's YYYY-MM).",
    )
    return parser.parse_args()


def iter_dump_lines(input_path: Path | None, url: str, timeout: float) -> Iterator[bytes]:
    """Yield raw lines from a local file or a streamed HTTP body, transparently gunzipping.

    Args:
        input_path: Local dump file. If set, `url` is ignored.
        url: Remote URL to stream when no local file is given.
        timeout: HTTP read timeout (only used when streaming).

    Yields:
        Each line as bytes, without the trailing newline.
    """
    if input_path is not None:
        with gzip.open(input_path, "rb") as f:
            yield from f
        return

    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        decompressor = gzip.GzipFile(fileobj=ResponseStream(response.iter_bytes(chunk_size=1 << 20)))
        yield from decompressor


class ResponseStream(io.RawIOBase):
    """File-like adapter that lets gzip stream from an httpx chunk iterator.

    httpx streams bytes chunks; gzip wants a file-like with `.read(n)`. This adapter
    buffers chunks internally and serves `.read` from that buffer, refilling as needed.
    """

    def __init__(self, chunks: Iterator[bytes]) -> None:
        """Wrap a byte-chunk iterator as a readable stream.

        Args:
            chunks: Iterator yielding bytes (typically from httpx.iter_bytes()).
        """
        super().__init__()
        self.chunks = chunks
        self.buffer = b""

    def readable(self) -> bool:
        """Mark the stream as readable so io machinery accepts it."""
        return True

    def readinto(self, buf: memoryview) -> int:  # type: ignore[override]
        """Fill `buf` with as many bytes as available, refilling from chunks if needed.

        Args:
            buf: Writable buffer to fill.

        Returns:
            Number of bytes written; 0 signals EOF.
        """
        while not self.buffer:
            try:
                self.buffer = next(self.chunks)
            except StopIteration:
                return 0
        n = min(len(buf), len(self.buffer))
        buf[:n] = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return n


def parse_dump_line(line: bytes) -> dict[str, Any] | None:
    """Parse one OL dump line; return the embedded JSON dict or None if malformed.

    Args:
        line: A single raw line from the dump (bytes).

    Returns:
        The parsed JSON record, or None if the row isn't 5 tab-separated fields or
        the JSON field doesn't decode.
    """
    parts = line.rstrip(b"\n").split(b"\t", 4)
    if len(parts) != 5:
        return None
    try:
        return json.loads(parts[4])
    except json.JSONDecodeError:
        return None


def is_keepable_work(record: dict[str, Any]) -> bool:
    """Coarse filter: keep works with a title + at least one subject + a description.

    Args:
        record: One parsed work record.

    Returns:
        True if worth keeping for downstream silver processing.
    """
    if not record.get("title"):
        return False
    if not record.get("subjects"):
        return False
    return bool(record.get("description"))


def write_manifest(
    out_dir: Path,
    *,
    source: str,
    n_scanned: int,
    n_kept: int,
    n_shards: int,
    duration_s: float,
) -> Path:
    """Write a JSON manifest describing this bulk-load run.

    Args:
        out_dir: Bronze dump output directory.
        source: URL or local path used as input.
        n_scanned: Total lines read from the dump.
        n_kept: Records that passed the keepable filter.
        n_shards: Number of JSONL shards written.
        duration_s: Wall-clock time taken.

    Returns:
        Path to the manifest file.
    """
    manifest = {
        "source": source,
        "loader": "mybookrec.ingest.bulk_openlibrary",
        "n_scanned": n_scanned,
        "n_kept": n_kept,
        "n_shards": n_shards,
        "duration_seconds": round(duration_s, 1),
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    path = out_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def run(
    input_path: Path | None,
    url: str,
    limit: int | None,
    shard_size: int,
    tag: str | None,
) -> tuple[Path, int]:
    """Stream the dump into sharded bronze JSONL + a manifest.

    Args:
        input_path: Local dump file path, or None to stream from URL.
        url: Source URL (used only when input_path is None).
        limit: Stop after this many kept records (None = no limit).
        shard_size: Records per output shard.
        tag: Subdirectory tag under bronze/openlibrary_dump/.

    Returns:
        Tuple of (output_directory, total_records_kept).
    """
    settings = get_settings()
    tag = tag or dt.datetime.now(dt.UTC).strftime("%Y-%m")
    out_dir = settings.bronze_dir / "openlibrary_dump" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    source = str(input_path) if input_path else url
    print(f"[bulk] source={source}  out={out_dir}  shard_size={shard_size:,}  limit={limit}")

    t_start = time.time()
    n_scanned = 0
    n_kept = 0
    shard_index = 0
    shard_file: io.TextIOWrapper | None = None

    try:
        for line in iter_dump_lines(input_path, url, timeout=settings.ingest_request_timeout_sec * 30):
            n_scanned += 1
            record = parse_dump_line(line)
            if record is None or not is_keepable_work(record):
                continue

            if shard_file is None or (n_kept and n_kept % shard_size == 0):
                if shard_file is not None:
                    shard_file.close()
                    shard_index += 1
                shard_path = out_dir / f"works_{shard_index:06d}.jsonl"
                shard_file = shard_path.open("w", encoding="utf-8")
                print(f"  shard {shard_index:>4d} → {shard_path.name}  (kept={n_kept:,})")

            shard_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_kept += 1

            if n_scanned % 250_000 == 0:
                rate = n_scanned / (time.time() - t_start)
                print(f"  [{time.time() - t_start:6.0f}s] scanned={n_scanned:,} kept={n_kept:,} ({rate:,.0f}/s)")

            if limit is not None and n_kept >= limit:
                break
    finally:
        if shard_file is not None:
            shard_file.close()

    n_shards = shard_index + (1 if shard_file is not None else 0)
    duration_s = time.time() - t_start
    manifest_path = write_manifest(
        out_dir,
        source=source,
        n_scanned=n_scanned,
        n_kept=n_kept,
        n_shards=n_shards,
        duration_s=duration_s,
    )
    print(f"[bulk] done: {n_kept:,} kept / {n_scanned:,} scanned in {duration_s:.0f}s → {n_shards} shards")
    print(f"       manifest: {manifest_path}")
    return out_dir, n_kept


def adapt_work(record: dict[str, Any]) -> SilverBook:
    """Convert one OL works-dump record into a SilverBook.

    The dump record shape differs from the Search API:
      - `subjects` (plural) instead of `subject`.
      - `authors` is a list of `{author: {key: "/authors/OL..."}}` — names live in the
        separate authors dump, which we don't process here. We use the author KEY
        (e.g. "OL26320A") as an opaque grouping identifier so the v4 batch-mean
        fallback can still cluster a single author's books together.
      - No ISBN — that's in the editions dump.
      - `description` is sometimes a `{type, value}` dict (SilverBook validator handles
        both shapes).

    Args:
        record: Parsed JSON record from one dump line.

    Returns:
        A SilverBook with description, subjects (as genres), and author keys.

    Raises:
        ValueError: If the record lacks a `key`.
    """
    key = (record.get("key") or "").rsplit("/", 1)[-1]
    if not key:
        raise ValueError("missing key")

    author_refs = record.get("authors") or []
    author_keys: list[str] = []
    for entry in author_refs:
        author = entry.get("author") if isinstance(entry, dict) else None
        if isinstance(author, dict):
            akey = (author.get("key") or "").rsplit("/", 1)[-1]
            if akey:
                author_keys.append(akey)

    return SilverBook(
        book_id=f"ol_{key}",
        raw_id=key,
        source="openlibrary",
        title=record.get("title", ""),
        description=record.get("description"),
        genres=list(record.get("subjects") or []),
        authors=author_keys,
    )


def to_silver(records: Iterable[dict[str, Any]]) -> Iterator[SilverBook]:
    """Convert raw dump records to SilverBook records, skipping malformed rows.

    Args:
        records: Iterable of parsed JSON dicts from dump shards.

    Yields:
        SilverBook for each record that has at minimum a key + title.
    """
    for record in records:
        try:
            yield adapt_work(record)
        except Exception:
            continue


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run(
        input_path=args.input,
        url=args.url,
        limit=args.limit,
        shard_size=args.shard_size,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
