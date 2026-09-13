"""Concatenate the per-day scraped match CSVs into one table.

The original ``merged/merge.py`` read every file with a bare ``csv.reader`` and
no header handling, so each file's header row was appended as a data record --
322 junk rows in the old merged output. This version reads with pandas, keeps
everything as text (the ``Votes`` and ``Stats`` columns hold Python literals
that must survive the round trip untouched), validates each file's schema and
drops duplicate matches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from football import paths

log = logging.getLogger(__name__)

#: Schema produced by ``webScrapper/getStats.py``.
EXPECTED_COLUMNS = [
    "Competition",
    "Country",
    "Home Team",
    "Home Score",
    "Away Team",
    "Away Score",
    "Date",
    "URL",
    "Votes",
    "Stats",
]


def merge_matches(
    source: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Merge every CSV under *source* into a single frame, written to *destination*.

    Returns the merged frame. Files whose header does not match
    :data:`EXPECTED_COLUMNS` are skipped with a warning rather than silently
    corrupting the output.
    """
    source = source or paths.MATCHES_SCRAPED
    destination = destination or paths.MATCHES_MERGED

    files = sorted(source.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {source}")

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for path in files:
        try:
            # dtype=str / keep_default_na=False: this stage is a faithful
            # passthrough. Parsing happens in filter.py.
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            skipped.append((path.name, "empty file"))
            continue
        except pd.errors.ParserError as exc:
            skipped.append((path.name, f"unparseable: {exc}"))
            continue

        missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
        if missing:
            skipped.append((path.name, f"missing columns {missing}"))
            continue

        if frame.empty:
            skipped.append((path.name, "no rows"))
            continue

        frames.append(frame[EXPECTED_COLUMNS])

    if not frames:
        raise ValueError(f"No readable match files under {source} ({len(skipped)} skipped)")

    merged = pd.concat(frames, ignore_index=True)
    total_read = len(merged)

    # The legacy scraper ran the same URL list in two threads, so the same match
    # can appear twice. URL is the natural key; fall back to the match tuple for
    # rows where it is blank.
    has_url = merged["URL"].str.strip().ne("")
    with_url = merged[has_url].drop_duplicates(subset="URL", keep="first")
    without_url = merged[~has_url].drop_duplicates(
        subset=["Date", "Home Team", "Away Team"], keep="first"
    )
    merged = pd.concat([with_url, without_url], ignore_index=True)
    merged = merged.sort_values(["Date", "Competition", "Home Team"], kind="stable")
    merged = merged.reset_index(drop=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False, encoding="utf-8")

    log.info("Read %d files (%d skipped)", len(frames), len(skipped))
    for name, reason in skipped:
        log.warning("  skipped %s: %s", name, reason)
    log.info("Merged %d rows -> %d after de-duplication", total_read, len(merged))
    log.info("Wrote %s", destination)

    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    merge_matches()


if __name__ == "__main__":
    main()
