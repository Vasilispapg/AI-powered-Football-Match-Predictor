"""Print a health report for a dataset produced by the pipeline.

Run this after every pipeline change. It is the check that would have caught
the market-value sentinel bug -- Arsenal sitting at 0.28 while Liverpool sat at
794.8 -- years before it reached the model.

    python -m football.data.audit                 # audits the processed dataset
    python -m football.data.audit data/interim/matches_filtered.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from football import paths

#: Value the legacy ``fix_the_price`` wrote over every unresolved market value.
LEGACY_SENTINEL = 0.28


def _rule(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _pct(count: int, total: int) -> str:
    return f"{count:>7,} ({count / total:6.1%})" if total else f"{count:>7,}"


def audit(path: Path) -> int:
    """Print the report for *path*. Returns the number of failed checks."""
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    total = len(frame)
    failures: list[str] = []

    print(f"Dataset: {path}")
    print(f"Rows:    {total:,}")
    print(f"Columns: {list(frame.columns)}")

    if total == 0:
        print("\nFAIL: dataset is empty")
        return 1

    # --- coverage -----------------------------------------------------------
    _rule("Missing values by column")
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        blank = 0
        # Not `dtype == object`: pandas 3.0 gives text columns a dedicated
        # string dtype, so that test silently skipped every column it was
        # written for.
        if not pd.api.types.is_numeric_dtype(frame[column]):
            blank = int(frame[column].astype(str).str.strip().isin({"{}", "[]"}).sum())
        label = f"{column:<20}"
        note = f"  (+{blank:,} empty literals)" if blank else ""
        print(f"  {label} {_pct(missing, total)}{note}")

    # --- dates --------------------------------------------------------------
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        _rule("Date range")
        print(f"  {dates.min():%Y-%m-%d}  ->  {dates.max():%Y-%m-%d}")
        print(f"  span: {(dates.max() - dates.min()).days} days")
        if dates.isna().any():
            failures.append(f"{int(dates.isna().sum())} unparseable dates")

    # --- teams --------------------------------------------------------------
    if {"home_team", "away_team"} <= set(frame.columns):
        appearances = pd.concat([frame["home_team"], frame["away_team"]])
        counts = appearances.value_counts()
        _rule("Teams")
        print(f"  distinct teams:        {counts.size:,}")
        print(f"  mean matches per team: {counts.mean():.1f}")
        print(f"  median:                {counts.median():.0f}")
        print(f"  teams with <5 matches: {_pct(int((counts < 5).sum()), counts.size)}")
        print("  fewest-played sample:  " + ", ".join(counts.tail(3).index.astype(str)))

    # --- outcome balance ----------------------------------------------------
    if {"home_score", "away_score"} <= set(frame.columns):
        home = pd.to_numeric(frame["home_score"], errors="coerce")
        away = pd.to_numeric(frame["away_score"], errors="coerce")
        scored = home.notna() & away.notna()
        wins = int((home > away).sum())
        draws = int((home == away).sum())
        losses = int((home < away).sum())
        played = int(scored.sum())
        _rule("Outcome balance")
        print(f"  home win: {_pct(wins, played)}")
        print(f"  draw:     {_pct(draws, played)}")
        print(f"  away win: {_pct(losses, played)}")
        print(f"\n  Baseline to beat (always predict home): {wins / played:.1%} accuracy")
        if played < total:
            failures.append(f"{total - played} rows without a usable score")

    # --- market values ------------------------------------------------------
    mv_columns = [c for c in ("mv_home", "mv_away") if c in frame.columns]
    if mv_columns:
        _rule("Market values")
        for column in mv_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            missing = int(values.isna().sum())
            print(f"  {column:<10} missing {_pct(missing, total)}   "
                  f"median {values.median():>8.2f}   max {values.max():>8.1f}")
            sentinel = int((values == LEGACY_SENTINEL).sum())
            if sentinel:
                failures.append(
                    f"{column}: {sentinel} rows still hold the legacy {LEGACY_SENTINEL} sentinel"
                )

        for column in ("mv_home_missing", "mv_away_missing"):
            if column in frame.columns:
                flagged = int(frame[column].astype(str).str.lower().isin({"true", "1"}).sum())
                print(f"  {column:<20} {_pct(flagged, total)}")

        # Rows with a value on one side only. This is expected when values are
        # independently missing -- with ~13% missing per side it should sit
        # near 2*p*(1-p), about 23% -- so it is reported, not failed. The old
        # two-pass append could also produce it by transposing columns, but
        # that is prevented structurally now, not detected here.
        if {"mv_home", "mv_away"} <= set(frame.columns):
            home_mv = pd.to_numeric(frame["mv_home"], errors="coerce")
            away_mv = pd.to_numeric(frame["mv_away"], errors="coerce")
            lopsided = int((home_mv.notna() ^ away_mv.notna()).sum())
            missing_rate = 1 - (home_mv.notna().sum() + away_mv.notna().sum()) / (2 * total)
            expected = 2 * missing_rate * (1 - missing_rate)
            print(
                f"  {'one-sided rows':<20} {_pct(lopsided, total)}"
                f"   expected ~{expected:.1%} if missingness is independent"
            )

    # --- structural checks --------------------------------------------------
    _rule("Checks")
    if "competition" in frame.columns:
        header_rows = int((frame["competition"].astype(str) == "Competition").sum())
        if header_rows:
            failures.append(f"{header_rows} header rows leaked into the data")
        print(f"  header rows as data:   {header_rows}")
        print(f"  distinct competitions: {frame['competition'].nunique():,}")

    if "url" in frame.columns:
        duplicates = int(frame["url"].duplicated().sum())
        if duplicates:
            failures.append(f"{duplicates} duplicate match URLs")
        print(f"  duplicate URLs:        {duplicates}")

    if {"home_team", "away_team"} <= set(frame.columns):
        self_play = int((frame["home_team"] == frame["away_team"]).sum())
        if self_play:
            failures.append(f"{self_play} rows where a team plays itself")
        print(f"  self-play rows:        {self_play}")

    _rule("Result")
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
    else:
        print("  All checks passed.")
    return len(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=paths.MATCHES,
        help="dataset to audit (default: the processed dataset)",
    )
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f"{args.path} does not exist -- run `python -m football.pipeline` first")

    sys.exit(1 if audit(args.path) else 0)


if __name__ == "__main__":
    main()
