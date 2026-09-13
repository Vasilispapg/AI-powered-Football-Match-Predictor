"""Attach team market values to matches.

Replaces ``marketValue/AddMV_into_filter.py``, which walked the 3,919 market
values against all 43,429 matches in two nested passes -- appending home values
in the first pass and away values in the second. When a home team failed to
match, nothing was appended, so that row's *away* value landed in the ``MV Home
Team`` column. 491 rows ended up with no home value and 1,082 with no away
value; an unknown number had the two silently transposed.

Two ``pd.merge`` calls cannot transpose columns, and turn an O(n*m) scan of
169 million comparisons into a hash join.

The join key is the raw team name. That is correct here, and deliberately not
fuzzy: the market-value table was seeded from the match data by the legacy
``f_to_mv`` step, so both sides share one namespace and every one of the 3,773
match teams is already present. Fuzzy matching would only create false
positives. Name normalisation earns its keep elsewhere -- building Transfermarkt
search queries for the re-scrape, and detecting one club filed under two names.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from football import paths
from football.teams.normalise import find_duplicate_candidates

log = logging.getLogger(__name__)


def join_market_values(
    matches: pd.DataFrame,
    market_values: pd.DataFrame,
) -> pd.DataFrame:
    """Return *matches* with ``mv_home`` / ``mv_away`` and their missing flags."""
    lookup = market_values.set_index("team")["market_value"]

    joined = matches.copy()
    joined["mv_home"] = joined["home_team"].map(lookup)
    joined["mv_away"] = joined["away_team"].map(lookup)
    joined["mv_home_missing"] = joined["mv_home"].isna()
    joined["mv_away_missing"] = joined["mv_away"].isna()

    unknown_home = set(joined.loc[~joined["home_team"].isin(lookup.index), "home_team"])
    unknown_away = set(joined.loc[~joined["away_team"].isin(lookup.index), "away_team"])
    unknown = unknown_home | unknown_away
    if unknown:
        log.warning(
            "%d team(s) absent from the market-value table entirely: %s",
            len(unknown),
            ", ".join(sorted(unknown)[:10]),
        )

    return joined


def build_dataset(
    matches_path: Path | None = None,
    market_values_path: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Join filtered matches to clean market values and write the dataset."""
    matches_path = matches_path or paths.MATCHES_FILTERED
    market_values_path = market_values_path or paths.MARKET_VALUES_CLEAN
    destination = destination or paths.MATCHES

    matches = pd.read_csv(matches_path, keep_default_na=False, na_values=[""])
    market_values = pd.read_csv(market_values_path)

    joined = join_market_values(matches, market_values)

    destination.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(destination, index=False, encoding="utf-8")

    total = len(joined)
    home_ok = int(joined["mv_home"].notna().sum())
    away_ok = int(joined["mv_away"].notna().sum())
    both_ok = int((joined["mv_home"].notna() & joined["mv_away"].notna()).sum())
    log.info("Joined %d matches", total)
    log.info("  home value present: %6d (%5.1f%%)", home_ok, 100 * home_ok / total)
    log.info("  away value present: %6d (%5.1f%%)", away_ok, 100 * away_ok / total)
    log.info("  both present:       %6d (%5.1f%%)", both_ok, 100 * both_ok / total)
    log.info("Wrote %s", destination)

    return joined


def write_duplicate_candidates(
    matches: pd.DataFrame,
    destination: Path | None = None,
) -> dict[str, list[str]]:
    """Flag clubs that appear under more than one spelling in the match data.

    A club filed under two names has its history split in two, which halves the
    rolling-form features Phase 3 will build for it. This only reports; merging
    names is a judgement call that needs a human (``Paris`` and ``PSG`` may or
    may not be the same club).
    """
    destination = destination or paths.DUPLICATE_TEAM_CANDIDATES

    names = pd.concat([matches["home_team"], matches["away_team"]]).dropna()
    counts = names.value_counts()
    groups = find_duplicate_candidates(names.unique().tolist())

    rows = [
        {"normalised": key, "team": team, "match_count": int(counts.get(team, 0))}
        for key, group in sorted(groups.items())
        for team in group
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["normalised", "team", "match_count"]).to_csv(
        destination, index=False, encoding="utf-8"
    )

    if groups:
        log.info("%d possible duplicate team identities -> %s", len(groups), destination)
    else:
        log.info("No duplicate team identities detected")
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dataset()


if __name__ == "__main__":
    main()
