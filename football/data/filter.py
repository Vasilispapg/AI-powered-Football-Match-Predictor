"""Filter merged matches down to the modelling population.

Changes from the original ``filter/filter_data.py``:

* column names are normalised to ``snake_case`` here, at the raw -> processed
  boundary, so downstream stages get one consistent schema;
* the six chained ``not in`` checks are a single token scan;
* scores and dates are validated explicitly instead of relying on the scraper
  having written the literal string ``"N/A"``;
* every dropped row is counted and reported by reason, rather than vanishing;
* the empty-input crash at ``filtered_rows[0].keys()`` is guarded.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from football import paths

log = logging.getLogger(__name__)

#: Competition name substrings that exclude a match from the modelling
#: population. Youth and women's football are dropped because market values and
#: crowd votes -- the two strongest features -- are largely unavailable for
#: them, not because they are uninteresting. Matches the original script's set.
EXCLUDED_COMPETITION_TOKENS = ("u17", "u18", "u19", "u20", "u21", "women")

COLUMN_RENAMES = {
    "Competition": "competition",
    "Country": "country",
    "Home Team": "home_team",
    "Home Score": "home_score",
    "Away Team": "away_team",
    "Away Score": "away_score",
    "Date": "date",
    "URL": "url",
    "Votes": "votes",
    "Stats": "stats",
}

#: Columns that must be present and non-placeholder for a row to be usable.
REQUIRED_COLUMNS = ("competition", "country", "home_team", "away_team", "date")


def classify_competition(competition: str) -> str:
    """Bucket a competition name into ``cup``, ``friendly`` or ``league``.

    The friendly test matches the stem ``friendl``, not ``friendly``. goal.com
    always writes the plural -- "Club Friendlies", "Friendlies", "Non-FIFA
    Friendlies" -- and ``"friendly" in "club friendlies"`` is False, so the
    original test classified all 463 friendlies in the dataset as leagues.
    """
    lowered = competition.lower()
    if "cup" in lowered:
        return "cup"
    if "friendl" in lowered:
        return "friendly"
    return "league"


def is_excluded(competition: str) -> bool:
    """True if the competition is youth or women's football."""
    lowered = competition.lower()
    return any(token in lowered for token in EXCLUDED_COMPETITION_TOKENS)


def filter_matches(
    source: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Filter *source* into the modelling population, written to *destination*."""
    source = source or paths.MATCHES_MERGED
    destination = destination or paths.MATCHES_FILTERED

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame = frame.rename(columns=COLUMN_RENAMES)
    total = len(frame)
    dropped: Counter[str] = Counter()

    # Strip whitespace on the text columns before any comparison.
    for column in frame.columns:
        frame[column] = frame[column].str.strip()

    # --- placeholder / missing values ---
    placeholder = frame[list(REQUIRED_COLUMNS)].apply(
        lambda column: column.str.lower().isin({"", "n/a", "na", "nan"})
    )
    bad_required = placeholder.any(axis=1)
    dropped["missing required field"] = int(bad_required.sum())
    frame = frame[~bad_required]

    # --- scores must be integers ---
    frame = frame.assign(
        home_score=pd.to_numeric(frame["home_score"], errors="coerce"),
        away_score=pd.to_numeric(frame["away_score"], errors="coerce"),
    )
    bad_scores = frame["home_score"].isna() | frame["away_score"].isna()
    dropped["unparseable score"] = int(bad_scores.sum())
    frame = frame[~bad_scores]
    frame = frame.astype({"home_score": int, "away_score": int})

    # --- dates must parse ---
    frame = frame.assign(
        date=pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    )
    bad_dates = frame["date"].isna()
    dropped["unparseable date"] = int(bad_dates.sum())
    frame = frame[~bad_dates]
    frame = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d"))

    # --- competition classification and exclusions ---
    frame = frame.assign(competition_type=frame["competition"].map(classify_competition))
    excluded = frame["competition"].map(is_excluded)
    dropped["youth or women's competition"] = int(excluded.sum())
    frame = frame[~excluded]

    # --- a team cannot play itself ---
    self_play = frame["home_team"] == frame["away_team"]
    dropped["home team == away team"] = int(self_play.sum())
    frame = frame[~self_play]

    frame = frame.sort_values(["date", "competition", "home_team"], kind="stable")
    frame = frame.reset_index(drop=True)

    if frame.empty:
        raise ValueError(
            f"Filtering removed every row from {source}. Dropped: {dict(dropped)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")

    log.info("Filtered %d rows -> %d", total, len(frame))
    for reason, count in dropped.most_common():
        if count:
            log.info("  dropped %6d: %s", count, reason)
    log.info("Wrote %s", destination)

    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    filter_matches()


if __name__ == "__main__":
    main()
