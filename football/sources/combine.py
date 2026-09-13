"""Combine the goal.com scrape with football-data.co.uk into one dataset.

    python -m football.sources.combine

The two sources name clubs differently -- ``Man Utd`` against ``Man United``,
``Beşiktaş`` against ``Besiktas`` -- so concatenating them blindly would file one
club under two identities, splitting its match history and halving the form and
Elo features built from it.

The bridge uses :func:`football.teams.normalise.normalise`, but only accepts
**unambiguous** mappings: a normalised key that resolves to exactly one name on
each side. That guard matters. ``normalise`` strips bare years, so ``CSKA 1948``
and ``CSKA`` both reduce to ``cska`` -- two genuinely different clubs. Any key
with more than one candidate on either side is left unbridged and reported,
because a wrong merge is far worse than a missed one: it silently fabricates a
history that never happened.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from football import paths
from football.teams.normalise import normalise

log = logging.getLogger(__name__)

COMBINED = paths.PROCESSED / "matches_combined.csv"
BRIDGE_REPORT = paths.INTERIM / "source_team_bridge.csv"

#: Columns the combined dataset carries. The odds columns are absent from the
#: goal.com side and stay NaN there, flagged by ``odds_missing``.
COLUMNS = [
    "competition",
    "country",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "date",
    "url",
    "votes",
    "stats",
    "competition_type",
    "mv_home",
    "mv_away",
    "mv_home_missing",
    "mv_away_missing",
    "odds_home_prob",
    "odds_draw_prob",
    "odds_away_prob",
    "odds_overround",
    "source",
]


def build_bridge(
    primary_teams: set[str], secondary_teams: set[str]
) -> tuple[dict[str, str], list[tuple[str, list[str], list[str]]]]:
    """Map *secondary* team names onto *primary* ones where unambiguous.

    Returns ``(mapping, ambiguous)``. ``mapping`` is secondary name -> primary
    name; ``ambiguous`` lists normalised keys that matched more than one name on
    either side and were therefore skipped.
    """
    primary_by_key: dict[str, list[str]] = defaultdict(list)
    for name in primary_teams:
        primary_by_key[normalise(name)].append(name)

    secondary_by_key: dict[str, list[str]] = defaultdict(list)
    for name in secondary_teams:
        secondary_by_key[normalise(name)].append(name)

    mapping: dict[str, str] = {}
    ambiguous: list[tuple[str, list[str], list[str]]] = []

    for key, secondary_names in secondary_by_key.items():
        primary_names = primary_by_key.get(key)
        if not primary_names:
            continue
        if len(primary_names) > 1 or len(secondary_names) > 1:
            ambiguous.append((key, sorted(primary_names), sorted(secondary_names)))
            continue
        secondary_name = secondary_names[0]
        primary_name = primary_names[0]
        if secondary_name != primary_name:
            mapping[secondary_name] = primary_name

    return mapping, ambiguous


def combine(
    goal_path: Path | None = None,
    footballdata_path: Path | None = None,
    extra_path: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Concatenate every source, bridging team names and dropping duplicates.

    Three sources, in descending priority where they describe the same match:
    the main football-data files (odds, statistics, opening and closing prices),
    the extra-leagues files (closing odds only, but 16 more countries), and the
    goal.com scrape (no odds, but 271 competitions).
    """
    goal_path = goal_path or paths.MATCHES
    footballdata_path = footballdata_path or paths.INTERIM / "footballdata_matches.csv"
    extra_path = extra_path or paths.INTERIM / "footballdata_extra.csv"
    destination = destination or COMBINED

    goal = pd.read_csv(goal_path, keep_default_na=False, na_values=[""])
    goal["source"] = "goal.com"

    if not footballdata_path.exists():
        raise FileNotFoundError(
            f"{footballdata_path} not found. Run:\n"
            f"    python -m football.sources.footballdata --fixtures"
        )
    extra = pd.read_csv(footballdata_path)

    pieces = [extra]
    if extra_path.exists():
        extra_leagues = pd.read_csv(extra_path)
        log.info("Adding %d extra-league matches from %s", len(extra_leagues), extra_path.name)
        pieces.append(extra_leagues)
    else:
        log.info(
            "No extra-leagues file at %s -- collect it with "
            "`python -m football.sources.footballdata --extra`",
            extra_path,
        )
    extra = pd.concat(pieces, ignore_index=True)

    goal_teams = set(goal["home_team"]) | set(goal["away_team"])
    extra_teams = set(extra["home_team"]) | set(extra["away_team"])
    mapping, ambiguous = build_bridge(goal_teams, extra_teams)

    log.info(
        "Bridged %d of %d football-data team name(s) onto goal.com names",
        len(mapping),
        len(extra_teams),
    )
    if ambiguous:
        log.warning(
            "%d normalised key(s) were ambiguous and left unbridged -- see %s",
            len(ambiguous),
            BRIDGE_REPORT,
        )

    extra["home_team"] = extra["home_team"].map(lambda name: mapping.get(name, name))
    extra["away_team"] = extra["away_team"].map(lambda name: mapping.get(name, name))

    combined = pd.concat([goal, extra], ignore_index=True)
    for column in COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    combined = combined[COLUMNS]

    # Where the two sources describe the same match, keep football-data: it
    # carries betting odds and richer statistics.
    before = len(combined)
    combined["_priority"] = (combined["source"] == "goal.com").astype(int)
    combined = combined.sort_values(["date", "_priority"], kind="stable")
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    combined = combined.drop(columns="_priority")
    combined = combined.sort_values(["date", "competition", "home_team"], kind="stable")
    combined = combined.reset_index(drop=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, encoding="utf-8")

    BRIDGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "normalised": key,
                "goal_com": " | ".join(primary),
                "football_data": " | ".join(secondary),
            }
            for key, primary, secondary in sorted(ambiguous)
        ],
        columns=["normalised", "goal_com", "football_data"],
    ).to_csv(BRIDGE_REPORT, index=False, encoding="utf-8")

    log.info("Combined %d rows -> %d after de-duplication", before, len(combined))
    log.info("Wrote %s", destination)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="destination CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    combined = combine(destination=args.out)

    odds = combined["odds_home_prob"].notna().sum()
    print()
    print(f"{len(combined):,} matches -> {args.out or COMBINED}")
    print(f"  date range   : {combined['date'].min()} -> {combined['date'].max()}")
    print(f"  with odds    : {odds:,} ({odds / len(combined):.1%})")
    print("  by source    :")
    for source, count in combined["source"].value_counts().items():
        print(f"      {source:<22} {count:,}")


if __name__ == "__main__":
    main()
