"""Ingest football-data.co.uk into the project schema.

    python -m football.sources.footballdata --seasons 2324 2425 2526 2627
    python -m football.sources.footballdata --fixtures

Why this source exists alongside the scraper:

* it is **current** -- results refreshed several times a week, plus a fixtures
  file for matches that have not been played yet, which is what the model needs
  to be useful at all;
* it is **static CSV**, so nothing breaks when a frontend is redeployed;
* it carries **betting odds**. Closing odds are the strongest single predictor
  in football outcome modelling: the market aggregates injuries, lineups,
  suspensions and everything else the public knows, none of which any other
  feature here sees.

The cost is coverage -- roughly 22 leagues rather than the scrape's 271
competitions -- so this augments the existing dataset rather than replacing it.
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from football import paths

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk"
SEASON_URL = f"{BASE_URL}/mmz4281/{{season}}/{{division}}.csv"
FIXTURES_URL = f"{BASE_URL}/fixtures.csv"

#: Identify the client honestly rather than impersonating a browser.
USER_AGENT = "football-predictor/0.1 (personal research; +https://github.com)"

#: Seconds between requests.
REQUEST_DELAY = 1.5

#: Division code -> (country, competition). The site's "main leagues", which are
#: the ones carrying match statistics as well as odds.
DIVISIONS: dict[str, tuple[str, str]] = {
    "E0": ("England", "Premier League"),
    "E1": ("England", "Championship"),
    "E2": ("England", "League One"),
    "E3": ("England", "League Two"),
    "EC": ("England", "National League"),
    "SC0": ("Scotland", "Premiership"),
    "SC1": ("Scotland", "Championship"),
    "SC2": ("Scotland", "League One"),
    "SC3": ("Scotland", "League Two"),
    "D1": ("Germany", "Bundesliga"),
    "D2": ("Germany", "2. Bundesliga"),
    "I1": ("Italy", "Serie A"),
    "I2": ("Italy", "Serie B"),
    "SP1": ("Spain", "La Liga"),
    "SP2": ("Spain", "Segunda Division"),
    "F1": ("France", "Ligue 1"),
    "F2": ("France", "Ligue 2"),
    "N1": ("Netherlands", "Eredivisie"),
    "B1": ("Belgium", "Jupiler League"),
    "P1": ("Portugal", "Primeira Liga"),
    "T1": ("Turkey", "Super Lig"),
    "G1": ("Greece", "Super League"),
}

#: Closing odds -- taken nearest kick-off, after the money has come in. This is
#: the market's final answer and the hardest thing in football to beat.
CLOSING_PREFERENCES: list[tuple[str, str, str]] = [
    ("AvgCH", "AvgCD", "AvgCA"),  # closing average across bookmakers
    ("PSCH", "PSCD", "PSCA"),  # closing Pinnacle
    ("B365CH", "B365CD", "B365CA"),  # closing Bet365
    ("MaxCH", "MaxCD", "MaxCA"),  # closing best price
]

#: Opening odds -- posted before the market has absorbed any money. These are
#: the prices you could actually have bet at early, which makes them the honest
#: input for a "can we beat the market" question.
OPENING_PREFERENCES: list[tuple[str, str, str]] = [
    ("AvgH", "AvgD", "AvgA"),  # opening average across bookmakers
    ("PSH", "PSD", "PSA"),  # opening Pinnacle
    ("B365H", "B365D", "B365A"),  # opening Bet365
    ("BbAvH", "BbAvD", "BbAvA"),  # older seasons
]

#: Kept for callers that just want "the best available odds", closing first.
ODDS_PREFERENCES: list[tuple[str, str, str]] = CLOSING_PREFERENCES + OPENING_PREFERENCES

#: The site's second dataset: one file per country holding every season since
#: 2012, in a different schema from the main European files. Closing odds only
#: -- no opening prices, no totals, no handicap, no match statistics -- but it
#: roughly doubles the number of competitions that carry a market at all.
EXTRA_URL = f"{BASE_URL}/new/{{country}}.csv"

EXTRA_COUNTRIES: tuple[str, ...] = (
    "ARG",
    "AUT",
    "BRA",
    "CHN",
    "DNK",
    "FIN",
    "IRL",
    "JPN",
    "MEX",
    "NOR",
    "POL",
    "ROU",
    "RUS",
    "SWE",
    "SWZ",
    "USA",
)

#: Column names in the extra files, which differ from the main ones.
EXTRA_RENAMES = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
}

#: Over/under 2.5 goals, in preference order. This is a second, independent view
#: of the market: 1X2 odds say who wins, these say how many goals are expected.
TOTALS_PREFERENCES: list[tuple[str, str]] = [
    ("AvgC>2.5", "AvgC<2.5"),
    ("AvgC>2.5", "AvgC<2.5"),
    ("Avg>2.5", "Avg<2.5"),
    ("P>2.5", "P<2.5"),
    ("B365>2.5", "B365<2.5"),
]

#: Asian handicap line columns, closing first. The line itself is the market's
#: own estimate of the goal difference -- a continuous number where 1X2 odds
#: give only three buckets.
HANDICAP_COLUMNS: list[str] = ["AHCh", "AHh", "BbAHh"]

#: Match statistics, mapped onto the same stems the scraper produced.
STAT_COLUMNS: dict[str, tuple[str, str]] = {
    "Total shots": ("HS", "AS"),
    "Shots on target": ("HST", "AST"),
    "Corners": ("HC", "AC"),
    "Fouls": ("HF", "AF"),
    "Yellow cards": ("HY", "AY"),
    "Red cards": ("HR", "AR"),
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read one of the site's CSVs.

    Encoding is not consistent across the site: the season files are plain
    latin-1 while ``fixtures.csv`` carries a UTF-8 BOM, which latin-1 decodes
    into a literal ``ï»¿`` glued to the first column name. Reading the fixtures
    as latin-1 therefore produced a frame with no ``Div`` column and silently
    yielded zero fixtures.
    """
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    frame.columns = [str(column).strip().lstrip("﻿") for column in frame.columns]
    return frame


def season_label(code: str) -> str:
    """``"2627"`` -> ``"2026/2027"``."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Season code must be four digits, got {code!r}")
    start, end = code[:2], code[2:]
    century = "20" if int(start) < 90 else "19"
    return f"{century}{start}/{century if int(end) > int(start) else '20'}{end}"


def fetch(url: str, timeout: int = 30) -> bytes:
    """Download *url* from football-data.co.uk over HTTPS.

    The scheme and host are checked rather than trusted: every caller builds
    its URL from the constants above, but an explicit guard means a future
    caller cannot accidentally turn this into a fetcher for ``file:`` URLs or
    arbitrary hosts.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc.endswith("football-data.co.uk"):
        raise ValueError(f"Refusing to fetch {url!r}: expected https://…football-data.co.uk")

    # S310 suppressed on both lines: the guard above already restricts the
    # scheme to https and the host to football-data.co.uk.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def download_season(season: str, division: str, directory: Path | None = None) -> Path | None:
    """Download one league-season CSV. Returns ``None`` when it does not exist."""
    directory = directory or paths.RAW / "footballdata"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{season}_{division}.csv"

    url = SEASON_URL.format(season=season, division=division)
    try:
        payload = fetch(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            log.debug("%s not published", url)
            return None
        log.warning("%s -> HTTP %s", url, error.code)
        return None
    except (urllib.error.URLError, TimeoutError) as error:
        log.warning("%s -> %s", url, error)
        return None

    if not payload.strip():
        return None
    destination.write_bytes(payload)
    return destination


def download_fixtures(directory: Path | None = None) -> Path | None:
    """Download the upcoming-fixtures file (with pre-match odds)."""
    directory = directory or paths.RAW / "footballdata"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "fixtures.csv"
    try:
        payload = fetch(FIXTURES_URL)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        log.error("Could not fetch fixtures: %s", error)
        return None
    destination.write_bytes(payload)
    return destination


# --------------------------------------------------------------------------
# Odds
# --------------------------------------------------------------------------


def pick_odds(
    row: pd.Series,
    preferences: list[tuple[str, str, str]] | None = None,
) -> tuple[float, float, float] | None:
    """First usable (home, draw, away) decimal odds triple from *row*.

    Pass :data:`CLOSING_PREFERENCES` or :data:`OPENING_PREFERENCES` to pin the
    result to one side of the market; the default takes closing where present
    and falls back to opening.
    """
    for home, draw, away in preferences or ODDS_PREFERENCES:
        if home not in row.index:
            continue
        values = (row.get(home), row.get(draw), row.get(away))
        if any(pd.isna(value) for value in values):
            continue
        try:
            numbers = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            continue
        if all(number > 1.0 for number in numbers):
            return numbers
    return None


def pick_totals(row: pd.Series) -> float | None:
    """Market probability that the match goes over 2.5 goals, margin removed.

    Complements the 1X2 odds: a match priced as a 1-0 grind and one priced as a
    4-3 shootout can carry identical win probabilities.
    """
    for over_column, under_column in TOTALS_PREFERENCES:
        if over_column not in row.index:
            continue
        over, under = row.get(over_column), row.get(under_column)
        if pd.isna(over) or pd.isna(under):
            continue
        try:
            over, under = float(over), float(under)
        except (TypeError, ValueError):
            continue
        if over <= 1.0 or under <= 1.0:
            continue
        implied_over, implied_under = 1.0 / over, 1.0 / under
        return implied_over / (implied_over + implied_under)
    return None


def pick_handicap(row: pd.Series) -> float | None:
    """The Asian handicap line, as goals the home side gives away.

    Negative means the home team is favoured. Signs follow the source.
    """
    for column in HANDICAP_COLUMNS:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        try:
            line = float(value)
        except (TypeError, ValueError):
            continue
        if -10.0 < line < 10.0:
            return line
    return None


def kickoff_hour(value: object) -> float:
    """Kick-off hour as a number, or NaN. Late kick-offs differ from lunchtime ones."""
    text = str(value).strip()
    if not text or ":" not in text:
        return float("nan")
    try:
        return float(int(text.split(":")[0]))
    except (TypeError, ValueError):
        return float("nan")


def odds_to_probabilities(
    odds: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Convert decimal odds to probabilities, removing the bookmaker's margin.

    Raw implied probabilities (1/odds) sum to more than 1 -- the excess is the
    overround, the bookmaker's built-in edge. Normalising by the total removes
    it proportionally, which is the standard first-order correction.

    Returns ``(home, draw, away, overround)``. The overround is returned as a
    feature in its own right: a wide margin signals a market the bookmaker is
    less confident in.
    """
    implied = [1.0 / value for value in odds]
    total = sum(implied)
    return (*(value / total for value in implied), total - 1.0)


# --------------------------------------------------------------------------
# Schema mapping
# --------------------------------------------------------------------------


def _stats_literal(row: pd.Series) -> str:
    """Build the same ``{name: [home, away]}`` literal the scraper wrote."""
    stats: dict[str, list[str]] = {}
    for name, (home_column, away_column) in STAT_COLUMNS.items():
        home, away = row.get(home_column), row.get(away_column)
        if pd.isna(home) or pd.isna(away):
            continue
        try:
            stats[name] = [str(int(float(home))), str(int(float(away)))]
        except (TypeError, ValueError):
            continue
    return repr(stats) if stats else "{}"


def to_matches(frame: pd.DataFrame, division: str) -> pd.DataFrame:
    """Map a football-data.co.uk league-season frame into the project schema."""
    country, competition = DIVISIONS.get(division, ("Unknown", division))

    dates = pd.to_datetime(frame["Date"], format="mixed", dayfirst=True, errors="coerce")

    rows = []
    for (_, row), date in zip(frame.iterrows(), dates, strict=True):
        if pd.isna(date) or pd.isna(row.get("HomeTeam")) or pd.isna(row.get("AwayTeam")):
            continue
        home_goals, away_goals = row.get("FTHG"), row.get("FTAG")
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue  # not played yet; fixtures are handled separately

        odds = pick_odds(row)
        if odds is None:
            probabilities = (np.nan, np.nan, np.nan, np.nan)
        else:
            probabilities = odds_to_probabilities(odds)

        opening = pick_odds(row, OPENING_PREFERENCES)
        closing = pick_odds(row, CLOSING_PREFERENCES)
        open_probs = odds_to_probabilities(opening) if opening else (np.nan,) * 4
        close_probs = odds_to_probabilities(closing) if closing else (np.nan,) * 4

        rows.append(
            {
                "competition": competition,
                "country": country,
                "home_team": str(row["HomeTeam"]).strip(),
                "away_team": str(row["AwayTeam"]).strip(),
                "home_score": int(float(home_goals)),
                "away_score": int(float(away_goals)),
                "date": date.strftime("%Y-%m-%d"),
                "url": "",
                "votes": "[]",
                "stats": _stats_literal(row),
                "competition_type": "league",
                "odds_home_prob": probabilities[0],
                "odds_draw_prob": probabilities[1],
                "odds_away_prob": probabilities[2],
                "odds_overround": probabilities[3],
                "odds_over25_prob": pick_totals(row),
                "odds_handicap": pick_handicap(row),
                # Opening and closing kept apart so the drift between them can
                # be studied. Closing is the market's final answer; opening is
                # the price you could actually have taken early.
                "open_home_prob": open_probs[0],
                "open_draw_prob": open_probs[1],
                "open_away_prob": open_probs[2],
                "open_overround": open_probs[3],
                "close_home_prob": close_probs[0],
                "close_draw_prob": close_probs[1],
                "close_away_prob": close_probs[2],
                "close_overround": close_probs[3],
                "referee": str(row.get("Referee", "")).strip(),
                "kickoff_hour": kickoff_hour(row.get("Time")),
                "source": "football-data.co.uk",
            }
        )

    return pd.DataFrame(rows)


def download_extra(country: str, directory: Path | None = None) -> Path | None:
    """Download one country's full history from the extra-leagues dataset."""
    directory = directory or paths.RAW / "footballdata"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"extra_{country}.csv"

    url = EXTRA_URL.format(country=country)
    try:
        payload = fetch(url)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            log.warning("%s -> HTTP %s", url, error.code)
        return None
    except (urllib.error.URLError, TimeoutError) as error:
        log.warning("%s -> %s", url, error)
        return None

    if not payload.strip():
        return None
    destination.write_bytes(payload)
    return destination


def to_matches_extra(frame: pd.DataFrame) -> pd.DataFrame:
    """Map an extra-leagues file into the project schema.

    These files carry several competitions each, so country and competition come
    from the rows rather than from a division code.
    """
    frame = frame.rename(columns=EXTRA_RENAMES)
    dates = pd.to_datetime(frame["Date"], format="mixed", dayfirst=True, errors="coerce")

    rows = []
    for (_, row), date in zip(frame.iterrows(), dates, strict=True):
        if pd.isna(date) or pd.isna(row.get("HomeTeam")) or pd.isna(row.get("AwayTeam")):
            continue
        home_goals, away_goals = row.get("FTHG"), row.get("FTAG")
        if pd.isna(home_goals) or pd.isna(away_goals):
            continue

        closing = pick_odds(row, CLOSING_PREFERENCES)
        probabilities = odds_to_probabilities(closing) if closing else (np.nan,) * 4

        rows.append(
            {
                "competition": str(row.get("League", "")).strip() or "Unknown",
                "country": str(row.get("Country", "")).strip() or "Unknown",
                "home_team": str(row["HomeTeam"]).strip(),
                "away_team": str(row["AwayTeam"]).strip(),
                "home_score": int(float(home_goals)),
                "away_score": int(float(away_goals)),
                "date": date.strftime("%Y-%m-%d"),
                "url": "",
                "votes": "[]",
                "stats": "{}",  # the extra files carry no match statistics
                "competition_type": "league",
                "odds_home_prob": probabilities[0],
                "odds_draw_prob": probabilities[1],
                "odds_away_prob": probabilities[2],
                "odds_overround": probabilities[3],
                "odds_over25_prob": np.nan,
                "odds_handicap": np.nan,
                "open_home_prob": np.nan,
                "open_draw_prob": np.nan,
                "open_away_prob": np.nan,
                "open_overround": np.nan,
                "close_home_prob": probabilities[0],
                "close_draw_prob": probabilities[1],
                "close_away_prob": probabilities[2],
                "close_overround": probabilities[3],
                "referee": "",
                "kickoff_hour": kickoff_hour(row.get("Time")),
                "source": "football-data.co.uk/extra",
            }
        )

    return pd.DataFrame(rows)


def collect_extra(
    countries: tuple[str, ...] = EXTRA_COUNTRIES,
    directory: Path | None = None,
) -> pd.DataFrame:
    """Download and map every extra-leagues country."""
    frames: list[pd.DataFrame] = []
    for country in countries:
        path = download_extra(country, directory)
        time.sleep(REQUEST_DELAY)
        if path is None:
            log.info("%-4s not published", country)
            continue
        try:
            mapped = to_matches_extra(read_csv(path))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, KeyError) as error:
            log.warning("%s unreadable: %s", path.name, error)
            continue
        if mapped.empty:
            continue
        frames.append(mapped)
        log.info(
            "%-4s %6d matches  %s -> %s  (%d competitions, %d with odds)",
            country,
            len(mapped),
            mapped["date"].min(),
            mapped["date"].max(),
            mapped["competition"].nunique(),
            int(mapped["odds_home_prob"].notna().sum()),
        )

    if not frames:
        raise SystemExit("No extra-league files downloaded")

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
        .sort_values(["date", "competition", "home_team"])
        .reset_index(drop=True)
    )


def fixtures_to_frame(path: Path) -> pd.DataFrame:
    """Read the fixtures file into the fixture shape ``predict`` consumes."""
    frame = read_csv(path)
    dates = pd.to_datetime(frame["Date"], format="mixed", dayfirst=True, errors="coerce")

    rows = []
    for (_, row), date in zip(frame.iterrows(), dates, strict=True):
        division = str(row.get("Div", "")).strip()
        if pd.isna(date) or division not in DIVISIONS:
            continue
        country, competition = DIVISIONS[division]
        odds = pick_odds(row)
        probabilities = odds_to_probabilities(odds) if odds else (np.nan, np.nan, np.nan, np.nan)
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "time": str(row.get("Time", "")).strip(),
                "country": country,
                "competition": competition,
                "competition_type": "league",
                "home_team": str(row.get("HomeTeam", "")).strip(),
                "away_team": str(row.get("AwayTeam", "")).strip(),
                "odds_home_prob": probabilities[0],
                "odds_draw_prob": probabilities[1],
                "odds_away_prob": probabilities[2],
                "odds_overround": probabilities[3],
                "odds_over25_prob": pick_totals(row),
                "odds_handicap": pick_handicap(row),
                "referee": str(row.get("Referee", "")).strip(),
                "kickoff_hour": kickoff_hour(row.get("Time")),
            }
        )
    return pd.DataFrame(rows)


def collect(
    seasons: list[str],
    divisions: list[str] | None = None,
    directory: Path | None = None,
) -> pd.DataFrame:
    """Download and map every requested league-season."""
    divisions = divisions or list(DIVISIONS)
    frames: list[pd.DataFrame] = []
    missing = 0

    for season in seasons:
        for division in divisions:
            path = download_season(season, division, directory)
            time.sleep(REQUEST_DELAY)
            if path is None:
                missing += 1
                continue
            try:
                raw = read_csv(path)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                log.warning("%s unreadable: %s", path.name, error)
                continue
            mapped = to_matches(raw, division)
            if not mapped.empty:
                frames.append(mapped)
                log.info(
                    "%s %-4s %5d matches  (%d with odds)",
                    season_label(season),
                    division,
                    len(mapped),
                    int(mapped["odds_home_prob"].notna().sum()),
                )

    if not frames:
        raise SystemExit("Nothing downloaded -- check the season codes and connectivity")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    combined = combined.sort_values(["date", "competition", "home_team"]).reset_index(drop=True)
    log.info("%d league-season file(s) not published", missing)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2324", "2425", "2526", "2627"],
        help="four-digit season codes, e.g. 2627 for 2026/2027",
    )
    parser.add_argument(
        "--divisions",
        nargs="+",
        default=None,
        help=f"division codes (default: all {len(DIVISIONS)})",
    )
    parser.add_argument("--fixtures", action="store_true", help="also download upcoming fixtures")
    parser.add_argument(
        "--extra",
        action="store_true",
        help="also collect the extra-leagues dataset (16 more countries, 2012 on)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.INTERIM / "footballdata_matches.csv",
        help="where to write the mapped matches",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    paths.ensure_dirs()

    matches = collect(args.seasons, args.divisions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.out, index=False, encoding="utf-8")

    with_odds = int(matches["odds_home_prob"].notna().sum())
    print()
    print(f"{len(matches):,} matches -> {args.out}")
    print(f"  date range : {matches['date'].min()} -> {matches['date'].max()}")
    print(f"  with odds  : {with_odds:,} ({with_odds / len(matches):.1%})")
    print(f"  leagues    : {matches['competition'].nunique()}")

    if args.extra:
        extra = collect_extra()
        destination = paths.INTERIM / "footballdata_extra.csv"
        extra.to_csv(destination, index=False, encoding="utf-8")
        with_odds = int(extra["odds_home_prob"].notna().sum())
        print()
        print(f"{len(extra):,} extra-league matches -> {destination}")
        print(f"  date range   : {extra['date'].min()} -> {extra['date'].max()}")
        print(f"  countries    : {extra['country'].nunique()}")
        print(f"  competitions : {extra['competition'].nunique()}")
        print(f"  with odds    : {with_odds:,} ({with_odds / len(extra):.1%})")

    if args.fixtures:
        path = download_fixtures()
        if path:
            fixtures = fixtures_to_frame(path)
            destination = paths.INTERIM / "fixtures.csv"
            fixtures.to_csv(destination, index=False, encoding="utf-8")
            print(f"\n{len(fixtures):,} upcoming fixture(s) -> {destination}")
            if not fixtures.empty:
                print(f"  {fixtures['date'].min()} -> {fixtures['date'].max()}")


if __name__ == "__main__":
    main()
