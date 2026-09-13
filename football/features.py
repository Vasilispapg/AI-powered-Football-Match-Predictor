"""Build the pre-match feature table.

This module is the fix for the defect that made the old model unusable: it
trained on ball possession, shots, corners and fouls *from the match being
predicted*. Those are known only after the final whistle, so the model could
never score a fixture that had not been played.

Everything here obeys one rule: **a feature for a match on date D may only use
data from matches played strictly before D.** Match statistics still appear, but
as rolling averages of each team's *previous* matches, which is legitimate and
genuinely predictive.

The rule is enforced structurally. Matches are processed one calendar day at a
time: every match on day D is featurised against team histories that end at
D-1, and only then are day D's results folded into those histories. Two teams
meeting twice on the same day cannot leak into each other, because the date
column cannot order same-day matches and so is not trusted to.

``tests/test_no_leakage.py`` checks the rule independently.
"""

from __future__ import annotations

import ast
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths

log = logging.getLogger(__name__)

#: Rolling windows, in matches, used for form features.
ROLLING_WINDOWS = (5, 10)

#: Longest history any feature needs. Histories are capped at this length.
MAX_HISTORY = max(ROLLING_WINDOWS)

#: Outcome encoding. Ordered so that the class index increases as the result
#: moves from home to away.
OUTCOME_LABELS = {0: "home_win", 1: "draw", 2: "away_win"}

#: Elo parameters. A rating system carries information that a fixed-window
#: rolling average cannot: it knows *who* a team beat. Two sides on 2.0 points
#: per game are indistinguishable to a form feature even if one played the
#: league leaders and the other played the bottom three.
ELO_START = 1500.0
ELO_K = 20.0
#: Home advantage, in rating points. ~60 is the conventional value and matches
#: this dataset's 44.5%/24.8%/30.8% outcome split closely.
ELO_HOME_ADVANTAGE = 60.0

#: Match statistics available in the scraped data, mapped to the feature stem
#: used for their rolling averages. ``Total shots`` and ``Ball possession`` are
#: present in roughly half as many rows as the other two, because
#: ``getStats.py`` returns from inside its category loop.
STAT_KEYS = {
    "Ball possession": "possession",
    "Total shots": "shots",
    "Corners": "corners",
    "Fouls": "fouls",
}


def parse_votes(raw: object) -> tuple[int, int, int] | None:
    """Parse the crowd-vote column into ``(home, draw, away)`` counts.

    Stored as ``[[home_name, 'Draw', away_name], [home_votes, draw_votes,
    away_votes]]``. Uses :func:`ast.literal_eval`, never :func:`eval` -- the old
    ``train_func.py`` ran ``eval()`` over strings built from scraped web pages.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text == "[]":
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
        return None
    counts = parsed[1]
    if not isinstance(counts, (list, tuple)) or len(counts) != 3:
        return None
    try:
        home, draw, away = (int(value) for value in counts)
    except (TypeError, ValueError):
        return None
    if min(home, draw, away) < 0:
        return None
    return home, draw, away


def parse_stats(raw: object) -> dict[str, tuple[float, float]]:
    """Parse the match-statistics column into ``{stem: (home, away)}``."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    text = str(raw).strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for key, stem in STAT_KEYS.items():
        values = parsed.get(key)
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            continue
        try:
            home = float(str(values[0]).replace("%", "").strip())
            away = float(str(values[1]).replace("%", "").strip())
        except (TypeError, ValueError):
            continue
        out[stem] = (home, away)
    return out


@dataclass(slots=True)
class MatchRecord:
    """One past match, from a single team's point of view."""

    date: pd.Timestamp
    points: int
    goals_for: int
    goals_against: int
    at_home: bool
    possession: float | None = None
    shots: float | None = None
    corners: float | None = None
    fouls: float | None = None


def elo_expected(
    home_rating: float, away_rating: float, home_advantage: float = ELO_HOME_ADVANTAGE
) -> float:
    """Elo's expected score for the home side, in [0, 1].

    A draw counts as half, so this is not a win probability -- it is the
    expected points share. The model learns the mapping to three outcomes.
    """
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - home_advantage) / 400.0))


def elo_update(
    home_rating: float,
    away_rating: float,
    home_score: int,
    away_score: int,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Ratings after a result. Zero-sum: what one side gains the other loses."""
    if home_score > away_score:
        actual = 1.0
    elif home_score == away_score:
        actual = 0.5
    else:
        actual = 0.0

    expected = elo_expected(home_rating, away_rating, home_advantage)
    # Scale by margin of victory, so a 5-0 moves the ratings further than a 1-0.
    margin = max(abs(home_score - away_score), 1)
    delta = k * math.sqrt(margin) * (actual - expected)
    return home_rating + delta, away_rating - delta


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _side_features(
    history: deque[MatchRecord],
    date: pd.Timestamp,
    at_home: bool,
    prefix: str,
) -> dict[str, float]:
    """Features for one team, from *history* only. Never reads the current match."""
    records = list(history)
    features: dict[str, float] = {f"{prefix}_matches_played": float(len(records))}

    for window in ROLLING_WINDOWS:
        recent = records[-window:]
        features[f"{prefix}_ppg_{window}"] = _mean([r.points for r in recent])
        if window == ROLLING_WINDOWS[0]:
            features[f"{prefix}_gf_{window}"] = _mean([r.goals_for for r in recent])
            features[f"{prefix}_ga_{window}"] = _mean([r.goals_against for r in recent])

    # Form in the same venue role: a team's home form predicts its next home
    # match better than its overall form does.
    same_venue = [r for r in records if r.at_home == at_home][-ROLLING_WINDOWS[0] :]
    features[f"{prefix}_venue_ppg"] = _mean([r.points for r in same_venue])
    features[f"{prefix}_venue_matches"] = float(len(same_venue))

    # Rolling averages of the match statistics. These are the same quantities
    # the old model read from the current match; here they come only from
    # earlier ones.
    window = ROLLING_WINDOWS[0]
    recent = records[-window:]
    for stem in ("possession", "shots", "corners", "fouls"):
        values = [getattr(record, stem) for record in recent if getattr(record, stem) is not None]
        features[f"{prefix}_{stem}_avg"] = _mean(values)

    if records:
        features[f"{prefix}_rest_days"] = float((date - records[-1].date).days)
    else:
        features[f"{prefix}_rest_days"] = np.nan

    return features


_EMPTY_HISTORY: deque[MatchRecord] = deque()


class TeamHistories:
    """Per-team match history, grown one calendar day at a time.

    Training and prediction both drive this same object, which is what makes
    train/serve skew detectable: ``tests/test_predict.py`` asserts that a
    fixture featurised through the prediction path is identical to the same
    fixture featurised during training.
    """

    __slots__ = ("_elo", "_histories", "_referees")

    def __init__(self) -> None:
        self._histories: dict[str, deque[MatchRecord]] = {}
        self._elo: dict[str, float] = {}
        # Referee -> [matches officiated, home wins]. Officials differ
        # measurably in how much home advantage they allow; the betting market
        # is usually set before the appointment is known, so this is one of the
        # few signals that is not already inside the odds.
        self._referees: dict[str, list[int]] = {}

    def referee_home_rate(self, referee: str) -> tuple[float, float]:
        """``(home win rate, matches officiated)`` before today."""
        if not referee:
            return float("nan"), 0.0
        record = self._referees.get(referee)
        if not record or record[0] == 0:
            return float("nan"), 0.0
        return record[1] / record[0], float(record[0])

    def get(self, team: str) -> deque[MatchRecord]:
        return self._histories.get(team, _EMPTY_HISTORY)

    def elo(self, team: str) -> float:
        """Current rating, or the starting rating for an unseen team."""
        return self._elo.get(team, ELO_START)

    def knows(self, team: str) -> bool:
        return team in self._histories

    def teams(self) -> set[str]:
        return set(self._histories)

    def __len__(self) -> int:
        return len(self._histories)

    def add_day(self, date: pd.Timestamp, day: pd.DataFrame) -> None:
        """Fold one day's results into every participating team's history."""
        # Ratings are read from a snapshot so that two matches on the same day
        # cannot feed each other -- the same reason histories are updated a day
        # at a time.
        elo_before = dict(self._elo)
        deltas: dict[str, float] = {}
        for _, match in day.iterrows():
            home, away = match["home_team"], match["away_team"]
            before_home = elo_before.get(home, ELO_START)
            before_away = elo_before.get(away, ELO_START)
            after_home, after_away = elo_update(
                before_home,
                before_away,
                int(match["home_score"]),
                int(match["away_score"]),
            )
            # Accumulate rather than assign: a team playing twice in one day
            # must keep both results, and each is rated against the same
            # start-of-day snapshot.
            deltas[home] = deltas.get(home, 0.0) + (after_home - before_home)
            deltas[away] = deltas.get(away, 0.0) + (after_away - before_away)

        for team, delta in deltas.items():
            self._elo[team] = elo_before.get(team, ELO_START) + delta

        for _, match in day.iterrows():
            referee = str(match.get("referee") or "").strip()
            if not referee:
                continue
            record = self._referees.setdefault(referee, [0, 0])
            record[0] += 1
            if int(match["home_score"]) > int(match["away_score"]):
                record[1] += 1

        for _, match in day.iterrows():
            home_score = int(match["home_score"])
            away_score = int(match["away_score"])
            stats = parse_stats(match.get("stats"))

            if home_score > away_score:
                home_points, away_points = 3, 0
            elif home_score == away_score:
                home_points, away_points = 1, 1
            else:
                home_points, away_points = 0, 3

            for team, at_home, points, scored, conceded, index in (
                (match["home_team"], True, home_points, home_score, away_score, 0),
                (match["away_team"], False, away_points, away_score, home_score, 1),
            ):
                self._histories.setdefault(team, deque(maxlen=MAX_HISTORY)).append(
                    MatchRecord(
                        date=date,
                        points=points,
                        goals_for=scored,
                        goals_against=conceded,
                        at_home=at_home,
                        possession=stats.get("possession", (None, None))[index],
                        shots=stats.get("shots", (None, None))[index],
                        corners=stats.get("corners", (None, None))[index],
                        fouls=stats.get("fouls", (None, None))[index],
                    )
                )


def fixture_features(
    histories: TeamHistories,
    fixture: pd.Series | dict,
    date: pd.Timestamp,
    include_votes: bool = True,
) -> dict[str, float]:
    """Features for one fixture, from *histories* as they stand.

    The caller is responsible for having folded in only matches before *date* --
    :func:`build_features` does that day by day, and the prediction path replays
    history up to the fixture date.
    """
    if isinstance(fixture, dict):
        fixture = pd.Series(fixture)

    features = _match_features(fixture, include_votes)
    features.update(_side_features(histories.get(fixture["home_team"]), date, True, "home"))
    features.update(_side_features(histories.get(fixture["away_team"]), date, False, "away"))
    referee_rate, referee_matches = histories.referee_home_rate(
        str(fixture.get("referee") or "").strip()
    )
    features["referee_home_rate"] = referee_rate
    features["referee_matches"] = referee_matches

    home_elo = histories.elo(fixture["home_team"])
    away_elo = histories.elo(fixture["away_team"])
    features["home_elo"] = home_elo
    features["away_elo"] = away_elo
    features["elo_diff"] = home_elo - away_elo
    features["elo_expected"] = elo_expected(home_elo, away_elo)

    features["ppg_diff_5"] = features["home_ppg_5"] - features["away_ppg_5"]
    features["rest_diff"] = features["home_rest_days"] - features["away_rest_days"]
    features["home_gd_5"] = features["home_gf_5"] - features["home_ga_5"]
    features["away_gd_5"] = features["away_gf_5"] - features["away_ga_5"]
    features["gd_diff_5"] = features["home_gd_5"] - features["away_gd_5"]
    return features


def _match_features(row: pd.Series, include_votes: bool) -> dict[str, float]:
    """Features that come from the fixture itself, all known before kick-off."""
    mv_home = row.get("mv_home")
    mv_away = row.get("mv_away")

    features: dict[str, float] = {
        "mv_home": float(mv_home) if pd.notna(mv_home) else np.nan,
        "mv_away": float(mv_away) if pd.notna(mv_away) else np.nan,
        "mv_home_missing": float(pd.isna(mv_home)),
        "mv_away_missing": float(pd.isna(mv_away)),
    }

    if pd.notna(mv_home) and pd.notna(mv_away):
        # Log ratio rather than difference: squad values span four orders of
        # magnitude, so a EUR 50m gap means something different at the top of
        # the Premier League than in the Scottish lower leagues.
        features["mv_log_ratio"] = math.log((float(mv_home) + 1.0) / (float(mv_away) + 1.0))
    else:
        features["mv_log_ratio"] = np.nan

    competition_type = str(row.get("competition_type", "league"))
    features["is_cup"] = float(competition_type == "cup")
    features["is_friendly"] = float(competition_type == "friendly")

    # Pre-match betting odds, already converted to overround-free probabilities
    # by the football-data source. These are the strongest single signal
    # available: the market prices in injuries, suspensions and lineups, none of
    # which any other feature here can see. Absent for the goal.com rows, which
    # is what odds_missing records.
    home_probability = row.get("odds_home_prob")
    if home_probability is None or pd.isna(home_probability):
        features.update(
            odds_home_prob=np.nan,
            odds_draw_prob=np.nan,
            odds_away_prob=np.nan,
            odds_overround=np.nan,
            odds_margin=np.nan,
            odds_missing=1.0,
        )
    else:
        draw = float(row.get("odds_draw_prob", np.nan))
        away = float(row.get("odds_away_prob", np.nan))
        features.update(
            odds_home_prob=float(home_probability),
            odds_draw_prob=draw,
            odds_away_prob=away,
            odds_overround=float(row.get("odds_overround", np.nan)),
            odds_margin=float(home_probability) - away,
            odds_missing=0.0,
        )

    # A second, independent view of the same market. 1X2 odds say who wins;
    # these say how the goals are expected to arrive. A 1-0 grind and a 4-3
    # shootout can carry identical win probabilities.
    over25 = row.get("odds_over25_prob")
    features["odds_over25_prob"] = np.nan if over25 is None or pd.isna(over25) else float(over25)
    handicap = row.get("odds_handicap")
    features["odds_handicap"] = np.nan if handicap is None or pd.isna(handicap) else float(handicap)

    hour = row.get("kickoff_hour")
    features["kickoff_hour"] = np.nan if hour is None or pd.isna(hour) else float(hour)

    # Opening and closing prices carried through separately so that
    # football.line_movement can ask whether the drift between them is
    # anticipatable. Absent from the goal.com rows.
    for column in (
        "open_home_prob",
        "open_draw_prob",
        "open_away_prob",
        "open_overround",
        "close_home_prob",
        "close_draw_prob",
        "close_away_prob",
        "close_overround",
    ):
        value = row.get(column)
        features[column] = np.nan if value is None or pd.isna(value) else float(value)

    if include_votes:
        votes = parse_votes(row.get("votes"))
        if votes is None:
            features.update(
                votes_total=np.nan,
                votes_home_share=np.nan,
                votes_draw_share=np.nan,
                votes_away_share=np.nan,
                votes_margin=np.nan,
                votes_missing=1.0,
            )
        else:
            home, draw, away = votes
            total = home + draw + away
            if total == 0:
                features.update(
                    votes_total=0.0,
                    votes_home_share=np.nan,
                    votes_draw_share=np.nan,
                    votes_away_share=np.nan,
                    votes_margin=np.nan,
                    votes_missing=1.0,
                )
            else:
                features.update(
                    votes_total=float(total),
                    votes_home_share=home / total,
                    votes_draw_share=draw / total,
                    votes_away_share=away / total,
                    votes_margin=(home - away) / total,
                    votes_missing=0.0,
                )

    return features


def _outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def build_features(
    matches: pd.DataFrame | None = None,
    *,
    include_votes: bool = True,
    min_matches_played: int = 0,
) -> pd.DataFrame:
    """Return one row of pre-match features per match, in date order.

    Args:
        matches: the processed dataset. Read from disk when omitted.
        include_votes: include crowd-vote features. Turn this off to measure
            how much of the model's skill rests on them -- they are collected
            from the same page as the result, and if the site keeps accepting
            votes after kick-off they would be leakage too.
        min_matches_played: drop matches where either side has fewer than this
            many prior matches. 0 keeps everything and lets the model use
            ``*_matches_played`` to discount thin histories itself.
    """
    if matches is None:
        matches = pd.read_csv(paths.MATCHES, keep_default_na=False, na_values=[""])

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "competition", "home_team"], kind="stable")

    histories = TeamHistories()
    rows: list[dict[str, float]] = []
    meta: list[dict[str, object]] = []

    for date, day in frame.groupby("date", sort=True):
        # --- featurise every match on this day against history up to D-1 ---
        for _, match in day.iterrows():
            rows.append(fixture_features(histories, match, date, include_votes))
            meta.append(
                {
                    "date": date,
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "competition": match["competition"],
                    "outcome": _outcome(int(match["home_score"]), int(match["away_score"])),
                }
            )

        # --- only now do this day's results become history ---
        histories.add_day(date, day)

    features_frame = pd.DataFrame(rows)
    meta_frame = pd.DataFrame(meta)
    result = pd.concat(
        [meta_frame.reset_index(drop=True), features_frame.reset_index(drop=True)], axis=1
    )

    if min_matches_played > 0:
        keep = (result["home_matches_played"] >= min_matches_played) & (
            result["away_matches_played"] >= min_matches_played
        )
        log.info(
            "Dropping %d matches with fewer than %d prior matches per side",
            int((~keep).sum()),
            min_matches_played,
        )
        result = result[keep].reset_index(drop=True)

    log.info("Built %d feature rows, %d features", len(result), len(feature_columns(result)))
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The model-input columns of a feature frame (everything but the metadata)."""
    metadata = {"date", "home_team", "away_team", "competition", "outcome"}
    return [column for column in frame.columns if column not in metadata]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    paths.ensure_dirs()
    features = build_features()
    destination = Path(paths.PROCESSED) / "features.csv"
    features.to_csv(destination, index=False, encoding="utf-8")
    log.info("Wrote %s", destination)


if __name__ == "__main__":
    main()
