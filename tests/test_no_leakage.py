"""The test that Phase 3 exists to satisfy.

A feature for a match on date D may use data only from matches played strictly
before D. The old model failed this: it read ball possession, shots, corners and
fouls from the match it was predicting, so it could not score an unplayed
fixture.

These tests check the property from the outside -- they do not trust the
implementation's own bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.features import build_features, feature_columns, parse_stats, parse_votes


def make_match(
    date: str,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    *,
    stats: str = "{}",
    votes: str = "[]",
    mv_home: float = 100.0,
    mv_away: float = 100.0,
) -> dict:
    return {
        "date": date,
        "competition": "Test League",
        "competition_type": "league",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "mv_home": mv_home,
        "mv_away": mv_away,
        "stats": stats,
        "votes": votes,
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


BIG_STATS = "{'Ball possession': ['90.0%', '10.0%'], 'Total shots': ['40', '1'], 'Corners': ['20', '0'], 'Fouls': ['1', '30']}"


class TestNoFutureData:
    def test_appending_future_matches_never_changes_past_features(self) -> None:
        """The strongest form of the check.

        If any feature peeked forward, adding matches *after* a row would change
        that row's features. Every value must be bit-identical.
        """
        past = [
            make_match("2023-01-01", "A", "B", 3, 0),
            make_match("2023-01-08", "B", "A", 1, 1),
            make_match("2023-01-15", "A", "C", 2, 1),
            make_match("2023-01-22", "C", "B", 0, 2),
        ]
        future = [
            make_match("2023-02-01", "A", "B", 5, 0, stats=BIG_STATS),
            make_match("2023-02-08", "C", "A", 4, 4),
        ]

        before = build_features(frame(past))
        after = build_features(frame(past + future))

        columns = feature_columns(before)
        pd.testing.assert_frame_equal(
            before[columns],
            after.iloc[: len(before)][columns].reset_index(drop=True),
            check_exact=True,
        )

    def test_first_ever_match_has_no_form(self) -> None:
        features = build_features(frame([make_match("2023-01-01", "A", "B", 3, 0)]))
        row = features.iloc[0]

        assert row["home_matches_played"] == 0
        assert row["away_matches_played"] == 0
        for column in ("home_ppg_5", "away_ppg_5", "home_gf_5", "home_rest_days"):
            assert np.isnan(row[column]), f"{column} should be NaN with no history"

    def test_current_match_stats_are_not_a_feature(self) -> None:
        """A blowout with extreme statistics must not describe itself."""
        features = build_features(
            frame([make_match("2023-01-01", "A", "B", 9, 0, stats=BIG_STATS)])
        )
        row = features.iloc[0]

        assert np.isnan(row["home_possession_avg"])
        assert np.isnan(row["home_shots_avg"])
        assert np.isnan(row["away_fouls_avg"])

    def test_result_of_the_match_is_not_a_feature(self) -> None:
        """Same fixture, opposite scorelines, identical features."""
        home_win = build_features(frame([make_match("2023-01-01", "A", "B", 5, 0)]))
        away_win = build_features(frame([make_match("2023-01-01", "A", "B", 0, 5)]))

        columns = feature_columns(home_win)
        pd.testing.assert_frame_equal(home_win[columns], away_win[columns])
        assert home_win.iloc[0]["outcome"] == 0
        assert away_win.iloc[0]["outcome"] == 2


class TestSameDayIsolation:
    def test_same_day_matches_do_not_feed_each_other(self) -> None:
        """The date column cannot order two matches on one day, so neither may
        appear in the other's history."""
        rows = [
            make_match("2023-01-01", "A", "B", 3, 0),
            make_match("2023-01-01", "A", "C", 3, 0),
        ]
        features = build_features(frame(rows))

        assert (features["home_matches_played"] == 0).all()
        assert features["home_ppg_5"].isna().all()

    def test_history_arrives_the_next_day(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "B", 3, 0),
            make_match("2023-01-02", "A", "C", 1, 1),
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)

        assert features.loc[0, "home_matches_played"] == 0
        assert features.loc[1, "home_matches_played"] == 1
        assert features.loc[1, "home_ppg_5"] == pytest.approx(3.0)


class TestFormIsComputedCorrectly:
    def test_points_per_game_uses_only_prior_results(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "X", 1, 0),  # A wins  -> 3
            make_match("2023-01-08", "A", "Y", 0, 0),  # A draws -> 1
            make_match("2023-01-15", "A", "Z", 0, 1),  # A loses -> 0
            make_match("2023-01-22", "A", "W", 0, 0),  # features here
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)

        assert features.loc[3, "home_matches_played"] == 3
        assert features.loc[3, "home_ppg_5"] == pytest.approx((3 + 1 + 0) / 3)

    def test_venue_form_separates_home_from_away(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "X", 3, 0),  # A at home: win
            make_match("2023-01-08", "Y", "A", 3, 0),  # A away:    loss
            make_match("2023-01-15", "A", "Z", 0, 0),  # features here
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)
        row = features.loc[2]

        assert row["home_venue_matches"] == 1
        assert row["home_venue_ppg"] == pytest.approx(3.0)
        assert row["home_ppg_5"] == pytest.approx(1.5)

    def test_rest_days_measures_the_gap(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "X", 1, 0),
            make_match("2023-01-11", "A", "Y", 1, 0),
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)
        assert features.loc[1, "home_rest_days"] == pytest.approx(10.0)

    def test_rolling_stats_come_from_previous_matches(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "X", 1, 0, stats=BIG_STATS),
            make_match("2023-01-08", "A", "Y", 1, 0),
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)

        assert np.isnan(features.loc[0, "home_possession_avg"])
        assert features.loc[1, "home_possession_avg"] == pytest.approx(90.0)
        assert features.loc[1, "home_shots_avg"] == pytest.approx(40.0)


class TestParsers:
    def test_parses_votes(self) -> None:
        raw = "[['Aris Limassol', 'Draw', 'AEK'], ['45', '61', '26']]"
        assert parse_votes(raw) == (45, 61, 26)

    @pytest.mark.parametrize("raw", ["", "[]", None, "not a list", "[[1,2,3]]"])
    def test_rejects_bad_votes(self, raw: object) -> None:
        assert parse_votes(raw) is None

    def test_parses_stats(self) -> None:
        parsed = parse_stats(BIG_STATS)
        assert parsed["possession"] == (90.0, 10.0)
        assert parsed["shots"] == (40.0, 1.0)
        assert parsed["fouls"] == (1.0, 30.0)

    @pytest.mark.parametrize("raw", ["", "{}", None, "[]"])
    def test_rejects_bad_stats(self, raw: object) -> None:
        assert parse_stats(raw) == {}

    def test_does_not_execute_the_string(self) -> None:
        """The old train_func.py ran eval() over scraped page content."""
        assert parse_votes("__import__('os').getcwd()") is None
        assert parse_stats("__import__('os').getcwd()") == {}


class TestVotesToggle:
    def test_votes_can_be_excluded(self) -> None:
        rows = [make_match("2023-01-01", "A", "B", 1, 0, votes="[['A','Draw','B'],['9','1','2']]")]

        with_votes = build_features(frame(rows), include_votes=True)
        without_votes = build_features(frame(rows), include_votes=False)

        assert "votes_home_share" in with_votes.columns
        assert not any(c.startswith("votes_") for c in without_votes.columns)
