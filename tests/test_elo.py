"""Tests for Elo ratings and the leak-free hyperparameter search."""

from __future__ import annotations

import pandas as pd
import pytest

from football.features import (
    ELO_HOME_ADVANTAGE,
    ELO_START,
    TeamHistories,
    build_features,
    elo_expected,
    elo_update,
)
from football.train import walk_forward_splits
from football.tune import development_slice, load_params, save_params
from tests.test_no_leakage import frame, make_match


class TestEloExpected:
    def test_equal_ratings_favour_the_home_side(self) -> None:
        assert elo_expected(1500, 1500) > 0.5

    def test_no_home_advantage_makes_equal_ratings_even(self) -> None:
        assert elo_expected(1500, 1500, home_advantage=0) == pytest.approx(0.5)

    def test_stronger_team_expects_more(self) -> None:
        assert elo_expected(1700, 1500) > elo_expected(1500, 1700)

    def test_bounded_between_zero_and_one(self) -> None:
        for home, away in ((3000, 1000), (1000, 3000), (1500, 1500)):
            assert 0.0 < elo_expected(home, away) < 1.0

    def test_home_advantage_is_worth_its_rating_points(self) -> None:
        """A home side rated 60 lower than its opponent should be near even."""
        assert elo_expected(1440, 1500, home_advantage=ELO_HOME_ADVANTAGE) == pytest.approx(
            0.5, abs=0.01
        )


class TestEloUpdate:
    def test_is_zero_sum(self) -> None:
        home, away = elo_update(1500, 1500, 3, 0)
        assert (home - 1500) == pytest.approx(-(away - 1500))

    def test_winner_gains_and_loser_loses(self) -> None:
        home, away = elo_update(1500, 1500, 2, 1)
        assert home > 1500
        assert away < 1500

    def test_a_draw_between_equals_barely_moves(self) -> None:
        home, away = elo_update(1500, 1500, 1, 1)
        # The home side was expected to win, so a draw costs it slightly.
        assert home < 1500
        assert away > 1500

    def test_beating_a_stronger_team_gains_more(self) -> None:
        upset, _ = elo_update(1400, 1700, 1, 0)
        expected_win, _ = elo_update(1700, 1400, 1, 0)
        assert (upset - 1400) > (expected_win - 1700)

    def test_margin_of_victory_matters(self) -> None:
        narrow, _ = elo_update(1500, 1500, 1, 0)
        thrashing, _ = elo_update(1500, 1500, 5, 0)
        assert thrashing > narrow


class TestEloInHistories:
    def test_unseen_team_starts_at_the_default(self) -> None:
        assert TeamHistories().elo("Nobody") == ELO_START

    def test_rating_moves_after_a_day_is_folded_in(self) -> None:
        histories = TeamHistories()
        day = frame([make_match("2023-01-01", "A", "B", 3, 0)])
        histories.add_day(pd.Timestamp("2023-01-01"), day)

        assert histories.elo("A") > ELO_START
        assert histories.elo("B") < ELO_START

    def test_same_day_double_header_keeps_both_results(self) -> None:
        """A team playing twice in one day must be rated for both matches.

        Assigning instead of accumulating would silently discard the first.
        """
        histories = TeamHistories()
        day = frame(
            [
                make_match("2023-01-01", "A", "B", 3, 0),
                make_match("2023-01-01", "A", "C", 3, 0),
            ]
        )
        histories.add_day(pd.Timestamp("2023-01-01"), day)

        one_win = TeamHistories()
        one_win.add_day(
            pd.Timestamp("2023-01-01"), frame([make_match("2023-01-01", "A", "B", 3, 0)])
        )

        assert histories.elo("A") > one_win.elo("A")

    def test_ratings_stay_zero_sum_across_a_season(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "B", 2, 0),
            make_match("2023-01-08", "B", "C", 1, 1),
            make_match("2023-01-15", "C", "A", 0, 3),
        ]
        histories = TeamHistories()
        for date, day in frame(rows).groupby("date"):
            histories.add_day(pd.Timestamp(date), day)

        total = sum(histories.elo(team) for team in ("A", "B", "C"))
        assert total == pytest.approx(3 * ELO_START)


class TestEloDoesNotLeak:
    def test_a_teams_first_match_is_rated_at_the_default(self) -> None:
        features = build_features(frame([make_match("2023-01-01", "A", "B", 5, 0)]))
        row = features.iloc[0]

        assert row["home_elo"] == ELO_START
        assert row["away_elo"] == ELO_START
        assert row["elo_diff"] == 0.0

    def test_the_result_does_not_reach_its_own_elo_feature(self) -> None:
        home_win = build_features(frame([make_match("2023-01-01", "A", "B", 5, 0)]))
        away_win = build_features(frame([make_match("2023-01-01", "A", "B", 0, 5)]))

        assert home_win.iloc[0]["elo_diff"] == away_win.iloc[0]["elo_diff"]

    def test_rating_reflects_earlier_matches_only(self) -> None:
        rows = [
            make_match("2023-01-01", "A", "B", 4, 0),
            make_match("2023-01-08", "A", "C", 0, 0),
        ]
        features = build_features(frame(rows)).sort_values("date").reset_index(drop=True)

        assert features.loc[0, "home_elo"] == ELO_START
        assert features.loc[1, "home_elo"] > ELO_START


class TestTuningSliceIsIsolated:
    def test_development_slice_excludes_every_evaluation_fold(self) -> None:
        """The property that keeps the walk-forward number honest.

        If the search could score on a match a later fold tests, the reported
        result would no longer be out-of-sample -- and nothing would raise.
        """
        rows = [
            make_match(
                (pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                "A",
                "B",
                day % 3,
                (day + 1) % 3,
            )
            for day in range(0, 400, 2)
        ]
        features = build_features(frame(rows))

        development = development_slice(features)
        splits = walk_forward_splits(features["date"])

        tested = {index for _, test_index, _ in splits for index in test_index}
        development_dates = set(pd.to_datetime(development["date"]))
        tested_dates = set(pd.to_datetime(features.iloc[sorted(tested)]["date"]))

        assert development_dates & tested_dates == set()

    def test_params_round_trip(self, tmp_path) -> None:
        path = tmp_path / "tuned.json"
        params = {"learning_rate": 0.03, "max_leaf_nodes": 15}

        save_params(params, path)
        assert load_params(path) == params

    def test_missing_params_file_is_none(self, tmp_path) -> None:
        assert load_params(tmp_path / "absent.json") is None

    def test_corrupt_params_file_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_params(path) is None
