"""Tests for the opening/closing odds split and the line-movement study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.ablation import bootstrap_difference, family_columns
from football.line_movement import CLOSE_COLUMNS, OPEN_COLUMNS, add_drift, market_probabilities
from football.sources.footballdata import (
    CLOSING_PREFERENCES,
    OPENING_PREFERENCES,
    kickoff_hour,
    pick_handicap,
    pick_odds,
    pick_totals,
)


class TestOpeningClosingSplit:
    def _row(self) -> pd.Series:
        return pd.Series(
            {
                "AvgH": 2.50,
                "AvgD": 3.40,
                "AvgA": 2.90,  # opening
                "AvgCH": 2.00,
                "AvgCD": 3.50,
                "AvgCA": 4.00,  # closing
            }
        )

    def test_closing_preference_takes_closing(self) -> None:
        assert pick_odds(self._row(), CLOSING_PREFERENCES) == (2.0, 3.5, 4.0)

    def test_opening_preference_takes_opening(self) -> None:
        assert pick_odds(self._row(), OPENING_PREFERENCES) == (2.5, 3.4, 2.9)

    def test_the_two_are_genuinely_different(self) -> None:
        """If these ever coincide the whole study is measuring nothing."""
        row = self._row()
        assert pick_odds(row, OPENING_PREFERENCES) != pick_odds(row, CLOSING_PREFERENCES)

    def test_default_prefers_closing(self) -> None:
        assert pick_odds(self._row()) == (2.0, 3.5, 4.0)

    def test_opening_is_none_when_absent(self) -> None:
        row = pd.Series({"AvgCH": 2.0, "AvgCD": 3.5, "AvgCA": 4.0})
        assert pick_odds(row, OPENING_PREFERENCES) is None
        assert pick_odds(row, CLOSING_PREFERENCES) == (2.0, 3.5, 4.0)


class TestSecondaryMarkets:
    def test_over_under_becomes_a_probability(self) -> None:
        probability = pick_totals(pd.Series({"Avg>2.5": 2.0, "Avg<2.5": 2.0}))
        assert probability == pytest.approx(0.5)

    def test_short_over_price_means_goals_expected(self) -> None:
        probability = pick_totals(pd.Series({"Avg>2.5": 1.3, "Avg<2.5": 3.5}))
        assert probability > 0.6

    def test_missing_totals_are_none(self) -> None:
        assert pick_totals(pd.Series({"HomeTeam": "A"})) is None

    def test_handicap_line_is_read(self) -> None:
        assert pick_handicap(pd.Series({"AHCh": -0.75})) == pytest.approx(-0.75)

    def test_absurd_handicap_is_rejected(self) -> None:
        assert pick_handicap(pd.Series({"AHCh": 99.0})) is None

    @pytest.mark.parametrize(
        ("value", "expected"), [("19:45", 19.0), ("12:30", 12.0), ("", None), ("x", None)]
    )
    def test_kickoff_hour(self, value: str, expected: float | None) -> None:
        result = kickoff_hour(value)
        if expected is None:
            assert np.isnan(result)
        else:
            assert result == expected


class TestDrift:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_home_prob": [0.40, 0.50],
                "open_draw_prob": [0.30, 0.25],
                "open_away_prob": [0.30, 0.25],
                "close_home_prob": [0.50, 0.45],
                "close_draw_prob": [0.25, 0.28],
                "close_away_prob": [0.25, 0.27],
            }
        )

    def test_drift_is_closing_minus_opening(self) -> None:
        drifted = add_drift(self._frame())
        assert drifted.loc[0, "drift_home"] == pytest.approx(0.10)
        assert drifted.loc[0, "drift_away"] == pytest.approx(-0.05)

    def test_magnitude_is_always_positive(self) -> None:
        drifted = add_drift(self._frame())
        assert (drifted["drift_magnitude"] >= 0).all()

    def test_no_movement_gives_zero_drift(self) -> None:
        frame = self._frame()
        for outcome in ("home", "draw", "away"):
            frame[f"close_{outcome}_prob"] = frame[f"open_{outcome}_prob"]

        drifted = add_drift(frame)
        assert (drifted["drift_magnitude"] == 0).all()


class TestMarketProbabilities:
    def test_rows_are_normalised(self) -> None:
        frame = pd.DataFrame(
            {"close_home_prob": [0.5], "close_draw_prob": [0.3], "close_away_prob": [0.3]}
        )
        probabilities = market_probabilities(frame, "close")
        assert probabilities.sum(axis=1)[0] == pytest.approx(1.0)

    def test_missing_prices_fall_back_to_uniform(self) -> None:
        frame = pd.DataFrame(
            {
                "close_home_prob": [np.nan],
                "close_draw_prob": [np.nan],
                "close_away_prob": [np.nan],
            }
        )
        assert market_probabilities(frame, "close")[0] == pytest.approx([1 / 3] * 3)

    def test_open_and_close_are_read_separately(self) -> None:
        frame = pd.DataFrame(
            {
                "open_home_prob": [0.4],
                "open_draw_prob": [0.3],
                "open_away_prob": [0.3],
                "close_home_prob": [0.6],
                "close_draw_prob": [0.2],
                "close_away_prob": [0.2],
            }
        )
        assert market_probabilities(frame, "open")[0][0] == pytest.approx(0.4)
        assert market_probabilities(frame, "close")[0][0] == pytest.approx(0.6)


class TestBootstrap:
    def test_identical_models_show_no_difference(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 500)
        probabilities = np.full((500, 3), 1 / 3)

        mean, low, high = bootstrap_difference(y, probabilities, probabilities, iterations=200)
        assert mean == pytest.approx(0.0)
        assert low <= 0 <= high

    def test_a_better_model_gives_a_negative_difference(self) -> None:
        """Negative means the first argument wins."""
        y = np.zeros(500, dtype=int)
        good = np.tile([0.8, 0.1, 0.1], (500, 1))
        bad = np.tile([0.2, 0.4, 0.4], (500, 1))

        mean, _, high = bootstrap_difference(y, good, bad, iterations=200)
        assert mean < 0
        assert high < 0

    def test_interval_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 800)
        a = rng.dirichlet([2, 2, 2], 800)
        b = rng.dirichlet([2, 2, 2], 800)

        mean, low, high = bootstrap_difference(y, a, b, iterations=400)
        assert low <= mean <= high


class TestFamilyColumns:
    def test_odds_family_excludes_open_and_close(self) -> None:
        """The ablation studies the combined odds view; opening and closing are
        the line-movement study's business."""
        columns = ["odds_home_prob", "open_home_prob", "close_home_prob", "elo_diff"]
        assert family_columns(columns, "odds") == ["odds_home_prob"]

    def test_elo_family_is_picked_up(self) -> None:
        columns = ["home_elo", "away_elo", "elo_diff", "elo_expected", "home_ppg_5"]
        assert set(family_columns(columns, "elo")) == {
            "home_elo",
            "away_elo",
            "elo_diff",
            "elo_expected",
        }

    def test_open_close_column_lists_are_disjoint(self) -> None:
        assert not set(OPEN_COLUMNS) & set(CLOSE_COLUMNS)
