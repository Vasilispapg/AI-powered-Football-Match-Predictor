"""Tests for the forecast-and-grade loop.

The property that matters here is ordering: a forecast must be written before
the matches are played and graded afterwards, from the file. Anything that lets
results influence the recorded prediction turns the whole exercise into the
thing tipster sites do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.forecast import STRONG, forecast_fixtures, load_forecast, save_forecast
from football.goals import GOAL_LINES


@pytest.fixture
def history() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    teams = ["A", "B", "C", "D", "E", "F"]
    rows = []
    day = pd.Timestamp("2025-01-01")
    for week in range(40):
        order = list(teams)
        rng.shuffle(order)
        for home, away in zip(order[::2], order[1::2], strict=True):
            rows.append(
                {
                    "date": day + pd.Timedelta(days=7 * week),
                    "competition": "Test League",
                    "country": "Testland",
                    "home_team": home,
                    "away_team": away,
                    "home_score": int(rng.integers(0, 4)),
                    "away_score": int(rng.integers(0, 3)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "time": "18:00",
                "competition": "Test League",
                "competition_type": "league",
                "home_team": "A",
                "away_team": "B",
                "odds_home_prob": 0.50,
                "odds_draw_prob": 0.27,
                "odds_away_prob": 0.23,
                "odds_over25_prob": 0.55,
                "odds_handicap": -0.5,
            },
            {
                "date": "2026-01-10",
                "time": "20:00",
                "competition": "Test League",
                "competition_type": "league",
                "home_team": "C",
                "away_team": "D",
                "odds_home_prob": 0.25,
                "odds_draw_prob": 0.25,
                "odds_away_prob": 0.50,
                "odds_over25_prob": 0.48,
                "odds_handicap": 0.25,
            },
        ]
    )


class TestForecast:
    def test_prices_every_fixture(self, fixtures, history) -> None:
        predictions = forecast_fixtures(fixtures, history)
        assert len(predictions) == 2

    def test_outcome_probabilities_sum_to_one(self, fixtures, history) -> None:
        predictions = forecast_fixtures(fixtures, history)
        totals = predictions[["p_home", "p_draw", "p_away"]].sum(axis=1)
        assert np.allclose(totals, 1.0, atol=0.01)

    def test_pick_matches_the_highest_probability(self, fixtures, history) -> None:
        for _, row in forecast_fixtures(fixtures, history).iterrows():
            best = max(
                ("1", row["p_home"]),
                ("X", row["p_draw"]),
                ("2", row["p_away"]),
                key=lambda kv: kv[1],
            )
            assert row["pick_1x2"] == best[0]
            assert row["conf_1x2"] == pytest.approx(best[1], abs=1e-4)

    def test_follows_the_market_it_was_given(self, fixtures, history) -> None:
        """A fixture priced as a home favourite must not come back as an away pick."""
        predictions = forecast_fixtures(fixtures, history)
        home_favourite = predictions.iloc[0]
        away_favourite = predictions.iloc[1]

        assert home_favourite["p_home"] > home_favourite["p_away"]
        assert away_favourite["p_away"] > away_favourite["p_home"]

    def test_every_totals_line_is_priced(self, fixtures, history) -> None:
        predictions = forecast_fixtures(fixtures, history)
        for line in GOAL_LINES:
            assert f"p_over_{line}" in predictions.columns

    def test_totals_decrease_as_the_line_rises(self, fixtures, history) -> None:
        """P(over 0.5) must exceed P(over 1.5), and so on -- they are nested."""
        for _, row in forecast_fixtures(fixtures, history).iterrows():
            values = [row[f"p_over_{line}"] for line in GOAL_LINES]
            assert values == sorted(values, reverse=True)

    def test_three_distinct_scorelines_are_offered(self, fixtures, history) -> None:
        for _, row in forecast_fixtures(fixtures, history).iterrows():
            assert len({row["score_1"], row["score_2"], row["score_3"]}) == 3
            assert row["p_score_1"] >= row["p_score_2"] >= row["p_score_3"]

    def test_skips_fixtures_without_a_market(self, fixtures, history) -> None:
        fixtures.loc[0, "odds_over25_prob"] = np.nan
        assert len(forecast_fixtures(fixtures, history)) == 1

    def test_only_uses_matches_before_the_fixture_date(self, fixtures, history) -> None:
        """Adding results after the fixture date must not change the forecast."""
        before = forecast_fixtures(fixtures, history)

        future = history.copy()
        future["date"] = future["date"] + pd.Timedelta(days=500)
        after = forecast_fixtures(fixtures, pd.concat([history, future], ignore_index=True))

        pd.testing.assert_frame_equal(before, after)


class TestSaving:
    def test_records_when_the_forecast_was_made(
        self, fixtures, history, tmp_path, monkeypatch
    ) -> None:
        """A prediction with no timestamp cannot be shown to predate the result."""
        import football.forecast as module

        monkeypatch.setattr(module, "FORECAST_DIR", tmp_path)
        predictions = forecast_fixtures(fixtures, history)

        destination = save_forecast(predictions, "2026-01-10")
        saved = pd.read_csv(destination)

        assert destination.name == "2026-01-10.csv"
        assert "forecast_made_at" in saved.columns
        assert saved["forecast_made_at"].notna().all()
        assert len(saved) == len(predictions)

    def test_saved_picks_survive_the_round_trip(
        self, fixtures, history, tmp_path, monkeypatch
    ) -> None:
        import football.forecast as module

        monkeypatch.setattr(module, "FORECAST_DIR", tmp_path)
        predictions = forecast_fixtures(fixtures, history)
        saved = load_forecast(save_forecast(predictions, "2026-01-10"))

        assert list(saved["pick_1x2"]) == list(predictions["pick_1x2"])
        assert list(saved["score_1"]) == list(predictions["score_1"])

    def test_picks_survive_a_day_with_no_draw(
        self, fixtures, history, tmp_path, monkeypatch
    ) -> None:
        """Regression: with picks of only "1" and "2", a plain read_csv returns
        integers and every pick then grades as wrong -- 0% instead of 51%."""
        import football.forecast as module

        monkeypatch.setattr(module, "FORECAST_DIR", tmp_path)
        predictions = forecast_fixtures(fixtures, history)
        assert "X" not in set(predictions["pick_1x2"])  # the conditions for the bug

        saved = load_forecast(save_forecast(predictions, "2026-01-10"))
        assert all(isinstance(pick, str) for pick in saved["pick_1x2"])


class TestConfidenceThreshold:
    def test_threshold_matches_the_measured_study(self) -> None:
        """0.60 is where the hit-rate study showed 1X2 rising to 71.6%."""
        assert STRONG == 0.60
