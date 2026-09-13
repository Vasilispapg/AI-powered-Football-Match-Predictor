"""Tests for regime routing and the specialist models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from football.ensemble import Specialist, build_specialists, route, score
from football.train import numeric_pipeline


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "odds_home_prob": np.where(rng.random(n) < 0.5, rng.uniform(0.2, 0.6, n), np.nan),
            "odds_draw_prob": rng.uniform(0.2, 0.3, n),
            "elo_diff": rng.normal(0, 100, n),
            "home_ppg_5": rng.uniform(0, 3, n),
        }
    )
    frame.loc[frame["odds_home_prob"].isna(), "odds_draw_prob"] = np.nan
    return frame, rng.integers(0, 3, n)


class TestApplies:
    def test_no_requirement_means_every_row(self, data) -> None:
        frame, _ = data
        specialist = Specialist("s", ["elo_diff"], numeric_pipeline(LogisticRegression()))
        assert specialist.applies(frame).all()

    def test_requirement_selects_rows_with_the_column_populated(self, data) -> None:
        frame, _ = data
        specialist = Specialist(
            "m", ["odds_home_prob"], numeric_pipeline(LogisticRegression()), ["odds_home_prob"]
        )
        applies = specialist.applies(frame)

        assert applies.sum() == int(frame["odds_home_prob"].notna().sum())
        assert not applies.all()

    def test_missing_column_means_no_rows(self, data) -> None:
        frame, _ = data
        specialist = Specialist(
            "x", ["elo_diff"], numeric_pipeline(LogisticRegression()), ["not_a_column"]
        )
        assert not specialist.applies(frame).any()


class TestSpecialistPredictions:
    def test_declines_where_it_does_not_apply(self, data) -> None:
        frame, y = data
        specialist = Specialist(
            "m",
            ["odds_home_prob", "odds_draw_prob"],
            numeric_pipeline(LogisticRegression(max_iter=1000)),
            ["odds_home_prob"],
        )
        specialist.fit(frame, y)
        probabilities = specialist.predict_proba(frame)

        missing = frame["odds_home_prob"].isna().to_numpy()
        assert np.isnan(probabilities[missing]).all()
        assert not np.isnan(probabilities[~missing]).any()

    def test_predictions_are_normalised(self, data) -> None:
        frame, y = data
        specialist = Specialist(
            "s", ["elo_diff", "home_ppg_5"], numeric_pipeline(LogisticRegression(max_iter=1000))
        )
        specialist.fit(frame, y)
        probabilities = specialist.predict_proba(frame)

        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_refuses_to_fit_on_too_few_rows(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        specialist = Specialist("s", ["a"], numeric_pipeline(LogisticRegression()))

        assert specialist.fit(frame, np.array([0, 1, 2])) is None
        assert np.isnan(specialist.predict_proba(frame)).all()

    def test_always_emits_three_columns(self, data) -> None:
        """Even if a training fold never saw a draw, the output stays (n, 3)."""
        frame, _ = data
        y = np.random.default_rng(1).choice([0, 2], len(frame))
        specialist = Specialist(
            "s", ["elo_diff"], numeric_pipeline(LogisticRegression(max_iter=1000))
        )
        specialist.fit(frame, y)

        assert specialist.predict_proba(frame).shape == (len(frame), 3)


class TestRouting:
    def test_prefers_the_first_specialist_that_applies(self) -> None:
        market = np.array([[0.7, 0.2, 0.1], [np.nan] * 3])
        strength = np.array([[0.4, 0.3, 0.3], [0.5, 0.25, 0.25]])

        routed = route({"market": market, "strength": strength}, ["market", "strength"], "strength")

        assert routed[0] == pytest.approx([0.7, 0.2, 0.1])  # market wins where present
        assert routed[1] == pytest.approx([0.5, 0.25, 0.25])  # falls back

    def test_falls_back_entirely_when_nothing_applies(self) -> None:
        market = np.full((3, 3), np.nan)
        strength = np.tile([0.4, 0.3, 0.3], (3, 1))

        routed = route({"market": market, "strength": strength}, ["market", "strength"], "strength")
        assert np.allclose(routed, strength)

    def test_router_never_returns_nan_when_the_fallback_is_complete(self) -> None:
        rng = np.random.default_rng(2)
        market = rng.random((50, 3))
        market[rng.random(50) < 0.6] = np.nan
        strength = rng.dirichlet([1, 1, 1], 50)

        routed = route({"market": market, "strength": strength}, ["market", "strength"], "strength")
        assert not np.isnan(routed).any()


class TestScoring:
    def test_ignores_rows_a_specialist_declined(self) -> None:
        y = np.array([0, 1, 2, 0])
        probabilities = np.array([[0.8, 0.1, 0.1], [np.nan] * 3, [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
        result = score("m", y, probabilities)

        assert result.n == 3
        assert 0.0 <= result.accuracy <= 1.0

    def test_confident_and_right_scores_well(self) -> None:
        y = np.zeros(10, dtype=int)
        probabilities = np.tile([0.9, 0.05, 0.05], (10, 1))

        result = score("m", y, probabilities)
        assert result.accuracy == 1.0
        assert result.log_loss < 0.2


class TestSpecialistConstruction:
    def test_splits_odds_from_strength_features(self) -> None:
        columns = ["odds_home_prob", "odds_draw_prob", "elo_diff", "home_ppg_5", "mv_home"]
        specialists = {s.name: s for s in build_specialists(columns)}

        assert set(specialists["market"].columns) == {"odds_home_prob", "odds_draw_prob"}
        assert "odds_home_prob" not in specialists["strength"].columns
        assert "elo_diff" in specialists["strength"].columns

    def test_strength_excludes_opening_and_closing_prices_too(self) -> None:
        columns = ["open_home_prob", "close_home_prob", "elo_diff"]
        specialists = {s.name: s for s in build_specialists(columns)}

        assert specialists["strength"].columns == ["elo_diff"]

    def test_market_specialist_is_absent_without_odds(self) -> None:
        specialists = build_specialists(["elo_diff", "home_ppg_5"])
        assert [s.name for s in specialists] == ["strength"]

    def test_market_specialist_is_tried_first(self) -> None:
        specialists = build_specialists(["odds_home_prob", "elo_diff"])
        assert specialists[0].name == "market"
