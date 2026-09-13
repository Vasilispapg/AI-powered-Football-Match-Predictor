"""Tests for the prediction path.

The critical one is the round trip: a fixture featurised through the prediction
path must be identical to the same fixture featurised during training. When
those two drift apart -- train/serve skew -- a model that evaluated well starts
returning nonsense in production, and nothing raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.features import build_features, feature_columns, fixture_features
from football.predict import Prediction, build_histories, predict_fixture
from football.train import CLASS_NAMES
from tests.test_no_leakage import frame, make_match


@pytest.fixture
def season() -> pd.DataFrame:
    """A synthetic league, big enough to fit a real estimator on.

    Scores come from a seeded RNG so every outcome class is represented and the
    fixture is reproducible.
    """
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D", "E", "F"]
    values = {team: 50.0 + 40 * index for index, team in enumerate(teams)}

    rows = []
    day = pd.Timestamp("2023-01-01")
    for week in range(30):
        order = list(teams)
        rng.shuffle(order)
        for home, away in zip(order[::2], order[1::2], strict=True):
            # Two thirds of matches carry statistics, roughly matching the real
            # dataset, so the rolling-stat features are exercised rather than
            # being uniformly NaN.
            stats = "{}"
            if rng.random() < 0.67:
                possession = round(float(rng.uniform(35, 65)), 1)
                stats = (
                    f"{{'Ball possession': ['{possession}%', "
                    f"'{round(100 - possession, 1)}%'], "
                    f"'Total shots': ['{rng.integers(4, 22)}', '{rng.integers(4, 22)}'], "
                    f"'Corners': ['{rng.integers(0, 12)}', '{rng.integers(0, 12)}'], "
                    f"'Fouls': ['{rng.integers(5, 22)}', '{rng.integers(5, 22)}']}}"
                )
            rows.append(
                make_match(
                    (day + pd.Timedelta(days=7 * week)).strftime("%Y-%m-%d"),
                    home,
                    away,
                    int(rng.integers(0, 4)),
                    int(rng.integers(0, 4)),
                    stats=stats,
                    mv_home=values[home],
                    mv_away=values[away],
                )
            )
    return frame(rows)


TRAIN_END = "2023-07-23"  # the last match day in the `season` fixture


def make_bundle(season: pd.DataFrame, include_votes: bool = False) -> dict:
    """A model bundle in the same shape ``football.train`` saves.

    Uses the logistic pipeline rather than gradient boosting: the point is to
    exercise the prediction path, and the boosting binner needs more rows than
    a synthetic fixture should have to carry.
    """
    from sklearn.linear_model import LogisticRegression

    from football.train import numeric_pipeline

    features = build_features(season, include_votes=include_votes)
    columns = feature_columns(features)
    model = numeric_pipeline(LogisticRegression(max_iter=1000))
    model.fit(features[columns], features["outcome"])
    return {
        "model": model,
        "feature_columns": columns,
        "classes": (0, 1, 2),
        "class_names": CLASS_NAMES,
        "trained_rows": len(features),
        "train_start": "2023-01-01",
        "train_end": TRAIN_END,
    }


class TestRoundTrip:
    def test_prediction_features_match_training_features(self, season) -> None:
        """The check that catches train/serve skew."""
        training = build_features(season, include_votes=False)

        target = training.iloc[-1]
        date = pd.Timestamp(target["date"])

        # build_features re-sorts within a day, so find the source row by key
        # rather than trusting positional alignment.
        source = season[
            (pd.to_datetime(season["date"]) == date)
            & (season["home_team"] == target["home_team"])
            & (season["away_team"] == target["away_team"])
        ].iloc[0]

        histories = build_histories(season, before=date)
        rebuilt = fixture_features(
            histories,
            {
                "home_team": target["home_team"],
                "away_team": target["away_team"],
                "competition_type": "league",
                "mv_home": source["mv_home"],
                "mv_away": source["mv_away"],
                "votes": "[]",
            },
            date,
            include_votes=False,
        )

        for column in feature_columns(training):
            expected = target[column]
            actual = rebuilt[column]
            if pd.isna(expected):
                assert pd.isna(actual), f"{column}: training NaN, prediction {actual}"
            else:
                assert actual == pytest.approx(expected), column

    def test_history_excludes_the_fixture_date_itself(self, season) -> None:
        date = pd.Timestamp("2023-02-05")
        histories = build_histories(season, before=date)

        replayed = season[pd.to_datetime(season["date"]) < date]
        assert len(histories) == len(set(replayed["home_team"]) | set(replayed["away_team"]))

    def test_empty_history_before_the_first_match(self, season) -> None:
        histories = build_histories(season, before=pd.Timestamp("2022-01-01"))
        assert len(histories) == 0


class TestPredictFixture:
    def _bundle(self, season, include_votes: bool = False) -> dict:
        return make_bundle(season, include_votes)

    def test_returns_three_probabilities_summing_to_one(self, season) -> None:
        prediction = predict_fixture(
            "A",
            "B",
            "2023-07-30",
            bundle=self._bundle(season),
            matches=season,
            market_values={"A": 120.0, "B": 110.0},
        )

        assert isinstance(prediction, Prediction)
        assert prediction.probabilities.shape == (3,)
        assert prediction.probabilities.sum() == pytest.approx(1.0)
        assert (prediction.probabilities >= 0).all()
        assert prediction.most_likely in CLASS_NAMES

    def test_warns_about_unknown_teams(self, season) -> None:
        prediction = predict_fixture(
            "Nonexistent United",
            "B",
            "2023-07-30",
            bundle=self._bundle(season),
            matches=season,
            market_values={},
        )
        joined = " ".join(prediction.warnings)
        assert "no match history" in joined
        assert "Nonexistent United" in joined

    def test_warns_when_the_fixture_is_far_past_training(self, season) -> None:
        prediction = predict_fixture(
            "A",
            "B",
            "2024-06-01",
            bundle=self._bundle(season),
            matches=season,
            market_values={"A": 120.0, "B": 110.0},
        )
        assert any("past the training data" in w for w in prediction.warnings)

    def test_warns_when_votes_are_missing_from_a_vote_model(self, season) -> None:
        prediction = predict_fixture(
            "A",
            "B",
            "2023-07-30",
            bundle=self._bundle(season, include_votes=True),
            matches=season,
            market_values={"A": 120.0, "B": 110.0},
        )
        assert any("crowd votes" in w for w in prediction.warnings)

    def test_accepts_supplied_votes(self, season) -> None:
        prediction = predict_fixture(
            "A",
            "B",
            "2023-07-30",
            votes=(45, 61, 26),
            bundle=self._bundle(season, include_votes=True),
            matches=season,
            market_values={"A": 120.0, "B": 110.0},
        )
        assert prediction.features["votes_total"] == pytest.approx(132.0)
        assert prediction.features["votes_missing"] == 0.0
        assert not any("crowd votes" in w for w in prediction.warnings)

    def test_defaults_to_the_day_after_training_ends(self, season) -> None:
        prediction = predict_fixture(
            "A", "B", None, bundle=self._bundle(season), matches=season, market_values={}
        )
        assert prediction.date == pd.Timestamp("2023-07-24")

    def test_raises_when_the_feature_contract_is_broken(self, season) -> None:
        bundle = self._bundle(season)
        bundle["feature_columns"] = [*bundle["feature_columns"], "a_column_that_left"]

        with pytest.raises(RuntimeError, match="Feature contract broken"):
            predict_fixture("A", "B", "2023-07-30", bundle=bundle, matches=season, market_values={})

    def test_render_is_printable(self, season) -> None:
        prediction = predict_fixture(
            "A",
            "B",
            "2023-07-30",
            bundle=self._bundle(season),
            matches=season,
            market_values={"A": 120.0, "B": 110.0},
        )
        text = prediction.render()
        assert "A" in text and "B" in text
        for name in CLASS_NAMES:
            assert name in text


class TestMarketValuesReachTheModel:
    def test_values_are_looked_up_per_team(self, season) -> None:
        bundle = make_bundle(season)

        prediction = predict_fixture(
            "A",
            "B",
            "2023-07-30",
            bundle=bundle,
            matches=season,
            market_values={"A": 900.0, "B": 5.0},
        )

        assert prediction.features["mv_home"] == pytest.approx(900.0)
        assert prediction.features["mv_away"] == pytest.approx(5.0)
        assert prediction.features["mv_log_ratio"] > 0
        assert not np.isnan(prediction.features["mv_log_ratio"])
