"""Tests for walk-forward validation and the evaluation metrics.

The split is the part that most easily goes wrong without anyone noticing: a
leaky split does not raise, it just reports a flattering number. The old
``train_seq.py`` used a random split on time-series data and nothing complained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.train import (
    CLASSES,
    AlwaysHome,
    PriorBaseline,
    Result,
    evaluate,
    multiclass_brier,
    plot_reliability,
    walk_forward_splits,
)


def make_result(y_true: np.ndarray, probabilities: np.ndarray) -> Result:
    return Result(
        name="test",
        log_loss=0.0,
        brier=0.0,
        accuracy=0.0,
        n=len(y_true),
        folds=1,
        y_true=y_true,
        probabilities=probabilities,
    )


@pytest.fixture
def dates() -> pd.Series:
    return pd.Series(pd.date_range("2023-01-01", periods=400, freq="D"))


class TestWalkForwardSplits:
    def test_training_data_always_precedes_the_test_window(self, dates) -> None:
        """The property the whole scheme rests on."""
        splits = walk_forward_splits(dates, warmup_days=30, window_days=14)
        assert splits

        for train_index, test_index, window_start in splits:
            assert dates.iloc[train_index].max() < window_start
            assert dates.iloc[test_index].min() >= window_start

    def test_train_and_test_never_overlap(self, dates) -> None:
        for train_index, test_index, _ in walk_forward_splits(
            dates, warmup_days=30, window_days=14
        ):
            assert not set(train_index) & set(test_index)

    def test_windows_are_chronological_and_disjoint(self, dates) -> None:
        splits = walk_forward_splits(dates, warmup_days=30, window_days=14)
        starts = [window_start for _, _, window_start in splits]
        assert starts == sorted(starts)

        seen: set[int] = set()
        for _, test_index, _ in splits:
            assert not seen & set(test_index)
            seen |= set(test_index)

    def test_training_set_grows(self, dates) -> None:
        sizes = [
            len(train_index)
            for train_index, _, _ in walk_forward_splits(dates, warmup_days=30, window_days=14)
        ]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_warmup_is_respected(self, dates) -> None:
        splits = walk_forward_splits(dates, warmup_days=100, window_days=14)
        first_window = splits[0][2]
        assert (first_window - dates.min()).days >= 100

    def test_eval_end_truncates(self, dates) -> None:
        cutoff = pd.Timestamp("2023-06-30")
        splits = walk_forward_splits(dates, warmup_days=30, window_days=14, eval_end=cutoff)
        for _, test_index, _ in splits:
            assert dates.iloc[test_index].max() <= cutoff

    def test_no_folds_when_history_is_too_short(self) -> None:
        short = pd.Series(pd.date_range("2023-01-01", periods=5, freq="D"))
        assert walk_forward_splits(short, warmup_days=120, window_days=14) == []


class TestMetrics:
    def test_brier_is_zero_for_perfect_confident_predictions(self) -> None:
        y = np.array([0, 1, 2])
        probabilities = np.eye(3)
        assert multiclass_brier(y, probabilities) == pytest.approx(0.0)

    def test_brier_is_two_for_confidently_wrong_predictions(self) -> None:
        y = np.array([0])
        probabilities = np.array([[0.0, 0.0, 1.0]])
        assert multiclass_brier(y, probabilities) == pytest.approx(2.0)

    def test_brier_rewards_honest_uncertainty(self) -> None:
        y = np.array([0, 1, 2])
        hedged = np.full((3, 3), 1 / 3)
        confident_wrong = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
        assert multiclass_brier(y, hedged) < multiclass_brier(y, confident_wrong)


class TestBaselines:
    def test_prior_baseline_returns_training_frequencies(self) -> None:
        y = np.array([0] * 50 + [1] * 30 + [2] * 20)
        model = PriorBaseline().fit(pd.DataFrame({"x": range(100)}), y)

        probabilities = model.predict_proba(pd.DataFrame({"x": range(4)}))
        assert probabilities.shape == (4, 3)
        assert probabilities[0] == pytest.approx([0.5, 0.3, 0.2])
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_always_home_predicts_class_zero(self) -> None:
        X = pd.DataFrame({"x": range(5)})
        model = AlwaysHome().fit(X, np.array([0, 1, 2, 0, 1]))
        assert (model.predict(X) == 0).all()
        assert model.predict_proba(X)[:, 0].min() > 0.99

    def test_prior_baseline_covers_every_class(self) -> None:
        """Even a fold whose training set is missing a class must emit 3 columns."""
        model = PriorBaseline().fit(pd.DataFrame({"x": [1, 2]}), np.array([0, 0]))
        assert model.predict_proba(pd.DataFrame({"x": [1]})).shape[1] == len(CLASSES)


class TestCalibration:
    def test_honest_model_has_near_zero_error(self) -> None:
        """A model that says 70% and is right 70% of the time is calibrated,
        even though it is wrong 30% of the time."""
        rng = np.random.default_rng(0)
        n = 20_000
        confidence = 0.7
        correct = rng.random(n) < confidence

        probabilities = np.zeros((n, 3))
        y_true = np.zeros(n, dtype=int)
        probabilities[:, 0] = confidence
        probabilities[:, 1] = (1 - confidence) / 2
        probabilities[:, 2] = (1 - confidence) / 2
        y_true[~correct] = 2

        assert make_result(y_true, probabilities).expected_calibration_error() < 0.02

    def test_overconfident_model_is_penalised(self) -> None:
        n = 1_000
        probabilities = np.tile([0.99, 0.005, 0.005], (n, 1))
        y_true = np.full(n, 2)  # always wrong, always certain

        assert make_result(y_true, probabilities).expected_calibration_error() > 0.9

    def test_missing_predictions_give_nan(self) -> None:
        result = Result(name="x", log_loss=0, brier=0, accuracy=0, n=0, folds=0)
        assert np.isnan(result.expected_calibration_error())

    def test_writes_a_reliability_diagram(self, tmp_path) -> None:
        rng = np.random.default_rng(1)
        n = 3_000
        raw = rng.random((n, 3)) + 0.1
        probabilities = raw / raw.sum(axis=1, keepdims=True)
        y_true = np.array([rng.choice(3, p=row) for row in probabilities])

        destination = plot_reliability(
            make_result(y_true, probabilities), tmp_path / "reliability.png"
        )

        assert destination.exists()
        assert destination.stat().st_size > 5_000

    def test_plot_requires_predictions(self, tmp_path) -> None:
        result = Result(name="x", log_loss=0, brier=0, accuracy=0, n=0, folds=0)
        with pytest.raises(ValueError, match="no pooled predictions"):
            plot_reliability(result, tmp_path / "x.png")


class TestEvaluate:
    def test_pools_out_of_sample_predictions_only(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        frame = pd.DataFrame({"x": rng.normal(size=n)})
        y = rng.integers(0, 3, size=n)
        dates = pd.Series(pd.date_range("2023-01-01", periods=n, freq="D"))

        splits = walk_forward_splits(dates, warmup_days=60, window_days=30)
        result = evaluate("prior", PriorBaseline(), frame, y, splits)

        assert result.folds == len(splits)
        assert result.n == sum(len(test_index) for _, test_index, _ in splits)
        assert result.n < n  # the warm-up period is never tested on
        assert 0.0 <= result.accuracy <= 1.0
        assert result.log_loss > 0
