"""Train and evaluate outcome models with walk-forward validation.

Three things the old ``train_seq.py`` got wrong, fixed here.

**The split.** It used ``train_test_split(random_state=101)`` on time-series
data, so matches from May trained a model tested on the previous October. Here
every fold trains only on matches that finished before its test window opens.

**The target.** It labelled ``home_score > away_score``, folding draws in with
away wins, while the README promised three outcomes. Draws are 24.8% of this
dataset; they get their own class.

**The metric.** Accuracy alone on a 3-class imbalanced problem hides almost
everything. Football outcomes are close to irreducibly noisy, so what matters is
whether the predicted probabilities are honest: log loss and Brier score lead,
accuracy follows, and every number is reported next to baselines that use no
features at all.

    python -m football.train                  # full run
    python -m football.train --no-votes       # measure what votes are worth
    python -m football.train --eval-end 2023-05-31
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football import paths, provenance
from football.features import build_features, feature_columns
from football.transformers import DropAllNaNColumns

log = logging.getLogger(__name__)

CLASSES = (0, 1, 2)
CLASS_NAMES = ("home win", "draw", "away win")

#: Days at the start of the dataset used only to build team history. No fold
#: tests here -- with a 10.6-month window, testing in week two would mostly
#: measure the cold start.
WARMUP_DAYS = 120

#: Length of each walk-forward test window.
TEST_WINDOW_DAYS = 14


class PriorBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the training set's class frequencies for every match.

    The honest probabilistic floor: any model that cannot beat this has learned
    nothing from its features.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(CLASSES)
        counts = np.array([(y == c).sum() for c in CLASSES], dtype=float)
        self.priors_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.priors_, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), int(np.argmax(self.priors_)))


class AlwaysHome(BaseEstimator, ClassifierMixin):
    """Always predicts a home win. The accuracy bar everyone quotes."""

    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), 3))
        probabilities[:, 0] = 1.0
        # Nudge off the boundary so log loss stays finite and comparable.
        return np.clip(probabilities, 1e-6, 1 - 1e-6)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and the one-hot truth."""
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


@dataclass
class Result:
    name: str
    log_loss: float
    brier: float
    accuracy: float
    n: int
    folds: int
    per_fold_log_loss: list[float] = field(default_factory=list)
    #: Pooled out-of-sample truth and probabilities, kept for calibration.
    y_true: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    def expected_calibration_error(self, bins: int = 10) -> float:
        """Mean gap between predicted confidence and observed accuracy.

        Accuracy says how often the argmax is right; this says whether a stated
        70% actually happens about 70% of the time. For a model whose output is
        a probability, that is the property that matters.
        """
        if self.y_true is None or self.probabilities is None:
            return float("nan")
        confidence = self.probabilities.max(axis=1)
        predicted = self.probabilities.argmax(axis=1)
        correct = (predicted == self.y_true).astype(float)

        edges = np.linspace(0.0, 1.0, bins + 1)
        error = 0.0
        for low, high in itertools.pairwise(edges):
            in_bin = (confidence > low) & (confidence <= high)
            if not in_bin.any():
                continue
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
        return float(error)


def walk_forward_splits(
    dates: pd.Series,
    warmup_days: int = WARMUP_DAYS,
    window_days: int = TEST_WINDOW_DAYS,
    eval_end: pd.Timestamp | None = None,
) -> list[tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    """Yield ``(train_index, test_index, window_start)`` in chronological order.

    Each fold trains on every match strictly before its window and tests inside
    it, so no fold can see its own future.
    """
    dates = pd.to_datetime(dates)
    start = dates.min() + pd.Timedelta(days=warmup_days)
    last = eval_end if eval_end is not None else dates.max()

    splits = []
    window_start = start
    while window_start <= last:
        window_end = window_start + pd.Timedelta(days=window_days)
        train_mask = dates < window_start
        test_mask = (dates >= window_start) & (dates < window_end) & (dates <= last)
        if test_mask.sum() > 0 and train_mask.sum() > 0:
            splits.append((np.where(train_mask)[0], np.where(test_mask)[0], window_start))
        window_start = window_end
    return splits


def evaluate(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray, pd.Timestamp]],
) -> Result:
    """Run *estimator* across every fold and pool the out-of-sample predictions."""
    all_true: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    per_fold: list[float] = []

    for train_index, test_index, _ in splits:
        model = clone_estimator(estimator)
        model.fit(X.iloc[train_index], y[train_index])

        raw = model.predict_proba(X.iloc[test_index])
        # Align to (home, draw, away) even if a fold's training set is missing
        # a class entirely.
        probabilities = np.zeros((len(test_index), 3))
        for position, class_label in enumerate(model.classes_):
            probabilities[:, int(class_label)] = raw[:, position]
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        truth = y[test_index]
        all_true.append(truth)
        all_probabilities.append(probabilities)
        per_fold.append(log_loss(truth, probabilities, labels=list(CLASSES)))

    truth = np.concatenate(all_true)
    probabilities = np.vstack(all_probabilities)

    return Result(
        name=name,
        log_loss=log_loss(truth, probabilities, labels=list(CLASSES)),
        brier=multiclass_brier(truth, probabilities),
        accuracy=accuracy_score(truth, probabilities.argmax(axis=1)),
        n=len(truth),
        folds=len(splits),
        per_fold_log_loss=per_fold,
        y_true=truth,
        probabilities=probabilities,
    )


def plot_reliability(result: Result, destination: Path, bins: int = 10) -> Path:
    """Write a reliability diagram for *result*.

    One panel per outcome. A perfectly calibrated model traces the diagonal:
    among the matches it called 60% home wins, 60% really are home wins. The
    histograms underneath show where the model actually spends its confidence --
    a curve hugging the diagonal means little if every prediction sits in one
    bin.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.y_true is None or result.probabilities is None:
        raise ValueError("Result carries no pooled predictions")

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), height_ratios=[2.2, 1])
    edges = np.linspace(0.0, 1.0, bins + 1)

    for index, class_name in enumerate(CLASS_NAMES):
        predicted = result.probabilities[:, index]
        actual = (result.y_true == index).astype(float)

        centres, observed = [], []
        for low, high in itertools.pairwise(edges):
            in_bin = (predicted > low) & (predicted <= high)
            if in_bin.sum() < 20:
                continue
            centres.append(predicted[in_bin].mean())
            observed.append(actual[in_bin].mean())

        top = axes[0, index]
        top.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        top.plot(centres, observed, "o-", color="#1f77b4", label=result.name)
        top.set_title(class_name)
        top.set_xlabel("predicted probability")
        top.set_ylabel("observed frequency" if index == 0 else "")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.grid(alpha=0.3)
        if index == 0:
            top.legend(loc="upper left", fontsize=8)

        bottom = axes[1, index]
        bottom.hist(predicted, bins=edges, color="#1f77b4", alpha=0.75)
        bottom.set_xlim(0, 1)
        bottom.set_xlabel("predicted probability")
        bottom.set_ylabel("matches" if index == 0 else "")
        bottom.grid(alpha=0.3)

    figure.suptitle(
        f"Reliability -- {result.name}  "
        f"({result.n:,} out-of-sample matches, ECE {result.expected_calibration_error():.3f})"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)
    return destination


def clone_estimator(estimator):
    from sklearn.base import clone

    return clone(estimator)


def numeric_pipeline(model) -> Pipeline:
    """Median imputation + scaling, for models that cannot handle NaN."""
    return Pipeline(
        [
            ("drop_empty", DropAllNaNColumns()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


class ColumnSubset(BaseEstimator):
    """Restrict a pipeline to a few columns, for the single-signal baselines."""

    def __init__(self, columns: list[str], estimator):
        self.columns = columns
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.base import clone

        self.estimator_ = clone(self.estimator)
        available = [c for c in self.columns if c in X.columns]
        self.available_ = available
        self.estimator_.fit(X[available], y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X[self.available_])

    def predict(self, X):
        return self.estimator_.predict(X[self.available_])


def build_candidates(columns: list[str]) -> dict[str, object]:
    """The models to compare, cheapest and most naive first."""
    vote_columns = [c for c in columns if c.startswith("votes_")]
    mv_columns = [c for c in columns if c.startswith("mv_")]
    form_columns = [
        c for c in columns if any(k in c for k in ("ppg", "gf_", "ga_", "rest", "venue"))
    ]

    candidates: dict[str, object] = {
        "baseline: class priors": PriorBaseline(),
        "baseline: always home": AlwaysHome(),
        "market value only": ColumnSubset(
            mv_columns, numeric_pipeline(LogisticRegression(max_iter=2000))
        ),
        "form only": ColumnSubset(
            form_columns, numeric_pipeline(LogisticRegression(max_iter=2000))
        ),
    }
    if vote_columns:
        candidates["votes only"] = ColumnSubset(
            vote_columns, numeric_pipeline(LogisticRegression(max_iter=2000))
        )

    # The betting market on its own. If the full model cannot beat this, every
    # feature in this project is redundant to a number anyone can look up.
    odds_columns = [c for c in columns if c.startswith("odds_")]
    if odds_columns:
        candidates["odds only"] = ColumnSubset(
            odds_columns, numeric_pipeline(LogisticRegression(max_iter=2000))
        )

    elo_columns = [c for c in columns if c.startswith(("elo_", "home_elo", "away_elo"))]
    if elo_columns:
        candidates["elo only"] = ColumnSubset(
            elo_columns, numeric_pipeline(LogisticRegression(max_iter=2000))
        )

    candidates["logistic regression"] = numeric_pipeline(LogisticRegression(max_iter=2000, C=1.0))

    # Hand-picked defaults, overridden by `python -m football.tune` when it has
    # been run. The search only ever sees matches from before the first
    # evaluation fold, so adopting its choice here does not compromise the
    # walk-forward number reported below.
    from football.tune import load_params

    boosting = {
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
    }
    tuned = load_params()
    if tuned:
        boosting.update(tuned)
        log.info("Using tuned hyperparameters: %s", tuned)

    candidates["gradient boosting"] = Pipeline(
        [
            ("drop_empty", DropAllNaNColumns()),
            (
                "model",
                HistGradientBoostingClassifier(
                    **boosting,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=provenance.SEED,
                ),
            ),
        ]
    )
    return candidates


def report(results: list[Result]) -> None:
    print()
    print(
        f"{'model':<24} {'log loss':>9} {'Brier':>8} {'accuracy':>9} {'ECE':>7} "
        f"{'n':>8} {'folds':>6}"
    )
    print("-" * 76)
    for result in sorted(results, key=lambda r: r.log_loss):
        print(
            f"{result.name:<24} {result.log_loss:>9.4f} {result.brier:>8.4f} "
            f"{result.accuracy:>8.1%} {result.expected_calibration_error():>7.3f} "
            f"{result.n:>8,} {result.folds:>6}"
        )

    best = min(results, key=lambda r: r.log_loss)
    prior = next((r for r in results if r.name == "baseline: class priors"), None)
    if prior is not None and best.name != prior.name:
        improvement = (prior.log_loss - best.log_loss) / prior.log_loss
        print()
        print(
            f"Best: {best.name} -- {improvement:.1%} lower log loss than predicting "
            f"the class priors, {best.accuracy:.1%} accuracy."
        )


def fit_final_model(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    dates: pd.Series,
    result: Result | None = None,
) -> Path:
    """Refit on everything and save the model with its feature contract.

    The bundle records the training date range so that a later prediction can
    be checked against it: a model trained to 2023-08-27 has nothing to say
    about a 2025 fixture, and the artifact should make that checkable rather
    than leaving it to memory.
    """
    import joblib

    model = clone_estimator(estimator)
    model.fit(X, y)

    dates = pd.to_datetime(dates)
    paths.ensure_dirs()
    destination = paths.MODELS / "outcome_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": columns,
            "classes": CLASSES,
            "class_names": CLASS_NAMES,
            "trained_rows": len(X),
            "train_start": dates.min().strftime("%Y-%m-%d"),
            "train_end": dates.max().strftime("%Y-%m-%d"),
            "provenance": provenance.collect(),
            "walk_forward": None
            if result is None
            else {
                "log_loss": result.log_loss,
                "brier": result.brier,
                "accuracy": result.accuracy,
                "ece": result.expected_calibration_error(),
                "n": result.n,
                "folds": result.folds,
            },
        },
        destination,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-votes",
        action="store_true",
        help="exclude crowd-vote features, to measure how much they carry",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default=None,
        help="last date to evaluate on, e.g. 2023-05-31 to stop before the "
        "summer friendly season changes the distribution",
    )
    parser.add_argument(
        "--min-matches", type=int, default=0, help="require N prior matches per side"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset to train on (default: data/processed/matches.csv; pass "
        "data/processed/matches_combined.csv for the goal.com + football-data union)",
    )
    parser.add_argument("--save", action="store_true", help="refit the best model and save it")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="write a reliability diagram for the best model to reports/",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = None
    if args.dataset is not None:
        matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
        log.info("Training on %s (%d rows)", args.dataset, len(matches))

    features = build_features(
        matches, include_votes=not args.no_votes, min_matches_played=args.min_matches
    )
    columns = feature_columns(features)
    X = features[columns]
    y = features["outcome"].to_numpy()

    eval_end = pd.Timestamp(args.eval_end) if args.eval_end else None
    splits = walk_forward_splits(features["date"], eval_end=eval_end)
    if not splits:
        raise SystemExit("No walk-forward folds -- not enough history in the dataset")

    log.info(
        "%d folds, first test window %s, last %s",
        len(splits),
        splits[0][2].date(),
        splits[-1][2].date(),
    )
    log.info("%d features, %d matches", len(columns), len(features))

    candidates = build_candidates(columns)
    results = []
    for name, estimator in candidates.items():
        log.info("Evaluating %s ...", name)
        results.append(evaluate(name, estimator, X, y, splits))

    report(results)

    best = min(results, key=lambda r: r.log_loss)

    if args.calibration:
        destination = plot_reliability(best, paths.ROOT / "reports" / "reliability.png")
        log.info("Wrote %s", destination)

    if args.save:
        if best.name.startswith("baseline"):
            log.warning("Best model is a baseline; not saving")
            return
        destination = fit_final_model(candidates[best.name], X, y, columns, features["date"], best)
        log.info("Saved %s (%s)", destination, best.name)


if __name__ == "__main__":
    main()
