"""Several specialist models, each doing what it is actually good at.

Running many models on the *same* target with the *same* information does not
help -- ``football.ablation`` measured that, and the answer was that nothing
adds anything on top of the market. Averaging ten views of the same odds returns
the odds.

Running models on different **regimes** is a different proposition, and this
dataset has two that barely overlap:

===================  =========  ==========================================
regime               matches    what is available
===================  =========  ==========================================
market priced        ~24,000    betting odds; the market is unbeatable here
no market            ~43,000    team history only; a model is all there is
===================  =========  ==========================================

Until now a single estimator handled both, with ``DropAllNaNColumns`` quietly
discarding the odds columns on every fold before the second source begins. That
works, but it means the model spends its capacity learning a compromise between
two situations that want opposite things.

This module makes the split explicit:

``MarketSpecialist``    reads the odds. Declines to predict without them.
``StrengthSpecialist``  team history, form, Elo, market values. Always applies.
``Router``              picks the specialist that applies, per match.
``Stack``               a meta-learner over both, trained only on *earlier*
                        folds' out-of-sample predictions, so it never sees its
                        own test data.

The evaluation reports each regime separately. One pooled number hides the whole
point: a market model scores brilliantly where it applies and not at all where
it does not.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from football import paths, provenance
from football.features import build_features, feature_columns
from football.train import CLASSES, multiclass_brier, numeric_pipeline, walk_forward_splits
from football.transformers import DropAllNaNColumns

log = logging.getLogger(__name__)


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-9, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _align(model, raw: np.ndarray, rows: int) -> np.ndarray:
    """Map an estimator's class order onto (home, draw, away)."""
    out = np.zeros((rows, 3))
    for position, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, position]
    return _normalise(out)


@dataclass
class Specialist:
    """One model plus the rule for when it is allowed to speak."""

    name: str
    columns: list[str]
    estimator: object
    requires: list[str] = field(default_factory=list)

    def applies(self, X: pd.DataFrame) -> np.ndarray:
        """Rows this specialist can actually predict."""
        if not self.requires:
            return np.ones(len(X), dtype=bool)
        mask = np.ones(len(X), dtype=bool)
        for column in self.requires:
            if column in X.columns:
                mask &= X[column].notna().to_numpy()
            else:
                mask[:] = False
        return mask

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Specialist | None:
        """Fit on the rows this specialist applies to. None if there are none."""
        usable = self.applies(X)
        if usable.sum() < 200 or len(np.unique(y[usable])) < 2:
            self.fitted_ = None
            return None
        self.fitted_ = clone(self.estimator)
        self.fitted_.fit(X.loc[usable, self.columns], y[usable])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities, with NaN rows where the specialist declines."""
        out = np.full((len(X), 3), np.nan)
        if getattr(self, "fitted_", None) is None:
            return out
        usable = self.applies(X)
        if not usable.any():
            return out
        raw = self.fitted_.predict_proba(X.loc[usable, self.columns])
        out[usable] = _align(self.fitted_, raw, int(usable.sum()))
        return out


def build_specialists(columns: list[str]) -> list[Specialist]:
    """The two regimes this dataset actually contains."""
    odds_columns = [c for c in columns if c.startswith("odds_")]
    strength_columns = [c for c in columns if not c.startswith(("odds_", "open_", "close_"))]

    specialists = [
        Specialist(
            name="strength",
            columns=strength_columns,
            # Same all-NaN guard the trainer uses: referee and kick-off hour
            # arrive only with the second source, so they are entirely empty on
            # every fold before 2023-07-28 and would crash the binner.
            estimator=Pipeline(
                [
                    ("drop_empty", DropAllNaNColumns()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=300,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            min_samples_leaf=80,
                            l2_regularization=1.0,
                            early_stopping=True,
                            validation_fraction=0.15,
                            random_state=provenance.SEED,
                        ),
                    ),
                ]
            ),
        )
    ]
    if odds_columns:
        specialists.insert(
            0,
            Specialist(
                name="market",
                columns=odds_columns,
                estimator=numeric_pipeline(LogisticRegression(max_iter=2000)),
                requires=["odds_home_prob"],
            ),
        )
    return specialists


def route(predictions: dict[str, np.ndarray], order: list[str], fallback: str) -> np.ndarray:
    """Take the first specialist that produced a prediction for each row.

    Walking *order* in reverse means the earliest-listed specialist overwrites
    later ones, so the list reads as a priority order.
    """
    out = predictions[fallback].copy()
    for name in reversed(order):
        available = ~np.isnan(predictions[name][:, 0])
        out[available] = predictions[name][available]
    return out


@dataclass
class RegimeResult:
    name: str
    log_loss: float
    brier: float
    accuracy: float
    n: int


def score(name: str, y_true: np.ndarray, probabilities: np.ndarray) -> RegimeResult:
    usable = ~np.isnan(probabilities[:, 0])
    truth = y_true[usable]
    predicted = _normalise(probabilities[usable])
    return RegimeResult(
        name=name,
        log_loss=log_loss(truth, predicted, labels=list(CLASSES)),
        brier=multiclass_brier(truth, predicted),
        accuracy=accuracy_score(truth, predicted.argmax(axis=1)),
        n=int(usable.sum()),
    )


def run(
    features: pd.DataFrame,
    warmup_days: int = 365,
    window_days: int = 30,
) -> None:
    """Walk forward, fitting every specialist, the router and the stack."""
    columns = feature_columns(features)
    X = features[columns]
    y = features["outcome"].to_numpy()
    splits = walk_forward_splits(features["date"], warmup_days=warmup_days, window_days=window_days)
    if not splits:
        raise SystemExit("Not enough history for walk-forward folds")

    specialists = build_specialists(columns)
    names = [specialist.name for specialist in specialists]
    log.info("%d fold(s), specialists: %s", len(splits), ", ".join(names))

    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    collected["router"] = []
    collected["stack"] = []
    truths: list[np.ndarray] = []
    has_market: list[np.ndarray] = []

    # Meta-training data accumulates from folds already scored, so the stack is
    # only ever fitted on predictions made out of sample.
    meta_X: list[np.ndarray] = []
    meta_y: list[np.ndarray] = []

    for fold, (train_index, test_index, _) in enumerate(splits):
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test = X.iloc[test_index]

        fold_predictions: dict[str, np.ndarray] = {}
        for specialist in specialists:
            specialist.fit(X_train, y_train)
            fold_predictions[specialist.name] = specialist.predict_proba(X_test)
            collected[specialist.name].append(fold_predictions[specialist.name])

        routed = route(fold_predictions, names, fallback="strength")
        collected["router"].append(routed)

        # --- stack -------------------------------------------------------
        stacked = np.full((len(test_index), 3), np.nan)
        flat = np.hstack([np.nan_to_num(fold_predictions[n], nan=1 / 3) for n in names])
        if len(meta_X) >= 2:
            # multi_class= was removed in sklearn 1.9; multinomial is the default.
            meta = LogisticRegression(max_iter=2000)
            meta.fit(np.vstack(meta_X), np.concatenate(meta_y))
            stacked = _align(meta, meta.predict_proba(flat), len(test_index))
        else:
            stacked = routed.copy()
        collected["stack"].append(stacked)

        meta_X.append(flat)
        meta_y.append(y[test_index])

        truths.append(y[test_index])
        market_available = fold_predictions.get("market")
        has_market.append(
            np.zeros(len(test_index), dtype=bool)
            if market_available is None
            else ~np.isnan(market_available[:, 0])
        )
        if fold % 10 == 0:
            log.info("  fold %d/%d", fold + 1, len(splits))

    truth = np.concatenate(truths)
    market_mask = np.concatenate(has_market)
    pooled = {name: np.vstack(values) for name, values in collected.items()}

    print()
    print(f"{'model':<12} {'regime':<16} {'log loss':>9} {'Brier':>8} {'accuracy':>9} {'n':>8}")
    print("-" * 68)

    regimes = [
        ("market priced", market_mask),
        ("no market", ~market_mask),
        ("all matches", np.ones_like(market_mask)),
    ]
    for regime_name, mask in regimes:
        if mask.sum() == 0:
            continue
        for model_name in [*names, "router", "stack"]:
            probabilities = pooled[model_name][mask]
            if np.isnan(probabilities[:, 0]).all():
                continue
            result = score(model_name, truth[mask], probabilities)
            print(
                f"{model_name:<12} {regime_name:<16} {result.log_loss:>9.4f} "
                f"{result.brier:>8.4f} {result.accuracy:>8.1%} {result.n:>8,}"
            )
        print()

    market_only = pooled.get("market")
    if market_only is not None and market_mask.any():
        market_score = score("market", truth[market_mask], market_only[market_mask])
        strength_here = score("strength", truth[market_mask], pooled["strength"][market_mask])
        print(
            f"Where the market exists it wins by {strength_here.log_loss - market_score.log_loss:+.4f} "
            f"log loss over the strength model."
        )
    if (~market_mask).any():
        strength_elsewhere = score(
            "strength", truth[~market_mask], pooled["strength"][~market_mask]
        )
        print(
            f"Where it does not, the strength model is the only option: "
            f"{strength_elsewhere.accuracy:.1%} accuracy on {strength_elsewhere.n:,} matches."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.PROCESSED / "matches_combined.csv",
        help="dataset spanning both regimes",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ignore matches before this date; history still builds from the full file",
    )
    parser.add_argument("--warmup", type=int, default=365, help="days before the first fold")
    parser.add_argument("--window", type=int, default=30, help="length of each test window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
    features = build_features(matches)

    if args.since:
        # Features are built from the whole file first, so team histories and
        # Elo are already warm; only the evaluation window is trimmed.
        cutoff = pd.Timestamp(args.since)
        before = len(features)
        features = features[pd.to_datetime(features["date"]) >= cutoff].reset_index(drop=True)
        log.info("Evaluating from %s: %d of %d matches", cutoff.date(), len(features), before)

    run(features, warmup_days=args.warmup, window_days=args.window)


if __name__ == "__main__":
    main()
