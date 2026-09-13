"""Hyperparameter search that cannot see the evaluation folds.

    python -m football.tune --iterations 60

Tuning is the easiest place to reintroduce the leakage Phase 3 removed. If the
search is scored on the same matches the walk-forward evaluation later reports,
the reported number is no longer out-of-sample -- the hyperparameters were
chosen by looking at it. Nothing raises; the number just quietly flatters.

So the search runs **only on matches before the first walk-forward test
window**, and within that slice it uses :class:`~sklearn.model_selection.TimeSeriesSplit`
so the inner folds are chronological too. The chosen parameters are written to
``data/interim/tuned_params.json``; :func:`football.train.build_candidates`
picks them up automatically on the next run, and the walk-forward evaluation
that follows is still genuinely out-of-sample.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from football import paths, provenance
from football.features import build_features, feature_columns
from football.train import walk_forward_splits
from football.transformers import DropAllNaNColumns

log = logging.getLogger(__name__)

TUNED_PARAMS = paths.INTERIM / "tuned_params.json"

#: Search space. Ranges are deliberately wide on regularisation: with 43k rows,
#: ~40 features and an irreducibly noisy target, over-fitting is the live risk,
#: not under-fitting.
SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [20, 40, 80, 160, 320],
    "l2_regularization": [0.0, 0.5, 1.0, 5.0, 10.0],
    "max_iter": [200, 300, 500],
    "max_features": [0.5, 0.7, 1.0],
}


def development_slice(features: pd.DataFrame) -> pd.DataFrame:
    """Matches that no walk-forward test fold will ever score.

    Everything before the first test window opens -- the warm-up period the
    evaluation already refuses to test on.
    """
    splits = walk_forward_splits(features["date"])
    if not splits:
        raise SystemExit("Not enough history to define a tuning slice")
    first_window = splits[0][2]
    development = features[pd.to_datetime(features["date"]) < first_window]
    log.info(
        "Tuning on %d matches before %s; the %d evaluation folds are untouched",
        len(development),
        first_window.date(),
        len(splits),
    )
    return development


def search(
    features: pd.DataFrame,
    iterations: int = 40,
    inner_folds: int = 4,
    n_jobs: int = -1,
) -> tuple[dict, float]:
    """Randomised search over :data:`SEARCH_SPACE`. Returns ``(params, log_loss)``."""
    development = development_slice(features)
    columns = feature_columns(development)
    X = development[columns]
    y = development["outcome"].to_numpy()

    if len(development) < 500:
        raise SystemExit(f"Only {len(development)} development matches -- too few to tune")

    # Same all-NaN guard the trainer uses: a column with no observed value
    # crashes the histogram binner (sklearn binning.py calls
    # sliding_window_view(distinct_values, 2), which needs at least two).
    estimator = Pipeline(
        [
            ("drop_empty", DropAllNaNColumns()),
            (
                "model",
                HistGradientBoostingClassifier(
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=provenance.SEED,
                ),
            ),
        ]
    )
    searcher = RandomizedSearchCV(
        estimator,
        param_distributions={f"model__{k}": v for k, v in SEARCH_SPACE.items()},
        n_iter=iterations,
        # Chronological inner folds: the search must not train on a match that
        # comes after the one it is scored on.
        cv=TimeSeriesSplit(n_splits=inner_folds),
        scoring="neg_log_loss",
        random_state=provenance.SEED,
        n_jobs=n_jobs,
        refit=False,
        verbose=0,
    )
    searcher.fit(X, y)

    # Strip the pipeline prefix so the saved file names plain estimator params.
    best = {key.removeprefix("model__"): value for key, value in searcher.best_params_.items()}
    best_loss = -float(searcher.best_score_)

    results = pd.DataFrame(searcher.cv_results_)
    results = results.sort_values("rank_test_score").head(5)
    log.info("Top 5 configurations by inner-fold log loss:")
    for _, row in results.iterrows():
        log.info("  %.4f  %s", -row["mean_test_score"], row["params"])

    return best, best_loss


def save_params(params: dict, path: Path | None = None) -> Path:
    path = path or TUNED_PARAMS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
    return path


def load_params(path: Path | None = None) -> dict | None:
    """Tuned parameters, or ``None`` if no search has been run."""
    path = path or TUNED_PARAMS
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        log.warning("Ignoring unreadable %s: %s", path, error)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=40, help="configurations to sample (default: 40)"
    )
    parser.add_argument(
        "--inner-folds", type=int, default=4, help="chronological folds inside the search"
    )
    parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs (default: all cores)")
    parser.add_argument(
        "--dataset", type=Path, default=None, help="dataset to tune on (default: processed matches)"
    )
    parser.add_argument("--no-votes", action="store_true", help="tune without vote features")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the best parameters without saving"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = None
    if args.dataset is not None:
        matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
        log.info("Tuning on %s (%d rows)", args.dataset, len(matches))

    features = build_features(matches, include_votes=not args.no_votes)
    best, best_loss = search(
        features,
        iterations=args.iterations,
        inner_folds=args.inner_folds,
        n_jobs=args.jobs,
    )

    print()
    print(f"Best inner-fold log loss: {best_loss:.4f}")
    print("Parameters:")
    for key, value in sorted(best.items()):
        print(f"  {key:<22} {value}")

    if args.dry_run:
        print("\n--dry-run: not saved")
        return

    path = save_params(best)
    print(f"\nSaved {path}")
    print("Re-run `python -m football.train` to evaluate them out-of-sample.")
    print(
        "The inner-fold score above is NOT the headline number -- it was used "
        "to choose these parameters, so only the walk-forward result is honest."
    )


if __name__ == "__main__":
    main()
