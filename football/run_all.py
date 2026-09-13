"""Run everything: data pipeline, features, training, saved model.

    python -m football.run_all

The original project needed eight manual steps in a specific order that was
recorded only in a gitignored notes file. This is that order, executed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from football import paths
from football.data.audit import audit

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-votes", action="store_true", help="build features without crowd votes"
    )
    parser.add_argument("--skip-train", action="store_true", help="stop after building features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    started = time.perf_counter()

    # Imported here so that `--help` does not pay for pandas and sklearn.
    from football.features import build_features, feature_columns
    from football.pipeline import run as run_pipeline
    from football.train import (
        build_candidates,
        evaluate,
        fit_final_model,
        plot_reliability,
        report,
        walk_forward_splits,
    )

    log.info("########## 1/4  data pipeline ##########")
    run_pipeline()

    log.info("########## 2/4  audit ##########")
    failures = audit(paths.MATCHES)
    if failures:
        log.error("Audit reported %d problem(s); stopping", failures)
        sys.exit(1)

    log.info("########## 3/4  features ##########")
    features = build_features(include_votes=not args.no_votes)
    columns = feature_columns(features)
    features.to_csv(paths.PROCESSED / "features.csv", index=False, encoding="utf-8")

    if args.skip_train:
        log.info("Done in %.1fs (training skipped)", time.perf_counter() - started)
        return

    log.info("########## 4/4  train ##########")
    X = features[columns]
    y = features["outcome"].to_numpy()
    splits = walk_forward_splits(features["date"])

    candidates = build_candidates(columns)
    results = [evaluate(name, estimator, X, y, splits) for name, estimator in candidates.items()]
    report(results)

    best = min(results, key=lambda r: r.log_loss)
    plot_reliability(best, paths.ROOT / "reports" / "reliability.png")

    if not best.name.startswith("baseline"):
        destination = fit_final_model(candidates[best.name], X, y, columns, features["date"], best)
        log.info("Saved %s (%s)", destination, best.name)

    log.info("Done in %.1fs", time.perf_counter() - started)
    print("\nPredict a fixture with:")
    print('  python -m football.predict --home "Arsenal" --away "Liverpool"')


if __name__ == "__main__":
    main()
