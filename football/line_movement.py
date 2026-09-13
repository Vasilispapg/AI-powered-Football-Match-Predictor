"""Can we anticipate where the betting line is going?

    python -m football.line_movement

Odds move between the moment they are posted and kick-off. Opening prices are
set by the bookmaker; closing prices are what they become once money has been
laid. The drift between them is the market learning something.

Three questions, only one of which is interesting:

1. **Does drift predict the result?** Trivially yes, and useless: drift is only
   known once the line has closed, by which point you are betting into the
   closing price anyway. Included as a sanity check.

2. **Does drift add anything to closing odds?** Almost certainly not -- closing
   odds *are* opening plus drift. Included to confirm the arithmetic.

3. **Can our features, given only the opening price, beat the closing price?**
   This is the one that matters. Opening odds are what you could actually have
   taken. If a model built from opening odds plus team history scores the
   result better than the closing line does, it is anticipating the market's own
   correction -- which is the only form of edge available from public data.

Question 3 is a high bar and it is supposed to be. Closing odds are among the
best-calibrated public forecasts of anything, and most published attempts to
beat them fail. A negative answer here is a real answer, not a failed
experiment: it tells you to stop spending money on data the market already has.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from football import paths, provenance
from football.ablation import bootstrap_difference
from football.features import build_features, feature_columns
from football.train import ColumnSubset, evaluate, numeric_pipeline, walk_forward_splits

log = logging.getLogger(__name__)

OPEN_COLUMNS = ["open_home_prob", "open_draw_prob", "open_away_prob", "open_overround"]
CLOSE_COLUMNS = ["close_home_prob", "close_draw_prob", "close_away_prob", "close_overround"]
DRIFT_COLUMNS = ["drift_home", "drift_draw", "drift_away", "drift_magnitude"]


def add_drift(features: pd.DataFrame) -> pd.DataFrame:
    """Attach closing-minus-opening probability drift."""
    frame = features.copy()
    for outcome in ("home", "draw", "away"):
        frame[f"drift_{outcome}"] = frame[f"close_{outcome}_prob"] - frame[f"open_{outcome}_prob"]
    frame["drift_magnitude"] = frame[DRIFT_COLUMNS[:3]].abs().sum(axis=1)
    return frame


def _model(columns: list[str]):
    return ColumnSubset(columns, numeric_pipeline(LogisticRegression(max_iter=2000)))


def market_probabilities(features: pd.DataFrame, prefix: str) -> np.ndarray:
    """The market's own three probabilities, used directly as a forecast."""
    probabilities = features[
        [f"{prefix}_home_prob", f"{prefix}_draw_prob", f"{prefix}_away_prob"]
    ].to_numpy(dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=1 / 3)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def study(features: pd.DataFrame, iterations: int = 2000) -> None:
    """Run the three questions and print the answers."""
    features = add_drift(features)
    usable = features[
        features["open_home_prob"].notna() & features["close_home_prob"].notna()
    ].reset_index(drop=True)

    log.info(
        "%d of %d matches have both opening and closing prices",
        len(usable),
        len(features),
    )
    if len(usable) < 2000:
        raise SystemExit("Too few matches with both prices to say anything")

    columns = feature_columns(usable)
    X = usable[columns]
    y = usable["outcome"].to_numpy()
    splits = walk_forward_splits(usable["date"])

    # --- the market's own forecasts, scored directly -----------------------
    tested = np.concatenate([test for _, test, _ in splits])
    truth = y[tested]
    opening_market = market_probabilities(usable, "open")[tested]
    closing_market = market_probabilities(usable, "close")[tested]

    from sklearn.metrics import log_loss

    opening_loss = log_loss(truth, opening_market, labels=[0, 1, 2])
    closing_loss = log_loss(truth, closing_market, labels=[0, 1, 2])

    print()
    print("The market, scored against itself")
    print("-" * 52)
    print(f"  opening price            log loss {opening_loss:.4f}")
    print(f"  closing price            log loss {closing_loss:.4f}")
    drift_value = opening_loss - closing_loss
    print(f"  value of the drift       {drift_value:+.4f}")
    if drift_value > 0:
        print("  -> the line moves toward the truth, as expected")

    # --- question 3: opening + our features, against the closing line ------
    open_and_features = [c for c in columns if not c.startswith(("close_", "drift_"))]
    challenger = evaluate("opening + features", _model(open_and_features), X, y, splits)

    mean, low, high = bootstrap_difference(
        truth, challenger.probabilities, closing_market, iterations
    )

    print()
    print("Can we anticipate the market? (the question that matters)")
    print("-" * 52)
    print(f"  closing line             log loss {closing_loss:.4f}")
    print(f"  opening + our features   log loss {challenger.log_loss:.4f}")
    print(f"  difference               {mean:+.4f}  [{low:+.4f}, {high:+.4f}]")
    print()
    if high < 0:
        print("  BEATS THE CLOSING LINE. Verify before believing it:")
        print("    - are the opening odds genuinely pre-movement in this source?")
        print("    - does any feature encode information from after the open?")
        print("    - does it survive on a later, untouched slice of matches?")
    elif low > 0:
        print("  Worse than the closing line, conclusively.")
        print("  The market's correction is not predictable from these features.")
    else:
        print("  No significant difference from the closing line.")
        print("  Not evidence of edge -- matching the market is the default outcome.")

    # --- questions 1 and 2: sanity checks ----------------------------------
    with_drift = evaluate("closing + drift", _model(CLOSE_COLUMNS + DRIFT_COLUMNS), X, y, splits)
    closing_only = evaluate("closing only", _model(CLOSE_COLUMNS), X, y, splits)
    drift_mean, drift_low, drift_high = bootstrap_difference(
        truth, with_drift.probabilities, closing_only.probabilities, iterations
    )

    print()
    print("Sanity checks")
    print("-" * 52)
    print(f"  closing only             log loss {closing_only.log_loss:.4f}")
    print(f"  closing + drift          log loss {with_drift.log_loss:.4f}")
    print(f"  drift adds               {drift_mean:+.4f}  [{drift_low:+.4f}, {drift_high:+.4f}]")
    if drift_high >= 0:
        print("  -> drift adds nothing once you already have the closing price,")
        print("     which is the arithmetic working as it should.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.INTERIM / "footballdata_matches.csv",
        help="dataset carrying opening and closing odds",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="bootstrap resamples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
    if "open_home_prob" not in matches.columns:
        raise SystemExit(
            "This dataset has no separate opening odds. Re-ingest with:\n"
            "    python -m football.sources.footballdata --seasons 2324 2425 2526 2627"
        )
    study(build_features(matches), iterations=args.iterations)


if __name__ == "__main__":
    main()
