"""Measure what each feature family is actually worth.

    python -m football.ablation --dataset data/interim/footballdata_matches.csv

The betting odds beat every model in this project, so "does this feature beat
the 44% baseline?" is the wrong question -- almost anything does. The question
that decides whether a feature earns its place is:

    does it add signal **on top of the odds**?

This module answers it two ways for each family:

*add*   odds + family, against odds alone. Does the family carry information
        the market has not already priced in?
*drop*  everything, minus family. Does removing it hurt?

Both are reported with a paired bootstrap interval on the log-loss difference,
because on ~20,000 matches a gain of 0.002 is well inside the noise and reading
it as an improvement is how feature sets quietly fill up with junk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths, provenance
from football.features import build_features, feature_columns
from football.train import evaluate, walk_forward_splits

log = logging.getLogger(__name__)

#: Feature families, by the prefixes and substrings that identify them.
FAMILIES: dict[str, tuple[str, ...]] = {
    "odds": ("odds_",),
    "elo": ("elo_", "home_elo", "away_elo"),
    "form": ("_ppg_", "_gf_", "_ga_", "_gd_", "ppg_diff", "gd_diff", "_venue_"),
    "match stats": ("_possession_avg", "_shots_avg", "_corners_avg", "_fouls_avg"),
    "market value": ("mv_",),
    "crowd votes": ("votes_",),
    "referee": ("referee_",),
    "schedule": ("_rest_days", "rest_diff", "kickoff_hour"),
    "context": ("is_cup", "is_friendly", "_matches_played"),
}


def family_columns(columns: list[str], family: str) -> list[str]:
    markers = FAMILIES[family]
    return [column for column in columns if any(marker in column for marker in markers)]


def bootstrap_difference(
    y_true: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    iterations: int = 2000,
    seed: int = provenance.SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap on the per-match log-loss difference (a - b).

    Negative means *a* is better. Returns ``(mean, low, high)`` at 95%.
    Pairing matters: both models score the same matches, so the comparison
    should not be swamped by which matches happened to be easy.
    """
    rows = np.arange(len(y_true))
    loss_a = -np.log(np.clip(probabilities_a[rows, y_true], 1e-15, 1.0))
    loss_b = -np.log(np.clip(probabilities_b[rows, y_true], 1e-15, 1.0))
    difference = loss_a - loss_b

    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, len(difference), len(difference))
        means[index] = difference[sample].mean()

    return (
        float(difference.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def run_ablation(
    features: pd.DataFrame,
    iterations: int = 2000,
) -> pd.DataFrame:
    """Evaluate each family's incremental contribution."""
    columns = feature_columns(features)
    X = features[columns]
    y = features["outcome"].to_numpy()
    splits = walk_forward_splits(features["date"])

    odds = family_columns(columns, "odds")
    if not odds:
        raise SystemExit(
            "No odds columns in this dataset -- ablation against the market needs "
            "football-data.co.uk rows"
        )

    def model_for(subset: list[str]):
        from sklearn.linear_model import LogisticRegression

        from football.train import ColumnSubset, numeric_pipeline

        return ColumnSubset(subset, numeric_pipeline(LogisticRegression(max_iter=2000)))

    log.info("Baseline: odds only (%d columns)", len(odds))
    baseline = evaluate("odds only", model_for(odds), X, y, splits)

    log.info("Full model: every family (%d columns)", len(columns))
    full = evaluate("everything", model_for(columns), X, y, splits)

    rows = []
    for family in FAMILIES:
        if family == "odds":
            continue
        family_cols = family_columns(columns, family)
        if not family_cols:
            continue

        added = evaluate(
            f"odds+{family}", model_for(sorted(set(odds) | set(family_cols))), X, y, splits
        )
        add_mean, add_low, add_high = bootstrap_difference(
            added.y_true, added.probabilities, baseline.probabilities, iterations
        )

        remaining = [c for c in columns if c not in family_cols]
        dropped = evaluate(f"all-{family}", model_for(remaining), X, y, splits)
        drop_mean, drop_low, drop_high = bootstrap_difference(
            full.y_true, full.probabilities, dropped.probabilities, iterations
        )

        rows.append(
            {
                "family": family,
                "columns": len(family_cols),
                "add_delta": add_mean,
                "add_low": add_low,
                "add_high": add_high,
                "add_helps": add_high < 0,
                "drop_delta": drop_mean,
                "drop_low": drop_low,
                "drop_high": drop_high,
                "drop_hurts": drop_high < 0,
            }
        )
        log.info(
            "  %-14s add %+.4f [%+.4f, %+.4f]",
            family,
            add_mean,
            add_low,
            add_high,
        )

    report = pd.DataFrame(rows).sort_values("add_delta")
    report.attrs["baseline_log_loss"] = baseline.log_loss
    report.attrs["full_log_loss"] = full.log_loss
    report.attrs["n"] = baseline.n
    return report


def print_report(report: pd.DataFrame) -> None:
    baseline = report.attrs["baseline_log_loss"]
    full = report.attrs["full_log_loss"]

    print()
    print(f"odds only   log loss {baseline:.4f}")
    print(f"everything  log loss {full:.4f}    ({report.attrs['n']:,} matches)")
    print()
    print("Added on top of odds alone (negative = better, 95% bootstrap interval):")
    print(f"  {'family':<14} {'cols':>5} {'delta':>9}  {'95% interval':>22}   verdict")
    print("  " + "-" * 66)
    for _, row in report.iterrows():
        verdict = "helps" if row["add_helps"] else "no evidence"
        print(
            f"  {row['family']:<14} {row['columns']:>5} {row['add_delta']:>+9.4f}  "
            f"[{row['add_low']:>+8.4f}, {row['add_high']:>+8.4f}]   {verdict}"
        )

    helping = report[report["add_helps"]]["family"].tolist()
    print()
    if helping:
        print("Families with evidence of value beyond the market: " + ", ".join(helping))
    else:
        print(
            "No family shows evidence of adding information the market has not\n"
            "already priced in. That is the expected result for public data --\n"
            "and the bar any new source has to clear."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.INTERIM / "footballdata_matches.csv",
        help="dataset to ablate on (needs odds columns)",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--out", type=Path, default=None, help="write the report as CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
    features = build_features(matches)
    report = run_ablation(features, iterations=args.iterations)
    print_report(report)

    destination = args.out or paths.INTERIM / "ablation.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, encoding="utf-8")
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
