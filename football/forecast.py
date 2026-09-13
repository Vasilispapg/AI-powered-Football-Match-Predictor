"""Forecast upcoming fixtures, and score the forecast once results arrive.

    python -m football.forecast --date 2026-09-14      # predict
    python -m football.forecast --score                # grade past forecasts

Every prediction is written to ``data/forecasts/<date>.csv`` before the matches
are played, and ``--score`` grades those files against results later. That order
matters: a forecast that can only be read after the fact is not a forecast.

It is also the one thing tipster sites do not do. Selecting which predictions to
show after the results are in turns any model, however bad, into a good one.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths, provenance
from football.goals import (
    GOAL_LINES,
    LeagueModels,
    derived_prices,
    rates_from_market_fit,
    score_matrix,
    top_scores,
)

log = logging.getLogger(__name__)

FORECAST_DIR = paths.DATA / "forecasts"

#: Confidence at which a pick is worth acting on, from the hit-rate study:
#: 1X2 rises from 51.0% to 71.6% above this, over/under 2.5 from 57.4% to 65.4%.
STRONG = 0.60


def load_history() -> pd.DataFrame:
    """Match history in the same team-name namespace as the fixtures.

    Deliberately not the combined dataset: ``combine`` renames football-data
    teams onto goal.com spellings, which would then fail to match the fixture
    list. These two files keep the original names.
    """
    pieces = []
    for name in ("footballdata_matches.csv", "footballdata_extra.csv"):
        path = paths.INTERIM / name
        if path.exists():
            pieces.append(pd.read_csv(path, keep_default_na=False, na_values=[""]))
    if not pieces:
        raise SystemExit(
            "No history. Run:\n"
            "    python -m football.sources.footballdata --seasons 2324 2425 2526 2627 --extra"
        )
    history = pd.concat(pieces, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    return history.sort_values("date", kind="stable").reset_index(drop=True)


def forecast_fixtures(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Price every market for each fixture from one score matrix."""
    cutoff = pd.to_datetime(fixtures["date"]).min()
    past = history[history["date"] < cutoff]
    log.info("Fitting on %d matches before %s", len(past), cutoff.date())
    models = LeagueModels().fit(past, reference_date=cutoff)

    rows = []
    for _, fixture in fixtures.iterrows():
        home_p = fixture.get("odds_home_prob")
        draw_p = fixture.get("odds_draw_prob")
        away_p = fixture.get("odds_away_prob")
        over_p = fixture.get("odds_over25_prob")
        if not all(pd.notna(v) for v in (home_p, draw_p, away_p, over_p)):
            continue

        competition = str(fixture.get("competition", ""))
        rho = models.rho(competition)
        handicap = fixture.get("odds_handicap")
        home_rate, away_rate = rates_from_market_fit(
            float(home_p),
            float(draw_p),
            float(away_p),
            float(over_p),
            handicap=float(handicap) if pd.notna(handicap) else None,
            rho=rho,
        )
        if not (np.isfinite(home_rate) and np.isfinite(away_rate)):
            continue

        matrix = score_matrix(home_rate, away_rate, rho)
        prices = derived_prices(matrix)
        scores = top_scores(matrix, 3)

        three = np.array([prices["home_win"], prices["draw"], prices["away_win"]])
        pick = int(three.argmax())

        row = {
            "date": fixture["date"],
            "time": fixture.get("time", ""),
            "competition": competition,
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "xg_home": round(home_rate, 2),
            "xg_away": round(away_rate, 2),
            "p_home": round(float(three[0]), 4),
            "p_draw": round(float(three[1]), 4),
            "p_away": round(float(three[2]), 4),
            "pick_1x2": ("1", "X", "2")[pick],
            "conf_1x2": round(float(three[pick]), 4),
            "score_1": f"{scores[0][0]}-{scores[0][1]}",
            "p_score_1": round(scores[0][2], 4),
            "score_2": f"{scores[1][0]}-{scores[1][1]}",
            "p_score_2": round(scores[1][2], 4),
            "score_3": f"{scores[2][0]}-{scores[2][1]}",
            "p_score_3": round(scores[2][2], 4),
            "p_btts": round(prices["btts_yes"], 4),
        }
        for line in GOAL_LINES:
            row[f"p_over_{line}"] = round(prices[f"over_{line}"], 4)
        rows.append(row)

    return pd.DataFrame(rows)


def save_forecast(predictions: pd.DataFrame, date: str) -> Path:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    destination = FORECAST_DIR / f"{date}.csv"
    stamped = predictions.copy()
    stamped["forecast_made_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    stamped["git_commit"] = provenance.git_revision()["commit"]
    stamped.to_csv(destination, index=False, encoding="utf-8")
    return destination


def load_forecast(path: Path) -> pd.DataFrame:
    """Read a saved forecast back with the pick columns intact.

    ``pick_1x2`` holds "1"/"X"/"2". On a day where none of the picks is a draw
    the column is all digits, and pandas reads it back as integers -- after
    which ``1 == "1"`` is False and every pick grades as wrong. Everything that
    reads a forecast file goes through here so the dtype cannot be forgotten in
    one place and not another.
    """
    return pd.read_csv(
        path, dtype={"pick_1x2": str, "score_1": str, "score_2": str, "score_3": str}
    )


def print_forecast(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        print("No fixtures to forecast.")
        return

    print()
    print(
        f"{'time':>5}  {'competition':<18} {'match':<36} "
        f"{'1':>5} {'X':>5} {'2':>5}  {'pick':>4}  {'score':>6} {'p':>5}  {'O2.5':>5} {'BTTS':>5}"
    )
    print("-" * 118)
    for _, row in predictions.iterrows():
        match = f"{row['home_team'][:16]} v {row['away_team'][:16]}"
        strong = "*" if row["conf_1x2"] >= STRONG else " "
        print(
            f"{row['time']:>5}  {row['competition'][:18]:<18} {match:<36} "
            f"{row['p_home']:>5.2f} {row['p_draw']:>5.2f} {row['p_away']:>5.2f}  "
            f"{row['pick_1x2']:>3}{strong}  {row['score_1']:>6} {row['p_score_1']:>5.2f}  "
            f"{row['p_over_2.5']:>5.2f} {row['p_btts']:>5.2f}"
        )

    strong_count = int((predictions["conf_1x2"] >= STRONG).sum())
    print()
    print(f"{len(predictions)} fixture(s); {strong_count} with 1X2 confidence >= {STRONG:.0%} (*)")
    print(
        "Expect roughly 51% of all 1X2 picks to land, and roughly 72% of the "
        "starred ones -- those are the measured out-of-sample rates, not targets."
    )


#: Typical bookmaker margin on a single 1X2 selection, and on a correct score.
#: Correct-score markets are thin and carry far more margin than the main market.
MARGIN_1X2 = 0.05
MARGIN_SCORE = 0.25


def build_parlay(
    predictions: pd.DataFrame, legs: int = 4, score_legs: int = 2
) -> tuple[pd.DataFrame, dict]:
    """Take the most confident picks and price the accumulator honestly.

    Legs are chosen by model confidence, which is the only defensible ordering.
    The arithmetic below is the part worth reading: combining selections
    multiplies the probabilities *and* the bookmaker's margin, which is why a
    slip that looks like four near-certainties is usually a long shot.
    """
    ranked = predictions.sort_values("conf_1x2", ascending=False)
    outcome_legs = ranked.head(legs)

    # Correct-score legs come from the remaining fixtures, most confident first,
    # so the same match never appears twice on one slip.
    remaining = ranked.iloc[legs:].sort_values("p_score_1", ascending=False)
    chosen_scores = remaining.head(score_legs)

    rows = []
    probability = 1.0
    fair_odds = 1.0
    offered_odds = 1.0

    for _, row in outcome_legs.iterrows():
        p = float(row["conf_1x2"])
        fair = 1.0 / p
        offered = fair * (1.0 - MARGIN_1X2)
        probability *= p
        fair_odds *= fair
        offered_odds *= offered
        rows.append(
            {
                "match": f"{row['home_team']} v {row['away_team']}",
                "market": "1X2",
                "selection": row["pick_1x2"],
                "probability": p,
                "fair_odds": fair,
                "likely_odds": offered,
            }
        )

    for _, row in chosen_scores.iterrows():
        p = float(row["p_score_1"])
        fair = 1.0 / p
        offered = fair * (1.0 - MARGIN_SCORE)
        probability *= p
        fair_odds *= fair
        offered_odds *= offered
        rows.append(
            {
                "match": f"{row['home_team']} v {row['away_team']}",
                "market": "correct score",
                "selection": row["score_1"],
                "probability": p,
                "fair_odds": fair,
                "likely_odds": offered,
            }
        )

    summary = {
        "probability": probability,
        "fair_odds": fair_odds,
        "likely_odds": offered_odds,
        "expected_value": probability * offered_odds - 1.0,
        "one_in": 1.0 / probability if probability else float("inf"),
    }
    return pd.DataFrame(rows), summary


def print_parlay(legs: pd.DataFrame, summary: dict) -> None:
    print()
    print(f"{'match':<38} {'market':<14} {'pick':>6} {'our p':>7} {'fair':>7} {'likely':>7}")
    print("-" * 84)
    for _, leg in legs.iterrows():
        print(
            f"{leg['match'][:38]:<38} {leg['market']:<14} {leg['selection']:>6} "
            f"{leg['probability']:>6.1%} {leg['fair_odds']:>7.2f} {leg['likely_odds']:>7.2f}"
        )

    print()
    print(f"  probability all {len(legs)} land : {summary['probability']:.2%}  "
          f"(about 1 in {summary['one_in']:.0f})")
    print(f"  fair odds                : {summary['fair_odds']:.1f}")
    print(f"  odds you would be offered: {summary['likely_odds']:.1f}")
    print(f"  expected value           : {summary['expected_value']:+.1%}")
    print()
    print(
        "  Combining selections multiplies the margin as well as the\n"
        "  probabilities. Each leg is priced fairly by our own model, and the\n"
        "  slip still carries a negative expectation -- that is the arithmetic\n"
        "  of accumulators, not a flaw in these particular picks."
    )


def score_forecasts(history: pd.DataFrame) -> None:
    """Grade every saved forecast against results that have since arrived."""
    if not FORECAST_DIR.exists():
        raise SystemExit(f"No forecasts saved under {FORECAST_DIR}")

    results = history.copy()
    results["date"] = pd.to_datetime(results["date"]).dt.strftime("%Y-%m-%d")
    key = results.set_index(["date", "home_team", "away_team"])

    total = hits = strong_total = strong_hits = 0
    exact_total = exact_hits = 0
    graded_files = 0

    for path in sorted(FORECAST_DIR.glob("*.csv")):
        predictions = load_forecast(path)
        graded = []
        for _, row in predictions.iterrows():
            try:
                actual = key.loc[(row["date"], row["home_team"], row["away_team"])]
            except KeyError:
                continue
            if isinstance(actual, pd.DataFrame):
                actual = actual.iloc[0]

            home_goals = int(actual["home_score"])
            away_goals = int(actual["away_score"])
            outcome = "1" if home_goals > away_goals else ("X" if home_goals == away_goals else "2")
            correct = row["pick_1x2"] == outcome

            total += 1
            hits += int(correct)
            if row["conf_1x2"] >= STRONG:
                strong_total += 1
                strong_hits += int(correct)
            exact_total += 1
            exact_hits += int(row["score_1"] == f"{home_goals}-{away_goals}")
            graded.append((row, f"{home_goals}-{away_goals}", outcome, correct))

        if graded:
            graded_files += 1
            print(f"\n{path.stem}  ({len(graded)} graded)")
            print("-" * 76)
            for row, score, outcome, correct in graded:
                mark = "OK " if correct else "  X"
                strong = "*" if row["conf_1x2"] >= STRONG else " "
                print(
                    f"  {mark} {row['home_team'][:16]:<16} v {row['away_team'][:16]:<16} "
                    f"pick {row['pick_1x2']}{strong} ({row['conf_1x2']:.0%})  "
                    f"actual {outcome} {score}  said {row['score_1']}"
                )

    if not total:
        print("No saved forecast has a result yet.")
        return

    print()
    print("=" * 60)
    print(f"1X2 picks      : {hits}/{total} = {hits / total:.1%}   (expected ~51%)")
    if strong_total:
        print(
            f"  of which >={STRONG:.0%}: {strong_hits}/{strong_total} = "
            f"{strong_hits / strong_total:.1%}   (expected ~72%)"
        )
    print(
        f"exact score    : {exact_hits}/{exact_total} = {exact_hits / exact_total:.1%}   (expected ~13%)"
    )
    print(f"files graded   : {graded_files}")
    if total < 100:
        print()
        print(
            f"With {total} matches this says almost nothing -- the standard error on a "
            f"51% rate at n={total} is about {(0.51 * 0.49 / total) ** 0.5:.1%}. "
            "Several hundred are needed before a hit rate means anything."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="fixture date to forecast (YYYY-MM-DD)")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=paths.INTERIM / "fixtures.csv",
        help="fixture list to read",
    )
    parser.add_argument(
        "--score", action="store_true", help="grade saved forecasts instead of making one"
    )
    parser.add_argument(
        "--parlay",
        type=int,
        nargs="?",
        const=4,
        default=None,
        help="also build an accumulator from the N most confident picks (default 4)",
    )
    parser.add_argument(
        "--parlay-scores",
        type=int,
        default=2,
        help="correct-score legs to add to the accumulator",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    history = load_history()

    if args.score:
        score_forecasts(history)
        return

    if not args.fixtures.exists():
        raise SystemExit(
            f"{args.fixtures} not found. Fetch it with:\n"
            "    python -m football.sources.footballdata --fixtures"
        )
    fixtures = pd.read_csv(args.fixtures, keep_default_na=False, na_values=[""])

    if args.date:
        fixtures = fixtures[fixtures["date"] == args.date]
        if fixtures.empty:
            available = ", ".join(sorted(pd.read_csv(args.fixtures)["date"].unique()))
            raise SystemExit(f"No fixtures on {args.date}. Available: {available}")

    predictions = forecast_fixtures(fixtures, history)
    print_forecast(predictions)

    if not predictions.empty:
        date = args.date or str(predictions["date"].min())
        destination = save_forecast(predictions, date)
        print(f"\nSaved {destination}")
        print("Grade it once the results are in:  python -m football.forecast --score")

    if args.parlay and not predictions.empty:
        legs, summary = build_parlay(predictions, args.parlay, args.parlay_scores)
        print_parlay(legs, summary)


if __name__ == "__main__":
    main()
