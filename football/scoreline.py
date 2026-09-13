"""Evaluate scoreline prediction.

    python -m football.scoreline

Exact-score accuracy is low by nature and that is not a failure of the model.
In this dataset 1-1 is the single most common result at 12.1%, so "always say
1-1" is the bar. Good published models land around 12-14%. Anything much above
that is leaking.

Three approaches are compared, all evaluated walk-forward so no fold sees its
own future:

*always 1-1*     the naive bar.
*Dixon-Coles*    attack/defence strengths fitted on earlier results only.
*market rates*   expected goals solved from the handicap and totals markets,
                 fed through the same Dixon-Coles machinery.

The market comparison is the interesting one. The 1X2 market is unbeatable with
public data -- that is settled elsewhere in this project. But the handicap and
totals markets price *goals*, and the question here is whether a proper
bivariate model turns those into better scorelines than a bookmaker's own
correct-score prices, which are typically generated from plain Poisson.

Metrics: exact-score hit rate, log loss over the scoreline distribution, and
ranked probability score on the 1X2 collapse. RPS is used rather than accuracy
because home/draw/away is ordered -- predicting a home win when it finished a
draw is a smaller error than predicting one when it finished an away win.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths, provenance
from football.goals import (
    MAX_GOALS,
    LeagueModels,
    outcome_probabilities,
    rates_from_market,
    rates_from_market_fit,
    score_matrix,
    top_scores,
)

log = logging.getLogger(__name__)

#: Days of history each fold trains on before its test window opens.
WARMUP_DAYS = 200

#: Length of each walk-forward test window.
WINDOW_DAYS = 21


@dataclass
class ScoreResult:
    name: str
    exact_hits: int = 0
    n: int = 0
    log_loss_sum: float = 0.0
    rps_sum: float = 0.0
    outcome_hits: int = 0
    top_scores_seen: dict[str, int] = field(default_factory=dict)

    @property
    def exact_rate(self) -> float:
        return self.exact_hits / self.n if self.n else float("nan")

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / self.n if self.n else float("nan")

    @property
    def rps(self) -> float:
        return self.rps_sum / self.n if self.n else float("nan")

    @property
    def outcome_rate(self) -> float:
        return self.outcome_hits / self.n if self.n else float("nan")

    def record(
        self,
        matrix: np.ndarray,
        home_goals: int,
        away_goals: int,
    ) -> None:
        self.n += 1

        predicted = top_scores(matrix, 1)[0]
        if (predicted[0], predicted[1]) == (home_goals, away_goals):
            self.exact_hits += 1
        label = f"{predicted[0]}-{predicted[1]}"
        self.top_scores_seen[label] = self.top_scores_seen.get(label, 0) + 1

        capped_home = min(home_goals, matrix.shape[0] - 1)
        capped_away = min(away_goals, matrix.shape[1] - 1)
        self.log_loss_sum -= float(np.log(max(matrix[capped_home, capped_away], 1e-12)))

        home, draw, away = outcome_probabilities(matrix)
        probabilities = np.array([home, draw, away])
        if home_goals > away_goals:
            actual = 0
        elif home_goals == away_goals:
            actual = 1
        else:
            actual = 2
        if int(np.argmax(probabilities)) == actual:
            self.outcome_hits += 1

        # Ranked probability score: squared error between the cumulative
        # predicted and actual distributions, which respects the ordering of
        # home / draw / away.
        observed = np.zeros(3)
        observed[actual] = 1.0
        self.rps_sum += float(
            np.sum((np.cumsum(probabilities)[:-1] - np.cumsum(observed)[:-1]) ** 2) / 2.0
        )


def always_draw_matrix(max_goals: int = MAX_GOALS) -> np.ndarray:
    """The naive bar: all mass on 1-1 (softened so log loss stays finite)."""
    matrix = np.full((max_goals + 1, max_goals + 1), 1e-6)
    matrix[1, 1] = 1.0
    return matrix / matrix.sum()


def evaluate_scorelines(
    matches: pd.DataFrame,
    half_life_days: float = 180.0,
    warmup_days: int = WARMUP_DAYS,
    window_days: int = WINDOW_DAYS,
) -> dict[str, ScoreResult]:
    """Walk forward through the season, refitting before each window."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)

    start = frame["date"].min() + pd.Timedelta(days=warmup_days)
    last = frame["date"].max()

    results = {
        "always 1-1": ScoreResult("always 1-1"),
        "Dixon-Coles": ScoreResult("Dixon-Coles"),
        "market rates": ScoreResult("market rates"),
        "market fit": ScoreResult("market fit"),
        "blend": ScoreResult("blend"),
    }
    naive = always_draw_matrix()

    window_start = start
    folds = 0
    while window_start <= last:
        window_end = window_start + pd.Timedelta(days=window_days)
        history = frame[frame["date"] < window_start]
        window = frame[(frame["date"] >= window_start) & (frame["date"] < window_end)]

        if window.empty or len(history) < 200:
            window_start = window_end
            continue

        folds += 1
        league_models = LeagueModels().fit(
            history, reference_date=window_start, half_life_days=half_life_days
        )

        for _, match in window.iterrows():
            home_goals = int(match["home_score"])
            away_goals = int(match["away_score"])
            competition = str(match["competition"])

            results["always 1-1"].record(naive, home_goals, away_goals)

            rates = league_models.rates(
                competition, str(match["home_team"]), str(match["away_team"])
            )
            if rates is not None:
                rho = league_models.rho(competition)
                results["Dixon-Coles"].record(
                    score_matrix(rates[0], rates[1], rho), home_goals, away_goals
                )

            handicap = match.get("odds_handicap")
            over_prob = match.get("odds_over25_prob")
            rho = league_models.rho(competition)

            if pd.notna(handicap) and pd.notna(over_prob):
                market_home, market_away = rates_from_market(float(handicap), float(over_prob))
                if np.isfinite(market_home) and np.isfinite(market_away):
                    results["market rates"].record(
                        score_matrix(market_home, market_away, rho),
                        home_goals,
                        away_goals,
                    )

            # Fit the rates to every published price at once, not just two.
            home_prob = match.get("odds_home_prob")
            draw_prob = match.get("odds_draw_prob")
            away_prob = match.get("odds_away_prob")
            if all(pd.notna(value) for value in (home_prob, draw_prob, away_prob, over_prob)):
                fit_home, fit_away = rates_from_market_fit(
                    float(home_prob),
                    float(draw_prob),
                    float(away_prob),
                    float(over_prob),
                    handicap=float(handicap) if pd.notna(handicap) else None,
                    rho=rho,
                )
                if np.isfinite(fit_home) and np.isfinite(fit_away):
                    fit_matrix = score_matrix(fit_home, fit_away, rho)
                    results["market fit"].record(fit_matrix, home_goals, away_goals)

                    # Average the market's goal expectation with the model's
                    # own. The market knows the overall level; team strengths
                    # may still refine how it splits between the sides.
                    if rates is not None:
                        results["blend"].record(
                            score_matrix(
                                0.7 * fit_home + 0.3 * rates[0],
                                0.7 * fit_away + 0.3 * rates[1],
                                rho,
                            ),
                            home_goals,
                            away_goals,
                        )

        window_start = window_end

    log.info("%d fold(s)", folds)
    return results


def report(results: dict[str, ScoreResult]) -> None:
    print()
    print(f"{'model':<16} {'exact score':>12} {'scoreline':>11} {'RPS':>8} {'1X2 acc':>9} {'n':>8}")
    print("-" * 70)
    for result in sorted(results.values(), key=lambda r: -r.exact_rate):
        if not result.n:
            continue
        print(
            f"{result.name:<16} {result.exact_rate:>11.1%} {result.log_loss:>11.4f} "
            f"{result.rps:>8.4f} {result.outcome_rate:>8.1%} {result.n:>8,}"
        )

    naive = results["always 1-1"]
    print()
    for result in results.values():
        if result.name == "always 1-1" or not result.n:
            continue
        lift = result.exact_rate - naive.exact_rate
        print(
            f"  {result.name}: {lift:+.1%} exact-score against 'always 1-1' "
            f"({result.exact_rate:.1%} vs {naive.exact_rate:.1%})"
        )

    print()
    print("Most-predicted scorelines:")
    for result in results.values():
        if result.name == "always 1-1" or not result.n:
            continue
        ordered = sorted(result.top_scores_seen.items(), key=lambda kv: -kv[1])[:6]
        share = ", ".join(f"{label} {count / result.n:.0%}" for label, count in ordered)
        print(f"  {result.name:<14} {share}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.INTERIM / "footballdata_matches.csv",
        help="dataset with scores, handicap and totals",
    )
    parser.add_argument(
        "--half-life", type=float, default=180.0, help="days until a match counts half"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
    results = evaluate_scorelines(matches, half_life_days=args.half_life)
    report(results)


if __name__ == "__main__":
    main()
