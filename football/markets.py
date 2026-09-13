"""Hit rate per market, against the bar that actually matters for each one.

    python -m football.markets

"Are we above 50%?" is the wrong question for most of these, and answering it
yes would be misleading. Three different bars apply:

**The number of outcomes differs.** 1X2 has three, so 50% is a long way above
chance. Over/under has two, so 50% is chance.

**The naive baseline differs.** Always betting *over 0.5 goals* wins 93% of the
time. That is not skill, it is arithmetic -- almost every match has a goal.

**Break-even differs, and it is never 50%.** At typical odds of 1.90 a binary
market needs **52.6%** to stop losing money, because the bookmaker's margin sits
between the true probability and the price. A model at 51% is beating a coin and
losing money at the same time.

So each market is reported against its own baseline and its own break-even, and
a "beats baseline" column that is not simply "above 50%".
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths, provenance
from football.goals import (
    LeagueModels,
    derived_prices,
    rates_from_market_fit,
    score_matrix,
    top_scores,
)

log = logging.getLogger(__name__)

#: Odds a binary market is typically available at, used for the break-even
#: column. 1.90 a side implies a 5.3% margin, which is normal for totals.
TYPICAL_BINARY_ODDS = 1.90

#: Confidence above which a pick is counted as "strong".
STRONG = 0.60


@dataclass
class MarketTally:
    """Hit rate for one market."""

    name: str
    picks: int = 0
    hits: int = 0
    #: How often the event actually happened. The baseline is derived from this
    #: rather than hardcoded: the honest naive strategy is "always pick whichever
    #: side wins more often", and which side that is has to be measured, not
    #: assumed. Guessing it wrong inflates the apparent lift.
    actual_true: int = 0
    probability_sum: float = 0.0
    strong_picks: int = 0
    strong_hits: int = 0
    outcomes: int = 2

    def record(self, predicted: bool, actual: bool, probability: float) -> None:
        self.picks += 1
        self.hits += int(predicted == actual)
        self.actual_true += int(actual)
        self.probability_sum += probability
        if probability >= STRONG:
            self.strong_picks += 1
            self.strong_hits += int(predicted == actual)

    @property
    def rate(self) -> float:
        return self.hits / self.picks if self.picks else float("nan")

    @property
    def baseline_rate(self) -> float:
        """Always picking the more common side -- measured, not assumed.

        For markets with more than two outcomes there is no "other side" to
        take, so the baseline is simply how often the modal outcome occurs.
        """
        if not self.picks:
            return float("nan")
        share = self.actual_true / self.picks
        return share if self.outcomes > 2 else max(share, 1.0 - share)

    @property
    def claimed(self) -> float:
        """Mean probability we assigned to our own pick -- a calibration check."""
        return self.probability_sum / self.picks if self.picks else float("nan")

    @property
    def strong_rate(self) -> float:
        return self.strong_hits / self.strong_picks if self.strong_picks else float("nan")


def evaluate_markets(
    matches: pd.DataFrame,
    warmup_days: int = 200,
    window_days: int = 21,
) -> dict[str, MarketTally]:
    """Walk forward, pricing every market from one score matrix per match."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)

    tallies: dict[str, MarketTally] = {}

    def tally(name: str, outcomes: int = 2) -> MarketTally:
        if name not in tallies:
            tallies[name] = MarketTally(name, outcomes=outcomes)
        return tallies[name]

    window_start = frame["date"].min() + pd.Timedelta(days=warmup_days)
    last = frame["date"].max()

    while window_start <= last:
        window_end = window_start + pd.Timedelta(days=window_days)
        history = frame[frame["date"] < window_start]
        window = frame[(frame["date"] >= window_start) & (frame["date"] < window_end)]
        if window.empty or len(history) < 200:
            window_start = window_end
            continue

        league_models = LeagueModels().fit(history, reference_date=window_start)

        for _, match in window.iterrows():
            home_goals = int(match["home_score"])
            away_goals = int(match["away_score"])
            total_goals = home_goals + away_goals

            home_p = match.get("odds_home_prob")
            draw_p = match.get("odds_draw_prob")
            away_p = match.get("odds_away_prob")
            over_p = match.get("odds_over25_prob")
            if not all(pd.notna(v) for v in (home_p, draw_p, away_p, over_p)):
                continue

            handicap = match.get("odds_handicap")
            rho = league_models.rho(str(match["competition"]))
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

            # --- 1X2: three outcomes, baseline is always-home ---------------
            three = np.array([prices["home_win"], prices["draw"], prices["away_win"]])
            if home_goals > away_goals:
                actual_outcome = 0
            elif home_goals == away_goals:
                actual_outcome = 1
            else:
                actual_outcome = 2
            picked = int(three.argmax())
            entry = tally("1X2 (result)", outcomes=3)
            entry.picks += 1
            entry.hits += int(picked == actual_outcome)
            entry.actual_true += int(actual_outcome == 0)  # home wins, the modal result
            entry.probability_sum += float(three[picked])
            if three[picked] >= STRONG:
                entry.strong_picks += 1
                entry.strong_hits += int(picked == actual_outcome)

            # --- totals, every line ------------------------------------------
            for line in (0.5, 1.5, 2.5, 3.5, 4.5):
                probability = prices[f"over_{line}"]
                predicted_over = probability >= 0.5
                actual_over = total_goals > line
                tally(f"over/under {line}").record(
                    predicted=predicted_over,
                    actual=actual_over,
                    probability=probability if predicted_over else 1 - probability,
                )

            # --- both teams to score ------------------------------------------
            btts_probability = prices["btts_yes"]
            predicted_btts = btts_probability >= 0.5
            actual_btts = home_goals > 0 and away_goals > 0
            tally("both teams score").record(
                predicted=predicted_btts,
                actual=actual_btts,
                probability=btts_probability if predicted_btts else 1 - btts_probability,
            )

            # --- double chance -------------------------------------------------
            double = {
                "home or draw": (prices["home_or_draw"], actual_outcome in (0, 1)),
                "away or draw": (prices["away_or_draw"], actual_outcome in (1, 2)),
            }
            for name, (probability, actual) in double.items():
                predicted = probability >= 0.5
                tally(f"double chance: {name}").record(
                    predicted=predicted,
                    actual=actual,
                    probability=probability if predicted else 1 - probability,
                )

            # --- exact score ----------------------------------------------------
            best = top_scores(matrix, 1)[0]
            entry = tally("exact score", outcomes=40)
            entry.picks += 1
            entry.hits += int((best[0], best[1]) == (home_goals, away_goals))
            entry.actual_true += int((home_goals, away_goals) == (1, 1))  # always 1-1
            entry.probability_sum += best[2]

        window_start = window_end

    return tallies


def report(tallies: dict[str, MarketTally]) -> None:
    break_even = 1.0 / TYPICAL_BINARY_ODDS

    print()
    print(
        f"{'market':<24} {'our rate':>9} {'baseline':>9} {'claimed':>8} {'when >60%':>10} {'n':>8}"
    )
    print("-" * 74)
    for tally in tallies.values():
        strong = f"{tally.strong_rate:.1%}" if tally.strong_picks else "-"
        print(
            f"{tally.name:<24} {tally.rate:>8.1%} {tally.baseline_rate:>9.1%} "
            f"{tally.claimed:>8.1%} {strong:>10} {tally.picks:>8,}"
        )

    print()
    print(f"Break-even on a binary market at {TYPICAL_BINARY_ODDS} is {break_even:.1%}.")
    print()
    print("Beating the naive baseline:")
    for tally in tallies.values():
        lift = tally.rate - tally.baseline_rate
        verdict = "yes" if lift > 0 else "no"
        print(f"  {tally.name:<24} {lift:>+7.1%}   {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=paths.INTERIM / "footballdata_matches.csv",
        help="dataset carrying totals and handicap markets",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    provenance.set_seeds()

    matches = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])
    report(evaluate_markets(matches))


if __name__ == "__main__":
    main()
