"""Predict the outcome of a fixture.

    python -m football.predict --home Arsenal --away Liverpool
    python -m football.predict --home "Real Madrid" --away Barcelona --date 2023-05-01

Replaces ``model/predict.py``, which scaled a list of column-*name strings* and
then called ``len()`` on an integer -- it raised on its first real call and had
evidently never been run.

The important property here is that this module does not reimplement feature
engineering. It replays match history up to the fixture date and calls the exact
same :func:`football.features.fixture_features` used to build the training set.
``tests/test_predict.py`` asserts the two paths agree value for value, which is
the check that catches train/serve skew -- the usual way a model that scored
well in evaluation quietly returns nonsense in production.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths
from football.features import TeamHistories, fixture_features

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    home_team: str
    away_team: str
    date: pd.Timestamp
    probabilities: np.ndarray
    class_names: tuple[str, ...]
    warnings: list[str]
    features: dict[str, float]

    @property
    def most_likely(self) -> str:
        return self.class_names[int(np.argmax(self.probabilities))]

    def render(self) -> str:
        lines = [
            f"{self.home_team}  vs  {self.away_team}     {self.date:%Y-%m-%d}",
            "",
        ]
        width = max(len(name) for name in self.class_names)
        for name, probability in zip(self.class_names, self.probabilities, strict=False):
            bar = "#" * round(probability * 40)
            lines.append(f"  {name:<{width}}  {probability:6.1%}  {bar}")

        lines += ["", f"  most likely: {self.most_likely}"]

        form = [
            ("home form (pts/game, last 5)", self.features.get("home_ppg_5")),
            ("away form (pts/game, last 5)", self.features.get("away_ppg_5")),
            ("home squad value (EUR m)", self.features.get("mv_home")),
            ("away squad value (EUR m)", self.features.get("mv_away")),
        ]
        lines.append("")
        for label, value in form:
            shown = "unknown" if value is None or pd.isna(value) else f"{value:,.2f}"
            lines.append(f"  {label:<30} {shown}")

        if self.warnings:
            lines.append("")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")
        return "\n".join(lines)


def load_model(path: Path | None = None) -> dict:
    """Load the saved model bundle."""
    import joblib

    path = path or paths.MODELS / "outcome_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one first:\n    python -m football.train --save"
        )
    return joblib.load(path)


def build_histories(matches: pd.DataFrame, before: pd.Timestamp) -> TeamHistories:
    """Replay every match strictly before *before* into a fresh history store."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] < before].sort_values("date", kind="stable")

    histories = TeamHistories()
    for date, day in frame.groupby("date", sort=True):
        histories.add_day(date, day)
    return histories


def market_value_lookup(path: Path | None = None) -> dict[str, float]:
    path = path or paths.MARKET_VALUES_CLEAN
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    values = frame.dropna(subset=["market_value"])
    return dict(zip(values["team"], values["market_value"], strict=True))


def predict_fixture(
    home_team: str,
    away_team: str,
    date: pd.Timestamp | str | None = None,
    *,
    competition_type: str = "league",
    votes: tuple[int, int, int] | None = None,
    bundle: dict | None = None,
    matches: pd.DataFrame | None = None,
    market_values: dict[str, float] | None = None,
) -> Prediction:
    """Predict one fixture's outcome probabilities."""
    bundle = bundle if bundle is not None else load_model()
    if matches is None:
        matches = pd.read_csv(paths.MATCHES, keep_default_na=False, na_values=[""])
    if market_values is None:
        market_values = market_value_lookup()

    columns = list(bundle["feature_columns"])
    uses_votes = any(column.startswith("votes_") for column in columns)

    if date is None:
        # Default to the day after the training data ends.
        date = pd.Timestamp(bundle["train_end"]) + pd.Timedelta(days=1)
    date = pd.Timestamp(date)

    warnings: list[str] = []
    histories = build_histories(matches, date)

    for team, role in ((home_team, "home"), (away_team, "away")):
        if not histories.knows(team):
            warnings.append(
                f"no match history for {role} team {team!r} before {date:%Y-%m-%d} "
                f"-- form features will be empty"
            )
        if team not in market_values:
            warnings.append(f"no market value for {team!r}")

    train_end = pd.Timestamp(bundle["train_end"])
    if date > train_end + pd.Timedelta(days=30):
        warnings.append(
            f"fixture is {(date - train_end).days} days past the training data "
            f"(ends {train_end:%Y-%m-%d}); form features come from stale history"
        )

    fixture = {
        "home_team": home_team,
        "away_team": away_team,
        "competition_type": competition_type,
        "mv_home": market_values.get(home_team, np.nan),
        "mv_away": market_values.get(away_team, np.nan),
        "votes": "[]"
        if votes is None
        else f"[['{home_team}', 'Draw', '{away_team}'], "
        f"['{votes[0]}', '{votes[1]}', '{votes[2]}']]",
    }

    features = fixture_features(histories, fixture, date, include_votes=uses_votes)

    if uses_votes and votes is None:
        warnings.append(
            "no crowd votes supplied; the model falls back to its vote-less "
            "regime, which scored 49.4% rather than 50.3% in evaluation"
        )

    missing = [column for column in columns if column not in features]
    if missing:
        raise RuntimeError(f"Feature contract broken -- model expects {missing}")

    X = pd.DataFrame([[features[column] for column in columns]], columns=columns)
    raw = bundle["model"].predict_proba(X)[0]

    probabilities = np.zeros(len(bundle["classes"]))
    for position, class_label in enumerate(bundle["model"].classes_):
        probabilities[int(class_label)] = raw[position]

    return Prediction(
        home_team=home_team,
        away_team=away_team,
        date=date,
        probabilities=probabilities,
        class_names=tuple(bundle["class_names"]),
        warnings=warnings,
        features=features,
    )


def predict_fixtures(
    fixtures: pd.DataFrame,
    bundle: dict | None = None,
    matches: pd.DataFrame | None = None,
    market_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score a whole fixture list, e.g. the coming weekend's matches.

    History is replayed once per distinct fixture date rather than once per
    fixture, so a 200-match card costs about the same as a single prediction.
    """
    bundle = bundle if bundle is not None else load_model()
    if matches is None:
        matches = pd.read_csv(paths.MATCHES, keep_default_na=False, na_values=[""])
    if market_values is None:
        market_values = market_value_lookup()

    columns = list(bundle["feature_columns"])
    uses_votes = any(column.startswith("votes_") for column in columns)

    history = matches.copy()
    history["date"] = pd.to_datetime(history["date"])

    rows: list[dict] = []
    for date_text, group in fixtures.groupby("date", sort=True):
        date = pd.Timestamp(date_text)
        histories = TeamHistories()
        earlier = history[history["date"] < date].sort_values("date", kind="stable")
        for day, day_matches in earlier.groupby("date", sort=True):
            histories.add_day(day, day_matches)

        for _, fixture in group.iterrows():
            home = str(fixture["home_team"])
            away = str(fixture["away_team"])
            payload = {
                "home_team": home,
                "away_team": away,
                "competition_type": fixture.get("competition_type", "league"),
                "mv_home": market_values.get(home, np.nan),
                "mv_away": market_values.get(away, np.nan),
                "votes": "[]",
                "odds_home_prob": fixture.get("odds_home_prob", np.nan),
                "odds_draw_prob": fixture.get("odds_draw_prob", np.nan),
                "odds_away_prob": fixture.get("odds_away_prob", np.nan),
                "odds_overround": fixture.get("odds_overround", np.nan),
            }
            features = fixture_features(histories, payload, date, include_votes=uses_votes)
            X = pd.DataFrame([[features.get(c, np.nan) for c in columns]], columns=columns)
            raw = bundle["model"].predict_proba(X)[0]

            probabilities = np.zeros(3)
            for position, label in enumerate(bundle["model"].classes_):
                probabilities[int(label)] = raw[position]

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "time": fixture.get("time", ""),
                    "competition": fixture.get("competition", ""),
                    "home_team": home,
                    "away_team": away,
                    "p_home": round(float(probabilities[0]), 4),
                    "p_draw": round(float(probabilities[1]), 4),
                    "p_away": round(float(probabilities[2]), 4),
                    "prediction": bundle["class_names"][int(probabilities.argmax())],
                    "known_home": histories.knows(home),
                    "known_away": histories.knows(away),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="home team, as named in the dataset")
    parser.add_argument("--away", help="away team, as named in the dataset")
    parser.add_argument(
        "--fixtures",
        nargs="?",
        const=str(paths.INTERIM / "fixtures.csv"),
        default=None,
        help="score a whole fixture list instead of one match",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="match history to replay (default: data/processed/matches.csv)",
    )
    parser.add_argument("--top", type=int, default=25, help="fixtures to print")
    parser.add_argument("--date", default=None, help="fixture date (YYYY-MM-DD)")
    parser.add_argument(
        "--competition-type",
        default="league",
        choices=("league", "cup", "friendly"),
        help="competition type (default: league)",
    )
    parser.add_argument(
        "--votes",
        default=None,
        help="crowd votes as home,draw,away -- e.g. 45,61,26",
    )
    parser.add_argument("--model", type=Path, default=None, help="model bundle to load")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")

    history = None
    if args.dataset is not None:
        history = pd.read_csv(args.dataset, keep_default_na=False, na_values=[""])

    if args.fixtures:
        path = Path(args.fixtures)
        if not path.exists():
            parser.error(
                f"{path} not found. Fetch it with:\n"
                f"    python -m football.sources.footballdata --fixtures"
            )
        fixtures = pd.read_csv(path)
        results = predict_fixtures(fixtures, bundle=load_model(args.model), matches=history)
        destination = paths.PROCESSED / "fixture_predictions.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(destination, index=False, encoding="utf-8")

        print(f"\n{len(results)} fixture(s)   ->  {destination}\n")
        shown = results.head(args.top)
        for _, row in shown.iterrows():
            flag = "" if row["known_home"] and row["known_away"] else "  (no history)"
            print(
                f"  {row['date']} {row['time']:>5}  {row['competition'][:18]:<18} "
                f"{row['home_team'][:20]:<20} v {row['away_team'][:20]:<20} "
                f"H {row['p_home']:.2f}  D {row['p_draw']:.2f}  A {row['p_away']:.2f}"
                f"{flag}"
            )
        if len(results) > len(shown):
            print(f"\n  ... {len(results) - len(shown)} more in {destination.name}")
        return

    if not args.home or not args.away:
        parser.error("--home and --away are required unless --fixtures is given")

    votes = None
    if args.votes:
        try:
            parts = tuple(int(part) for part in args.votes.split(","))
        except ValueError:
            parser.error("--votes must be three integers, e.g. 45,61,26")
        if len(parts) != 3:
            parser.error("--votes must be three integers, e.g. 45,61,26")
        votes = parts

    prediction = predict_fixture(
        args.home,
        args.away,
        args.date,
        competition_type=args.competition_type,
        votes=votes,
        bundle=load_model(args.model),
        matches=history,
    )
    print()
    print(prediction.render())
    print()


if __name__ == "__main__":
    main()
