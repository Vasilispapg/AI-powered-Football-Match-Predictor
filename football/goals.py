"""Model goals, so that scorelines fall out of a distribution.

Predicting the exact score as a classification problem does not work: there are
effectively unbounded classes, the tail is desperately sparse, and nothing ties
3-1 to 3-2. The standard approach models the *goals* instead and reads every
scoreline off the resulting joint distribution.

Two ingredients, both from the literature:

**Poisson with team strengths** (Maher, 1982). Each team gets an attack and a
defence parameter; the home side gets a bonus. Home goals are Poisson with
``exp(attack_home - defence_away + home_advantage)``, away goals likewise
without the bonus.

**The Dixon-Coles correction** (1997). Plain Poisson treats the two scores as
independent and gets low-scoring games wrong -- it under-predicts 0-0 and 1-1
and over-predicts 1-0 and 0-1. Dixon-Coles multiplies those four cells by a
fitted factor ``rho``, and weights recent matches more heavily via exponential
time decay, because form from two seasons ago should not count equally.

Why this is worth doing when the 1X2 market cannot be beaten: the correct-score
market is a different market. It is thin, carries bookmaker margins of 20-30%
against 2-5% on 1X2, and its prices are often generated from a naive Poisson --
the very model this one is a documented improvement on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

log = logging.getLogger(__name__)

#: Goals considered when building a score matrix. P(5+ goals for one side) is
#: small but not negligible; 10 keeps the matrix summing to ~1.
MAX_GOALS = 10

#: Exponential time decay. A match this many days old counts half as much as
#: one played today. Dixon-Coles used roughly half a season.
DEFAULT_HALF_LIFE_DAYS = 180.0

#: Ridge penalty on attack/defence. Two jobs: it makes the parameters
#: identifiable (the likelihood is otherwise flat along "add one to every
#: attack, subtract one from every defence"), and it shrinks teams with few
#: matches toward league average instead of letting one 4-0 define them.
DEFAULT_REGULARISATION = 1e-3


@dataclass
class GoalsModel:
    """Fitted attack/defence strengths for one competition."""

    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    matches_fitted: int = 0
    converged: bool = True
    league_mean_attack: float = 0.0
    league_mean_defence: float = 0.0

    def rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals ``(home, away)`` for a fixture.

        Unknown teams fall back to league-average strength rather than raising:
        a promoted side with no history should look like a typical team, not
        like a missing value.
        """
        attack_home = self.attack.get(home_team, self.league_mean_attack)
        attack_away = self.attack.get(away_team, self.league_mean_attack)
        defence_home = self.defence.get(home_team, self.league_mean_defence)
        defence_away = self.defence.get(away_team, self.league_mean_defence)

        home_rate = np.exp(attack_home - defence_away + self.home_advantage)
        away_rate = np.exp(attack_away - defence_home)
        return float(home_rate), float(away_rate)

    def knows(self, team: str) -> bool:
        return team in self.attack


def dixon_coles_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    """The low-score correction factor.

    Only the four cells at or below 1-1 are adjusted; everything else is left
    to plain Poisson. This is where the dependence between the two scores --
    a team defending a 1-0, a game opening up at 1-1 -- gets represented.
    """
    tau = np.ones_like(home_rate, dtype=float)

    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    tau[zero_zero] = 1.0 - home_rate[zero_zero] * away_rate[zero_zero] * rho
    tau[zero_one] = 1.0 + home_rate[zero_one] * rho
    tau[one_zero] = 1.0 + away_rate[one_zero] * rho
    tau[one_one] = 1.0 - rho
    return tau


def time_weights(dates: pd.Series, reference: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Exponential decay weights: recent matches count more."""
    age_days = (reference - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    age_days = np.maximum(age_days, 0.0)
    return np.exp(-np.log(2.0) * age_days / half_life_days)


def _negative_log_likelihood(
    params: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    regularisation: float,
) -> float:
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    home_advantage = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    log_home_rate = attack[home_index] - defence[away_index] + home_advantage
    log_away_rate = attack[away_index] - defence[home_index]
    home_rate = np.exp(log_home_rate)
    away_rate = np.exp(log_away_rate)

    log_likelihood = (
        home_goals * log_home_rate
        - home_rate
        - gammaln(home_goals + 1.0)
        + away_goals * log_away_rate
        - away_rate
        - gammaln(away_goals + 1.0)
    )

    tau = dixon_coles_tau(home_goals, away_goals, home_rate, away_rate, rho)
    # tau can go non-positive for extreme rho; clip rather than return NaN so
    # the optimiser is pushed back instead of crashing.
    log_likelihood = log_likelihood + np.log(np.clip(tau, 1e-10, None))

    penalty = regularisation * (np.sum(attack**2) + np.sum(defence**2))
    return float(-np.sum(weights * log_likelihood) + penalty)


def fit_goals_model(
    matches: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    regularisation: float = DEFAULT_REGULARISATION,
    max_iterations: int = 400,
) -> GoalsModel:
    """Fit attack/defence strengths by weighted maximum likelihood.

    *matches* must all come from one competition -- teams that never meet carry
    no information about each other, so fitting across leagues would compare
    strengths that share no common opponents.
    """
    if matches.empty:
        raise ValueError("No matches to fit")

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    index_of = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)

    home_index = matches["home_team"].map(index_of).to_numpy()
    away_index = matches["away_team"].map(index_of).to_numpy()
    home_goals = matches["home_score"].to_numpy(dtype=float)
    away_goals = matches["away_score"].to_numpy(dtype=float)

    reference = reference_date or pd.to_datetime(matches["date"]).max()
    weights = time_weights(matches["date"], reference, half_life_days)

    initial = np.zeros(2 * n_teams + 2)
    initial[2 * n_teams] = 0.25  # a sensible starting home advantage
    initial[2 * n_teams + 1] = 0.0  # rho

    bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 1.5), (-0.2, 0.2)]

    result = minimize(
        _negative_log_likelihood,
        initial,
        args=(
            home_index,
            away_index,
            home_goals,
            away_goals,
            weights,
            n_teams,
            regularisation,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations},
    )

    attack = dict(zip(teams, result.x[:n_teams], strict=True))
    defence = dict(zip(teams, result.x[n_teams : 2 * n_teams], strict=True))

    return GoalsModel(
        teams=teams,
        attack=attack,
        defence=defence,
        home_advantage=float(result.x[2 * n_teams]),
        rho=float(result.x[2 * n_teams + 1]),
        matches_fitted=len(matches),
        converged=bool(result.success),
        league_mean_attack=float(np.mean(list(attack.values()))),
        league_mean_defence=float(np.mean(list(defence.values()))),
    )


def score_matrix(
    home_rate: float, away_rate: float, rho: float = 0.0, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Joint probability of every scoreline up to *max_goals*.

    ``matrix[i, j]`` is P(home scores i, away scores j). Rows and columns are
    renormalised so the matrix sums to 1 after truncation and correction.
    """
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(goals * np.log(home_rate) - home_rate - gammaln(goals + 1.0))
    away_pmf = np.exp(goals * np.log(away_rate) - away_rate - gammaln(goals + 1.0))

    matrix = np.outer(home_pmf, away_pmf)

    if rho:
        matrix[0, 0] *= 1.0 - home_rate * away_rate * rho
        matrix[0, 1] *= 1.0 + home_rate * rho
        matrix[1, 0] *= 1.0 + away_rate * rho
        matrix[1, 1] *= 1.0 - rho

    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """``(home win, draw, away win)`` from a score matrix."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


def over_probability(matrix: np.ndarray, line: float = 2.5) -> float:
    """P(total goals above *line*), for checking against the totals market."""
    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    return float(matrix[goals > line].sum())


#: Totals lines a bookmaker normally posts.
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)


def derived_prices(matrix: np.ndarray) -> dict[str, float]:
    """Every market a score matrix implies, priced consistently.

    This is the real product of a goals model. One distribution prices the
    result, every totals line, both teams to score, the winning margin and each
    correct score -- and they cannot contradict each other, because they are all
    read off the same matrix.

    Bookmakers price the main market sharply and derive the rest with fatter
    margins, so internal inconsistencies between their own prices are where a
    coherent model earns its keep.
    """
    home, draw, away = outcome_probabilities(matrix)
    totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))

    prices: dict[str, float] = {
        "home_win": home,
        "draw": draw,
        "away_win": away,
        "home_or_draw": home + draw,
        "away_or_draw": away + draw,
        "home_or_away": home + away,
    }

    for line in GOAL_LINES:
        over = float(matrix[totals > line].sum())
        prices[f"over_{line}"] = over
        prices[f"under_{line}"] = 1.0 - over

    both_score = float(matrix[1:, 1:].sum())
    prices["btts_yes"] = both_score
    prices["btts_no"] = 1.0 - both_score

    prices["home_clean_sheet"] = float(matrix[:, 0].sum())
    prices["away_clean_sheet"] = float(matrix[0, :].sum())

    return prices


def top_scores(matrix: np.ndarray, count: int = 5) -> list[tuple[int, int, float]]:
    """The *count* most likely scorelines, most likely first."""
    flat = np.argsort(matrix, axis=None)[::-1][:count]
    rows, columns = np.unravel_index(flat, matrix.shape)
    return [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
    ]


def total_goals_from_over_probability(
    over_prob: float, line: float = 2.5, tolerance: float = 1e-8
) -> float:
    """Invert the totals market: what Poisson mean implies this over-probability?

    Monotone in the mean, so a bisection is exact and cheap.
    """
    if not np.isfinite(over_prob) or not 0.0 < over_prob < 1.0:
        return float("nan")

    threshold = int(np.floor(line))

    def over_for(mean: float) -> float:
        goals = np.arange(threshold + 1)
        under = np.exp(goals * np.log(mean) - mean - gammaln(goals + 1.0)).sum()
        return 1.0 - under

    low, high = 0.05, 12.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if over_for(middle) < over_prob:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def rates_from_market(handicap: float, over_prob: float, line: float = 2.5) -> tuple[float, float]:
    """Expected goals implied by the handicap and totals markets.

    The Asian handicap prices the expected goal *difference*; the totals market
    prices the expected *sum*. Two equations, two unknowns:

        home + away = total        (from over/under)
        home - away = supremacy    (from the handicap)

    The handicap is quoted as goals given to the home side, so a home favourite
    carries a negative line and supremacy is its negation.
    """
    total = total_goals_from_over_probability(over_prob, line)
    if not np.isfinite(total) or not np.isfinite(handicap):
        return float("nan"), float("nan")

    supremacy = -float(handicap)
    home_rate = (total + supremacy) / 2.0
    away_rate = (total - supremacy) / 2.0
    # A market can imply a negative rate for a heavy mismatch; floor it.
    return max(home_rate, 0.05), max(away_rate, 0.05)


def rates_from_market_fit(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_prob: float,
    handicap: float | None = None,
    rho: float = 0.0,
    line: float = 2.5,
    max_goals: int = 8,
) -> tuple[float, float]:
    """Expected goals that best reproduce *every* price the market publishes.

    :func:`rates_from_market` uses two signals: the totals market for the sum of
    goals and the handicap for the difference. But the handicap is quoted in
    quarter-goal steps, which is a coarse instrument, and the 1X2 prices carry
    information neither of the other two contains.

    This fits ``(home_rate, away_rate)`` so the resulting score matrix matches
    the home/draw/away probabilities *and* the over/under probability at once --
    an over-determined problem solved in least squares. More constraints, finer
    rates, and the same bivariate structure underneath.

    This is the practical route to beating a bookmaker's correct-score prices:
    those are usually generated from plain independent Poisson, while this reads
    the same public numbers through a model that handles low-scoring dependence.
    """
    from scipy.optimize import minimize

    if not all(np.isfinite([home_prob, draw_prob, away_prob, over_prob])):
        return float("nan"), float("nan")

    if handicap is not None and np.isfinite(handicap):
        start_home, start_away = rates_from_market(handicap, over_prob, line)
    else:
        total = total_goals_from_over_probability(over_prob, line)
        start_home = start_away = total / 2.0
    if not np.isfinite(start_home) or not np.isfinite(start_away):
        return float("nan"), float("nan")

    def objective(log_rates: np.ndarray) -> float:
        home_rate, away_rate = np.exp(log_rates)
        matrix = score_matrix(home_rate, away_rate, rho, max_goals)
        home, draw, away = outcome_probabilities(matrix)
        over = over_probability(matrix, line)
        return (
            (home - home_prob) ** 2
            + (draw - draw_prob) ** 2
            + (away - away_prob) ** 2
            + (over - over_prob) ** 2
        )

    result = minimize(
        objective,
        np.log([max(start_home, 0.05), max(start_away, 0.05)]),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 200},
    )
    home_rate, away_rate = np.exp(result.x)
    return float(np.clip(home_rate, 0.05, 8.0)), float(np.clip(away_rate, 0.05, 8.0))


@dataclass
class LeagueModels:
    """One fitted model per competition, fitted on matches before a cutoff."""

    models: dict[str, GoalsModel] = field(default_factory=dict)

    def fit(
        self,
        matches: pd.DataFrame,
        reference_date: pd.Timestamp,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        min_matches: int = 40,
    ) -> LeagueModels:
        self.models = {}
        for competition, group in matches.groupby("competition"):
            if len(group) < min_matches:
                continue
            try:
                self.models[str(competition)] = fit_goals_model(
                    group, reference_date=reference_date, half_life_days=half_life_days
                )
            except (ValueError, FloatingPointError) as error:
                log.warning("Could not fit %s: %s", competition, error)
        return self

    def rates(self, competition: str, home: str, away: str) -> tuple[float, float] | None:
        model = self.models.get(competition)
        if model is None or not (model.knows(home) and model.knows(away)):
            return None
        return model.rates(home, away)

    def rho(self, competition: str) -> float:
        model = self.models.get(competition)
        return model.rho if model else 0.0
