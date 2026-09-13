"""Tests for the football-data.co.uk source and the cross-source team bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.sources.combine import build_bridge
from football.sources.footballdata import (
    DIVISIONS,
    odds_to_probabilities,
    pick_odds,
    read_csv,
    season_label,
    to_matches,
)
from football.transformers import DropAllNaNColumns


class TestSeasonLabel:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [("2627", "2026/2027"), ("2324", "2023/2024"), ("9900", "1999/2000")],
    )
    def test_expands_codes(self, code: str, expected: str) -> None:
        assert season_label(code) == expected

    @pytest.mark.parametrize("code", ["26", "abcd", "", "20267"])
    def test_rejects_bad_codes(self, code: str) -> None:
        with pytest.raises(ValueError, match="four digits"):
            season_label(code)


class TestOdds:
    def test_probabilities_sum_to_one(self) -> None:
        home, draw, away, _ = odds_to_probabilities((2.0, 3.5, 4.0))
        assert home + draw + away == pytest.approx(1.0)

    def test_overround_is_the_bookmakers_margin(self) -> None:
        """Fair odds on a coin flip are 2.0/2.0 and carry no margin."""
        *_, overround = odds_to_probabilities((2.0, 1e12, 2.0))
        assert overround == pytest.approx(0.0, abs=1e-6)

    def test_a_real_market_has_a_positive_margin(self) -> None:
        *_, overround = odds_to_probabilities((2.0, 3.5, 4.0))
        assert overround > 0

    def test_shorter_odds_mean_higher_probability(self) -> None:
        home, _, away, _ = odds_to_probabilities((1.5, 4.0, 6.0))
        assert home > away

    def test_prefers_closing_odds_over_opening(self) -> None:
        """Closing odds are taken nearest kick-off and carry more information."""
        row = pd.Series(
            {"AvgCH": 2.0, "AvgCD": 3.5, "AvgCA": 4.0, "AvgH": 9.0, "AvgD": 9.0, "AvgA": 9.0}
        )
        assert pick_odds(row) == (2.0, 3.5, 4.0)

    def test_falls_back_when_closing_odds_are_absent(self) -> None:
        """Fixtures carry opening odds only -- the match has not closed yet."""
        row = pd.Series(
            {
                "AvgCH": np.nan,
                "AvgCD": np.nan,
                "AvgCA": np.nan,
                "B365H": 2.5,
                "B365D": 3.2,
                "B365A": 2.9,
            }
        )
        assert pick_odds(row) == (2.5, 3.2, 2.9)

    def test_returns_none_when_nothing_is_usable(self) -> None:
        assert pick_odds(pd.Series({"AvgCH": np.nan, "HomeTeam": "A"})) is None

    def test_rejects_impossible_odds(self) -> None:
        """Decimal odds below 1.0 would imply a probability above 1."""
        row = pd.Series({"AvgCH": 0.5, "AvgCD": 3.5, "AvgCA": 4.0})
        assert pick_odds(row) is None


class TestSchemaMapping:
    def _raw(self, **overrides) -> pd.DataFrame:
        row = {
            "Date": "11/09/2026",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 2,
            "FTAG": 1,
            "HS": 14,
            "AS": 9,
            "HST": 6,
            "AST": 3,
            "HC": 7,
            "AC": 4,
            "HF": 11,
            "AF": 13,
            "AvgCH": 2.0,
            "AvgCD": 3.5,
            "AvgCA": 4.0,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_maps_into_the_project_schema(self) -> None:
        mapped = to_matches(self._raw(), "E0")
        row = mapped.iloc[0]

        assert row["home_team"] == "Arsenal"
        assert row["home_score"] == 2
        assert row["competition"] == "Premier League"
        assert row["country"] == "England"
        assert row["date"] == "2026-09-11"

    def test_parses_day_first_dates(self) -> None:
        """The site writes dd/mm/yyyy; read as mm/dd this becomes 9 November."""
        mapped = to_matches(self._raw(Date="11/09/2026"), "E0")
        assert mapped.iloc[0]["date"] == "2026-09-11"

    def test_statistics_round_trip_through_the_feature_parser(self) -> None:
        from football.features import parse_stats

        mapped = to_matches(self._raw(), "E0")
        stats = parse_stats(mapped.iloc[0]["stats"])

        assert stats["shots"] == (14.0, 9.0)
        assert stats["corners"] == (7.0, 4.0)
        assert stats["fouls"] == (11.0, 13.0)

    def test_skips_matches_that_have_not_been_played(self) -> None:
        assert to_matches(self._raw(FTHG=np.nan, FTAG=np.nan), "E0").empty

    def test_every_division_has_a_country_and_competition(self) -> None:
        for code, (country, competition) in DIVISIONS.items():
            assert country and competition, code


class TestExtraLeagues:
    def _raw(self, **overrides) -> pd.DataFrame:
        row = {
            "Country": "Argentina",
            "League": "Liga Profesional",
            "Season": "2026",
            "Date": "03/08/2026",
            "Time": "23:00",
            "Home": "Arsenal Sarandi",
            "Away": "Union de Santa Fe",
            "HG": 1,
            "AG": 0,
            "Res": "H",
            "AvgCH": 1.76,
            "AvgCD": 3.30,
            "AvgCA": 4.74,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_maps_the_alternative_schema(self) -> None:
        """The extra files use Home/Away/HG/AG, not HomeTeam/AwayTeam/FTHG/FTAG."""
        from football.sources.footballdata import to_matches_extra

        mapped = to_matches_extra(self._raw())
        row = mapped.iloc[0]

        assert row["home_team"] == "Arsenal Sarandi"
        assert row["home_score"] == 1
        assert row["away_score"] == 0
        assert row["date"] == "2026-08-03"

    def test_takes_country_and_competition_from_the_rows(self) -> None:
        """One file holds several competitions, so there is no division code."""
        from football.sources.footballdata import to_matches_extra

        row = to_matches_extra(self._raw()).iloc[0]
        assert row["country"] == "Argentina"
        assert row["competition"] == "Liga Profesional"

    def test_odds_become_probabilities(self) -> None:
        from football.sources.footballdata import to_matches_extra

        row = to_matches_extra(self._raw()).iloc[0]
        total = row["odds_home_prob"] + row["odds_draw_prob"] + row["odds_away_prob"]
        assert total == pytest.approx(1.0)
        assert row["odds_home_prob"] > row["odds_away_prob"]

    def test_absent_markets_are_missing_not_zero(self) -> None:
        """These files carry no totals, handicap, statistics or opening prices."""
        from football.sources.footballdata import to_matches_extra

        row = to_matches_extra(self._raw()).iloc[0]
        assert np.isnan(row["odds_over25_prob"])
        assert np.isnan(row["odds_handicap"])
        assert np.isnan(row["open_home_prob"])
        assert row["stats"] == "{}"

    def test_closing_prices_are_populated(self) -> None:
        from football.sources.footballdata import to_matches_extra

        row = to_matches_extra(self._raw()).iloc[0]
        assert row["close_home_prob"] == pytest.approx(row["odds_home_prob"])

    def test_skips_unplayed_matches(self) -> None:
        from football.sources.footballdata import to_matches_extra

        assert to_matches_extra(self._raw(HG=np.nan, AG=np.nan)).empty

    def test_is_tagged_with_its_own_source(self) -> None:
        """combine() prioritises sources, so they must be distinguishable."""
        from football.sources.footballdata import to_matches_extra

        assert to_matches_extra(self._raw()).iloc[0]["source"] == "football-data.co.uk/extra"


class TestReadCsv:
    def test_strips_a_utf8_bom(self, tmp_path) -> None:
        """fixtures.csv carries a BOM while the season files do not; reading it
        as latin-1 glued 'ï»¿' onto the first column name and silently produced
        a frame with no Div column."""
        path = tmp_path / "fixtures.csv"
        path.write_bytes("﻿Div,Date\nE0,11/09/2026\n".encode())

        frame = read_csv(path)
        assert "Div" in frame.columns

    def test_reads_latin1_season_files(self, tmp_path) -> None:
        path = tmp_path / "season.csv"
        path.write_bytes("Div,HomeTeam\nSP1,Alav\xe9s\n".encode("latin-1"))

        assert "Div" in read_csv(path).columns


class TestFetchGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "http://www.football-data.co.uk/x.csv",
            "https://evil.test/x.csv",
            "https://football-data.co.uk.evil.test/x.csv",
        ],
    )
    def test_refuses_anything_but_https_to_the_expected_host(self, url: str) -> None:
        from football.sources.footballdata import fetch

        with pytest.raises(ValueError, match="Refusing to fetch"):
            fetch(url)


class TestTeamBridge:
    def test_bridges_names_that_differ_only_by_accent(self) -> None:
        mapping, _ = build_bridge({"Beşiktaş"}, {"Besiktas"})
        assert mapping == {"Besiktas": "Beşiktaş"}

    def test_bridges_abbreviations(self) -> None:
        mapping, _ = build_bridge({"Dundee Utd"}, {"Dundee United"})
        assert mapping == {"Dundee United": "Dundee Utd"}

    def test_refuses_ambiguous_matches(self) -> None:
        """`normalise` strips bare years, so CSKA 1948 and CSKA both reduce to
        'cska' -- two different clubs. Merging them would fabricate a history
        that never happened, so the pair is skipped and reported."""
        mapping, ambiguous = build_bridge({"CSKA", "CSKA 1948"}, {"CSKA 1948"})

        assert mapping == {}
        assert len(ambiguous) == 1
        key, primary, secondary = ambiguous[0]
        assert key == "cska"
        assert primary == ["CSKA", "CSKA 1948"]
        assert secondary == ["CSKA 1948"]

    def test_leaves_identical_names_alone(self) -> None:
        mapping, _ = build_bridge({"Arsenal"}, {"Arsenal"})
        assert mapping == {}

    def test_ignores_teams_with_no_counterpart(self) -> None:
        mapping, _ = build_bridge({"Arsenal"}, {"Alloa Athletic"})
        assert mapping == {}


class TestDropAllNaNColumns:
    def test_drops_a_column_with_no_observed_value(self) -> None:
        """Regression: sklearn's histogram binner calls
        sliding_window_view(distinct_values, 2), which raises when a column is
        entirely NaN -- as the odds columns are in every pre-2023-07-28 fold."""
        frame = pd.DataFrame({"a": [1.0, 2.0], "odds": [np.nan, np.nan]})
        transformer = DropAllNaNColumns().fit(frame)

        assert transformer.keep_ == ["a"]
        assert transformer.dropped_ == ["odds"]
        assert list(transformer.transform(frame).columns) == ["a"]

    def test_keeps_partially_missing_columns(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan]})
        assert DropAllNaNColumns().fit(frame).keep_ == ["a"]

    def test_transform_reuses_the_fitted_choice(self) -> None:
        """Inference must see the training shape even if the new frame happens
        to have a value where training had none."""
        train = pd.DataFrame({"a": [1.0, 2.0], "odds": [np.nan, np.nan]})
        transformer = DropAllNaNColumns().fit(train)

        serve = pd.DataFrame({"a": [3.0], "odds": [0.5]})
        assert list(transformer.transform(serve).columns) == ["a"]

    def test_lets_a_boosting_model_fit_through_an_all_nan_column(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline

        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
        frame["odds"] = np.nan
        y = rng.integers(0, 3, size=400)

        pipeline = Pipeline(
            [
                ("drop_empty", DropAllNaNColumns()),
                ("model", HistGradientBoostingClassifier(max_iter=10, random_state=0)),
            ]
        )
        pipeline.fit(frame, y)  # raises without the transformer
        assert pipeline.predict_proba(frame).shape == (400, 3)
