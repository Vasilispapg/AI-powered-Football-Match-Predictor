"""Tests for the merge, filter and join stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.data.filter import classify_competition, filter_matches, is_excluded
from football.data.merge import EXPECTED_COLUMNS, merge_matches
from football.marketvalue.join import join_market_values


def _match_row(**overrides) -> dict:
    row = {
        "Competition": "Premier League",
        "Country": "England",
        "Home Team": "Arsenal",
        "Home Score": "2",
        "Away Team": "Liverpool",
        "Away Score": "1",
        "Date": "2023-01-15",
        "URL": "https://example.test/arsenal-liverpool",
        "Votes": "[]",
        "Stats": "{}",
    }
    row.update(overrides)
    return row


def _write_day(directory, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(
        directory / name, index=False, encoding="utf-8"
    )


class TestMerge:
    def test_header_rows_do_not_become_data(self, tmp_path) -> None:
        """Regression: the legacy merge used a bare csv.reader with no header
        handling, so each file's header row was appended as a record -- 322 junk
        rows in the old merged output."""
        source = tmp_path / "in"
        source.mkdir()
        _write_day(source, "a.csv", [_match_row()])
        _write_day(source, "b.csv", [_match_row(URL="https://example.test/b")])

        merged = merge_matches(source, tmp_path / "merged.csv")

        assert len(merged) == 2
        assert (merged["Competition"] == "Competition").sum() == 0

    def test_deduplicates_on_url(self, tmp_path) -> None:
        """The legacy scraper ran the same URL list in two threads."""
        source = tmp_path / "in"
        source.mkdir()
        _write_day(source, "a.csv", [_match_row()])
        _write_day(source, "b.csv", [_match_row()])

        assert len(merge_matches(source, tmp_path / "merged.csv")) == 1

    def test_skips_files_with_the_wrong_schema(self, tmp_path) -> None:
        """Regression: webScrapper/football.py writes a 7-column header for
        8-column rows, so a re-scrape silently produces files with no URL."""
        source = tmp_path / "in"
        source.mkdir()
        _write_day(source, "good.csv", [_match_row()])
        pd.DataFrame([{"Competition": "X", "Country": "Y"}]).to_csv(source / "bad.csv", index=False)

        merged = merge_matches(source, tmp_path / "merged.csv")
        assert len(merged) == 1

    def test_raises_when_there_is_nothing_to_merge(self, tmp_path) -> None:
        source = tmp_path / "empty"
        source.mkdir()
        with pytest.raises(FileNotFoundError):
            merge_matches(source, tmp_path / "merged.csv")


class TestCompetitionClassification:
    @pytest.mark.parametrize(
        ("competition", "expected"),
        [
            ("FA Cup", "cup"),
            ("Carabao Cup", "cup"),
            ("Club Friendlies", "friendly"),
            ("Premier League", "league"),
            ("Serie A", "league"),
        ],
    )
    def test_classifies(self, competition: str, expected: str) -> None:
        assert classify_competition(competition) == expected

    @pytest.mark.parametrize(
        "competition",
        ["UEFA U19 Championship", "Women's Super League", "U21 Premier League"],
    )
    def test_excludes_youth_and_womens(self, competition: str) -> None:
        assert is_excluded(competition)

    def test_keeps_senior_mens_football(self) -> None:
        assert not is_excluded("Premier League")


class TestFilter:
    def _run(self, tmp_path, rows):
        source = tmp_path / "merged.csv"
        pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(source, index=False)
        return filter_matches(source, tmp_path / "filtered.csv")

    def test_renames_columns_to_snake_case(self, tmp_path) -> None:
        result = self._run(tmp_path, [_match_row()])
        assert "home_team" in result.columns
        assert "Home Team" not in result.columns

    def test_drops_unparseable_scores(self, tmp_path) -> None:
        rows = [
            _match_row(),
            _match_row(**{"Home Score": "N/A", "URL": "https://example.test/2"}),
            _match_row(**{"Away Score": "", "URL": "https://example.test/3"}),
        ]
        assert len(self._run(tmp_path, rows)) == 1

    def test_drops_self_play(self, tmp_path) -> None:
        rows = [_match_row(**{"Away Team": "Arsenal"})]
        with pytest.raises(ValueError, match="removed every row"):
            self._run(tmp_path, rows)

    def test_raises_instead_of_indexerror_on_empty_result(self, tmp_path) -> None:
        """Regression: the legacy filter crashed with IndexError on
        ``filtered_rows[0].keys()`` when everything was filtered out."""
        rows = [_match_row(Competition="Women's Super League")]
        with pytest.raises(ValueError, match="removed every row"):
            self._run(tmp_path, rows)

    def test_scores_are_integers(self, tmp_path) -> None:
        result = self._run(tmp_path, [_match_row()])
        assert result["home_score"].dtype.kind == "i"


class TestJoin:
    def test_never_transposes_columns(self) -> None:
        """Regression: the legacy two-pass append put the away value in the
        ``MV Home Team`` column whenever the home team failed to match."""
        matches = pd.DataFrame(
            {"home_team": ["Unknown FC", "Arsenal"], "away_team": ["Liverpool", "Chelsea"]}
        )
        values = pd.DataFrame(
            {
                "team": ["Arsenal", "Liverpool", "Chelsea"],
                "market_value": [1240.0, 794.8, 961.95],
            }
        )

        joined = join_market_values(matches, values)

        # The unmatched home team leaves a gap; it does not shift Liverpool's
        # value into the home column.
        assert np.isnan(joined.loc[0, "mv_home"])
        assert joined.loc[0, "mv_away"] == pytest.approx(794.8)
        assert joined.loc[1, "mv_home"] == pytest.approx(1240.0)
        assert joined.loc[1, "mv_away"] == pytest.approx(961.95)

    def test_sets_missing_flags(self) -> None:
        matches = pd.DataFrame({"home_team": ["Unknown FC"], "away_team": ["Arsenal"]})
        values = pd.DataFrame({"team": ["Arsenal"], "market_value": [1240.0]})

        joined = join_market_values(matches, values)

        assert bool(joined.loc[0, "mv_home_missing"]) is True
        assert bool(joined.loc[0, "mv_away_missing"]) is False

    def test_missing_value_propagates_as_missing(self) -> None:
        """A team present in the table but without a value must stay missing,
        not become zero."""
        matches = pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Liverpool"]})
        values = pd.DataFrame({"team": ["Arsenal", "Liverpool"], "market_value": [np.nan, 794.8]})

        joined = join_market_values(matches, values)

        assert np.isnan(joined.loc[0, "mv_home"])
        assert bool(joined.loc[0, "mv_home_missing"]) is True
