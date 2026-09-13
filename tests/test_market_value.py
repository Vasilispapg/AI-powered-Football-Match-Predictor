"""Tests for market-value parsing and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football.marketvalue.clean import (
    LEGACY_SENTINEL,
    clean_market_values,
    parse_market_value,
)


class TestParseMarketValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("996.00m", 996.0),
            ("794.8", 794.8),
            ("125k", 0.125),
            ("1.5m", 1.5),
            ("€308.45m", 308.45),
            ("1,240.00m", 1240.0),
            ("  45.2m  ", 45.2),
            ("0.01", 0.01),
        ],
    )
    def test_parses_known_formats(self, raw: str, expected: float) -> None:
        assert parse_market_value(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1.24bn", 1240.0), ("1bn", 1000.0), ("1.05BN", 1050.0), ("2.5b", 2500.0)],
    )
    def test_parses_billions(self, raw: str, expected: float) -> None:
        """Regression: the legacy parser mapped 'b' but tested endswith('b'),
        so "1.24bn" (ending in 'n') raised and every club worth EUR 1bn or more
        was silently written as zero. The scraped table has a hard ceiling at
        998 as a result."""
        assert parse_market_value(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "-", "n/a", "N/A", "none", "abc", "1.2.3m", None])
    def test_rejects_unparseable(self, raw: object) -> None:
        assert parse_market_value(raw) is None

    def test_rejects_nan(self) -> None:
        assert parse_market_value(float("nan")) is None

    def test_billion_outranks_million(self) -> None:
        assert parse_market_value("1bn") > parse_market_value("999m")


class TestCleanMarketValues:
    def _write(self, tmp_path, rows):
        source = tmp_path / "scraped.csv"
        pd.DataFrame(rows, columns=["Team Name", "Market Value"]).to_csv(source, index=False)
        return source, tmp_path / "clean.csv"

    def test_flags_sentinel_as_missing(self, tmp_path) -> None:
        source, destination = self._write(
            tmp_path, [["Arsenal", str(LEGACY_SENTINEL)], ["Liverpool", "794.8"]]
        )
        cleaned = clean_market_values(source, destination).set_index("team")

        assert cleaned.loc["Arsenal", "status"] == "sentinel"
        assert np.isnan(cleaned.loc["Arsenal", "market_value"])
        assert cleaned.loc["Liverpool", "status"] == "scraped"
        assert cleaned.loc["Liverpool", "market_value"] == pytest.approx(794.8)

    def test_does_not_impute(self, tmp_path) -> None:
        """Missing values stay missing. Imputation is the model's decision, and a
        flag that survives it is more useful than a silently filled number."""
        source, destination = self._write(
            tmp_path, [["A", "0.28"], ["B", "10"], ["C", "20"], ["D", "30"]]
        )
        cleaned = clean_market_values(source, destination).set_index("team")
        assert np.isnan(cleaned.loc["A", "market_value"])

    def test_median_is_not_the_fractional_part(self, tmp_path) -> None:
        """Regression: the legacy code computed ``median - int(median)``, turning
        a 2.28m median into 0.28 and stamping it onto 938 teams."""
        source, destination = self._write(tmp_path, [["A", "2.0"], ["B", "2.28"], ["C", "3.0"]])
        cleaned = clean_market_values(source, destination)
        usable = cleaned.loc[cleaned["status"] == "scraped", "market_value"]
        assert usable.median() == pytest.approx(2.28)
        assert usable.median() > 1

    def test_drops_duplicate_teams(self, tmp_path) -> None:
        source, destination = self._write(tmp_path, [["A", "10"], ["A", "20"], ["B", "5"]])
        cleaned = clean_market_values(source, destination)
        assert len(cleaned) == 2
        assert cleaned.set_index("team").loc["A", "market_value"] == pytest.approx(10.0)

    def test_zero_is_missing_not_a_value(self, tmp_path) -> None:
        source, destination = self._write(tmp_path, [["A", "0"], ["B", "5"]])
        cleaned = clean_market_values(source, destination).set_index("team")
        assert cleaned.loc["A", "status"] == "zero"
        assert np.isnan(cleaned.loc["A", "market_value"])
