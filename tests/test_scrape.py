"""Tests for the scraping layer.

Only the parts that do not need a browser: rate limiting, robots.txt, failure
accounting, selector configuration, and the pure parsing helpers. Live
collection is not exercised -- it depends on third-party markup that changes
without notice, which is exactly why the selectors live in a config file.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from football.scrape.browser import Failures, RobotsPolicy, Throttle, load_selectors
from football.scrape.goal import (
    COLUMNS,
    Match,
    generate_dates,
    output_path,
    parse_score,
    split_competition,
    write_day,
)


class TestThrottle:
    def test_first_request_does_not_wait(self) -> None:
        throttle = Throttle(delay=5.0, jitter=0)
        slept: list[float] = []

        throttle.wait("https://example.test/a", clock=lambda: 100.0, sleep=slept.append)
        assert slept == []

    def test_second_request_to_the_same_host_waits(self) -> None:
        throttle = Throttle(delay=5.0, jitter=0)
        slept: list[float] = []
        clock = iter([100.0, 100.0, 101.0, 101.0])

        throttle.wait("https://example.test/a", clock=lambda: next(clock), sleep=slept.append)
        throttle.wait("https://example.test/b", clock=lambda: next(clock), sleep=slept.append)

        assert slept and slept[0] == pytest.approx(4.0)

    def test_different_hosts_are_independent(self) -> None:
        throttle = Throttle(delay=5.0, jitter=0)
        slept: list[float] = []
        clock = iter([100.0, 100.0, 100.0, 100.0])

        throttle.wait("https://a.test/x", clock=lambda: next(clock), sleep=slept.append)
        throttle.wait("https://b.test/x", clock=lambda: next(clock), sleep=slept.append)

        assert slept == []

    def test_jitter_adds_delay(self) -> None:
        throttle = Throttle(delay=0, jitter=1.0)
        slept: list[float] = []

        throttle.wait("https://example.test/", clock=lambda: 0.0, sleep=slept.append)
        assert len(slept) == 1
        assert 0.0 <= slept[0] <= 1.0


class TestRobotsPolicy:
    def test_allows_when_robots_cannot_be_read(self, monkeypatch) -> None:
        """A network failure fetching robots.txt is not a prohibition."""
        policy = RobotsPolicy()
        monkeypatch.setattr(policy, "_parser", lambda url: None)
        assert policy.allows("https://example.test/anything")

    def test_honours_a_disallow(self) -> None:
        import urllib.robotparser

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /private"])

        policy = RobotsPolicy()
        policy._parsers["https://example.test"] = parser

        assert policy.allows("https://example.test/public")
        assert not policy.allows("https://example.test/private/page")


class TestFailures:
    def test_counts_by_reason(self) -> None:
        failures = Failures()
        failures.record("timeout", "https://a.test")
        failures.record("timeout", "https://b.test")
        failures.record("403")

        assert failures.counts == {"timeout": 2, "403": 1}
        assert failures.total() == 3

    def test_keeps_the_first_example_only(self) -> None:
        failures = Failures()
        failures.record("timeout", "first")
        failures.record("timeout", "second")

        assert failures.examples["timeout"] == "first"

    def test_reports_silence_when_clean(self) -> None:
        assert Failures().report() == "No failures."

    def test_report_names_every_reason(self) -> None:
        failures = Failures()
        failures.record("timeout")
        failures.record("blocked")

        report = failures.report()
        assert "timeout" in report and "blocked" in report


class TestSelectorConfig:
    def test_loads_both_sites(self) -> None:
        selectors = load_selectors()
        assert "goal" in selectors
        assert "transfermarkt" in selectors

    def test_comments_are_stripped(self) -> None:
        assert not any(key.startswith("_") for key in load_selectors())

    def test_goal_has_every_selector_the_scraper_uses(self) -> None:
        goal = load_selectors()["goal"]
        for key in (
            "results_url",
            "competition_block",
            "competition_name",
            "match_row",
            "home_team",
            "away_team",
            "score",
            "match_link",
            "stats_group",
            "stats_value",
            "predictor",
            "vote_wrapper",
            "vote_count",
        ):
            assert key in goal, key

    def test_transfermarkt_has_every_selector(self) -> None:
        site = load_selectors()["transfermarkt"]
        for key in ("search_url", "results_main", "section_heading", "market_value"):
            assert key in site, key

    def test_config_is_valid_json(self) -> None:
        from football.scrape.browser import SELECTORS_PATH

        json.loads(SELECTORS_PATH.read_text(encoding="utf-8"))


class TestDateGeneration:
    def test_inclusive_of_both_ends(self) -> None:
        dates = generate_dates(date(2023, 1, 1), date(2023, 1, 5))
        assert len(dates) == 5
        assert dates[0] == date(2023, 1, 1)
        assert dates[-1] == date(2023, 1, 5)

    def test_single_day(self) -> None:
        assert generate_dates(date(2023, 1, 1), date(2023, 1, 1)) == [date(2023, 1, 1)]

    def test_rejects_a_reversed_range(self) -> None:
        with pytest.raises(ValueError, match="before start"):
            generate_dates(date(2023, 5, 1), date(2023, 1, 1))


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2 - 1", ("2", "1")),
            ("0-0", ("0", "0")),
            ("10 - 3", ("10", "3")),
        ],
    )
    def test_parses_scores(self, text: str, expected: tuple[str, str]) -> None:
        assert parse_score(text) == expected

    @pytest.mark.parametrize("text", ["", "vs", "- ", "TBD", "Postponed", "a - b"])
    def test_rejects_non_scores(self, text: str) -> None:
        assert parse_score(text) == ("N/A", "N/A")

    def test_splits_competition_heading(self) -> None:
        assert split_competition("England - Premier League") == ("England", "Premier League")

    def test_handles_a_heading_with_no_country(self) -> None:
        assert split_competition("Club Friendlies") == ("N/A", "Club Friendlies")


class TestDayOutput:
    def test_header_matches_the_rows(self, tmp_path) -> None:
        """Regression: the original wrote a 7-column header for 8-column rows,
        omitting URL, so the next stage could not find the match links."""
        matches = [
            Match(
                date="2023-01-15",
                competition="Premier League",
                country="England",
                home_team="Arsenal",
                home_score="2",
                away_team="Liverpool",
                away_score="1",
                url="https://example.test/m",
            )
        ]
        destination = write_day(matches, date(2023, 1, 15), tmp_path)

        lines = destination.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",") == COLUMNS
        assert len(lines[1].split(",")) == len(COLUMNS)
        assert "https://example.test/m" in lines[1]

    def test_empty_day_still_writes_a_header(self, tmp_path) -> None:
        destination = write_day([], date(2023, 1, 15), tmp_path)
        assert destination.read_text(encoding="utf-8").strip() == ",".join(COLUMNS)

    def test_path_is_derived_from_the_date(self, tmp_path) -> None:
        assert output_path(date(2023, 1, 5), tmp_path).name == "matches_data_2023-01-05.csv"

    def test_empty_stats_and_votes_round_trip(self, tmp_path) -> None:
        """They must serialise as {} and [], which the feature parsers expect."""
        row = Match(date="2023-01-15").as_row()
        assert row["Stats"] == "{}"
        assert row["Votes"] == "[]"

    def test_populated_stats_and_votes_round_trip(self) -> None:
        from football.features import parse_stats, parse_votes

        match = Match(
            date="2023-01-15",
            home_team="A",
            away_team="B",
            stats={"Ball possession": ["60%", "40%"]},
            votes=[["A", "Draw", "B"], ["10", "5", "3"]],
        )
        row = match.as_row()

        assert parse_stats(row["Stats"])["possession"] == (60.0, 40.0)
        assert parse_votes(row["Votes"]) == (10, 5, 3)
