"""Tests for team-name normalisation."""

from __future__ import annotations

import pytest

from football.teams.normalise import (
    find_duplicate_candidates,
    normalise,
    search_query,
    strip_accents,
)


class TestStripAccents:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Beşiktaş", "Besiktas"),
            ("Stabæk", "Stabaek"),
            ("Köln", "Koln"),
            ("Atlético", "Atletico"),
            ("América MG", "America MG"),
            ("Arsenal", "Arsenal"),
        ],
    )
    def test_removes_diacritics(self, raw: str, expected: str) -> None:
        assert strip_accents(raw) == expected


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("FC Barcelona", "barcelona"),
            ("1. FC Köln", "koln"),
            ("Man Utd", "man united"),
            ("Real Madrid", "real madrid"),
            ("Borussia M'gladbach", "borussia m gladbach"),
            # "AS" is left alone: it is a common word as well as a club form,
            # and Transfermarkt resolves "AS Roma" perfectly well.
            ("AS Roma", "as roma"),
        ],
    )
    def test_produces_stable_key(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    def test_never_returns_empty(self) -> None:
        """A name made entirely of club-form tokens must keep them rather than
        normalise to nothing -- an empty key would collide with every other
        empty key."""
        assert normalise("FC") == "fc"
        assert normalise("1899") == "1899"

    def test_does_not_merge_distinct_clubs(self) -> None:
        """Over-stripping is worse than under-stripping: it silently fuses two
        clubs' histories."""
        assert normalise("Atletico Madrid") != normalise("Athletic Bilbao")
        assert normalise("Sporting CP") != normalise("Sporting Gijon")
        assert normalise("Manchester City") != normalise("Manchester United")

    def test_is_idempotent(self) -> None:
        for name in ("FC Barcelona", "1. FC Köln", "Man Utd"):
            assert normalise(normalise(name)) == normalise(name)


class TestSearchQuery:
    def test_uses_alias_when_present(self) -> None:
        aliases = {"M Haifa": "Maccabi Haifa"}
        assert search_query("M Haifa", aliases) == "Maccabi Haifa"

    def test_falls_back_to_the_name(self) -> None:
        assert search_query("Liverpool", {}) == "Liverpool"

    def test_alias_survives_punctuation_differences(self) -> None:
        aliases = {"R. Union SG": "Royale Union Saint-Gilloise"}
        assert search_query("R Union SG", aliases) == "Royale Union Saint-Gilloise"

    def test_shipped_alias_table_loads(self) -> None:
        """The real table must parse and cover the clubs the billion bug ate."""
        assert search_query("Arsenal") == "Arsenal FC"
        assert search_query("PSG") == "Paris Saint-Germain"


class TestDuplicateCandidates:
    def test_groups_spelling_variants(self) -> None:
        groups = find_duplicate_candidates(["Man Utd", "Man United", "Liverpool"])
        assert groups == {"man united": ["Man United", "Man Utd"]}

    def test_silent_when_names_are_distinct(self) -> None:
        assert find_duplicate_candidates(["Arsenal", "Chelsea", "Liverpool"]) == {}
