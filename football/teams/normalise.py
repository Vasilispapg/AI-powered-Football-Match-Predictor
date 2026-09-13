"""Team-name normalisation.

Two jobs, deliberately kept apart:

``normalise``
    A mechanical comparison key -- accents, punctuation, case and club-form
    tokens removed. Used to spot the *same* club written two ways inside one
    source.

``search_query``
    The string to type into Transfermarkt when re-scraping a market value. The
    match data is full of broadcast abbreviations (``M Haifa``, ``Def y
    Justicia``, ``Gimn La Plata``) that no search engine will resolve. Mechanical
    normalisation cannot fix those -- only the hand-maintained alias table in
    ``aliases.csv`` can.

The join between matches and market values does *not* need any of this: the
market-value table was seeded from the match data by the legacy ``f_to_mv``
step, so both sides already share one namespace and join on an exact key.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from football import paths

#: Tokens that denote the legal or sporting form of a club and carry no
#: identity of their own. Stripped only when they stand alone and something is
#: left afterwards, so "Sporting CP" keeps "sporting" but "FC Barcelona" becomes
#: "barcelona". Deliberately conservative: over-stripping merges distinct clubs,
#: which is far worse than leaving a name unmatched.
CLUB_FORM_TOKENS = frozenset(
    {
        "fc",
        "cf",
        "afc",
        "ac",
        "sc",
        "cd",
        "ud",
        "rc",
        "rcd",
        "ss",
        "ssc",
        "sv",
        "tsv",
        "tsg",
        "ssv",
        "msv",
        "vfb",
        "vfl",
        "fsv",
        "bsc",
        "spvgg",
        "fk",
        "nk",
        "hk",
        "sk",
        "bk",
        "ik",
        "if",
        "ff",
        "kf",
        "kv",
        "os",
        "club",
        "clube",
        "calcio",
        "cfr",
        "cs",
        "csa",
        "acf",
        "asd",
    }
)

#: Expanded before normalisation. Only unambiguous, purely orthographic
#: shortenings belong here -- anything requiring knowledge of *which* club is
#: meant belongs in aliases.csv.
ABBREVIATIONS = {
    "utd": "united",
    "intl": "international",
    "cty": "city",
    "acad": "academy",
}

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_STANDALONE_NUMBER = re.compile(r"^\d{1,4}$")


def strip_accents(text: str) -> str:
    """Remove diacritics: ``Beşiktaş`` -> ``Besiktas``, ``Stabæk`` -> ``Stabaek``."""
    # Handle ligatures that NFKD does not decompose on their own.
    for source, target in (
        ("ß", "ss"),
        ("æ", "ae"),
        ("ø", "o"),
        ("å", "a"),
        ("đ", "d"),
        ("ł", "l"),
    ):
        text = text.replace(source, target).replace(source.upper(), target.upper())
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalise(name: str) -> str:
    """Return a canonical comparison key for *name*.

    >>> normalise("1. FC Köln")
    'koln'
    >>> normalise("Man Utd")
    'man united'
    >>> normalise("Sporting CP")
    'sporting cp'
    """
    text = strip_accents(str(name)).lower()
    text = text.replace("&", " and ")
    text = _PUNCTUATION.sub(" ", text)
    tokens = _WHITESPACE.sub(" ", text).strip().split()
    tokens = [ABBREVIATIONS.get(token, token) for token in tokens]

    # Drop club-form tokens and bare founding years, but never everything.
    kept = [
        token
        for token in tokens
        if token not in CLUB_FORM_TOKENS and not _STANDALONE_NUMBER.match(token)
    ]
    return " ".join(kept) if kept else " ".join(tokens)


@lru_cache(maxsize=1)
def load_aliases(path: Path | None = None) -> dict[str, str]:
    """Load ``team name -> Transfermarkt search query`` from the alias table."""
    path = path or paths.TEAM_ALIASES
    if not path.exists():
        return {}
    aliases: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            team = (row.get("team") or "").strip()
            query = (row.get("search_query") or "").strip()
            if team and query:
                aliases[team] = query
    return aliases


def search_query(name: str, aliases: dict[str, str] | None = None) -> str:
    """Return the string to search Transfermarkt with for *name*.

    Falls back to the name itself when no alias is recorded, which is correct
    for the large majority of clubs whose broadcast name is also their real one.
    """
    aliases = load_aliases() if aliases is None else aliases
    if name in aliases:
        return aliases[name]
    # Try the normalised form too, so one alias covers punctuation variants.
    normalised = normalise(name)
    for team, query in aliases.items():
        if normalise(team) == normalised:
            return query
    return name


def find_duplicate_candidates(names: list[str]) -> dict[str, list[str]]:
    """Group *names* that normalise to the same key.

    A non-empty result means one club is present under several spellings, which
    would split its match history in two and silently halve its form features.
    """
    groups: dict[str, list[str]] = {}
    for name in sorted(set(names)):
        groups.setdefault(normalise(name), []).append(name)
    return {key: group for key, group in groups.items() if len(group) > 1}
