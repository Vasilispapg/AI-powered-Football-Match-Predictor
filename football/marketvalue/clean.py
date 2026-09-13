"""Clean the scraped team market-value table.

Two bugs in the legacy ``marketValue/fix_the_price.py`` are fixed here.

**The billion bug.** The old suffix table mapped ``'b' -> 1e3`` and tested
``value.endswith(suffix)``. Transfermarkt writes billions as ``"1.24bn"``, which
ends in ``"n"``, so no suffix ever matched, ``float("1.24bn")`` raised, the
function returned ``None`` and the value was stored as ``0``. Every club worth
EUR 1bn or more was silently zeroed -- which is why the scraped table has a hard
ceiling at 998 with fifteen clubs piled up in 900-999, and why Arsenal, Man City
and PSG (the three clubs over a billion in that window) all sit at the sentinel.

**The median bug.** The old code computed ``median - int(median)``, keeping only
the *fractional* part, so a EUR 2.28m median was written as ``0.28``. That value
was stamped onto 938 of 3,919 teams.

This module does not impute. It parses what can be parsed, marks the rest as
missing with a reason, and leaves the choice of imputation to the modelling
stage -- scikit-learn's gradient boosting handles NaN natively, and a flag that
survives imputation is more useful to the model than a silently filled number.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from football import paths
from football.teams.normalise import search_query

log = logging.getLogger(__name__)

#: Value the legacy pipeline stamped onto every unresolved market value.
#: 938 teams carry it. Real values this low exist (398 teams sit below it), but
#: not 938 of them at exactly this figure.
LEGACY_SENTINEL = 0.28

#: Multipliers to convert a suffixed figure into millions of euros. Checked
#: longest-first so "bn" wins over "b" -- the bug that started all this.
SUFFIX_MULTIPLIERS: dict[str, float] = {
    "bn": 1_000.0,
    "b": 1_000.0,
    "mil": 1.0,
    "m": 1.0,
    "th": 1e-3,
    "k": 1e-3,
}

_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_market_value(raw: object) -> float | None:
    """Parse a Transfermarkt market value into millions of euros.

    >>> parse_market_value("1.24bn")
    1240.0
    >>> parse_market_value("996.00m")
    996.0
    >>> parse_market_value("125k")
    0.125
    >>> parse_market_value("794.8")
    794.8
    >>> parse_market_value("-") is None
    True
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None

    text = str(raw).strip().lower()
    for symbol in ("€", "eur", "£", "$", " ", "\xa0"):
        text = text.replace(symbol, "")
    text = text.replace(",", "")
    if not text or text in {"-", "n/a", "na", "none"}:
        return None

    for suffix in sorted(SUFFIX_MULTIPLIERS, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            if _NUMERIC.match(number):
                return float(number) * SUFFIX_MULTIPLIERS[suffix]
            return None

    if _NUMERIC.match(text):
        return float(text)
    return None


def clean_market_values(
    source: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Parse and validate the scraped market-value table.

    Returns a frame of ``team``, ``market_value`` (NaN where unknown) and
    ``status`` (``scraped`` / ``sentinel`` / ``zero`` / ``unparseable``).
    """
    source = source or paths.MARKET_VALUES_SCRAPED
    destination = destination or paths.MARKET_VALUES_CLEAN

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame = frame.rename(columns={"Team Name": "team", "Market Value": "raw_value"})
    if "team" not in frame or "raw_value" not in frame:
        raise ValueError(f"{source} must have 'Team Name' and 'Market Value' columns")

    frame["team"] = frame["team"].str.strip()
    frame = frame[frame["team"] != ""]

    duplicates = int(frame["team"].duplicated().sum())
    if duplicates:
        log.warning("Dropping %d duplicate team rows (keeping the first)", duplicates)
        frame = frame.drop_duplicates(subset="team", keep="first")

    parsed = frame["raw_value"].map(parse_market_value)

    status = pd.Series("scraped", index=frame.index, dtype=object)
    status[parsed.isna()] = "unparseable"
    status[parsed.notna() & (parsed == 0)] = "zero"
    # Exact equality is the right test: the sentinel was written as a literal.
    status[parsed.notna() & np.isclose(parsed.astype(float), LEGACY_SENTINEL)] = "sentinel"

    market_value = parsed.astype(float).where(status == "scraped", other=np.nan)

    cleaned = (
        pd.DataFrame({"team": frame["team"], "market_value": market_value, "status": status})
        .sort_values("team", kind="stable")
        .reset_index(drop=True)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False, encoding="utf-8")

    usable = cleaned["market_value"].notna()
    log.info("Cleaned %d teams", len(cleaned))
    for name, count in cleaned["status"].value_counts().items():
        log.info("  %-12s %6d (%5.1f%%)", name, count, 100 * count / len(cleaned))
    if usable.any():
        log.info(
            "  usable values: median %.2fm, max %.1fm",
            cleaned.loc[usable, "market_value"].median(),
            cleaned.loc[usable, "market_value"].max(),
        )
    log.info("Wrote %s", destination)

    return cleaned


def write_rescrape_targets(
    matches: pd.DataFrame,
    market_values: pd.DataFrame,
    destination: Path | None = None,
) -> pd.DataFrame:
    """List the teams whose market value must be re-scraped, worst first.

    Ordered by how many matches each team appears in, so the top of the file is
    where re-scraping buys the most. ``search_query`` is what to type into
    Transfermarkt -- see ``football/teams/aliases.csv``.
    """
    destination = destination or paths.RESCRAPE_TARGETS

    appearances = pd.concat([matches["home_team"], matches["away_team"]])
    counts = appearances.value_counts().rename("match_count")

    missing = market_values[market_values["market_value"].isna()].copy()
    missing = missing.join(counts, on="team")
    missing["match_count"] = missing["match_count"].fillna(0).astype(int)
    missing["search_query"] = missing["team"].map(search_query)
    missing = missing.sort_values(["match_count", "team"], ascending=[False, True]).reset_index(
        drop=True
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    missing[["team", "search_query", "status", "match_count"]].to_csv(
        destination, index=False, encoding="utf-8"
    )

    affected = int(missing["match_count"].sum())
    log.info(
        "%d teams need a market value, covering %d team-appearances",
        len(missing),
        affected,
    )
    log.info("Wrote %s", destination)
    return missing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    clean_market_values()


if __name__ == "__main__":
    main()
