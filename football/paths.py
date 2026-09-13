"""Filesystem layout for the project.

Every path used anywhere in the codebase is derived from ``ROOT`` so that
scripts behave identically regardless of the current working directory. The
original scripts mixed hardcoded absolute paths (``C:\\Users\\vasil\\Downloads\\...``)
with root-relative ones, which meant no single working directory could run the
whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Read-only inputs -------------------------------------------------------
# Output of the scrapers. This is the only match data that cannot be
# regenerated without re-scraping, so nothing in the pipeline ever writes here.
MATCHES_SCRAPED = ROOT / "matches_detailed_processed"

# --- Generated data ---------------------------------------------------------
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"

# Frozen snapshot of the scraped market values. The legacy pipeline rewrote
# ``marketValue/teams_market_value.csv`` in place on every run, which is how the
# original scraped values were lost; this copy is written once and then only
# read.
MARKET_VALUES_SCRAPED = RAW / "market_values_scraped.csv"

MATCHES_MERGED = INTERIM / "matches_merged.csv"
MATCHES_FILTERED = INTERIM / "matches_filtered.csv"
MARKET_VALUES_CLEAN = INTERIM / "market_values_clean.csv"
RESCRAPE_TARGETS = INTERIM / "market_value_rescrape_targets.csv"
DUPLICATE_TEAM_CANDIDATES = INTERIM / "duplicate_team_candidates.csv"

# Phase 2 output: matches joined to market values. Phase 3 reads this.
MATCHES = PROCESSED / "matches.csv"

TEAM_ALIASES = ROOT / "football" / "teams" / "aliases.csv"


def ensure_dirs() -> None:
    """Create the generated-data directories if they do not exist."""
    for directory in (RAW, INTERIM, PROCESSED, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
