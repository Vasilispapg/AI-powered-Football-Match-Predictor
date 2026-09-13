"""Run the data pipeline end to end.

    python -m football.pipeline

Reads only ``matches_detailed_processed/`` and a frozen snapshot of the scraped
market values; writes everything else under ``data/``. Nothing in here mutates
its own inputs -- the legacy pipeline rewrote ``teams_market_value.csv`` in
place on every run, which is how the original scraped values were lost.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

import pandas as pd

from football import paths
from football.data.audit import audit
from football.data.filter import filter_matches
from football.data.merge import merge_matches
from football.marketvalue.clean import clean_market_values, write_rescrape_targets
from football.marketvalue.join import build_dataset, write_duplicate_candidates

log = logging.getLogger(__name__)

#: The scraped market values still live in the legacy location. They are copied
#: into data/raw/ once and then treated as read-only.
LEGACY_MARKET_VALUES = paths.ROOT / "marketValue" / "teams_market_value.csv"


def snapshot_market_values(force: bool = False) -> None:
    """Copy the scraped market-value table into ``data/raw/`` if not already there."""
    if paths.MARKET_VALUES_SCRAPED.exists() and not force:
        log.info("Using existing snapshot %s", paths.MARKET_VALUES_SCRAPED)
        return
    if not LEGACY_MARKET_VALUES.exists():
        raise FileNotFoundError(
            f"No market values at {LEGACY_MARKET_VALUES} and no snapshot at "
            f"{paths.MARKET_VALUES_SCRAPED}"
        )
    paths.MARKET_VALUES_SCRAPED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_MARKET_VALUES, paths.MARKET_VALUES_SCRAPED)
    log.info("Snapshotted %s -> %s", LEGACY_MARKET_VALUES, paths.MARKET_VALUES_SCRAPED)


def run() -> pd.DataFrame:
    """Run every stage and return the processed dataset."""
    paths.ensure_dirs()
    snapshot_market_values()

    log.info("=== Stage 1/4: merge ===")
    merge_matches()

    log.info("=== Stage 2/4: filter ===")
    matches = filter_matches()

    log.info("=== Stage 3/4: clean market values ===")
    market_values = clean_market_values()
    write_rescrape_targets(matches, market_values)

    log.info("=== Stage 4/4: join ===")
    dataset = build_dataset()
    write_duplicate_candidates(matches)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-audit", action="store_true", help="skip the audit report at the end")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    run()

    if args.no_audit:
        return

    print("\n" + "=" * 72)
    failures = audit(paths.MATCHES)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
