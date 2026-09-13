"""Re-scrape the missing team market values.

    python -m football.scrape.transfermarkt --check
    python -m football.scrape.transfermarkt --limit 50

Driven by ``data/interim/market_value_rescrape_targets.csv``, which the pipeline
writes ordered by how many matches each team appears in -- so a partial run
fixes the most valuable gaps first. 938 teams (24%) currently have no value,
including Arsenal, Man City and PSG.

Two rules this module keeps that the original broke:

* **It never overwrites its input.** The original rewrote
  ``teams_market_value.csv`` in place on every run, so the raw scraped strings
  were lost and the placeholder values got baked back in as if they were real.
  Results go to a separate file and are merged explicitly.
* **A failure is a failure, not a zero.** The original caught every exception --
  including 403s from hammering the site -- and wrote ``0``, which the next
  stage then replaced with a plausible-looking number.

Transfermarkt's terms of service prohibit automated collection.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football import paths
from football.marketvalue.clean import parse_market_value
from football.scrape.browser import Failures, PoliteBrowser, Throttle, load_selectors, make_driver

log = logging.getLogger(__name__)

#: Results land here, never on top of the raw snapshot.
RESCRAPED = paths.RAW / "market_values_rescraped.csv"

#: Incremental checkpoint, so an interrupted run resumes without re-requesting.
CHECKPOINT = paths.INTERIM / "market_value_rescrape_progress.json"


@dataclass
class Scraped:
    team: str
    query: str
    raw_value: str | None
    market_value: float | None
    status: str


def load_targets(path: Path | None = None) -> pd.DataFrame:
    path = path or paths.RESCRAPE_TARGETS
    if not path.exists():
        raise FileNotFoundError(
            f"No re-scrape list at {path}. Run `python -m football.pipeline` first."
        )
    return pd.read_csv(path)


def load_checkpoint(path: Path | None = None) -> dict[str, dict]:
    path = path or CHECKPOINT
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(state: dict[str, dict], path: Path | None = None) -> None:
    path = path or CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=1)


def find_club_value(driver, selectors: dict, failures: Failures) -> str | None:
    """Read the market value from a Transfermarkt quick-search results page.

    The page lists several result sections (players, clubs, officials); the one
    headed "club" is the one we want.
    """
    from selenium.webdriver.common.by import By

    try:
        main = driver.find_element(By.CSS_SELECTOR, selectors["results_main"])
    except Exception as error:
        failures.record("no results container", str(error))
        return None

    headings = main.find_elements(By.CSS_SELECTOR, selectors["section_heading"])
    wanted = selectors["club_section_text"].lower()
    for heading in headings:
        if wanted not in heading.text.strip().lower():
            continue
        try:
            parent = heading.find_element(By.XPATH, "..")
            text = parent.find_element(By.CSS_SELECTOR, selectors["market_value"]).text.strip()
        except Exception as error:
            failures.record("club section has no value cell", str(error))
            return None
        return text or None

    failures.record("no club section in search results")
    return None


def scrape_team(browser: PoliteBrowser, team: str, query: str, selectors: dict) -> Scraped:
    """Look up one club. Never returns a fabricated number."""
    url = selectors["search_url"].format(query=query.replace(" ", "%20"))
    if not browser.get(url):
        return Scraped(team, query, None, None, "navigation failed")

    raw = find_club_value(browser.driver, selectors, browser.failures)
    if raw is None:
        return Scraped(team, query, None, None, "not found")

    value = parse_market_value(raw.replace("€", ""))
    if value is None:
        browser.failures.record("value did not parse", raw)
        return Scraped(team, query, raw, None, "unparseable")
    return Scraped(team, query, raw, value, "scraped")


def merge_results(
    scraped: list[Scraped],
    base: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    """Fold newly scraped values into the cleaned market-value table.

    Writes a new file; the raw snapshot is left untouched.
    """
    base = base or paths.MARKET_VALUES_CLEAN
    destination = destination or RESCRAPED

    existing = (
        pd.read_csv(base)
        if base.exists()
        else pd.DataFrame(columns=["team", "market_value", "status"])
    )
    found = {item.team: item for item in scraped if item.market_value is not None}

    merged = existing.copy()
    merged["market_value"] = merged.apply(
        lambda row: (
            found[row["team"]].market_value
            if row["team"] in found and pd.isna(row["market_value"])
            else row["market_value"]
        ),
        axis=1,
    )
    merged["status"] = merged.apply(
        lambda row: (
            "rescraped" if row["team"] in found and row["status"] != "scraped" else row["status"]
        ),
        axis=1,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False, encoding="utf-8")
    log.info("Merged %d new value(s) -> %s", len(found), destination)
    return merged


def run(
    limit: int | None = None,
    *,
    headless: bool = True,
    delay: float = 3.0,
    respect_robots: bool = True,
) -> list[Scraped]:
    """Work through the re-scrape list, most-played teams first."""
    selectors = load_selectors()["transfermarkt"]
    targets = load_targets()
    checkpoint = load_checkpoint()

    pending = [row for _, row in targets.iterrows() if row["team"] not in checkpoint]
    if limit:
        pending = pending[:limit]

    log.info(
        "%d team(s) on the list, %d already attempted, %d this run",
        len(targets),
        len(checkpoint),
        len(pending),
    )
    if not pending:
        return []

    browser = PoliteBrowser(
        driver=make_driver(headless=headless),
        throttle=Throttle(delay=delay),
        respect_robots=respect_robots,
    )
    results: list[Scraped] = []
    try:
        for index, row in enumerate(pending, start=1):
            team = str(row["team"])
            query = str(row.get("search_query") or team)
            result = scrape_team(browser, team, query, selectors)
            results.append(result)
            checkpoint[team] = {
                "query": query,
                "raw_value": result.raw_value,
                "market_value": result.market_value,
                "status": result.status,
            }
            log.info(
                "[%d/%d] %-28s %-12s %s",
                index,
                len(pending),
                team[:28],
                result.status,
                "" if result.market_value is None else f"{result.market_value:,.2f}m",
            )
            if index % 10 == 0:
                save_checkpoint(checkpoint)
    finally:
        save_checkpoint(checkpoint)
        browser.close()

    print()
    print(browser.failures.report())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="stop after N teams")
    parser.add_argument("--no-headless", action="store_true", help="show the browser")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    parser.add_argument("--ignore-robots", action="store_true", help="proceed despite robots.txt")
    parser.add_argument(
        "--check", action="store_true", help="show the top of the re-scrape list and exit"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="merge an existing checkpoint into the values table without scraping",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.check:
        targets = load_targets()
        print(f"{len(targets)} team(s) need a market value. Top 20 by match count:\n")
        print(targets.head(20).to_string(index=False))
        return

    if args.merge_only:
        checkpoint = load_checkpoint()
        scraped = [
            Scraped(team, d["query"], d["raw_value"], d["market_value"], d["status"])
            for team, d in checkpoint.items()
        ]
        merge_results(scraped)
        return

    log.warning(
        "Transfermarkt's terms of service prohibit automated collection. "
        "Requests are rate limited to one per %.1fs and robots.txt is %s.",
        args.delay,
        "ignored" if args.ignore_robots else "honoured",
    )

    results = run(
        limit=args.limit,
        headless=not args.no_headless,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
    )
    if results:
        merge_results(results)
        print(
            f"\nMerged into {RESCRAPED}.\n"
            f"Review it, then copy over {paths.MARKET_VALUES_CLEAN.name} and re-run:\n"
            f"    python -m football.run_all"
        )


if __name__ == "__main__":
    main()
