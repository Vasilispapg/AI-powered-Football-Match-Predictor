"""Collect match results, statistics and crowd votes.

    python -m football.scrape.goal --check                    # verify selectors
    python -m football.scrape.goal --start 2023-09-01 --end 2023-09-30

Run ``--check`` first. The selectors are hashed CSS-module class names captured
in 2023 and are regenerated on every frontend deploy, so they have very likely
stopped matching. ``--check`` loads one page and reports which ones still find
anything, instead of letting a full run produce a directory of empty CSVs.

goal.com's terms of service prohibit automated collection. The defaults here are
deliberately slow and robots.txt is honoured, but neither of those grants
permission -- that is yours to establish.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from football import paths
from football.scrape.browser import (
    Failures,
    PoliteBrowser,
    Throttle,
    load_selectors,
    make_driver,
)

log = logging.getLogger(__name__)

#: Column order written to each day's CSV. The original wrote a seven-column
#: header for eight-column rows, omitting "URL" -- so a re-scrape silently
#: produced files whose match links had no header, and the next stage could not
#: read them.
COLUMNS = [
    "Competition",
    "Country",
    "Home Team",
    "Home Score",
    "Away Team",
    "Away Score",
    "Date",
    "URL",
    "Votes",
    "Stats",
]


@dataclass
class Match:
    date: str
    competition: str = "N/A"
    country: str = "N/A"
    home_team: str = "N/A"
    home_score: str = "N/A"
    away_team: str = "N/A"
    away_score: str = "N/A"
    url: str = ""
    votes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def as_row(self) -> dict[str, str]:
        return {
            "Competition": self.competition,
            "Country": self.country,
            "Home Team": self.home_team,
            "Home Score": self.home_score,
            "Away Team": self.away_team,
            "Away Score": self.away_score,
            "Date": self.date,
            "URL": self.url,
            "Votes": repr(self.votes) if self.votes else "[]",
            "Stats": repr(self.stats) if self.stats else "{}",
        }


# --------------------------------------------------------------------------
# Pure helpers -- no browser, directly testable
# --------------------------------------------------------------------------


def generate_dates(start: Date, end: Date) -> list[Date]:
    """Every date from *start* to *end* inclusive."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def parse_score(text: str) -> tuple[str, str]:
    """Split a scoreline. Returns ``("N/A", "N/A")`` when it is not a result."""
    if not text or "-" not in text:
        return "N/A", "N/A"
    home, _, away = text.partition("-")
    home, away = home.strip(), away.strip()
    if home.isdigit() and away.isdigit():
        return home, away
    return "N/A", "N/A"


def split_competition(text: str) -> tuple[str, str]:
    """``"England - Premier League"`` -> ``("England", "Premier League")``."""
    if " - " not in text:
        return "N/A", text.strip() or "N/A"
    country, _, competition = text.partition(" - ")
    return country.strip() or "N/A", competition.strip() or "N/A"


def output_path(date: Date, directory: Path) -> Path:
    return directory / f"matches_data_{date:%Y-%m-%d}.csv"


def write_day(matches: list[Match], date: Date, directory: Path) -> Path:
    """Write one day's matches. Header and rows always agree."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = output_path(date, directory)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(match.as_row() for match in matches)
    return destination


# --------------------------------------------------------------------------
# Browser-driven extraction
# --------------------------------------------------------------------------


def extract_stats(driver, selectors: dict, failures: Failures) -> dict[str, list[str]]:
    """Read every statistic group on a match page.

    The original returned from *inside* its category loop, so only the first
    group was ever collected -- which is why ``Total shots`` and ``Ball
    possession`` appear in half as many rows as ``Fouls`` and ``Corners``.
    """
    from selenium.webdriver.common.by import By

    stats: dict[str, list[str]] = {}
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, f"{selectors['stats_group']} li")
    except Exception as error:
        failures.record("stats container not found", str(error))
        return stats

    for item in rows:
        try:
            name = item.find_element(By.CSS_SELECTOR, selectors["stats_heading"]).text.strip()
            values = [
                value.text.strip()
                for value in item.find_elements(By.CSS_SELECTOR, selectors["stats_value"])
            ]
        except Exception as error:
            failures.record("stat row unreadable", str(error))
            continue
        if name and values:
            stats[name] = values
    # No early return: every group is collected.
    return stats


def extract_votes(driver, selectors: dict, home: str, away: str, failures: Failures) -> list:
    """Read the crowd predictor. Returns ``[]`` when it is absent."""
    from selenium.webdriver.common.by import By

    try:
        predictor = driver.find_element(By.CSS_SELECTOR, selectors["predictor"])
    except Exception:
        failures.record("no predictor widget")
        return []

    counts: list[str] = []
    for wrapper in predictor.find_elements(By.CSS_SELECTOR, selectors["vote_wrapper"]):
        try:
            text = wrapper.find_element(By.CSS_SELECTOR, selectors["vote_count"]).text
        except Exception as error:
            # Expected: not every wrapper carries a count. The len check below
            # is what decides whether the widget was actually readable.
            log.debug("vote wrapper without a count: %s", error)
            continue
        counts.append(text.split(" ")[0])

    if len(counts) != 3:
        failures.record(f"expected 3 vote counts, got {len(counts)}")
        return []
    return [[home, "Draw", away], counts]


def accept_cookies(browser: PoliteBrowser, selectors: dict) -> None:
    """Dismiss the consent banner if present. Declines non-essential cookies
    where the banner offers that; otherwise clicks the primary action."""
    from selenium.webdriver.common.by import By

    for how, what in (
        (By.XPATH, "//button[contains(., 'Reject All') or contains(., 'Reject all')]"),
        (By.XPATH, selectors.get("cookie_accept", "")),
    ):
        if not what:
            continue
        try:
            element = browser.driver.find_element(how, what)
            element.click()
        except Exception as error:
            # Expected: only one of these banner variants is ever present.
            log.debug("consent banner selector %r did not work: %s", what, error)
            continue
        return


def fetch_day(browser: PoliteBrowser, date: Date, selectors: dict) -> list[Match]:
    """Collect every match listed for *date*."""
    from selenium.webdriver.common.by import By

    url = selectors["results_url"].format(date=f"{date:%Y-%m-%d}")
    if not browser.get(url):
        return []
    accept_cookies(browser, selectors)

    matches: list[Match] = []
    blocks = browser.driver.find_elements(By.CSS_SELECTOR, selectors["competition_block"])
    if not blocks:
        browser.failures.record("no competition blocks on results page", url)

    for block in blocks:
        try:
            heading = block.find_element(
                By.CSS_SELECTOR, selectors["competition_name"]
            ).text.strip()
            country, competition = split_competition(heading)
        except Exception:
            country, competition = "N/A", "N/A"
            browser.failures.record("competition heading unreadable", url)

        for row in block.find_elements(By.CSS_SELECTOR, selectors["match_row"]):
            match = Match(date=f"{date:%Y-%m-%d}", competition=competition, country=country)

            for attribute, selector in (
                ("home_team", selectors["home_team"]),
                ("away_team", selectors["away_team"]),
            ):
                try:
                    setattr(
                        match, attribute, row.find_element(By.CSS_SELECTOR, selector).text.strip()
                    )
                except Exception:
                    browser.failures.record(f"{attribute} unreadable", url)

            try:
                text = row.find_element(By.CSS_SELECTOR, selectors["score"]).text
                match.home_score, match.away_score = parse_score(text)
            except Exception:
                browser.failures.record("score unreadable", url)

            try:
                link = row.find_element(By.CSS_SELECTOR, selectors["match_link"])
                match.url = link.get_attribute("href") or ""
            except Exception:
                browser.failures.record("match link unreadable", url)

            matches.append(match)

    return matches


def fetch_details(browser: PoliteBrowser, match: Match, selectors: dict) -> None:
    """Visit a match page and attach its statistics and votes, in place."""
    if not match.url:
        browser.failures.record("no match URL to follow")
        return
    if not browser.get(match.url):
        return
    match.stats = extract_stats(browser.driver, selectors, browser.failures)
    match.votes = extract_votes(
        browser.driver, selectors, match.home_team, match.away_team, browser.failures
    )


def check_selectors(url: str, selectors: dict, headless: bool = True) -> dict[str, int]:
    """Load *url* and count what each selector matches. Diagnoses stale markup."""
    from selenium.webdriver.common.by import By

    browser = PoliteBrowser(driver=make_driver(headless=headless))
    found: dict[str, int] = {}
    try:
        if not browser.get(url):
            raise RuntimeError(f"could not load {url}")
        accept_cookies(browser, selectors)
        for name, selector in selectors.items():
            if not isinstance(selector, str) or selector.startswith(("http", "//")):
                continue
            try:
                found[name] = len(browser.driver.find_elements(By.CSS_SELECTOR, selector))
            except Exception as error:
                log.debug("selector %s is not valid CSS: %s", name, error)
                found[name] = -1
    finally:
        browser.close()
    return found


def collect(
    start: Date,
    end: Date,
    directory: Path,
    *,
    with_details: bool = True,
    headless: bool = True,
    delay: float = 2.5,
    respect_robots: bool = True,
) -> Failures:
    """Collect every day between *start* and *end*, skipping days already on disk.

    One day per file, written only once that day is complete, so an interrupted
    run resumes cleanly. Unlike the original there is no threading: two threads
    walking the same URL list in opposite directions duplicated every request.
    """
    selectors = load_selectors()["goal"]
    dates = [d for d in generate_dates(start, end) if not output_path(d, directory).exists()]
    if not dates:
        log.info("Every requested day is already on disk")
        return Failures()

    browser = PoliteBrowser(
        driver=make_driver(headless=headless),
        throttle=Throttle(delay=delay),
        respect_robots=respect_robots,
    )
    try:
        for index, date in enumerate(dates, start=1):
            log.info("[%d/%d] %s", index, len(dates), date)
            matches = fetch_day(browser, date, selectors)
            if with_details:
                for match in matches:
                    fetch_details(browser, match, selectors)
            destination = write_day(matches, date, directory)
            log.info("    %d match(es) -> %s", len(matches), destination.name)
    finally:
        browser.close()

    return browser.failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", help="last date, YYYY-MM-DD")
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.MATCHES_SCRAPED,
        help="output directory (default: matches_detailed_processed/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load one page and report which selectors still match, then exit",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="results only; skip the per-match statistics and votes pages",
    )
    parser.add_argument("--no-headless", action="store_true", help="show the browser")
    parser.add_argument("--delay", type=float, default=2.5, help="minimum seconds between requests")
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="proceed even where robots.txt disallows the path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    selectors = load_selectors()["goal"]

    if args.check:
        url = selectors["results_url"].format(date="2023-08-27")
        log.info("Checking selectors against %s", url)
        results = check_selectors(url, selectors, headless=not args.no_headless)
        print(json.dumps(results, indent=2))
        stale = [name for name, count in results.items() if count == 0]
        if stale:
            print(f"\n{len(stale)} selector(s) matched nothing: {', '.join(stale)}")
            print("Repair football/scrape/selectors.json before collecting.")
        else:
            print("\nAll selectors matched at least one element.")
        return

    if not args.start or not args.end:
        parser.error("--start and --end are required unless --check is given")

    log.warning(
        "goal.com's terms of service prohibit automated collection. Requests are "
        "rate limited to one per %.1fs and robots.txt is %s.",
        args.delay,
        "ignored" if args.ignore_robots else "honoured",
    )

    failures = collect(
        Date.fromisoformat(args.start),
        Date.fromisoformat(args.end),
        args.out,
        with_details=not args.no_details,
        headless=not args.no_headless,
        delay=args.delay,
        respect_robots=not args.ignore_robots,
    )
    print()
    print(failures.report())


if __name__ == "__main__":
    main()
