"""Polite browsing: rate limiting, robots.txt, retries and a Selenium 4 driver.

The originals had none of this. ``marketValue.py`` requested as fast as Selenium
allowed and handled the resulting 403s by writing a zero, which is how 938 teams
ended up without a market value.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

SELECTORS_PATH = Path(__file__).with_name("selectors.json")

#: Minimum seconds between requests to one host. Deliberately slow.
DEFAULT_DELAY = 2.5

#: Extra random delay on top, so requests are not perfectly periodic.
DEFAULT_JITTER = 1.0


def load_selectors(path: Path | None = None) -> dict:
    """Load the selector configuration."""
    path = path or SELECTORS_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {key: value for key, value in config.items() if not key.startswith("_")}


class Throttle:
    """Enforces a minimum gap between requests, per host."""

    def __init__(self, delay: float = DEFAULT_DELAY, jitter: float = DEFAULT_JITTER):
        self.delay = delay
        self.jitter = jitter
        self._last: dict[str, float] = {}

    def wait(self, url: str, *, clock=time.monotonic, sleep=time.sleep) -> float:
        """Block until it is polite to request *url*. Returns seconds waited."""
        host = urlparse(url).netloc
        now = clock()
        earliest = self._last.get(host, 0.0) + self.delay
        waited = 0.0
        if now < earliest:
            waited = earliest - now
            sleep(waited)
        if self.jitter:
            # Jitter only de-synchronises the request pattern; it is not a
            # security control, so the fast PRNG is the right choice.
            extra = random.uniform(0, self.jitter)  # noqa: S311
            sleep(extra)
            waited += extra
        self._last[host] = clock()
        return waited


class RobotsPolicy:
    """Checks robots.txt before fetching.

    A site's terms of service are a separate matter from robots.txt, and this
    checks only the latter. Respecting robots.txt is necessary, not sufficient.
    """

    def __init__(self, user_agent: str = "*"):
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _parser(self, url: str):
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._parsers:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(f"{origin}/robots.txt")
            try:
                parser.read()
            except Exception as error:  # network failure, not a policy decision
                log.warning("Could not read robots.txt for %s: %s", origin, error)
                self._parsers[origin] = None
            else:
                self._parsers[origin] = parser
        return self._parsers[origin]

    def allows(self, url: str) -> bool:
        """True if robots.txt permits fetching *url*, or could not be read."""
        parser = self._parser(url)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)


@dataclass
class Failures:
    """Counts what went wrong, by reason.

    Replaces the bare ``except: pass`` blocks. A run that collected nothing and
    a run that collected everything used to look identical in the logs.
    """

    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)

    def record(self, reason: str, detail: str = "") -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if detail and reason not in self.examples:
            self.examples[reason] = detail

    def total(self) -> int:
        return sum(self.counts.values())

    def report(self) -> str:
        if not self.counts:
            return "No failures."
        lines = [f"{self.total()} failure(s):"]
        for reason, count in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {count:6d}  {reason}")
            if reason in self.examples:
                lines.append(f"          e.g. {self.examples[reason][:120]}")
        return "\n".join(lines)


def make_driver(headless: bool = True, page_load_timeout: int = 45):
    """Create a Firefox WebDriver using the Selenium 4 API.

    The originals used ``executable_path=`` and ``options.headless``, both
    removed in Selenium 4.3; on any current Selenium they raise immediately.
    Selenium 4.6+ resolves geckodriver itself, so no driver path is needed.
    """
    from selenium import webdriver

    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("-headless")
    options.set_preference("permissions.default.image", 2)  # skip images
    options.set_preference("dom.webnotifications.enabled", False)

    driver = webdriver.Firefox(options=options)
    driver.set_page_load_timeout(page_load_timeout)
    return driver


class PoliteBrowser:
    """A driver wrapped in rate limiting, robots.txt checks and retries."""

    def __init__(
        self,
        driver=None,
        throttle: Throttle | None = None,
        robots: RobotsPolicy | None = None,
        failures: Failures | None = None,
        max_retries: int = 3,
        respect_robots: bool = True,
    ):
        self.driver = driver
        self.throttle = throttle or Throttle()
        self.robots = robots or RobotsPolicy()
        self.failures = failures or Failures()
        self.max_retries = max_retries
        self.respect_robots = respect_robots

    def get(self, url: str) -> bool:
        """Navigate to *url*. Returns False (and records why) on failure."""
        if self.respect_robots and not self.robots.allows(url):
            self.failures.record("blocked by robots.txt", url)
            log.warning("robots.txt disallows %s -- skipping", url)
            return False

        for attempt in range(1, self.max_retries + 1):
            self.throttle.wait(url)
            try:
                self.driver.get(url)
            except Exception as error:
                reason = type(error).__name__
                if attempt == self.max_retries:
                    self.failures.record(f"navigation failed ({reason})", url)
                    log.warning("Giving up on %s after %d attempts", url, attempt)
                    return False
                # Back off before trying again.
                time.sleep(self.throttle.delay * attempt)
            else:
                return True
        return False

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception as error:
                log.debug("Driver shutdown: %s", error)
