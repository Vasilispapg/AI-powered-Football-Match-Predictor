"""Data collection.

Rebuilt from ``webScrapper/`` and ``marketValue/marketValue.py``. The originals
worked -- they produced the 43,423-match dataset this project runs on -- but had
four structural problems this package fixes:

* **Selectors were hardcoded.** They are hashed CSS-module build artefacts
  (``row_row__UQmGm``), regenerated on every frontend deploy, so the scrapers
  break on someone else's release schedule and fixing them meant editing Python.
  They now live in ``selectors.json``.
* **Failures were silent.** Bare ``except:`` turned every timeout, block and
  markup change into the string ``"N/A"``. A third of the dataset has no match
  statistics and there is no record of why. Every failure is now counted and
  logged by reason.
* **No rate limiting.** The original hit Transfermarkt as fast as Selenium
  allowed and handled the resulting 403s by writing zeros.
* **Duplicated work.** ``football.py`` ran the same URL list forwards and
  backwards in two threads, and its "already fetched" check looked in the wrong
  directory, so nothing was ever skipped.

Both sites prohibit automated collection in their terms of service. Read them
before running anything here; the polite defaults in this package reduce load
but do not grant permission.
"""
