"""Football match outcome prediction.

Pipeline stages, in order:

* ``football.data``        merge and filter the scraped match CSVs
* ``football.marketvalue`` clean team market values and join them to matches
* ``football.teams``       team-name normalisation and search-query building

Run the whole thing with ``python -m football.pipeline``.
"""

__version__ = "0.1.0"
