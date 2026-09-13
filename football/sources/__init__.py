"""External data sources.

The goal.com scrape in ``football.scrape`` remains the only source for the 271
competitions in the original dataset, but it stops at 2023-08-27 and its
selectors are hashed build artefacts that no longer match.

``footballdata`` reads football-data.co.uk instead: static CSVs, free for
personal use, updated several times a week, covering ~22 major European leagues
with richer per-match data than the scrape ever collected -- including
pre-match betting odds, which are the strongest single predictor available for
football outcomes.
"""
