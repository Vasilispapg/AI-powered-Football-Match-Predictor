"""Estimator components shared by training and prediction.

These live outside ``train.py`` deliberately. A class defined in the module that
happens to be ``__main__`` pickles as ``__main__.ClassName``, so a model saved by
``python -m football.train`` could not be loaded by ``python -m football.predict``
-- unpickling would look for the class on the wrong module and fail with
``AttributeError``. Defining them here gives every saved model a stable, portable
import path.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing in the training fold.

    ``HistGradientBoostingClassifier`` handles missing values natively but
    raises ``ValueError: window shape cannot be larger than input array shape``
    when a column has *no* observed value at all: its binner calls
    ``sliding_window_view(distinct_values, 2)``, and an all-NaN column yields
    zero distinct values.

    That is not hypothetical here. The betting-odds columns do not exist before
    2023-07-28, when the second data source starts, so every earlier
    walk-forward fold trains on a frame where they are entirely empty.

    Columns to keep are chosen at ``fit`` time and reused at ``transform`` time,
    so training and inference always agree on the shape.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.keep_ = [column for column in frame.columns if frame[column].notna().any()]
        self.dropped_ = [column for column in frame.columns if column not in self.keep_]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]
