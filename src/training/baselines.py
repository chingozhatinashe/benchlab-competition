"""Reference models for training.evaluate. They set the score a real model must beat."""

import numpy as np

from training.evaluate import HORIZON


class Persistence:
    """Repeat the last observed hourly speed for all 72 hours."""

    def fit(self, train):
        # Fallback for a history with no valid hours at all.
        self.fallback = float(np.nanmean(train.to_numpy()))
        return self

    def predict(self, history, t0):
        observed = history.dropna()
        last = observed.iloc[-1] if len(observed) else self.fallback
        return np.full(HORIZON, last)
