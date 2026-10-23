"""Walk-forward validation folds over the training years (never the eval year).

Each fold is (validation start, validation end): a model is fitted on the hours
before the start (from its history start) and scored on the fold's 6-hourly runs.

    OLD     the two folds used by every selection so far (2023-10 -> 2025-10)
    YEARLY  six yearly folds, 2019-10 -> 2025-10, possible since the history was
            extended back to 2016-10 (training.datasets.DATA_START)

SELECTION is the set the selection scripts use: YEARLY since the 9-year
re-selection (branch reselection_9y); OLD before.
"""

import pandas as pd

from training.datasets import TRAIN_END

YEAR = pd.Timedelta(days=365)
OLD = [(TRAIN_END - 2 * YEAR, TRAIN_END - YEAR), (TRAIN_END - YEAR, TRAIN_END)]
YEARLY = [(pd.Timestamp(f"{y}-10-20T00:00Z"), pd.Timestamp(f"{y + 1}-10-20T00:00Z"))
          for y in range(2019, 2025)]
SELECTION = YEARLY  # the folds model selection uses (training.linear_model, xgboost_model)
HISTORY_STARTS = {"since 2016": pd.Timestamp("2016-10-20T00:00Z"),
                  "since 2021": pd.Timestamp("2021-10-20T00:00Z")}
