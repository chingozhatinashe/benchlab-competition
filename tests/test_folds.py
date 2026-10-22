"""training.folds: walk-forward folds stay inside the training years."""

import pytest

from training import datasets, folds


@pytest.mark.parametrize("which", [folds.OLD, folds.YEARLY])
def test_folds_are_ordered_disjoint_and_inside_the_training_years(which):
    for (start, end), (next_start, _) in zip(which, which[1:], strict=False):
        assert start < end <= next_start
    assert which[0][0] > datasets.DATA_START
    assert which[-1][1] <= datasets.TRAIN_END


def test_yearly_folds_cover_2019_to_the_eval_year():
    assert len(folds.YEARLY) == 6
    assert folds.YEARLY[0][0].year == 2019 and folds.YEARLY[-1][1] == datasets.TRAIN_END
    assert all(h >= datasets.DATA_START for h in folds.HISTORY_STARTS.values())
