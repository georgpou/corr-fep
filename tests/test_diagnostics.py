import numpy as np
import pytest

from corr_fep.diagnostics import free_energy_reduced, importance_ess, summarize_work


def test_constant_work_identity():
    work = np.full(20, 5.3)
    assert free_energy_reduced(work) == pytest.approx(5.3)
    assert importance_ess(work) == pytest.approx(20)


def test_shift_invariance_and_ess():
    work = np.array([-3.0, 0.0, 4.0, 7.0])
    assert free_energy_reduced(work + 1234) == pytest.approx(
        free_energy_reduced(work) + 1234
    )
    assert importance_ess(work + 1234) == pytest.approx(importance_ess(work))


def test_large_work_values_remain_finite():
    work = np.array([1e5, 1e5 + 2, 1e5 + 5])
    result = summarize_work(work)
    assert np.isfinite(result["delta_f_reduced"])
    assert 1 <= result["importance_ess"] <= 3


@pytest.mark.parametrize("work", [[], [[1, 2]], [float("nan")], [float("inf")]])
def test_invalid_work(work):
    with pytest.raises(ValueError):
        free_energy_reduced(work)


def test_extreme_weight_dominance():
    work = np.array([-1000.0, 1000.0, 1000.0])
    assert importance_ess(work) == pytest.approx(1)
