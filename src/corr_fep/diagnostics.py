"""Numerically stable free-energy and importance-weight diagnostics.

All work values are dimensionless (reduced units, k_B T). These
diagnostics do not account for MD autocorrelation: analyze independent
trajectories or use an appropriate block/trajectory bootstrap.
"""

import numpy as np


def _validate_work(work):
    values = np.asarray(work, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("work must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("work values must be finite")
    return values


def _logsumexp(values):
    top = float(np.max(values))
    return top + float(np.log(np.sum(np.exp(values - top))))


def free_energy_reduced(work):
    """Unidirectional reduced free-energy difference: -log(mean(exp(-w)))."""
    values = _validate_work(work)
    return -(_logsumexp(-values) - np.log(values.size))


def importance_ess(work):
    """Kish ESS of normalized exponential importance weights, in [1, N].

    This is NOT an autocorrelation-corrected effective sample size and is
    not proof of convergence or absence of missing target modes.
    """
    values = _validate_work(work)
    logweights = -values
    return float(np.exp(2 * _logsumexp(logweights) - _logsumexp(2 * logweights)))


def summarize_work(work):
    """Basic diagnostic summary, with work and free energy in k_B T."""
    values = _validate_work(work)
    return {
        "n": int(values.size),
        "delta_f_reduced": free_energy_reduced(values),
        "mean_work_reduced": float(values.mean()),
        "std_work_reduced": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "importance_ess": importance_ess(values),
        "importance_ess_fraction": importance_ess(values) / values.size,
    }
