"""Statistical utilities for variance-minimised estimator combination.

Implements the optimal linear combination approach from:
J. Chem. Phys. 154, 191101 (2021).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_lambda_weights(
    variance: NDArray[np.floating],
    covariance: NDArray[np.floating],
    *,
    zero_variance_replacement: float = 0.0,
) -> NDArray[np.floating]:
    """
    Compute optimal combination weights from variance and covariance.

    Calculates lambda = Cov(delta, estimator) / Var(delta), with handling
    for edge cases where variance is zero or results are non-finite.

    Parameters
    ----------
    variance : ndarray
        Variance of the difference between estimators, Var(delta).
    covariance : ndarray
        Covariance of delta with one of the estimators.
    zero_variance_replacement : float, default 0.0
        Value to use for lambda where variance is zero.

    Returns
    -------
    ndarray
        Optimal combination weights, same shape as inputs. Non-finite
        values are replaced with zero_variance_replacement.
    """
    # Avoid division by zero: substitute 1.0 for zero variance
    variance_safe = np.where(variance == 0, 1.0, variance)
    weights = np.divide(covariance, variance_safe)

    # Replace non-finite results and zero-variance locations
    weights = np.nan_to_num(
        weights,
        nan=zero_variance_replacement,
        posinf=zero_variance_replacement,
        neginf=zero_variance_replacement,
    )

    # Where variance was zero, use the replacement value
    weights = np.where(variance == 0, zero_variance_replacement, weights)

    return weights


def combine_estimators(
    estimator_a: NDArray[np.floating],
    estimator_b: NDArray[np.floating],
    weights: NDArray[np.floating] | float,
    *,
    sanitise_output: bool = True,
) -> NDArray[np.floating]:
    """
    Linearly combine two estimators using precomputed weights.

    Computes: result = estimator_a * (1 - weights) + estimator_b * weights

    Parameters
    ----------
    estimator_a : ndarray
        First estimator (receives weight 1 - lambda).
    estimator_b : ndarray
        Second estimator (receives weight lambda).
    weights : ndarray or float
        Combination weights (same shape as estimators, or broadcastable).
    sanitise_output : bool, default True
        If True, replace NaN/Inf in output with 0.0.

    Returns
    -------
    ndarray
        Combined estimator, same shape as inputs.
    """
    result = estimator_a * (1 - weights) + estimator_b * weights

    if sanitise_output:
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
