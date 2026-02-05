"""Statistical utilities for variance-minimised estimator combination.

Implements the optimal linear combination approach from:
J. Chem. Phys. 154, 191101 (2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import Any


class WelfordAccumulator3D:
    """
    Online variance/covariance accumulator for 3D grids.

    Uses Welford's algorithm to compute running mean, variance, and covariance
    in a single pass, without storing all samples. This enables variance
    estimation across multiple trajectories without memory explosion.

    For lambda estimation, we track:
    - delta = rho_force - rho_count (per section)
    - rho_force (per section)

    And compute:
    - Var(delta) across sections
    - Cov(delta, rho_force) across sections

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D grid (nx, ny, nz).

    Attributes
    ----------
    count : int
        Number of samples (sections) accumulated.
    mean_delta : ndarray
        Running mean of delta across sections.
    mean_rho_force : ndarray
        Running mean of rho_force across sections.
    M2_delta : ndarray
        Sum of squared deviations for variance calculation.
    C_delta_force : ndarray
        Sum of cross-deviations for covariance calculation.

    Examples
    --------
    >>> acc = WelfordAccumulator3D((10, 10, 10))
    >>> for delta, rho_force in section_data:
    ...     acc.update(delta, rho_force)
    >>> variance, covariance = acc.finalise()
    """

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.shape = shape
        self.count = 0
        self.mean_delta: NDArray[np.floating[Any]] = np.zeros(shape)
        self.mean_rho_force: NDArray[np.floating[Any]] = np.zeros(shape)
        self.M2_delta: NDArray[np.floating[Any]] = np.zeros(shape)
        self.C_delta_force: NDArray[np.floating[Any]] = np.zeros(shape)

    def update(
        self,
        delta: NDArray[np.floating[Any]],
        rho_force: NDArray[np.floating[Any]],
    ) -> None:
        """
        Add one section's densities to the running statistics.

        Parameters
        ----------
        delta : ndarray
            The difference rho_force - rho_count for this section.
        rho_force : ndarray
            The force-based density for this section.
        """
        self.count += 1

        # Welford update for delta mean and variance
        d_delta = delta - self.mean_delta
        self.mean_delta += d_delta / self.count
        d_delta2 = delta - self.mean_delta  # uses updated mean
        self.M2_delta += d_delta * d_delta2

        # Update mean_rho_force
        d_force = rho_force - self.mean_rho_force
        self.mean_rho_force += d_force / self.count

        # Covariance update: dx * (y - mean_y_new)
        self.C_delta_force += d_delta * (rho_force - self.mean_rho_force)

    def finalise(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Return variance and covariance arrays.

        Returns
        -------
        variance : ndarray
            Var(delta) across all sections (population variance).
        covariance : ndarray
            Cov(delta, rho_force) across all sections.

        Raises
        ------
        ValueError
            If fewer than 2 sections have been accumulated.
        """
        if self.count < 2:
            msg = (
                f"Need at least 2 sections for variance estimation, "
                f"got {self.count}"
            )
            raise ValueError(msg)

        # Population variance (divide by count, not count-1)
        variance = self.M2_delta / self.count
        covariance = self.C_delta_force / self.count
        return variance, covariance

    @property
    def has_data(self) -> bool:
        """Return True if any sections have been accumulated."""
        return self.count > 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.count = 0
        self.mean_delta.fill(0)
        self.mean_rho_force.fill(0)
        self.M2_delta.fill(0)
        self.C_delta_force.fill(0)


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
