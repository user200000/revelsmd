"""Statistical utilities for variance-minimised estimator combination.

Implements the optimal linear combination approach from:
J. Chem. Phys. 154, 191101 (2021).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class WelfordAccumulator3D:
    """
    Online variance/covariance accumulator for 3D grids.

    Uses Welford's algorithm to compute running mean, variance, and covariance
    in a single pass, without storing all samples. This enables variance
    estimation across multiple trajectories without memory explosion.

    Callers provide per-block values of:
    - delta = rho_force - rho_count
    - rho_force

    And compute:
    - Var(delta) across blocks
    - Cov(delta, rho_force) across blocks

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D grid (nx, ny, nz).

    Attributes
    ----------
    count : int
        Number of blocks accumulated.
    sum_weights : float
        Sum of weights across all blocks.

    Examples
    --------
    >>> acc = WelfordAccumulator3D((10, 10, 10))
    >>> for delta, rho_force, n_frames in block_data:
    ...     acc.update(delta, rho_force, weight=n_frames)
    >>> variance, covariance = acc.finalise()
    """

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.shape = shape
        self.count = 0
        self.sum_weights: float = 0.0
        self._mean_delta: NDArray[np.floating] = np.zeros(shape)
        self._mean_rho_force: NDArray[np.floating] = np.zeros(shape)
        self._M2_delta: NDArray[np.floating] = np.zeros(shape)
        self._C_delta_force: NDArray[np.floating] = np.zeros(shape)

        # Scratch buffers for in-place update (avoids temporary allocations)
        self._scratch_a: NDArray[np.floating] = np.empty(shape)
        self._scratch_b: NDArray[np.floating] = np.empty(shape)
        self._scratch_c: NDArray[np.floating] = np.empty(shape)

    def update(
        self,
        delta: NDArray[np.floating],
        rho_force: NDArray[np.floating],
        weight: float = 1.0,
    ) -> None:
        """
        Add one block's densities to the running statistics.

        Parameters
        ----------
        delta : ndarray
            The difference rho_force - rho_count for this block.
        rho_force : ndarray
            The force-based density for this block.
        weight : float, optional
            Weight for this block (e.g. number of frames). Blocks with
            more frames contribute proportionally more to the mean and
            variance estimates. Default is 1.0 (unweighted).
        """
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        self.count += 1
        self.sum_weights += weight
        ratio = weight / self.sum_weights
        sa, sb, sc = self._scratch_a, self._scratch_b, self._scratch_c

        # d_delta = delta - mean_delta  (keep in sa for covariance later)
        np.subtract(delta, self._mean_delta, out=sa)
        # mean_delta += d_delta * ratio
        np.multiply(sa, ratio, out=sb)
        self._mean_delta += sb
        # d_delta2 = delta - mean_delta  (updated mean)
        np.subtract(delta, self._mean_delta, out=sb)
        # M2_delta += weight * d_delta * d_delta2
        np.multiply(sa, sb, out=sc)
        sc *= weight
        self._M2_delta += sc

        # d_force = rho_force - mean_rho_force
        np.subtract(rho_force, self._mean_rho_force, out=sb)
        # mean_rho_force += d_force * ratio
        np.multiply(sb, ratio, out=sc)
        self._mean_rho_force += sc
        # C_delta_force += weight * d_delta * (rho_force - mean_rho_force_new)
        np.subtract(rho_force, self._mean_rho_force, out=sb)
        np.multiply(sa, sb, out=sc)
        sc *= weight
        self._C_delta_force += sc

    def finalise(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Return weighted population variance and covariance arrays.

        Returns
        -------
        variance : ndarray
            Weighted population Var(delta) across all blocks.
        covariance : ndarray
            Weighted population Cov(delta, rho_force) across all blocks.

        Raises
        ------
        ValueError
            If fewer than 2 blocks have been accumulated.
        """
        if self.count < 2:
            msg = (
                f"Need at least 2 blocks for variance estimation, "
                f"but only {self.count} have been accumulated."
            )
            raise ValueError(msg)

        # Population variance (divide by sum_weights, not count)
        variance = self._M2_delta / self.sum_weights
        covariance = self._C_delta_force / self.sum_weights
        return variance, covariance

    @property
    def has_data(self) -> bool:
        """Return True if any blocks have been accumulated."""
        return self.count > 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.count = 0
        self.sum_weights = 0.0
        self._mean_delta.fill(0)
        self._mean_rho_force.fill(0)
        self._M2_delta.fill(0)
        self._C_delta_force.fill(0)


#: A standard deviation of Delta at most this fraction of the scale (the size
#: of the numbers whose rounding sets that of Delta) is rounding, not
#: sampling, so the variance counts as zero.
NEGLIGIBLE_SPREAD = 1e-10


def control_variate_combination(
    mean_e0: NDArray[np.floating],
    mean_e1: NDArray[np.floating],
    var_delta: NDArray[np.floating],
    cov_e0_delta: NDArray[np.floating],
    *,
    scale: float,
    fallback: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Combine two estimators of the same quantity with the variance-minimising weight.

    Implements Coles et al., J. Chem. Phys. 154, 191101 (2021), Eqs. 3-4:

        E_lambda = (1 - lambda) * E_0 + lambda * E_1,
        lambda   = -Cov(E_0, Delta) / Var(Delta),    Delta = E_1 - E_0,

    This lambda minimises the variance of E_0 + lambda * Delta over the
    blocks or frames the moments come from. E_0 sets both the base term and
    the covariance, so relabelling the estimators (with their moments, and
    with ``fallback`` replaced by 1 - ``fallback``) gives the same estimate
    with weight 1 - lambda rather than an inverted one.

    Where Var(Delta) is negligible, lambda is ``fallback``. A variance is
    negligible where the standard deviation of Delta is at most
    ``NEGLIGIBLE_SPREAD`` times ``scale``: it is then rounding, and dividing
    by it would give an arbitrarily large lambda. Non-finite inputs are not
    hidden: they propagate into the result.

    Parameters
    ----------
    mean_e0, mean_e1 : ndarray
        Means of E_0 and E_1 over the blocks or frames.
    var_delta : ndarray
        Var(Delta) over the blocks or frames.
    cov_e0_delta : ndarray
        Cov(E_0, Delta) over the blocks or frames.
    scale : float
        One number: the size of the numbers whose rounding sets that of
        Delta, everywhere in the arrays.
    fallback : float
        lambda where Var(Delta) is negligible: 0 gives E_0, 1 gives E_1.

    Returns
    -------
    combined : ndarray
        The combined estimate.
    lam : ndarray
        lambda, the weight on E_1.
    """
    # One number for all the arrays: a scale per voxel or bin lets degenerate
    # ones through, since their rounding is set by the largest values.
    scale = float(scale)
    negligible = var_delta <= (NEGLIGIBLE_SPREAD * scale) ** 2
    lam = np.where(negligible, fallback, -cov_e0_delta / np.where(negligible, 1.0, var_delta))
    return (1 - lam) * mean_e0 + lam * mean_e1, lam
