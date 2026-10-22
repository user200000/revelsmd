"""Statistical utilities for variance-minimised estimator combination.

Implements the optimal linear combination approach from:
J. Chem. Phys. 154, 191101 (2021).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Each fold's lambda is a regression slope, -Cov(E_0, Delta) / Var(Delta),
# over its n samples. Its variance scales as E[1 / sum (Delta_i - mean)^2],
# which for roughly Gaussian Delta is proportional to 1 / (n - 3): infinite
# for n <= 3, since n samples bunch together, making the denominator small,
# too often.
MIN_LAMBDA_SAMPLES = 4


class WelfordAccumulator:
    """
    Online variance/covariance accumulator for arrays of any shape.

    Uses Welford's algorithm to compute running mean, variance, and covariance
    in a single pass, without storing all samples, so memory does not grow
    with the number of samples. Each sample (a block of frames for a density
    grid, a frame for an RDF) gives, at every point of the array:

    - delta, the difference between the two estimators
    - other, a second quantity to correlate with delta

    and the accumulator returns Var(delta) and Cov(delta, other) across the
    samples, each sample weighted by the weight passed to update().

    Parameters
    ----------
    shape : tuple of int
        Shape of the arrays, e.g. (nx, ny, nz) for a grid or (n,) for g(r).

    Attributes
    ----------
    count : int
        Number of samples accumulated.
    sum_weights : float
        Sum of weights across all samples.

    Examples
    --------
    >>> acc = WelfordAccumulator((10, 10, 10))
    >>> for delta, other, weight in samples:
    ...     acc.update(delta, other, weight=weight)
    >>> variance, covariance = acc.finalise()
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.count = 0
        self.sum_weights: float = 0.0
        self._mean_delta: NDArray[np.floating] = np.zeros(shape)
        self._mean_other: NDArray[np.floating] = np.zeros(shape)
        self._M2_delta: NDArray[np.floating] = np.zeros(shape)
        self._C_delta_other: NDArray[np.floating] = np.zeros(shape)

        # Scratch buffers for in-place update (avoids temporary allocations)
        self._scratch_a: NDArray[np.floating] = np.empty(shape)
        self._scratch_b: NDArray[np.floating] = np.empty(shape)
        self._scratch_c: NDArray[np.floating] = np.empty(shape)

    def update(
        self,
        delta: NDArray[np.floating],
        other: NDArray[np.floating],
        weight: float = 1.0,
    ) -> None:
        """
        Add one sample to the running statistics.

        Parameters
        ----------
        delta : ndarray
            The difference between the two estimators for this sample.
        other : ndarray
            The quantity whose covariance with delta is wanted, for this
            sample.
        weight : float, optional
            Weight for this sample, e.g. its number of frames. A sample
            contributes to the means, the variance and the covariance in
            proportion to its weight. Default is 1.0.
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

        # d_other = other - mean_other
        np.subtract(other, self._mean_other, out=sb)
        # mean_other += d_other * ratio
        np.multiply(sb, ratio, out=sc)
        self._mean_other += sc
        # C_delta_other += weight * d_delta * (other - mean_other_new)
        np.subtract(other, self._mean_other, out=sb)
        np.multiply(sa, sb, out=sc)
        sc *= weight
        self._C_delta_other += sc

    def finalise(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Return weighted population variance and covariance arrays.

        Returns
        -------
        variance : ndarray
            Weighted population Var(delta) across all samples.
        covariance : ndarray
            Weighted population Cov(delta, other) across all samples.

        Raises
        ------
        ValueError
            If fewer than 2 samples have been accumulated.
        """
        if self.count < 2:
            msg = (
                f"need at least 2 samples for variance estimation, "
                f"but only {self.count} have been accumulated"
            )
            raise ValueError(msg)

        # Population variance (divide by sum_weights, not count)
        variance = self._M2_delta / self.sum_weights
        covariance = self._C_delta_other / self.sum_weights
        return variance, covariance

    def means(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the weighted means of delta and other across the samples.

        These are the accumulator's own arrays, updated in place by update():
        copy them to keep them, and do not modify them.
        """
        return self._mean_delta, self._mean_other

    @property
    def has_data(self) -> bool:
        """Return True if any samples have been accumulated."""
        return self.count > 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.count = 0
        self.sum_weights = 0.0
        self._mean_delta.fill(0)
        self._mean_other.fill(0)
        self._M2_delta.fill(0)
        self._C_delta_other.fill(0)


class FoldMoments(NamedTuple):
    """One fold's moments, as ``cross_fitted_combination`` takes them."""

    #: Total weight of the fold's samples, e.g. their number of frames.
    weight: float
    #: Mean of E_0 over the fold's samples.
    mean_e0: NDArray[np.floating]
    #: Mean of E_1 over the fold's samples.
    mean_e1: NDArray[np.floating]
    #: Var(Delta), Delta = E_1 - E_0, across the fold's samples.
    var_delta: NDArray[np.floating]
    #: Cov(E_0, Delta) across the fold's samples.
    cov_e0_delta: NDArray[np.floating]


class CrossFitStatistics:
    """
    Running lambda statistics in two folds of samples.

    Fold 0 holds the samples from the first half in time of each
    accumulation and fold 1 those from the second half. lambda is estimated
    in each fold and applied to the other fold's means (see
    ``cross_fitted_combination``), so it is never estimated from the samples
    it weights. Folds split by time are close to independent, since only the
    samples near the boundary between the halves are correlated; folds of
    alternating samples would not be.

    Each sample (a block of frames for a density grid, a frame for an RDF)
    gives, at every point of the array, Delta = E_1 - E_0 and E_0, the
    estimators in the notation of Coles et al. (2021).

    Parameters
    ----------
    shape : tuple of int
        Shape of the arrays, e.g. (nx, ny, nz) for a grid or (n,) for g(r).
    samples : str, optional
        What the samples are, for error messages: e.g. "blocks" or "frames".
        Default "samples".

    Attributes
    ----------
    folds : tuple of WelfordAccumulator
        The two folds' running statistics.
    """

    def __init__(self, shape: tuple[int, ...], *, samples: str = "samples") -> None:
        self.folds = (WelfordAccumulator(shape), WelfordAccumulator(shape))
        self._samples = samples
        # One sample is added at a time, so the folds can share scratch space.
        first, second = self.folds
        second._scratch_a = first._scratch_a
        second._scratch_b = first._scratch_b
        second._scratch_c = first._scratch_c

    def update(
        self,
        fold: int,
        delta: NDArray[np.floating],
        e0: NDArray[np.floating],
        weight: float = 1.0,
    ) -> None:
        """Add one sample's Delta = E_1 - E_0 and E_0 to fold 0 or fold 1.

        See ``WelfordAccumulator.update`` for ``weight``.

        Raises
        ------
        ValueError
            If ``fold`` is not 0 or 1.
        """
        if fold not in (0, 1):
            raise ValueError(f"fold must be 0 or 1, got {fold}")
        self.folds[fold].update(delta, e0, weight)

    def finalise(self, *, remedy: str | None = None) -> tuple[FoldMoments, FoldMoments]:
        """Return each fold's moments, for ``cross_fitted_combination``.

        Parameters
        ----------
        remedy : str, optional
            What to do about too few samples, appended to the error below.

        Raises
        ------
        ValueError
            If either fold has fewer than ``MIN_LAMBDA_SAMPLES`` samples.
        """
        counts = [fold.count for fold in self.folds]
        if min(counts) < MIN_LAMBDA_SAMPLES:
            message = (
                f"lambda needs at least {MIN_LAMBDA_SAMPLES} {self._samples} in each half of "
                f"the frames (have {counts[0]} and {counts[1]}), since with fewer its variance "
                "is infinite; each accumulate() call's frames are split into two halves, so "
                "a call of one frame adds only to the first"
            )
            if remedy is not None:
                message = f"{message}. {remedy}"
            raise ValueError(message)
        moments = []
        for fold in self.folds:
            var_delta, cov_e0_delta = fold.finalise()
            mean_delta, mean_e0 = fold.means()
            moments.append(FoldMoments(
                fold.sum_weights, mean_e0.copy(), mean_e0 + mean_delta, var_delta, cov_e0_delta,
            ))
        return moments[0], moments[1]

    @property
    def count(self) -> int:
        """Number of samples accumulated, in both folds."""
        return sum(fold.count for fold in self.folds)

    @property
    def sum_weights(self) -> float:
        """Sum of the weights of the samples, in both folds."""
        return sum(fold.sum_weights for fold in self.folds)

    @property
    def has_data(self) -> bool:
        """Return True if any samples have been accumulated."""
        return self.count > 0

    def reset(self) -> None:
        """Clear all accumulated state."""
        for fold in self.folds:
            fold.reset()


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


def cross_fitted_combination(
    folds: tuple[FoldMoments, FoldMoments],
    mean_e0: NDArray[np.floating],
    mean_e1: NDArray[np.floating],
    *,
    scale: float,
    fallback: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Combine two estimators with lambda estimated in one fold and applied to the other.

    lambda from the samples it weights is correlated with them, which biases
    the combination: E[lambda * Delta] = lambda E[Delta] + Cov(lambda, Delta).
    Here each fold's lambda (Coles et al. (2021), Eq. 4, through
    ``control_variate_combination``, with its negligible-variance
    ``fallback``) weights the other fold's means, which it is independent of,
    and the two results are averaged by weight. With fold weights
    w_A + w_B = 1::

        E = w_A (E_0,A + lambda_B Delta_A) + w_B (E_0,B + lambda_A Delta_B)
          = (1 - mu) E_0 + mu E_1 + w_A w_B (lambda_B - lambda_A)(Delta_A - Delta_B),
        mu = w_A lambda_B + w_B lambda_A,

    where E_0 and E_1 are the means over both folds. The second form is the
    one evaluated: where the folds' lambdas agree it is exactly
    (1 - lambda) E_0 + lambda E_1, so lambda = 1 gives E_1 exactly.

    Parameters
    ----------
    folds : pair of FoldMoments
        Each fold's total weight, means of E_0 and E_1, Var(Delta) and
        Cov(E_0, Delta), e.g. from ``CrossFitStatistics.finalise``.
    mean_e0, mean_e1 : ndarray
        Means of E_0 and E_1 over both folds. They must be the folds' means
        weighted by the folds' weights, which is what the identity above
        assumes; they are passed rather than computed so that the result
        matches the caller's own estimators exactly where lambda is 0 or 1.
    scale, fallback : float
        As for ``control_variate_combination``.

    Returns
    -------
    combined : ndarray
        The combined estimate.
    lam : ndarray
        mu, the weight the combination puts on E_1 overall. The combined
        estimate is (1 - mu) E_0 + mu E_1 only where the folds' lambdas agree.
    """
    a, b = folds
    lam_a = control_variate_combination(
        a.mean_e0, a.mean_e1, a.var_delta, a.cov_e0_delta, scale=scale, fallback=fallback,
    )[1]
    lam_b = control_variate_combination(
        b.mean_e0, b.mean_e1, b.var_delta, b.cov_e0_delta, scale=scale, fallback=fallback,
    )[1]
    w_a = a.weight / (a.weight + b.weight)
    # Written so that equal lambdas give mu exactly equal to them.
    mu = lam_a + w_a * (lam_b - lam_a)
    cross = w_a * (1 - w_a) * (lam_b - lam_a) * ((a.mean_e1 - a.mean_e0) - (b.mean_e1 - b.mean_e0))
    return (1 - mu) * mean_e0 + mu * mean_e1 + cross, mu
