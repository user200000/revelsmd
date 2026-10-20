"""Tests for revelsMD.statistics module."""

import numpy as np
import pytest

from revelsMD.statistics import (
    WelfordAccumulator,
    control_variate_combination,
)


def _moments(e0, e1):
    """Means of E_0 and E_1, and population moments of Delta, over axis 0."""
    delta = e1 - e0
    d_delta = delta - delta.mean(axis=0)
    return (
        e0.mean(axis=0),
        e1.mean(axis=0),
        np.mean(d_delta**2, axis=0),
        np.mean((e0 - e0.mean(axis=0)) * d_delta, axis=0),
    )


def _samples(seed, n_blocks=12, shape=(5,)):
    """Correlated per-block estimates E_0 and E_1 of the same quantity."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_blocks, *shape))
    e0 = 2.0 + common + 0.5 * rng.normal(size=(n_blocks, *shape))
    e1 = 2.0 + 0.3 * common + 0.2 * rng.normal(size=(n_blocks, *shape))
    return e0, e1


class TestControlVariateCombination:
    """Tests for control_variate_combination (Coles et al. (2021), Eqs. 3-4)."""

    def test_formula(self):
        combined, lam = control_variate_combination(
            np.array([1.0, 2.0]), np.array([1.5, 1.0]), np.array([2.0, 4.0]),
            np.array([-1.0, 1.0]), scale=1.0, fallback=0.0,
        )
        np.testing.assert_allclose(lam, [0.5, -0.25])
        np.testing.assert_allclose(combined, [1.25, 2.25])

    @pytest.mark.parametrize("seed", range(5))
    def test_lambda_minimises_the_variance_of_the_combination(self, seed):
        # In-sample: no other weight gives a smaller spread of E_0 + w * Delta.
        e0, e1 = _samples(seed)
        combined, lam = control_variate_combination(*_moments(e0, e1), scale=1.0, fallback=0.0)
        trial = np.linspace(-3, 3, 6001)[:, None]
        spreads = np.var(e0[:, None, :] + trial[None] * (e1 - e0)[:, None, :], axis=0)
        best = np.var(e0 + lam * (e1 - e0), axis=0)
        assert np.all(best <= spreads.min(axis=0) + 1e-12)
        np.testing.assert_allclose(combined, np.mean(e0 + lam * (e1 - e0), axis=0))

    @pytest.mark.parametrize("seed", range(5))
    def test_relabelling_the_estimators_gives_the_same_combination(self, seed):
        # Passing E_1 as E_0, with the fallback flipped too, gives the same
        # estimate with weight 1 - lambda, so swapping the estimators cannot
        # silently invert the result. Element 0 has a negligible variance.
        e0, e1 = _samples(seed)
        e1[:, 0] = e0[:, 0] + 0.7
        combined, lam = control_variate_combination(*_moments(e0, e1), scale=1.0, fallback=0.0)
        swapped, lam_swapped = control_variate_combination(*_moments(e1, e0), scale=1.0, fallback=1.0)
        assert lam[0] == 0.0 and lam_swapped[0] == 1.0
        np.testing.assert_allclose(swapped, combined, rtol=1e-12)
        np.testing.assert_allclose(lam_swapped, 1 - lam, rtol=1e-12)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    @pytest.mark.parametrize("fallback", [0.0, 1.0])
    def test_negligible_variance_takes_the_fallback(self, scale, fallback):
        # A standard deviation of delta at most 1e-10 of the scale is
        # rounding, not sampling. The result does not depend on units.
        var_delta = np.array([(1e-11 * scale) ** 2, (1e-9 * scale) ** 2, 0.0])
        cov_e0_delta = -np.array([3.0, 2.0, 3.0]) * var_delta - np.array([0.0, 0.0, scale**2])
        combined, lam = control_variate_combination(
            np.full(3, scale), np.full(3, 1.5 * scale), var_delta, cov_e0_delta,
            scale=scale, fallback=fallback,
        )
        np.testing.assert_allclose(lam, [fallback, 2.0, fallback])
        # The fallback reproduces the chosen estimator exactly.
        chosen = 1.5 * scale if fallback else scale
        np.testing.assert_allclose(combined, [chosen, 2.0 * scale, chosen])
        assert combined[0] == combined[2] == chosen

    @pytest.mark.filterwarnings("ignore:invalid value encountered")  # numpy reports the NaN
    def test_non_finite_moments_are_not_hidden(self):
        combined, lam = control_variate_combination(
            np.ones(2), np.full(2, 1.1), np.array([np.nan, 1.0]), np.array([0.5, np.inf]),
            scale=1.0, fallback=1.0,
        )
        assert np.isnan(lam[0]) and np.isinf(lam[1])
        assert not np.isfinite(combined).any()

    def test_scale_is_one_number(self):
        # A scale per element would let degenerate elements through.
        with pytest.raises(TypeError):
            control_variate_combination(
                np.ones(2), np.ones(2), np.ones(2), np.ones(2), scale=np.ones(2), fallback=0.0
            )

    def test_non_finite_estimates_are_not_hidden(self):
        combined, _ = control_variate_combination(
            np.array([np.nan, np.inf]), np.ones(2), np.ones(2), np.zeros(2), scale=1.0, fallback=0.0,
        )
        assert np.isnan(combined[0]) and np.isinf(combined[1])

    def test_works_on_grids(self):
        e0, e1 = _samples(7, shape=(3, 4, 5))
        combined, lam = control_variate_combination(*_moments(e0, e1), scale=1.0, fallback=0.0)
        assert combined.shape == lam.shape == (3, 4, 5)

    @pytest.mark.parametrize("missing", ["scale", "fallback"])
    def test_scale_and_fallback_are_required(self, missing):
        keywords = {"scale": 1.0, "fallback": 0.0}
        del keywords[missing]
        with pytest.raises(TypeError, match=missing):
            control_variate_combination(np.ones(2), np.ones(2), np.ones(2), np.ones(2), **keywords)


@pytest.mark.parametrize("shape", [(20,), (6, 7), (20, 20, 20)])
def test_welford_inplace_matches_reference(shape):
    """In-place Welford update matches a direct weighted calculation, for any shape."""
    rng = np.random.default_rng(42)
    n_samples = 10

    deltas = [rng.standard_normal(shape) for _ in range(n_samples)]
    others = [rng.standard_normal(shape) for _ in range(n_samples)]
    weights = rng.integers(1, 10, size=n_samples).astype(float)

    acc = WelfordAccumulator(shape)
    for delta, other, w in zip(deltas, others, weights):
        acc.update(delta, other, weight=w)

    variance, covariance = acc.finalise()

    # Compare against manual numpy computation
    total_w = weights.sum()
    mean_delta = sum(w * d for w, d in zip(weights, deltas)) / total_w
    mean_other = sum(w * o for w, o in zip(weights, others)) / total_w
    var_expected = sum(w * (d - mean_delta)**2 for w, d in zip(weights, deltas)) / total_w
    cov_expected = sum(w * (d - mean_delta) * (o - mean_other) for w, d, o in zip(weights, deltas, others)) / total_w

    np.testing.assert_allclose(variance, var_expected, rtol=1e-10)
    np.testing.assert_allclose(covariance, cov_expected, rtol=1e-10)


class TestWelfordAccumulator:
    """Tests for WelfordAccumulator online variance/covariance accumulator."""

    def test_has_data_false_initially(self):
        """has_data is False before any updates."""
        acc = WelfordAccumulator((2, 2, 2))
        assert acc.has_data is False

    def test_has_data_true_after_update(self):
        """has_data is True after first update."""
        acc = WelfordAccumulator((2, 2, 2))
        acc.update(np.ones((2, 2, 2)), np.ones((2, 2, 2)))
        assert acc.has_data is True

    def test_count_increments(self):
        """Count increments with each update call."""
        acc = WelfordAccumulator((2, 2, 2))
        for i in range(5):
            acc.update(np.ones((2, 2, 2)) * i, np.ones((2, 2, 2)) * i)
        assert acc.count == 5

    def test_finalise_raises_with_fewer_than_two_samples(self):
        """finalise() requires at least 2 samples."""
        acc = WelfordAccumulator((2, 2, 2))
        with pytest.raises(ValueError, match="at least 2 samples"):
            acc.finalise()

        acc.update(np.ones((2, 2, 2)), np.ones((2, 2, 2)))
        with pytest.raises(ValueError, match="at least 2 samples"):
            acc.finalise()

    def test_variance_matches_numpy_simple(self):
        """Welford variance matches numpy.var for simple 1D data at each voxel."""
        np.random.seed(42)
        n_sections = 20
        shape = (3, 3, 3)

        # Generate random data
        deltas = np.random.randn(n_sections, *shape)
        forces = np.random.randn(n_sections, *shape)

        # Compute with Welford
        acc = WelfordAccumulator(shape)
        for i in range(n_sections):
            acc.update(deltas[i], forces[i])

        variance, covariance = acc.finalise()

        # Compare to numpy (population variance, ddof=0)
        expected_variance = np.var(deltas, axis=0, ddof=0)
        np.testing.assert_allclose(variance, expected_variance, rtol=1e-10)

    def test_covariance_matches_numpy(self):
        """Welford covariance matches manually computed covariance."""
        np.random.seed(42)
        n_sections = 20
        shape = (3, 3, 3)

        # Generate correlated data
        deltas = np.random.randn(n_sections, *shape)
        forces = deltas * 0.5 + np.random.randn(n_sections, *shape) * 0.1

        # Compute with Welford
        acc = WelfordAccumulator(shape)
        for i in range(n_sections):
            acc.update(deltas[i], forces[i])

        variance, covariance = acc.finalise()

        # Compute expected covariance manually (population covariance)
        mean_delta = np.mean(deltas, axis=0)
        mean_force = np.mean(forces, axis=0)
        expected_covariance = np.mean(
            (deltas - mean_delta) * (forces - mean_force), axis=0
        )
        np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-10)

    def test_zero_variance_when_constant(self):
        """Variance is zero when all samples are identical."""
        shape = (2, 2, 2)
        acc = WelfordAccumulator(shape)

        constant_delta = np.ones(shape) * 3.0
        constant_force = np.ones(shape) * 5.0

        for _ in range(10):
            acc.update(constant_delta, constant_force)

        variance, covariance = acc.finalise()
        np.testing.assert_allclose(variance, 0.0, atol=1e-15)
        np.testing.assert_allclose(covariance, 0.0, atol=1e-15)

    def test_mean_delta_correct(self):
        """Mean delta is correctly computed."""
        shape = (2, 2, 2)
        acc = WelfordAccumulator(shape)

        deltas = [
            np.ones(shape) * 1.0,
            np.ones(shape) * 2.0,
            np.ones(shape) * 3.0,
        ]
        forces = [np.zeros(shape)] * 3

        for d, f in zip(deltas, forces):
            acc.update(d, f)

        expected_mean = np.ones(shape) * 2.0  # (1+2+3)/3
        np.testing.assert_allclose(acc._mean_delta, expected_mean)

    def test_numerical_stability_large_values(self):
        """Welford remains stable with large values."""
        shape = (2, 2, 2)
        acc = WelfordAccumulator(shape)

        # Large values that would cause issues with naive variance
        base = 1e10
        deltas = np.array([
            np.ones(shape) * (base + 1),
            np.ones(shape) * (base + 2),
            np.ones(shape) * (base + 3),
        ])
        forces = np.zeros((3, *shape))

        for i in range(3):
            acc.update(deltas[i], forces[i])

        variance, _ = acc.finalise()
        # Variance of [1, 2, 3] is 2/3 (population variance)
        expected = 2.0 / 3.0
        np.testing.assert_allclose(variance, expected, rtol=1e-10)

    def test_reset_clears_state(self):
        """reset() clears all accumulated state."""
        shape = (2, 2, 2)
        acc = WelfordAccumulator(shape)

        # Add some data
        acc.update(np.ones(shape) * 5, np.ones(shape) * 3)
        acc.update(np.ones(shape) * 10, np.ones(shape) * 6)
        assert acc.count == 2
        assert acc.has_data is True

        # After reset, the accumulator behaves as a fresh one.
        acc.reset()
        assert acc.count == 0 and acc.has_data is False
        rng = np.random.default_rng(3)
        samples = [(rng.standard_normal(shape), rng.standard_normal(shape)) for _ in range(4)]
        fresh = WelfordAccumulator(shape)
        for delta, other in samples:
            acc.update(delta, other, weight=2.0)
            fresh.update(delta, other, weight=2.0)
        for after_reset, from_fresh in zip(acc.finalise(), fresh.finalise()):
            np.testing.assert_array_equal(after_reset, from_fresh)

    def test_weighted_variance_matches_numpy(self):
        """Weighted Welford variance matches numpy weighted average."""
        shape = (3, 3, 3)
        np.random.seed(42)

        # Generate blocks with different weights (simulating different block sizes)
        deltas = np.random.randn(4, *shape)
        forces = np.random.randn(4, *shape)
        weights = [10.0, 10.0, 10.0, 3.0]  # last block is a remainder

        acc = WelfordAccumulator(shape)
        for i in range(4):
            acc.update(deltas[i], forces[i], weight=weights[i])

        variance, covariance = acc.finalise()

        # Compute expected weighted population variance manually
        w = np.array(weights)
        w_sum = w.sum()
        weighted_mean = np.sum(w[:, None, None, None] * deltas, axis=0) / w_sum
        expected_variance = np.sum(
            w[:, None, None, None] * (deltas - weighted_mean) ** 2, axis=0
        ) / w_sum
        np.testing.assert_allclose(variance, expected_variance, rtol=1e-10)

        # Compute expected weighted covariance
        weighted_mean_force = np.sum(
            w[:, None, None, None] * forces, axis=0
        ) / w_sum
        expected_covariance = np.sum(
            w[:, None, None, None]
            * (deltas - weighted_mean)
            * (forces - weighted_mean_force),
            axis=0,
        ) / w_sum
        np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-10)

    def test_update_rejects_zero_weight(self):
        """update() raises ValueError for zero weight."""
        acc = WelfordAccumulator((2, 2, 2))
        with pytest.raises(ValueError, match="weight must be positive"):
            acc.update(np.ones((2, 2, 2)), np.ones((2, 2, 2)), weight=0.0)

    def test_update_rejects_negative_weight(self):
        """update() raises ValueError for negative weight."""
        acc = WelfordAccumulator((2, 2, 2))
        with pytest.raises(ValueError, match="weight must be positive"):
            acc.update(np.ones((2, 2, 2)), np.ones((2, 2, 2)), weight=-1.0)

    def test_finalise_returns_correct_shapes(self):
        """finalise() returns arrays matching input shape."""
        shape = (5, 6, 7)
        acc = WelfordAccumulator(shape)

        acc.update(np.random.randn(*shape), np.random.randn(*shape))
        acc.update(np.random.randn(*shape), np.random.randn(*shape))

        variance, covariance = acc.finalise()
        assert variance.shape == shape
        assert covariance.shape == shape
