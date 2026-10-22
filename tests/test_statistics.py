"""Tests for revelsMD.statistics module."""

import numpy as np
import pytest

from revelsMD.statistics import (
    CrossFitStatistics,
    FoldMoments,
    WelfordAccumulator,
    control_variate_combination,
    cross_fitted_combination,
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


def _fold_moments(e0, e1, weight):
    """FoldMoments of per-sample estimates E_0 and E_1, with the fold's total weight."""
    return FoldMoments(weight, *_moments(e0, e1))


def _two_folds(seed, sizes=(7, 5)):
    """Per-sample E_0 and E_1 for two folds, of sizes[0] and sizes[1] samples."""
    e0, e1 = _samples(seed, n_blocks=sum(sizes))
    split = sizes[0]
    return (e0[:split], e1[:split]), (e0[split:], e1[split:])


def _means_over(folds):
    """Means of E_0 and E_1 over the per-sample estimates of both folds."""
    e0 = np.concatenate([fold[0] for fold in folds]).mean(axis=0)
    e1 = np.concatenate([fold[1] for fold in folds]).mean(axis=0)
    return e0, e1


class TestCrossFittedCombination:
    """Tests for cross_fitted_combination: each fold's lambda weights the other fold's means."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_each_fold_is_weighted_by_the_other_folds_lambda(self, seed):
        (e0_a, e1_a), (e0_b, e1_b) = folds = _two_folds(seed)
        moments = (_fold_moments(e0_a, e1_a, 7.0), _fold_moments(e0_b, e1_b, 5.0))
        combined, mu = cross_fitted_combination(moments, *_means_over(folds), scale=1.0, fallback=0.0)

        lam_a, lam_b = (-m.cov_e0_delta / m.var_delta for m in moments)
        w_a, w_b = 7 / 12, 5 / 12
        expected = (w_a * (e0_a.mean(axis=0) + lam_b * (e1_a - e0_a).mean(axis=0))
                    + w_b * (e0_b.mean(axis=0) + lam_a * (e1_b - e0_b).mean(axis=0)))
        np.testing.assert_allclose(combined, expected, rtol=1e-12)
        np.testing.assert_allclose(mu, w_a * lam_b + w_b * lam_a, rtol=1e-12)

    def test_swapping_the_folds_gives_the_same_result(self):
        (e0_a, e1_a), (e0_b, e1_b) = folds = _two_folds(3)
        a, b = _fold_moments(e0_a, e1_a, 7.0), _fold_moments(e0_b, e1_b, 5.0)
        means = _means_over(folds)
        combined, mu = cross_fitted_combination((a, b), *means, scale=1.0, fallback=0.0)
        swapped, mu_swapped = cross_fitted_combination((b, a), *means, scale=1.0, fallback=0.0)
        np.testing.assert_allclose(swapped, combined, rtol=1e-12)
        np.testing.assert_allclose(mu_swapped, mu, rtol=1e-12)

    def test_equal_lambdas_give_the_plain_combination_exactly(self):
        """Where the folds' lambdas agree, the result is (1 - lambda) E_0 + lambda E_1 bit for bit."""
        e0, e1 = _samples(4, n_blocks=6)
        a = _fold_moments(e0, e1, 6.0)
        b = FoldMoments(4.0, a.mean_e0 + 0.5, a.mean_e1 - 0.25, a.var_delta, a.cov_e0_delta)
        mean_e0 = (6 * a.mean_e0 + 4 * b.mean_e0) / 10
        mean_e1 = (6 * a.mean_e1 + 4 * b.mean_e1) / 10
        combined, mu = cross_fitted_combination((a, b), mean_e0, mean_e1, scale=1.0, fallback=0.0)

        lam = -a.cov_e0_delta / a.var_delta
        np.testing.assert_array_equal(mu, lam)
        np.testing.assert_array_equal(combined, (1 - lam) * mean_e0 + lam * mean_e1)

    def test_negligible_variance_in_both_folds_gives_the_fallback_estimator_exactly(self):
        rng = np.random.default_rng(5)
        zeros = np.zeros(5)
        a = FoldMoments(3.0, rng.normal(size=5), rng.normal(size=5), zeros, zeros)
        b = FoldMoments(2.0, rng.normal(size=5), rng.normal(size=5), zeros, zeros)
        mean_e0 = (3 * a.mean_e0 + 2 * b.mean_e0) / 5
        mean_e1 = (3 * a.mean_e1 + 2 * b.mean_e1) / 5
        combined, mu = cross_fitted_combination((a, b), mean_e0, mean_e1, scale=1.0, fallback=1.0)
        np.testing.assert_array_equal(mu, 1.0)
        np.testing.assert_array_equal(combined, mean_e1)

    def test_fallback_applies_in_the_degenerate_fold_only(self):
        (e0_a, e1_a), (e0_b, e1_b) = folds = _two_folds(6)
        zeros = np.zeros(5)
        degenerate = FoldMoments(7.0, e0_a.mean(axis=0), e1_a.mean(axis=0), zeros, zeros)
        live = _fold_moments(e0_b, e1_b, 5.0)
        _, mu = cross_fitted_combination((degenerate, live), *_means_over(folds), scale=1.0, fallback=1.0)

        lam_live = -live.cov_e0_delta / live.var_delta
        np.testing.assert_allclose(mu, (7 / 12) * lam_live + (5 / 12) * 1.0, rtol=1e-12)


class TestCrossFitStatistics:
    """Tests for CrossFitStatistics: running moments in two folds for cross_fitted_combination."""

    def _filled(self, seed=0, sizes=(4, 5), weight=1.0):
        statistics = CrossFitStatistics((5,), samples="blocks")
        folds = _two_folds(seed, sizes)
        for fold, (e0, e1) in enumerate(folds):
            for sample_e0, sample_e1 in zip(e0, e1):
                statistics.update(fold, sample_e1 - sample_e0, sample_e0, weight=weight)
        return statistics, folds

    def test_no_data_initially(self):
        statistics = CrossFitStatistics((5,))
        assert statistics.count == 0
        assert statistics.sum_weights == 0.0
        assert statistics.has_data is False

    def test_count_and_sum_weights_sum_over_both_folds(self):
        statistics, _ = self._filled(weight=2.0)
        assert statistics.count == 9
        assert statistics.sum_weights == 18.0
        assert statistics.has_data is True

    def test_finalise_gives_each_folds_moments_with_other_as_e0(self):
        statistics, folds = self._filled(weight=2.0)
        for got, (e0, e1), n in zip(statistics.finalise(), folds, (4, 5)):
            expected = FoldMoments(2.0 * n, *_moments(e0, e1))
            assert got.weight == expected.weight
            for name in ("mean_e0", "mean_e1", "var_delta", "cov_e0_delta"):
                np.testing.assert_allclose(getattr(got, name), getattr(expected, name), rtol=1e-12,
                                           err_msg=name)

    @pytest.mark.parametrize("sizes", [(3, 4), (4, 3), (0, 4), (2, 2)])
    def test_finalise_raises_with_fewer_than_four_samples_in_either_fold(self, sizes):
        """Below 4 samples the variance of a fold's lambda, a regression slope, is infinite."""
        statistics, _ = self._filled(sizes=sizes)
        with pytest.raises(ValueError, match=rf"at least 4 blocks in each half of the frames \(have {sizes[0]} and {sizes[1]}\)"):
            statistics.finalise()

    def test_finalise_accepts_four_samples_in_each_fold(self):
        statistics, _ = self._filled(sizes=(4, 4))
        statistics.finalise()

    def test_finalise_appends_the_remedy_to_the_error(self):
        statistics, _ = self._filled(sizes=(1, 1))
        with pytest.raises(ValueError, match=r"\(have 1 and 1\).*\. Use more frames$"):
            statistics.finalise(remedy="Use more frames")

    @pytest.mark.parametrize("fold", [-1, 2])
    def test_update_rejects_a_fold_other_than_0_or_1(self, fold):
        statistics = CrossFitStatistics((2,))
        with pytest.raises(ValueError, match="fold must be 0 or 1"):
            statistics.update(fold, np.ones(2), np.ones(2))

    def test_reset_clears_both_folds(self):
        statistics, _ = self._filled()
        statistics.reset()
        assert statistics.count == 0
        assert statistics.has_data is False
        with pytest.raises(ValueError, match=r"\(have 0 and 0\)"):
            statistics.finalise()
