"""Tests for revelsMD.statistics module."""

import numpy as np

from revelsMD.statistics import combine_estimators, compute_lambda_weights


class TestComputeLambdaWeights:
    """Tests for compute_lambda_weights function."""

    def test_basic_computation(self):
        """Standard case: lambda = cov / var."""
        variance = np.array([1.0, 2.0, 4.0])
        covariance = np.array([0.5, 1.0, 2.0])
        result = compute_lambda_weights(variance, covariance)
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_zero_variance_uses_replacement(self):
        """Zero variance locations get replacement value."""
        variance = np.array([1.0, 0.0, 2.0])
        covariance = np.array([0.5, 0.5, 1.0])
        result = compute_lambda_weights(variance, covariance)
        expected = np.array([0.5, 0.0, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_zero_variance_custom_replacement(self):
        """Custom replacement value for zero variance."""
        variance = np.array([0.0])
        covariance = np.array([1.0])
        result = compute_lambda_weights(
            variance, covariance, zero_variance_replacement=0.5
        )
        assert result[0] == 0.5

    def test_nan_in_covariance_produces_replacement(self):
        """NaN in covariance produces replacement value."""
        variance = np.array([1.0, 1.0])
        covariance = np.array([0.5, np.nan])
        result = compute_lambda_weights(variance, covariance)
        assert result[0] == 0.5
        assert result[1] == 0.0

    def test_inf_result_produces_replacement(self):
        """Inf from division produces replacement value."""
        import warnings
        # Very small variance can produce inf
        variance = np.array([1e-320])  # subnormal
        covariance = np.array([1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = compute_lambda_weights(variance, covariance)
        assert np.isfinite(result[0])

    def test_negative_covariance(self):
        """Negative covariance produces negative weights."""
        variance = np.array([1.0])
        covariance = np.array([-0.5])
        result = compute_lambda_weights(variance, covariance)
        assert result[0] == -0.5

    def test_multidimensional_1d_array(self):
        """Works with 1D arrays."""
        variance = np.ones(10)
        covariance = np.full(10, 0.5)
        result = compute_lambda_weights(variance, covariance)
        assert result.shape == (10,)
        np.testing.assert_allclose(result, 0.5)

    def test_multidimensional_3d_array(self):
        """Works with 3D arrays (typical for density grids)."""
        variance = np.ones((3, 4, 5))
        covariance = np.full((3, 4, 5), 0.5)
        result = compute_lambda_weights(variance, covariance)
        assert result.shape == (3, 4, 5)
        np.testing.assert_allclose(result, 0.5)

    def test_mixed_zero_and_nonzero_variance(self):
        """Handles arrays with mix of zero and non-zero variance."""
        variance = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        covariance = np.array([1.0, 1.0, 2.0, 1.0, 3.0])
        result = compute_lambda_weights(variance, covariance)
        expected = np.array([0.0, 1.0, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_all_zero_variance(self):
        """All zero variance returns all replacement values."""
        variance = np.zeros(5)
        covariance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_lambda_weights(variance, covariance)
        np.testing.assert_allclose(result, 0.0)

    def test_preserves_dtype_float64(self):
        """Output dtype matches input dtype."""
        variance = np.array([1.0, 2.0], dtype=np.float64)
        covariance = np.array([0.5, 1.0], dtype=np.float64)
        result = compute_lambda_weights(variance, covariance)
        assert result.dtype == np.float64


class TestCombineEstimators:
    """Tests for combine_estimators function."""

    def test_basic_combination(self):
        """Standard linear combination."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        weights = np.array([0.5, 0.5])
        result = combine_estimators(a, b, weights)
        # (1*0.5 + 3*0.5) = 2.0, (2*0.5 + 4*0.5) = 3.0
        expected = np.array([2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_weights_zero_returns_estimator_a(self):
        """Weight 0 returns estimator_a unchanged."""
        a = np.array([1.0, 2.0])
        b = np.array([10.0, 20.0])
        weights = np.array([0.0, 0.0])
        result = combine_estimators(a, b, weights)
        np.testing.assert_allclose(result, a)

    def test_weights_one_returns_estimator_b(self):
        """Weight 1 returns estimator_b unchanged."""
        a = np.array([1.0, 2.0])
        b = np.array([10.0, 20.0])
        weights = np.array([1.0, 1.0])
        result = combine_estimators(a, b, weights)
        np.testing.assert_allclose(result, b)

    def test_sanitise_replaces_nan(self):
        """NaN in input is replaced with 0.0 when sanitise_output=True."""
        a = np.array([np.nan, 2.0])
        b = np.array([3.0, 4.0])
        weights = np.array([0.5, 0.5])
        result = combine_estimators(a, b, weights, sanitise_output=True)
        assert np.isfinite(result[0])
        assert result[0] == 0.0

    def test_sanitise_replaces_inf(self):
        """Inf in output is replaced with 0.0 when sanitise_output=True."""
        a = np.array([np.inf, 2.0])
        b = np.array([3.0, 4.0])
        weights = np.array([0.5, 0.5])
        result = combine_estimators(a, b, weights, sanitise_output=True)
        assert np.isfinite(result[0])
        assert result[0] == 0.0

    def test_sanitise_disabled_preserves_nan(self):
        """NaN is preserved when sanitise_output=False."""
        a = np.array([np.nan])
        b = np.array([1.0])
        weights = np.array([0.0])  # Should return a
        result = combine_estimators(a, b, weights, sanitise_output=False)
        assert np.isnan(result[0])

    def test_sanitise_disabled_preserves_inf(self):
        """Inf is preserved when sanitise_output=False."""
        a = np.array([np.inf])
        b = np.array([1.0])
        weights = np.array([0.0])
        result = combine_estimators(a, b, weights, sanitise_output=False)
        assert np.isinf(result[0])

    def test_broadcasting_scalar_weight(self):
        """Scalar weight broadcasts to array."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        result = combine_estimators(a, b, 0.5)
        expected = np.array([5.5, 11.0, 16.5])
        np.testing.assert_allclose(result, expected)

    def test_multidimensional_3d_array(self):
        """Works with 3D arrays."""
        a = np.ones((2, 3, 4))
        b = np.full((2, 3, 4), 3.0)
        weights = np.full((2, 3, 4), 0.5)
        result = combine_estimators(a, b, weights)
        assert result.shape == (2, 3, 4)
        np.testing.assert_allclose(result, 2.0)

    def test_varying_weights(self):
        """Different weights at different positions."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([5.0, 5.0, 5.0])
        weights = np.array([0.0, 0.5, 1.0])
        result = combine_estimators(a, b, weights)
        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_negative_weights(self):
        """Negative weights extrapolate beyond estimator_a."""
        a = np.array([2.0])
        b = np.array([4.0])
        weights = np.array([-0.5])
        result = combine_estimators(a, b, weights)
        # 2.0 * (1 - (-0.5)) + 4.0 * (-0.5) = 2.0 * 1.5 - 2.0 = 1.0
        expected = np.array([1.0])
        np.testing.assert_allclose(result, expected)

    def test_weights_greater_than_one(self):
        """Weights > 1 extrapolate beyond estimator_b."""
        a = np.array([2.0])
        b = np.array([4.0])
        weights = np.array([1.5])
        result = combine_estimators(a, b, weights)
        # 2.0 * (1 - 1.5) + 4.0 * 1.5 = 2.0 * (-0.5) + 6.0 = 5.0
        expected = np.array([5.0])
        np.testing.assert_allclose(result, expected)


class TestIntegration:
    """Integration tests combining both functions."""

    def test_rdf_style_workflow(self):
        """Simulate RDF-style lambda computation workflow."""
        # Simulate per-frame data
        n_frames = 10
        n_bins = 50
        np.random.seed(42)

        # Generate mock estimator data
        base_zero = np.random.randn(n_frames, n_bins) * 0.1 + 0.5
        base_inf = np.random.randn(n_frames, n_bins) * 0.1 + 0.5

        # Compute expectations
        exp_zero = np.mean(base_zero, axis=0)
        exp_inf = np.mean(base_inf, axis=0)
        exp_delta = exp_inf - exp_zero

        # Compute per-frame delta
        base_delta = base_inf - base_zero

        # Compute variance and covariance
        var_del = np.mean((base_delta - exp_delta) ** 2, axis=0)
        cov_inf = np.mean((base_delta - exp_delta) * (base_inf - exp_inf), axis=0)

        # Use statistics module
        weights = compute_lambda_weights(var_del, cov_inf)
        per_frame_combined = combine_estimators(base_inf, base_zero, weights)
        result = np.mean(per_frame_combined, axis=0)

        # Verify output is finite and reasonable
        assert np.all(np.isfinite(weights))
        assert np.all(np.isfinite(result))
        assert result.shape == (n_bins,)

    def test_density_style_workflow(self):
        """Simulate density grid lambda computation workflow."""
        # Simulate 3D grid data
        grid_shape = (5, 5, 5)
        np.random.seed(42)

        # Generate mock density data
        expected_rho = np.random.randn(*grid_shape) * 0.1 + 1.0
        expected_particle = np.random.randn(*grid_shape) * 0.1 + 1.0

        # Simulate variance and covariance buffers
        var_buffer = np.abs(np.random.randn(*grid_shape)) + 0.01
        cov_buffer_force = np.random.randn(*grid_shape) * 0.1

        # Add some zero-variance voxels (edge case)
        var_buffer[0, 0, 0] = 0.0
        var_buffer[2, 2, 2] = 0.0

        # Use statistics module (density convention: 1 - lambda)
        lambda_raw = compute_lambda_weights(var_buffer, cov_buffer_force)
        combination = 1.0 - lambda_raw
        optimal_density = combine_estimators(
            expected_particle, expected_rho, combination
        )

        # Verify output is finite
        assert np.all(np.isfinite(lambda_raw))
        assert np.all(np.isfinite(combination))
        assert np.all(np.isfinite(optimal_density))

        # Verify zero-variance voxels were handled
        assert lambda_raw[0, 0, 0] == 0.0
        assert lambda_raw[2, 2, 2] == 0.0
