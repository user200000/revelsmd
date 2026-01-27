"""
Pipeline integration tests for RDF calculations (Example 1).

These tests exercise the full RDF workflow using the Example 1 LJ data:
- LammpsTrajectoryState loading
- run_rdf() with different integration directions
- run_rdf_lambda() for optimal linear combination
- Regression against stored reference data
"""

import pytest
import numpy as np
from pathlib import Path

from revelsMD.revels_rdf import RevelsRDF
from .conftest import load_reference_data, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_example1
class TestRDFPipelineExample1:
    """Full pipeline tests using Example 1 LJ data."""

    def test_trajectory_loads_correctly(self, example1_trajectory):
        """Verify Example 1 trajectory loads with expected properties."""
        ts = example1_trajectory

        from revelsMD.trajectory_states import LammpsTrajectoryState
        assert isinstance(ts, LammpsTrajectoryState)
        assert ts.units == 'lj'
        assert ts.frames > 0
        assert ts.box_x > 0
        assert ts.box_y > 0
        assert ts.box_z > 0

        # Check we can get atom indices
        type1_indices = ts.get_indices('1')
        assert len(type1_indices) > 0

    def test_rdf_like_pairs_from_zero(self, example1_trajectory):
        """RDF calculation for like pairs with forward integration."""
        ts = example1_trajectory

        rdf = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005, from_zero=True
        )

        assert rdf is not None
        assert rdf.shape[0] == 2  # [r, g(r)]
        assert np.all(np.isfinite(rdf))

        # g(r) should be zero at r=0 (from_zero enforces this)
        assert rdf[1, 0] < 0.1, "g(r=0) should be near zero for from_zero=True"

        # g(r) should have a first peak (LJ fluid)
        max_gr = np.max(rdf[1])
        assert max_gr > 1.0, "LJ fluid should have g(r) peak > 1"

    def test_rdf_like_pairs_from_infinity(self, example1_trajectory):
        """RDF calculation for like pairs with backward integration."""
        ts = example1_trajectory

        rdf = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005, from_zero=False
        )

        assert rdf is not None
        assert rdf.shape[0] == 2
        assert np.all(np.isfinite(rdf))

        # Should still have LJ peak structure
        max_gr = np.max(rdf[1])
        assert max_gr > 1.0

    def test_rdf_unlike_pairs(self, example1_trajectory):
        """RDF calculation for unlike pairs (type 1 - type 2)."""
        ts = example1_trajectory

        # Check if type 2 exists
        try:
            type2_indices = ts.get_indices('2')
            has_type2 = len(type2_indices) > 0
        except (ValueError, KeyError):
            has_type2 = False

        if not has_type2:
            pytest.skip("Example 1 data has only one atom type")

        rdf = RevelsRDF.run_rdf(
            ts, '1', '2', temp=1.35,
            period=1, delr=0.005, from_zero=True
        )

        assert rdf is not None
        assert np.all(np.isfinite(rdf))

    def test_rdf_lambda_calculation(self, example1_trajectory):
        """Lambda-combined RDF calculation produces valid output."""
        ts = example1_trajectory

        rdf_lambda = RevelsRDF.run_rdf_lambda(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005
        )

        assert rdf_lambda is not None
        assert rdf_lambda.shape[1] == 3  # [r, g_lambda, lambda]
        assert np.all(np.isfinite(rdf_lambda))

        r = rdf_lambda[:, 0]
        g_lambda = rdf_lambda[:, 1]
        lambda_vals = rdf_lambda[:, 2]

        # r should be monotonically increasing
        assert np.all(np.diff(r) > 0), "r values should be increasing"

        # Lambda should approach 1 at r=0 and 0 at large r
        # (with some tolerance for numerical effects)
        assert lambda_vals[0] > 0.5, f"Lambda at r=0 should be near 1, got {lambda_vals[0]}"

        # g_lambda should have similar structure to regular g(r)
        assert np.max(g_lambda) > 1.0, "Lambda-combined g(r) should have peaks"

    def test_rdf_frame_subset(self, example1_trajectory):
        """RDF calculation on frame subset works correctly."""
        ts = example1_trajectory

        # Use only first 10 frames
        rdf_subset = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            start=0, stop=10, period=1, delr=0.01
        )

        assert rdf_subset is not None
        assert np.all(np.isfinite(rdf_subset))

    def test_rdf_with_stride(self, example1_trajectory):
        """RDF calculation with frame stride works correctly."""
        ts = example1_trajectory

        # Use every 5th frame
        rdf_strided = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=5, delr=0.01
        )

        assert rdf_strided is not None
        assert np.all(np.isfinite(rdf_strided))

    def test_instantaneous_rdf(self, example1_trajectory):
        """Single-frame (instantaneous) RDF calculation."""
        ts = example1_trajectory

        # Use period = total frames to get single-frame RDF
        rdf_instant = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=ts.frames, delr=0.01
        )

        assert rdf_instant is not None
        assert np.all(np.isfinite(rdf_instant))

        # Instantaneous RDF should still show LJ structure
        # (though noisier than averaged)
        assert np.max(rdf_instant[1]) > 0.5

    @pytest.mark.regression
    def test_rdf_regression_like_pairs(self, example1_trajectory):
        """RDF matches stored reference data."""
        ts = example1_trajectory

        rdf = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005, from_zero=True
        )

        ref = load_reference_data("rdf_example1", "rdf_like_type1.npz")

        if ref is None:
            pytest.skip("Reference data not generated yet")

        assert_arrays_close(rdf[0], ref['r'], rtol=1e-10, context="r values")
        assert_arrays_close(rdf[1], ref['g_r'], rtol=1e-6, context="g(r) values")

    @pytest.mark.regression
    def test_rdf_lambda_regression(self, example1_trajectory):
        """Lambda-combined RDF matches stored reference data."""
        ts = example1_trajectory

        rdf_lambda = RevelsRDF.run_rdf_lambda(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005
        )

        ref = load_reference_data("rdf_example1", "rdf_lambda_type1.npz")

        if ref is None:
            pytest.skip("Reference data not generated yet")

        assert_arrays_close(rdf_lambda[:, 0], ref['r'], rtol=1e-10, context="r values")
        assert_arrays_close(rdf_lambda[:, 1], ref['g_lambda'], rtol=1e-6, context="g_lambda values")
        assert_arrays_close(rdf_lambda[:, 2], ref['lambda_'], rtol=1e-6, context="lambda values")


@pytest.mark.integration
@pytest.mark.requires_example1
class TestRDFPhysicalProperties:
    """Tests validating physical properties of RDF results."""

    def test_rdf_normalisation(self, example1_trajectory):
        """RDF should approach 1 at large r for homogeneous fluid."""
        ts = example1_trajectory

        rdf = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005, from_zero=True
        )

        # Find bulk region (r > 3 sigma for LJ fluid)
        # LJ sigma ~ 1.0 in reduced units
        bulk_mask = rdf[0] > 3.0

        if np.any(bulk_mask):
            bulk_gr = rdf[1][bulk_mask]
            mean_bulk = np.mean(bulk_gr)

            # Should be close to 1 for homogeneous fluid
            assert abs(mean_bulk - 1.0) < 0.2, \
                f"Bulk g(r) = {mean_bulk}, expected ~1.0"

    def test_rdf_first_peak_position(self, example1_trajectory):
        """First RDF peak should be near LJ sigma (r ~ 1.0 in reduced units)."""
        ts = example1_trajectory

        rdf = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.005, from_zero=True
        )

        # Find first peak (maximum in r < 2)
        short_range_mask = rdf[0] < 2.0
        short_range_r = rdf[0][short_range_mask]
        short_range_gr = rdf[1][short_range_mask]

        if len(short_range_gr) > 0:
            peak_idx = np.argmax(short_range_gr)
            peak_r = short_range_r[peak_idx]

            # First peak should be near r ~ 1.0 (LJ sigma)
            assert 0.8 < peak_r < 1.5, \
                f"First peak at r = {peak_r}, expected near 1.0"

    def test_forward_backward_bulk_agreement(self, example1_trajectory):
        """Forward and backward integration should agree in bulk region."""
        ts = example1_trajectory

        rdf_forward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.01, from_zero=True
        )

        rdf_backward = RevelsRDF.run_rdf(
            ts, '1', '1', temp=1.35,
            period=1, delr=0.01, from_zero=False
        )

        # Compare in bulk region
        bulk_mask = (rdf_forward[0] > 2.5) & (rdf_forward[0] < 4.0)

        if np.any(bulk_mask):
            mean_forward = np.mean(rdf_forward[1][bulk_mask])
            mean_backward = np.mean(rdf_backward[1][bulk_mask])

            # Should be similar in bulk
            assert abs(mean_forward - mean_backward) < 0.15, \
                f"Forward ({mean_forward:.3f}) and backward ({mean_backward:.3f}) disagree in bulk"
