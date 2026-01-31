"""
Pipeline integration tests for RDF calculations (Example 1).

These tests exercise additional RDF workflow scenarios not covered by regression tests:
- Trajectory loading verification
- Backward integration
- Unlike pairs
- Lambda combination
- Frame subsets and striding

Note: Forward integration with physical property checks is in test_regression.py
"""

import pytest
import numpy as np

from revelsMD.rdf import run_rdf, run_rdf_lambda


@pytest.mark.integration
@pytest.mark.requires_example1
class TestRDFPipelineExample1:
    """Full pipeline tests using Example 1 LJ data."""

    def test_trajectory_loads_correctly(self, example1_trajectory):
        """Verify Example 1 trajectory loads with expected properties."""
        ts = example1_trajectory

        from revelsMD.trajectories import LammpsTrajectory
        assert isinstance(ts, LammpsTrajectory)
        assert ts.units == 'lj'
        assert ts.frames > 0
        assert ts.box_x > 0
        assert ts.box_y > 0
        assert ts.box_z > 0

        # Check we can get atom indices
        type1_indices = ts.get_indices('1')
        assert len(type1_indices) > 0

    def test_rdf_backward_integration(self, example1_trajectory):
        """RDF calculation with backward integration."""
        ts = example1_trajectory

        rdf = run_rdf(
            ts, '1', '1',
            period=1, delr=0.02, from_zero=False,
            start=0, stop=5
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

        rdf = run_rdf(
            ts, '1', '2',
            period=1, delr=0.02, from_zero=True,
            start=0, stop=5
        )

        assert rdf is not None
        assert np.all(np.isfinite(rdf))

    def test_rdf_lambda_calculation(self, example1_trajectory):
        """Lambda-combined RDF calculation produces valid output."""
        ts = example1_trajectory

        rdf_lambda = run_rdf_lambda(
            ts, '1', '1',
            period=1, delr=0.02,
            start=0, stop=5
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
        assert lambda_vals[0] > 0.5, f"Lambda at r=0 should be near 1, got {lambda_vals[0]}"

        # g_lambda should have similar structure to regular g(r)
        assert np.max(g_lambda) > 1.0, "Lambda-combined g(r) should have peaks"

    def test_rdf_frame_subset(self, example1_trajectory):
        """RDF calculation on frame subset works correctly."""
        ts = example1_trajectory

        # Use only first 10 frames with coarser resolution
        rdf_subset = run_rdf(
            ts, '1', '1',
            start=0, stop=5, period=1, delr=0.05
        )

        assert rdf_subset is not None
        assert np.all(np.isfinite(rdf_subset))

    def test_rdf_with_stride(self, example1_trajectory):
        """RDF calculation with frame stride works correctly."""
        ts = example1_trajectory

        # Use every 5th frame with coarser resolution
        rdf_strided = run_rdf(
            ts, '1', '1',
            period=5, delr=0.05, start=0, stop=25
        )

        assert rdf_strided is not None
        assert np.all(np.isfinite(rdf_strided))
