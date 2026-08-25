"""
Pipeline integration tests for 3D number density (Example 2).

These tests exercise workflow scenarios not covered by regression tests:
- accumulate() with the box kernel
- Lambda estimation via compute_lambda
"""

import pytest
import numpy as np

from revelsMD.density import DensityGrid


@pytest.mark.integration
@pytest.mark.requires_example2
class TestNumberDensityPipelineExample2:
    """Full pipeline tests using Example 2 LJ 3D data."""

    def test_box_kernel_alternative(self, example2_trajectory):
        """Box kernel produces valid (though higher variance) output."""
        ts = example2_trajectory

        gs = DensityGrid(ts, 'number', nbins=50)
        gs.accumulate(ts, '2', kernel='box', rigid=False, start=0, stop=5)

        assert gs.rho_force is not None
        assert np.all(np.isfinite(gs.rho_force))

    def test_lambda_combination(self, example2_trajectory):
        """Lambda combination produces valid optimal density."""
        ts = example2_trajectory

        gs = DensityGrid(ts, 'number', nbins=30)
        gs.accumulate(
            ts, '2', kernel='triangular', rigid=False,
            start=0, stop=10, compute_lambda=True,
        )

        assert gs.rho_lambda is not None
        assert np.all(np.isfinite(gs.rho_lambda))
