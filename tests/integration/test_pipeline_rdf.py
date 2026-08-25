"""
Pipeline integration tests for RDF calculations (Example 1).

These tests exercise additional RDF workflow scenarios not covered by regression tests:
- Unlike pairs
- Frame striding

Note: Forward integration with physical property checks is in test_regression.py
"""

import pytest
import numpy as np

from revelsMD.rdf import compute_rdf


@pytest.mark.integration
@pytest.mark.requires_example1
class TestRDFPipelineExample1:
    """Full pipeline tests using Example 1 LJ data."""

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

        rdf = compute_rdf(
            ts, '1', '2',
            period=1, delr=0.02, integration='forward',
            start=0, stop=5
        )

        assert rdf.r is not None
        assert np.all(np.isfinite(rdf.g))

    def test_rdf_with_stride(self, example1_trajectory):
        """RDF calculation with frame stride works correctly."""
        ts = example1_trajectory

        # Use every 2nd frame with coarser resolution
        rdf = compute_rdf(
            ts, '1', '1',
            period=2, delr=0.05, start=0, stop=10
        )

        assert rdf.r is not None
        assert np.all(np.isfinite(rdf.g))
