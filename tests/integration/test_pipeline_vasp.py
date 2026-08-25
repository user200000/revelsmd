"""
Pipeline integration tests for VASP trajectories (Example 3).

These tests exercise the VASP workflow using BaSnF4 solid electrolyte data.
"""

import pytest
import numpy as np

from revelsMD.rdf import compute_rdf


@pytest.mark.integration
@pytest.mark.requires_vasp
class TestVASPPipelineExample3:
    """Full pipeline tests using VASP BaSnF4 data."""

    def test_ba_f_rdf(self, vasp_trajectory):
        """Ba-F RDF calculation (cation-anion correlation)."""
        ts = vasp_trajectory

        rdf = compute_rdf(
            ts, 'Ba', 'F',
            period=1, delr=0.1
        )

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))
