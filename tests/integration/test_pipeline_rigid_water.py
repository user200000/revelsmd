"""
Pipeline integration tests for rigid water molecules (Example 4).

These tests check charge data availability and charge neutrality, which are
loader properties not covered by regression tests.
"""

import pytest
import numpy as np


@pytest.mark.integration
@pytest.mark.requires_example4
class TestRigidWaterPipelineExample4:
    """Full pipeline tests using Example 4 rigid water data."""

    def test_charges_available(self, example4_trajectory):
        """Water trajectory should have charge data."""
        ts = example4_trajectory

        # MDATrajectory gets charges via MDAnalysis universe
        ow_charges = ts.get_charges('Ow')
        hw1_charges = ts.get_charges('Hw1')

        assert len(ow_charges) > 0
        assert len(hw1_charges) > 0

        # Check charge magnitudes are reasonable for water
        # SPC/E: O ~ -0.8476, H ~ +0.4238
        assert np.all(ow_charges < 0), "Oxygen should have negative charge"
        assert np.all(hw1_charges > 0), "Hydrogen should have positive charge"


@pytest.mark.integration
@pytest.mark.requires_example4
class TestRigidWaterPhysicalProperties:
    """Tests validating physical properties of rigid water results."""

    def test_charge_neutrality(self, example4_trajectory):
        """Total system charge should be neutral."""
        ts = example4_trajectory

        ow_charges = ts.get_charges('Ow')
        hw1_charges = ts.get_charges('Hw1')
        hw2_charges = ts.get_charges('Hw2')

        total_charge = np.sum(ow_charges) + np.sum(hw1_charges) + np.sum(hw2_charges)

        assert abs(total_charge) < 1e-6, \
            f"System should be charge neutral, got total charge {total_charge}"
