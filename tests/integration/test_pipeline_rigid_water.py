"""
Pipeline integration tests for rigid water molecules (Example 4).

These tests check charge data availability, charge neutrality, and the
rigid charge-density neutrality invariant -- properties not covered by
the baseline regression tests.
"""

import pytest
import numpy as np

from revelsMD.density import DensityGrid


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

    def test_rigid_charge_density_is_identically_zero(self, example4_trajectory):
        """Rigid whole-molecule charge density of SPC/E water is exactly zero.

        Rigid deposition weights each molecule by its SUMMED charge, and
        SPC/E molecules are exactly neutral, so every deposit carries zero
        weight: the counting estimator is exactly zero, and because the same
        weight multiplies the deposited forces, the force estimator is
        exactly zero by construction too. Any non-zero voxel means the
        deposition mixed atomic and molecular identities (e.g. deposited
        per-atom charges instead of the molecular sum). This is an invariant
        test, not a baseline: a regression baseline of this field would pin
        nothing, since any multiplicative error maps zero to zero.
        """
        gs = DensityGrid(example4_trajectory, 'charge', nbins=30)
        gs.accumulate(
            example4_trajectory, ['Ow', 'Hw1', 'Hw2'], kernel='triangular',
            rigid=True, start=0, stop=2
        )

        assert np.all(gs.rho_count == 0), \
            "Rigid charge counting density of neutral water must be exactly zero"
        assert np.all(gs.rho_force == 0), \
            "Rigid charge force density of neutral water must be exactly zero"
