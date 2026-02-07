"""
Pipeline integration tests for rigid water molecules (Example 4).

These tests exercise the full rigid molecule workflow using the Example 4 GROMACS data:
- MDATrajectoryState loading from trr/tpr files
- Number density at centre of mass
- Polarisation density calculation
- Cube file output
"""

import pytest
import numpy as np
from pathlib import Path

from revelsMD.density import DensityGrid
from revelsMD.rdf import RDF, compute_rdf
from .conftest import load_reference_data, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_example4
class TestRigidWaterPipelineExample4:
    """Full pipeline tests using Example 4 rigid water data."""

    def test_trajectory_loads_correctly(self, example4_trajectory):
        """Verify Example 4 trajectory loads with expected properties."""
        ts = example4_trajectory

        from revelsMD.trajectories import MDATrajectory
        assert isinstance(ts, MDATrajectory)
        assert ts.frames > 0
        assert ts.box_x > 0

        # Check we can get water atom indices
        ow_indices = ts.get_indices('Ow')
        hw1_indices = ts.get_indices('Hw1')
        hw2_indices = ts.get_indices('Hw2')

        assert len(ow_indices) > 0, "Should have oxygen atoms"
        assert len(hw1_indices) > 0, "Should have H1 atoms"
        assert len(hw2_indices) > 0, "Should have H2 atoms"

        # Water stoichiometry: O:H should be 1:2
        n_water = len(ow_indices)
        assert len(hw1_indices) == n_water, f"H1 count ({len(hw1_indices)}) should match O count ({n_water})"
        assert len(hw2_indices) == n_water, f"H2 count ({len(hw2_indices)}) should match O count ({n_water})"

    def test_charges_available(self, example4_trajectory):
        """Water trajectory should have charge data."""
        ts = example4_trajectory

        # MDATrajectoryState gets charges via MDAnalysis universe
        ow_charges = ts.get_charges('Ow')
        hw1_charges = ts.get_charges('Hw1')

        assert len(ow_charges) > 0
        assert len(hw1_charges) > 0

        # Check charge magnitudes are reasonable for water
        # SPC/E: O ~ -0.8476, H ~ +0.4238
        assert np.all(ow_charges < 0), "Oxygen should have negative charge"
        assert np.all(hw1_charges > 0), "Hydrogen should have positive charge"

    def test_number_density_at_com_small_subset(self, example4_trajectory):
        """Number density for rigid water at COM (small frame subset)."""
        ts = example4_trajectory

        gs = DensityGrid(ts, 'number', nbins=50)

        # Use very small subset for fast test
        gs.accumulate(
            ts, ['Ow', 'Hw1', 'Hw2'],
            kernel='triangular', rigid=True,
            start=0, stop=5, period=1
        )

        assert gs.count > 0  # Data has been accumulated
        assert gs.count == 5

        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert gs.rho_force.shape == (50, 50, 50)
        assert np.all(np.isfinite(gs.rho_force))

    def test_polarisation_density_x(self, example4_trajectory):
        """Polarisation density calculation (x-component)."""
        ts = example4_trajectory

        gs = DensityGrid(ts, 'polarisation', nbins=50)

        gs.accumulate(
            ts, ['Ow', 'Hw1', 'Hw2'],
            kernel='triangular', rigid=True,
            start=0, stop=5, period=1
        )

        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert np.all(np.isfinite(gs.rho_force))

        # Polarisation can be positive or negative
        # Check it has some structure (non-zero variance)
        assert np.std(gs.rho_force) > 0, "Polarisation density should have spatial variation"

    def test_number_density_larger_subset(self, example4_trajectory):
        """Number density with larger frame subset for better statistics."""
        ts = example4_trajectory

        gs = DensityGrid(ts, 'number', nbins=50)

        # Use 10 frames - enough for reasonable statistics, fast enough for tests
        gs.accumulate(
            ts, ['Ow', 'Hw1', 'Hw2'],
            kernel='triangular', rigid=True,
            start=0, stop=10, period=1
        )

        gs.get_real_density()

        assert gs.count == 10
        assert np.all(np.isfinite(gs.rho_force))

    def test_write_cube_output(self, example4_trajectory, tmp_path):
        """Cube file output produces valid file."""
        ts = example4_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(
            ts, ['Ow', 'Hw1', 'Hw2'],
            rigid=True, start=0, stop=3, period=1
        )
        gs.get_real_density()

        # Create minimal ASE atoms for cube file
        from ase import Atoms
        atoms = Atoms('OH2', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        cube_file = tmp_path / "test_water_density.cube"
        gs.write_to_cube(atoms, gs.rho_force, str(cube_file))

        assert cube_file.exists()
        assert cube_file.stat().st_size > 0

        # Basic validation of cube file format
        with open(cube_file, 'r') as f:
            lines = f.readlines()
            # Cube files start with 2 comment lines, then atom count line
            assert len(lines) > 6, "Cube file should have header and data"

    # Note: Regression tests are in test_regression.py which uses the correct
    # reference data paths (mda_example4, not rigid_water_example4).


@pytest.mark.integration
@pytest.mark.requires_example4
class TestRigidWaterRDF:
    """RDF tests using Example 4 water trajectory."""

    def test_oxygen_oxygen_rdf(self, example4_trajectory):
        """O-O RDF calculation for water."""
        ts = example4_trajectory

        rdf = compute_rdf(
            ts, 'Ow', 'Ow',
            period=1, delr=0.1, start=0, stop=5
        )

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))

        # Water O-O RDF should have first peak near 2.8 Angstrom
        # Find first peak
        peak_idx = np.argmax(rdf.g)
        peak_r = rdf.r[peak_idx]

        # Peak should be in reasonable range for water (2.5-3.5 A)
        assert 2.0 < peak_r < 4.0, \
            f"O-O RDF peak at {peak_r} A, expected near 2.8 A"

    def test_oxygen_hydrogen_rdf(self, example4_trajectory):
        """O-H RDF calculation for water."""
        ts = example4_trajectory

        rdf = compute_rdf(
            ts, 'Ow', 'Hw1',
            period=1, delr=0.1, start=0, stop=5
        )

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))


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

    def test_water_geometry_consistency(self, example4_trajectory):
        """Water molecule geometry should be consistent."""
        ts = example4_trajectory

        # Load first frame positions
        universe = ts.mdanalysis_universe
        universe.trajectory[0]

        ow_positions = universe.select_atoms('name Ow').positions
        hw1_positions = universe.select_atoms('name Hw1').positions
        hw2_positions = universe.select_atoms('name Hw2').positions

        # Check O-H distances for first few molecules
        n_check = min(10, len(ow_positions))

        for i in range(n_check):
            d_oh1 = np.linalg.norm(hw1_positions[i] - ow_positions[i])
            d_oh2 = np.linalg.norm(hw2_positions[i] - ow_positions[i])

            # O-H bond length should be ~1.0 A for SPC/E
            assert 0.8 < d_oh1 < 1.2, f"O-H1 distance {d_oh1} A out of range"
            assert 0.8 < d_oh2 < 1.2, f"O-H2 distance {d_oh2} A out of range"

    def test_density_has_solvation_structure(self, example4_trajectory):
        """Number density should show solvation shell structure."""
        ts = example4_trajectory

        gs = DensityGrid(ts, 'number', nbins=50)
        gs.accumulate(
            ts, ['Ow', 'Hw1', 'Hw2'],
            kernel='triangular', rigid=True,
            start=0, stop=5, period=1
        )
        gs.get_real_density()

        # Check that density has spatial variation (not flat)
        # This indicates solvation structure
        mean_rho = np.mean(gs.rho_force)
        std_rho = np.std(gs.rho_force)

        if mean_rho > 0:
            cv = std_rho / mean_rho
            # Should have some structure, not perfectly flat
            assert cv > 0.01, "Density should show spatial structure"
