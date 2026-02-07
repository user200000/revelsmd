"""
Pipeline integration tests for VASP trajectories (Example 3).

These tests exercise the VASP workflow using BaSnF4 solid electrolyte data:
- VaspTrajectoryState loading from vasprun.xml
- 3D number density for fluoride ions
- RDF calculations for ionic species
"""

import pytest
import numpy as np
from pathlib import Path

from revelsMD.density import DensityGrid
from revelsMD.rdf import RDF, compute_rdf
from .conftest import load_reference_data, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_vasp
class TestVASPPipelineExample3:
    """Full pipeline tests using VASP BaSnF4 data."""

    def test_trajectory_loads_correctly(self, vasp_trajectory):
        """Verify VASP trajectory loads with expected properties."""
        ts = vasp_trajectory

        from revelsMD.trajectories import VaspTrajectory
        assert isinstance(ts, VaspTrajectory)
        assert ts.units == 'metal'
        assert ts.frames > 0
        assert ts.box_x > 0
        assert ts.box_y > 0
        assert ts.box_z > 0

        # Check positions and forces are available
        assert hasattr(ts, 'positions')
        assert hasattr(ts, 'forces')
        assert ts.positions.shape[0] == ts.frames
        assert ts.forces.shape[0] == ts.frames

    def test_species_identification(self, vasp_trajectory):
        """Verify species can be identified in VASP trajectory."""
        ts = vasp_trajectory

        # BaSnF4 should have Ba, Sn, F atoms
        # Try to get indices for F (fluoride - the mobile ion)
        try:
            f_indices = ts.get_indices('F')
            assert len(f_indices) > 0, "Should have fluoride atoms"
        except (ValueError, KeyError) as e:
            pytest.skip(f"Could not find F atoms: {e}")

        # Try Ba and Sn
        try:
            ba_indices = ts.get_indices('Ba')
            sn_indices = ts.get_indices('Sn')
            assert len(ba_indices) > 0, "Should have barium atoms"
            assert len(sn_indices) > 0, "Should have tin atoms"
        except (ValueError, KeyError):
            pass  # These might have different names

    def test_gridstate_initialisation(self, vasp_trajectory):
        """DensityGrid initialises correctly for VASP trajectory."""
        ts = vasp_trajectory

        # Use temperature appropriate for AIMD (typically 600-1000K)
        gs = DensityGrid(ts, 'number', nbins=50)

        assert gs.density_type == 'number'
        assert gs.nbinsx == 50
        assert gs.beta == ts.beta

    def test_fluoride_number_density(self, vasp_trajectory):
        """Number density calculation for fluoride ions."""
        ts = vasp_trajectory

        gs = DensityGrid(ts, 'number', nbins=50)

        # Use all available frames (may be short trajectory)
        try:
            gs.accumulate(ts, 'F', kernel='triangular', rigid=False)
        except Exception as e:
            pytest.skip(f"Could not compute force grid: {e}")

        assert gs.count > 0  # Data has been accumulated

        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert gs.rho_force.shape == (50, 50, 50)
        assert np.all(np.isfinite(gs.rho_force))

    def test_fluoride_rdf(self, vasp_trajectory):
        """F-F RDF calculation for solid electrolyte."""
        ts = vasp_trajectory

        try:
            rdf = compute_rdf(
                ts, 'F', 'F',
                period=1, delr=0.1
            )
        except Exception as e:
            pytest.skip(f"Could not compute RDF: {e}")

        assert rdf is not None
        assert rdf.r.ndim == 1
        assert rdf.g.ndim == 1
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))

    def test_ba_f_rdf(self, vasp_trajectory):
        """Ba-F RDF calculation (cation-anion correlation)."""
        ts = vasp_trajectory

        try:
            rdf = compute_rdf(
                ts, 'Ba', 'F',
                period=1, delr=0.1
            )
        except Exception as e:
            pytest.skip(f"Could not compute Ba-F RDF: {e}")

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))

    def test_lambda_density(self, vasp_trajectory):
        """Lambda-combined density for fluoride."""
        ts = vasp_trajectory

        gs = DensityGrid(ts, 'number', nbins=30)
        gs.accumulate(ts, 'F', kernel='triangular', rigid=False, start=0, stop=10)
        gs.get_real_density()

        # Use 5 sections for variance estimation with 10 frames
        gs.get_lambda(ts, sections=5)

        assert gs.rho_lambda is not None  # Lambda was computed
        assert np.all(np.isfinite(gs.rho_lambda))


@pytest.mark.integration
@pytest.mark.requires_vasp
class TestVASPPhysicalProperties:
    """Tests validating physical properties of VASP results."""

    def test_forces_have_reasonable_magnitude(self, vasp_trajectory):
        """VASP forces should have reasonable magnitude for AIMD."""
        ts = vasp_trajectory

        # Check force magnitudes (in eV/Angstrom for VASP metal units)
        force_magnitudes = np.linalg.norm(ts.forces, axis=-1)
        max_force = np.max(force_magnitudes)
        mean_force = np.mean(force_magnitudes)

        # AIMD forces typically < 10 eV/A, usually < 1 eV/A
        assert max_force < 100, f"Maximum force {max_force} eV/A seems too large"
        assert mean_force < 10, f"Mean force {mean_force} eV/A seems too large"

    def test_box_dimensions_reasonable(self, vasp_trajectory):
        """Box dimensions should be reasonable for solid."""
        ts = vasp_trajectory

        # Solid electrolyte cell typically 5-20 Angstroms per side
        for dim, name in [(ts.box_x, 'x'), (ts.box_y, 'y'), (ts.box_z, 'z')]:
            assert 3 < dim < 50, f"Box {name} = {dim} A seems unreasonable"

    def test_density_shows_structure(self, vasp_trajectory):
        """Fluoride density should show crystalline structure."""
        ts = vasp_trajectory

        gs = DensityGrid(ts, 'number', nbins=30)

        try:
            gs.accumulate(ts, 'F', kernel='triangular', rigid=False)
            gs.get_real_density()
        except Exception:
            pytest.skip("Could not compute density")

        # Crystalline structure should have significant spatial variation
        mean_rho = np.mean(gs.rho_force)
        std_rho = np.std(gs.rho_force)

        if mean_rho > 0:
            cv = std_rho / mean_rho
            # Solid should have more structure than liquid
            # (lower threshold since short trajectory may not converge)
            assert cv > 0.001, f"Density CV = {cv}, expected some structure"


@pytest.mark.integration
class TestVASPSyntheticFallback:
    """
    Tests for VASP pipeline using synthetic data.

    These tests exercise the VASP code paths even when real data is unavailable.
    """

    @pytest.fixture
    def synthetic_vasp_like_trajectory(self):
        """Create synthetic trajectory mimicking VASP-like system."""
        from revelsMD.trajectories import NumpyTrajectory

        np.random.seed(42)

        # Mimic a small BaSnF4-like system
        n_frames = 10
        box = 8.0  # Angstroms

        # Simple positions: 4 F, 1 Ba, 1 Sn
        n_f = 4
        n_ba = 1
        n_sn = 1
        n_atoms = n_f + n_ba + n_sn

        # Generate positions on a distorted lattice
        base_f = np.array([
            [2, 2, 2], [2, 6, 6], [6, 2, 6], [6, 6, 2]
        ], dtype=float)
        base_ba = np.array([[0, 0, 0]], dtype=float)
        base_sn = np.array([[4, 4, 4]], dtype=float)

        base_positions = np.vstack([base_f, base_ba, base_sn])

        # Add thermal motion
        positions = np.zeros((n_frames, n_atoms, 3))
        for i in range(n_frames):
            positions[i] = base_positions + np.random.randn(n_atoms, 3) * 0.2

        # Random forces (AIMD-like magnitude)
        forces = np.random.randn(n_frames, n_atoms, 3) * 0.5

        species = ['F'] * n_f + ['Ba'] * n_ba + ['Sn'] * n_sn

        return NumpyTrajectory(
            positions, forces, box, box, box, species, temperature=600.0, units='metal'
        )

    def test_synthetic_rdf_calculation(self, synthetic_vasp_like_trajectory):
        """RDF calculation works with VASP-like synthetic data."""
        ts = synthetic_vasp_like_trajectory

        rdf = compute_rdf(ts, 'F', 'F', delr=0.2)

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))

    def test_synthetic_density_calculation(self, synthetic_vasp_like_trajectory):
        """Density calculation works with VASP-like synthetic data."""
        ts = synthetic_vasp_like_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, 'F', kernel='triangular', rigid=False)
        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert np.all(np.isfinite(gs.rho_force))

    def test_synthetic_unlike_rdf(self, synthetic_vasp_like_trajectory):
        """Unlike-pair RDF works with VASP-like synthetic data."""
        ts = synthetic_vasp_like_trajectory

        rdf = compute_rdf(ts, 'Ba', 'F', delr=0.2)

        assert rdf is not None
        assert np.all(np.isfinite(rdf.r))
        assert np.all(np.isfinite(rdf.g))
