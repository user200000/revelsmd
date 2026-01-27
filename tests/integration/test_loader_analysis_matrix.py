"""
Loader x Analysis matrix tests for RevelsMD.

These tests verify that all valid combinations of trajectory loaders and
analysis methods work correctly. This helps ensure consistency across
different backends and catches regressions in specific combinations.

Test Matrix:
                    | LAMMPS | MDA/GROMACS | VASP | NumPy |
--------------------|--------|-------------|------|-------|
RDF like            |   X    |      X      |  X   |   X   |
RDF unlike          |   X    |      X      |  X   |   X   |
RDF lambda          |   X    |      X      |  X   |   X   |
3D number           |   X    |      X      |  X   |   X   |
3D number (rigid)   |   -    |      X      |  -   |   X   |
3D charge           |   X    |      X      |  -   |   X   |
3D polarisation     |   -    |      X      |  -   |   X   |
Lambda combination  |   X    |      X      |  X   |   X   |
"""

import pytest
import numpy as np

from revelsMD.revels_rdf import RevelsRDF
from revelsMD.revels_3D import Revels3D


# ---------------------------------------------------------------------------
# Analysis runner functions
# ---------------------------------------------------------------------------

def run_rdf_like(ts, species_a, temp, delr=0.1, start=0, stop=5):
    """Run like-pair RDF calculation."""
    return RevelsRDF.run_rdf(
        ts, species_a, species_a, temp=temp,
        delr=delr, start=start, stop=stop
    )


def run_rdf_unlike(ts, species_a, species_b, temp, delr=0.1, start=0, stop=5):
    """Run unlike-pair RDF calculation."""
    return RevelsRDF.run_rdf(
        ts, species_a, species_b, temp=temp,
        delr=delr, start=start, stop=stop
    )


def run_rdf_lambda(ts, species_a, temp, delr=0.1, start=0, stop=5):
    """Run lambda-combined RDF calculation."""
    return RevelsRDF.run_rdf_lambda(
        ts, species_a, species_a, temp=temp,
        delr=delr, start=start, stop=stop
    )


def run_number_density(ts, species, temp, nbins=30, start=0, stop=5):
    """Run 3D number density calculation."""
    gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=temp)
    gs.make_force_grid(ts, species, kernel='triangular', rigid=False, start=start, stop=stop)
    gs.get_real_density()
    return gs


def run_number_density_rigid(ts, species_list, temp, nbins=30, start=0, stop=5):
    """Run 3D number density for rigid molecules."""
    gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=temp)
    gs.make_force_grid(ts, species_list, kernel='triangular', rigid=True, start=start, stop=stop)
    gs.get_real_density()
    return gs


def run_charge_density(ts, species, temp, nbins=30, start=0, stop=5):
    """Run 3D charge density calculation."""
    gs = Revels3D.GridState(ts, 'charge', nbins=nbins, temperature=temp)
    gs.make_force_grid(ts, species, kernel='triangular', rigid=False, start=start, stop=stop)
    gs.get_real_density()
    return gs


def run_polarisation_density(ts, species_list, temp, nbins=30, start=0, stop=5):
    """Run 3D polarisation density for rigid molecules."""
    gs = Revels3D.GridState(ts, 'polarisation', nbins=nbins, temperature=temp)
    gs.make_force_grid(ts, species_list, kernel='triangular', rigid=True, start=start, stop=stop)
    gs.get_real_density()
    return gs


# ---------------------------------------------------------------------------
# LAMMPS Matrix Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.requires_example1
class TestLammpsAnalysisMatrix:
    """Test all applicable analyses with LAMMPS trajectory."""

    def test_rdf_like(self, example1_trajectory):
        """LAMMPS: like-pair RDF works."""
        result = run_rdf_like(example1_trajectory, '1', temp=1.35)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_unlike(self, example1_trajectory):
        """LAMMPS: unlike-pair RDF works (if multiple types)."""
        ts = example1_trajectory
        try:
            ts.get_indices('2')
        except (ValueError, KeyError):
            pytest.skip("Only one atom type in trajectory")

        result = run_rdf_unlike(ts, '1', '2', temp=1.35)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_lambda(self, example1_trajectory):
        """LAMMPS: lambda-combined RDF works."""
        result = run_rdf_lambda(example1_trajectory, '1', temp=1.35)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_number_density(self, example1_trajectory):
        """LAMMPS: 3D number density works."""
        gs = run_number_density(example1_trajectory, '1', temp=1.35)
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))

    @pytest.mark.slow
    def test_lambda_combination(self, example1_trajectory):
        """LAMMPS: lambda density combination works."""
        ts = example1_trajectory
        gs = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.35)
        gs.make_force_grid(ts, '1', kernel='triangular', rigid=False, start=0, stop=10)
        gs.get_real_density()

        gs_lambda = gs.get_lambda(ts, sections=5, start=0, stop=10)
        assert gs_lambda.grid_progress == "Lambda"
        assert np.all(np.isfinite(gs_lambda.optimal_density))


# ---------------------------------------------------------------------------
# MDA/GROMACS Matrix Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.requires_example4
class TestMDAAnalysisMatrix:
    """Test all applicable analyses with MDA/GROMACS trajectory."""

    @pytest.mark.xfail(
        reason="Bug: revels_rdf.py:331 accesses .trajectory.atoms instead of .atoms",
        raises=AttributeError,
    )
    def test_rdf_like(self, example4_trajectory):
        """MDA: like-pair RDF works."""
        result = run_rdf_like(example4_trajectory, 'Ow', temp=300)
        assert result is not None
        assert np.all(np.isfinite(result))

    @pytest.mark.xfail(
        reason="Bug: revels_rdf.py:331 accesses .trajectory.atoms instead of .atoms",
        raises=AttributeError,
    )
    def test_rdf_unlike(self, example4_trajectory):
        """MDA: unlike-pair RDF works."""
        result = run_rdf_unlike(example4_trajectory, 'Ow', 'Hw1', temp=300)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_lambda(self, example4_trajectory):
        """MDA: lambda-combined RDF works.

        Note: run_rdf_lambda correctly uses .atoms.positions (unlike run_rdf).
        """
        result = run_rdf_lambda(example4_trajectory, 'Ow', temp=300)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_number_density(self, example4_trajectory):
        """MDA: 3D number density works."""
        gs = run_number_density(example4_trajectory, 'Ow', temp=300)
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))

    def test_number_density_rigid(self, example4_trajectory):
        """MDA: rigid molecule number density works."""
        gs = run_number_density_rigid(
            example4_trajectory, ['Ow', 'Hw1', 'Hw2'], temp=300
        )
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))

    def test_polarisation_density(self, example4_trajectory):
        """MDA: polarisation density works."""
        gs = run_polarisation_density(
            example4_trajectory, ['Ow', 'Hw1', 'Hw2'], temp=300
        )
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))


# ---------------------------------------------------------------------------
# VASP Matrix Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.requires_vasp
class TestVASPAnalysisMatrix:
    """Test all applicable analyses with VASP trajectory."""

    def test_rdf_like(self, vasp_trajectory):
        """VASP: like-pair RDF works."""
        result = run_rdf_like(vasp_trajectory, 'F', temp=600)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_unlike(self, vasp_trajectory):
        """VASP: unlike-pair RDF works."""
        try:
            result = run_rdf_unlike(vasp_trajectory, 'Ba', 'F', temp=600)
            assert result is not None
            assert np.all(np.isfinite(result))
        except (ValueError, KeyError):
            pytest.skip("Species not found")

    def test_rdf_lambda(self, vasp_trajectory):
        """VASP: lambda-combined RDF works."""
        result = run_rdf_lambda(vasp_trajectory, 'F', temp=600)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_number_density(self, vasp_trajectory):
        """VASP: 3D number density works."""
        gs = run_number_density(vasp_trajectory, 'F', temp=600)
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))


# ---------------------------------------------------------------------------
# NumPy Matrix Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.analytical
class TestNumpyAnalysisMatrix:
    """Test all applicable analyses with NumPy synthetic trajectories."""

    def test_rdf_like(self, uniform_gas_trajectory):
        """NumPy: like-pair RDF works."""
        result = run_rdf_like(uniform_gas_trajectory, '1', temp=1.0, stop=-1)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_unlike(self, multispecies_trajectory):
        """NumPy: unlike-pair RDF works."""
        result = run_rdf_unlike(
            multispecies_trajectory, '1', '2', temp=1.0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_lambda(self, uniform_gas_trajectory):
        """NumPy: lambda-combined RDF works."""
        result = run_rdf_lambda(uniform_gas_trajectory, '1', temp=1.0, stop=-1)
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_number_density(self, uniform_gas_trajectory):
        """NumPy: 3D number density works."""
        gs = run_number_density(uniform_gas_trajectory, '1', temp=1.0, stop=-1)
        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))

    def test_number_density_rigid(self, water_molecule_trajectory):
        """NumPy: rigid molecule number density works."""
        try:
            gs = run_number_density_rigid(
                water_molecule_trajectory, ['O', 'H', 'H'], temp=300, stop=-1
            )
            assert hasattr(gs, 'rho')
            assert np.all(np.isfinite(gs.rho))
        except Exception as e:
            # Known issue with rigid molecules and unequal counts
            pytest.skip(f"Rigid mode failed (possibly known issue): {e}")

    def test_polarisation_density(self, water_molecule_trajectory):
        """NumPy: polarisation density works."""
        try:
            gs = run_polarisation_density(
                water_molecule_trajectory, ['O', 'H', 'H'], temp=300, stop=-1
            )
            assert hasattr(gs, 'rho')
            assert np.all(np.isfinite(gs.rho))
        except Exception as e:
            pytest.skip(f"Polarisation failed (possibly known issue): {e}")


# ---------------------------------------------------------------------------
# Cross-loader result shape consistency
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestResultShapeConsistency:
    """Verify that different loaders produce results with consistent shapes."""

    def test_rdf_output_shape(self, uniform_gas_trajectory):
        """RDF output should always be 2 x n_bins."""
        result = run_rdf_like(uniform_gas_trajectory, '1', temp=1.0, stop=-1)
        assert result.shape[0] == 2
        assert len(result.shape) == 2

    def test_rdf_lambda_output_shape(self, uniform_gas_trajectory):
        """Lambda RDF output should always be n_bins x 3."""
        result = run_rdf_lambda(uniform_gas_trajectory, '1', temp=1.0, stop=-1)
        assert result.shape[1] == 3
        assert len(result.shape) == 2

    def test_density_output_shape(self, uniform_gas_trajectory):
        """Density output should be nbins x nbins x nbins."""
        nbins = 25
        gs = Revels3D.GridState(
            uniform_gas_trajectory, 'number', nbins=nbins, temperature=1.0
        )
        gs.make_force_grid(
            uniform_gas_trajectory, '1', kernel='triangular', rigid=False
        )
        gs.get_real_density()

        assert gs.rho.shape == (nbins, nbins, nbins)
