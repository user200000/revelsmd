"""
Loader x Analysis matrix tests for RevelsMD.

These tests verify result shape consistency and NumPy backend functionality.
LAMMPS, MDA, and VASP backends are tested in their respective pipeline tests.

Test Matrix (NumPy only - others covered by pipeline tests):
                    | NumPy |
--------------------|-------|
RDF like            |   X   |
RDF unlike          |   X   |
3D number           |   X   |
"""

import pytest
import numpy as np

from revelsMD.revels_rdf import RevelsRDF
from revelsMD.revels_3D import Revels3D


# ---------------------------------------------------------------------------
# NumPy Matrix Tests (unique - not tested elsewhere)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.analytical
class TestNumpyAnalysisMatrix:
    """Test analyses with NumPy synthetic trajectories."""

    def test_rdf_like_and_shape(self, uniform_gas_trajectory):
        """NumPy: like-pair RDF works and has correct shape."""
        result = RevelsRDF.run_rdf(
            uniform_gas_trajectory, '1', '1',
            delr=0.1, start=0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result))
        # Shape check: RDF output should be 2 x n_bins
        assert result.shape[0] == 2
        assert len(result.shape) == 2

    def test_rdf_unlike(self, multispecies_trajectory):
        """NumPy: unlike-pair RDF works."""
        result = RevelsRDF.run_rdf(
            multispecies_trajectory, '1', '2',
            delr=0.1, start=0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_rdf_lambda_and_shape(self, uniform_gas_trajectory):
        """NumPy: lambda-combined RDF works and has correct shape."""
        result = RevelsRDF.run_rdf_lambda(
            uniform_gas_trajectory, '1', '1',
            delr=0.1, start=0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result))
        # Shape check: Lambda RDF output should be n_bins x 3
        assert result.shape[1] == 3
        assert len(result.shape) == 2

    def test_number_density_and_shape(self, uniform_gas_trajectory):
        """NumPy: 3D number density works and has correct shape."""
        nbins = 25
        gs = Revels3D.GridState(
            uniform_gas_trajectory, 'number', nbins=nbins
        )
        gs.make_force_grid(
            uniform_gas_trajectory, '1', kernel='triangular', rigid=False
        )
        gs.get_real_density()

        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))
        # Shape check: Density should be nbins x nbins x nbins
        assert gs.rho.shape == (nbins, nbins, nbins)
