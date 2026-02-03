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

from revelsMD.rdf import RDF, compute_rdf
from revelsMD.density import DensityGrid


# ---------------------------------------------------------------------------
# NumPy Matrix Tests (unique - not tested elsewhere)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.analytical
class TestNumpyAnalysisMatrix:
    """Test analyses with NumPy synthetic trajectories."""

    def test_rdf_like_and_shape(self, uniform_gas_trajectory):
        """NumPy: like-pair RDF works and has correct shape."""
        result = compute_rdf(
            uniform_gas_trajectory, '1', '1',
            delr=0.1, start=0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result.r))
        assert np.all(np.isfinite(result.g))
        # Shape check: RDF r and g should be 1D arrays of same length
        assert result.r.ndim == 1
        assert result.g.ndim == 1
        assert len(result.r) == len(result.g)

    def test_rdf_unlike(self, multispecies_trajectory):
        """NumPy: unlike-pair RDF works."""
        result = compute_rdf(
            multispecies_trajectory, '1', '2',
            delr=0.1, start=0, stop=-1
        )
        assert result is not None
        assert np.all(np.isfinite(result.r))
        assert np.all(np.isfinite(result.g))

    def test_rdf_lambda_and_shape(self, uniform_gas_trajectory):
        """NumPy: lambda-combined RDF works and has correct shape."""
        result = compute_rdf(
            uniform_gas_trajectory, '1', '1',
            delr=0.1, start=0, stop=-1, integration='lambda'
        )
        assert result is not None
        assert np.all(np.isfinite(result.r))
        assert np.all(np.isfinite(result.g))
        assert np.all(np.isfinite(result.lam))
        # Shape check: r, g, lam should all be 1D arrays of same length
        assert result.r.ndim == 1
        assert result.g.ndim == 1
        assert result.lam.ndim == 1
        assert len(result.r) == len(result.g) == len(result.lam)

    def test_number_density_and_shape(self, uniform_gas_trajectory):
        """NumPy: 3D number density works and has correct shape."""
        nbins = 25
        gs = DensityGrid(
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
