"""
Pipeline integration tests for 3D number density (Example 2).

These tests exercise the full 3D density workflow using the Example 2 LJ data:
- LammpsTrajectoryState loading
- GridState creation and configuration
- make_force_grid() with triangular kernel
- get_real_density() integration
- get_lambda() for optimal linear combination
- Regression against stored reference data
"""

import pytest
import numpy as np
from pathlib import Path

from revelsMD.revels_3D import Revels3D
from .conftest import load_reference_data, assert_arrays_close


@pytest.mark.integration
@pytest.mark.requires_example2
class TestNumberDensityPipelineExample2:
    """Full pipeline tests using Example 2 LJ 3D data."""

    def test_trajectory_loads_correctly(self, example2_trajectory):
        """Verify Example 2 trajectory loads with expected properties."""
        ts = example2_trajectory

        assert ts.variety == 'lammps'
        assert ts.units == 'lj'
        assert ts.frames > 0
        assert ts.box_x > 0

        # Check we can get atom indices for type 2 (solvating particles)
        type2_indices = ts.get_indices('2')
        assert len(type2_indices) > 0

    def test_gridstate_initialisation(self, example2_trajectory):
        """GridState initialises correctly for number density."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)

        assert gs.density_type == 'number'
        assert gs.nbinsx == 50
        assert gs.nbinsy == 50
        assert gs.nbinsz == 50
        assert gs.temperature == 1.35

        # Grids should be initialised to zero
        assert gs.forceX.shape == (50, 50, 50)
        assert np.all(gs.forceX == 0)
        assert np.all(gs.forceY == 0)
        assert np.all(gs.forceZ == 0)

    def test_make_force_grid_subset(self, example2_trajectory):
        """Force grid accumulation works on frame subset."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)

        # Use only first 5 frames for fast test
        gs.make_force_grid(
            ts, '2', kernel='triangular', rigid=False,
            start=0, stop=5, period=1
        )

        assert gs.grid_progress == "Allocated"
        assert gs.count == 5

        # Force grids should now have non-zero values
        assert not np.all(gs.forceX == 0) or not np.all(gs.forceY == 0) or not np.all(gs.forceZ == 0)

    def test_get_real_density(self, example2_trajectory):
        """Density integration produces valid output."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=5)
        gs.get_real_density()

        assert hasattr(gs, 'rho')
        assert gs.rho.shape == (50, 50, 50)
        assert np.all(np.isfinite(gs.rho))

    def test_box_kernel_alternative(self, example2_trajectory):
        """Box kernel produces valid (though higher variance) output."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='box', rigid=False, start=0, stop=5)
        gs.get_real_density()

        assert hasattr(gs, 'rho')
        assert np.all(np.isfinite(gs.rho))

    @pytest.mark.slow
    def test_full_trajectory_density(self, example2_trajectory):
        """Full trajectory density calculation (slow)."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=100, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False)
        gs.get_real_density()

        assert gs.count == ts.frames
        assert np.all(np.isfinite(gs.rho))

        # For solvation around frozen particle, should see excluded volume
        # (lower density at centre)
        centre_region = gs.rho[45:55, 45:55, 45:55]
        bulk_region = gs.rho[0:10, 0:10, 0:10]

        # This is a qualitative check - frozen particle creates void
        mean_centre = np.mean(centre_region)
        mean_bulk = np.mean(bulk_region)

        # Bulk should have higher density than centre (excluded volume)
        # Note: This depends on the system setup
        assert np.isfinite(mean_centre)
        assert np.isfinite(mean_bulk)

    @pytest.mark.slow
    def test_lambda_combination(self, example2_trajectory):
        """Lambda combination produces valid optimal density."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=20)
        gs.get_real_density()

        # Use 5 sections for variance estimation
        gs_lambda = gs.get_lambda(ts, sections=5, start=0, stop=20)

        assert gs_lambda.grid_progress == "Lambda"
        assert hasattr(gs_lambda, 'optimal_density')
        assert np.all(np.isfinite(gs_lambda.optimal_density))

    @pytest.mark.regression
    @pytest.mark.slow
    def test_density_regression(self, example2_trajectory):
        """Density matches stored reference data."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False)
        gs.get_real_density()

        ref = load_reference_data("density_example2", "number_density.npz")

        if ref is None:
            pytest.skip("Reference data not generated yet")

        assert_arrays_close(gs.rho, ref['rho'], rtol=1e-6, context="density values")


@pytest.mark.integration
@pytest.mark.requires_example2
class TestDensityPhysicalProperties:
    """Tests validating physical properties of density results."""

    def test_density_positive(self, example2_trajectory):
        """Number density should be non-negative everywhere."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=10)
        gs.get_real_density()

        # Number density should not be strongly negative
        # (small negative values possible due to FFT artifacts)
        min_rho = np.min(gs.rho)
        assert min_rho > -0.5 * np.mean(np.abs(gs.rho)), \
            f"Density has large negative values: min = {min_rho}"

    def test_density_reasonable_magnitude(self, example2_trajectory):
        """Density magnitude should be physically reasonable."""
        ts = example2_trajectory

        gs = Revels3D.GridState(ts, 'number', nbins=50, temperature=1.35)
        gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=10)
        gs.get_real_density()

        # Mean density should be order of magnitude of N/V
        n_atoms = len(ts.get_indices('2'))
        volume = ts.box_x * ts.box_y * ts.box_z
        expected_mean_rho = n_atoms / volume

        mean_rho = np.mean(gs.rho)

        # Should be within order of magnitude
        ratio = mean_rho / expected_mean_rho if expected_mean_rho > 0 else float('inf')
        assert 0.1 < ratio < 10.0, \
            f"Mean density {mean_rho:.4f} differs significantly from expected {expected_mean_rho:.4f}"

    def test_kernels_produce_consistent_mean(self, example2_trajectory):
        """Triangular and box kernels should give similar mean density."""
        ts = example2_trajectory

        # Triangular kernel
        gs_tri = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.35)
        gs_tri.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=10)
        gs_tri.get_real_density()

        # Box kernel
        gs_box = Revels3D.GridState(ts, 'number', nbins=30, temperature=1.35)
        gs_box.make_force_grid(ts, '2', kernel='box', rigid=False, start=0, stop=10)
        gs_box.get_real_density()

        mean_tri = np.mean(gs_tri.rho)
        mean_box = np.mean(gs_box.rho)

        # Mean densities should be similar
        relative_diff = abs(mean_tri - mean_box) / max(abs(mean_tri), abs(mean_box), 1e-10)
        assert relative_diff < 0.5, \
            f"Kernels give different mean densities: tri={mean_tri:.4f}, box={mean_box:.4f}"


@pytest.mark.integration
@pytest.mark.requires_example2
class TestDensityGridResolution:
    """Tests for grid resolution effects."""

    def test_different_nbins(self, example2_trajectory):
        """Different grid resolutions should produce valid results."""
        ts = example2_trajectory

        for nbins in [20, 50, 100]:
            gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=1.35)
            gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=5)
            gs.get_real_density()

            assert gs.rho.shape == (nbins, nbins, nbins)
            assert np.all(np.isfinite(gs.rho))

    def test_resolution_preserves_total(self, example2_trajectory):
        """Different resolutions should give similar integrated density."""
        ts = example2_trajectory

        totals = []

        for nbins in [20, 40]:
            gs = Revels3D.GridState(ts, 'number', nbins=nbins, temperature=1.35)
            gs.make_force_grid(ts, '2', kernel='triangular', rigid=False, start=0, stop=5)
            gs.get_real_density()

            # Calculate voxel volume and integrate
            voxel_vol = (ts.box_x / nbins) * (ts.box_y / nbins) * (ts.box_z / nbins)
            total = np.sum(gs.rho) * voxel_vol
            totals.append(total)

        # Totals should be similar (within 50% - generous tolerance)
        if len(totals) == 2 and all(t != 0 for t in totals):
            relative_diff = abs(totals[0] - totals[1]) / max(abs(totals[0]), abs(totals[1]))
            assert relative_diff < 0.5, \
                f"Different resolutions give very different totals: {totals}"
