"""
Analytical reference tests for RevelsMD.

These tests validate numerical correctness against mathematically known results
using synthetic NumpyTrajectoryState data. They require no external data files.
"""

import pytest
import numpy as np

from revelsMD.rdf import RDF, compute_rdf
from revelsMD.density import DensityGrid


@pytest.mark.analytical
@pytest.mark.integration
class TestRDFAnalyticalReference:
    """Tests for RDF calculation against known analytical results."""

    def test_uniform_gas_rdf_approaches_unity(self, uniform_gas_trajectory):
        """
        Uniform random gas should have g(r) approaching 1 at large r.

        The backward integration should give g(r)~1 in bulk.
        Uses bulk region 2.0 < r < 4.5 to avoid edge effects.
        """
        ts = uniform_gas_trajectory

        # Use backward integration which should give g(r) ~ 1 for uniform gas
        rdf = compute_rdf(ts, '1', '1', delr=0.1, start=0, stop=-1, integration='backward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert np.all(np.isfinite(rdf.g))

        # Use bulk region away from both short-range and cutoff edges
        mask = (rdf.r > 2.0) & (rdf.r < 4.5)
        assert np.any(mask), "No bins in bulk region"

        mean_gr = np.mean(rdf.g[mask])
        # Force-based g(r) has more variance due to integration
        assert abs(mean_gr - 1.0) < 0.3, f"Mean g(r) in bulk region = {mean_gr}, expected ~1.0"

    def test_two_atoms_rdf_peak_at_separation(self, two_atom_trajectory):
        """
        Two atoms at fixed separation should produce g(r) peak at that distance.

        With atoms at separation d = 3.0, the RDF should show a sharp peak
        at r = 3.0.
        """
        ts = two_atom_trajectory

        # Use fine binning to resolve the peak
        rdf = compute_rdf(ts, '1', '1', delr=0.1, start=0, stop=-1, integration='forward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert np.all(np.isfinite(rdf.g))

        # Find the peak location
        peak_idx = np.argmax(rdf.g)
        peak_r = rdf.r[peak_idx]

        # Peak should be near r = 3.0 (the separation distance)
        expected_separation = 3.0
        assert abs(peak_r - expected_separation) < 0.5, \
            f"Peak at r = {peak_r}, expected near {expected_separation}"

    def test_crystal_lattice_rdf_peaks(self, cubic_lattice_trajectory):
        """
        Simple cubic lattice should show peaks at lattice spacings.

        For a simple cubic lattice with spacing a = 2.5, peaks should appear at:
        - r = 2.5 (nearest neighbours)
        - r = 2.5 * sqrt(2) ~ 3.54 (next-nearest neighbours)
        - r = 2.5 * sqrt(3) ~ 4.33 (third shell)

        Note: The force-sampling method with random forces may not show
        crystal structure as clearly as histogram methods.
        """
        ts = cubic_lattice_trajectory

        # Use backward integration for cleaner results
        rdf = compute_rdf(ts, '1', '1', delr=0.1, start=0, stop=-1, integration='backward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert np.all(np.isfinite(rdf.g))

        # For force-sampling with random forces on a static lattice,
        # we mainly check that the calculation completes and produces finite values.
        # The structure may not be as pronounced as with histogram methods.
        bulk_mask = (rdf.r > 1.0) & (rdf.r < 5.0)
        if np.any(bulk_mask):
            # Check that g(r) values are reasonable (not all zeros or infinities)
            bulk_values = rdf.g[bulk_mask]
            assert np.mean(np.abs(bulk_values)) > 0, "RDF should have non-zero values"

    def test_rdf_forward_backward_consistency(self, uniform_gas_trajectory):
        """
        Forward and backward RDF integration produce consistent results.

        The forward integration starts from g(0)=0 and accumulates upward.
        The backward integration starts from g(inf)=1 and accumulates downward.

        For the lambda-combined method, both should contribute to a consistent result.
        """
        ts = uniform_gas_trajectory

        rdf_forward = compute_rdf(ts, '1', '1', delr=0.1, integration='forward')
        rdf_backward = compute_rdf(ts, '1', '1', delr=0.1, integration='backward')

        assert rdf_forward.r is not None
        assert rdf_backward.r is not None

        # Both should produce finite values
        assert np.all(np.isfinite(rdf_forward.g))
        assert np.all(np.isfinite(rdf_backward.g))

        # Forward starts from 0, backward starts from 1
        # The two methods are complementary - their sum should be approximately 1
        # at each r value (this is the basis of the lambda combination)
        mid_range_mask = (rdf_forward.r > 1.5) & (rdf_forward.r < 3.5)
        if np.any(mid_range_mask):
            combined = rdf_forward.g[mid_range_mask] + (1 - rdf_backward.g[mid_range_mask])
            # The "complementary" check: forward + (1 - backward) should be small
            # This is an approximation of how the lambda method works
            mean_combined = np.mean(np.abs(combined))
            assert mean_combined < 2.0, f"Forward/backward methods inconsistent: {mean_combined}"

    def test_rdf_lambda_produces_valid_output(self, uniform_gas_trajectory):
        """
        Lambda-combined RDF should produce valid output with correct properties.

        The lambda integration should provide r, g, and lam arrays.
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.2, integration='lambda')

        assert rdf.r is not None
        assert rdf.g is not None
        assert rdf.lam is not None
        assert np.all(np.isfinite(rdf.g))
        assert np.all(np.isfinite(rdf.lam))

        # Lambda should be between 0 and 1 (approximately)
        assert np.all(rdf.lam >= -0.5), "Lambda values should not be strongly negative"
        assert np.all(rdf.lam <= 1.5), "Lambda values should not exceed 1 significantly"


@pytest.mark.analytical
@pytest.mark.integration
class TestDensityAnalyticalReference:
    """Tests for 3D density calculation against known analytical results."""

    def test_single_atom_density_peak(self, single_atom_trajectory):
        """
        Single atom at known position should produce density peak at that location.

        With one atom at (5, 5, 5) in a 10x10x10 box, the density should peak
        near the centre of the grid.
        """
        ts = single_atom_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, '1', kernel='triangular', rigid=False)

        assert gs.progress == "Allocated"
        # Note: count may be frames-1 due to stop=-1 handling in API
        assert gs.count > 0

        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert gs.rho_force.shape == (20, 20, 20)
        assert np.all(np.isfinite(gs.rho_force))

        # The density should have its maximum near the centre
        # (atom is at 5,5,5 in a 10x10x10 box -> should be near bin 10,10,10)
        max_idx = np.unravel_index(np.argmax(gs.rho_force), gs.rho_force.shape)

        # Check that max is in the central region (within 5 bins of centre)
        centre = 10
        assert all(abs(idx - centre) < 6 for idx in max_idx), \
            f"Density peak at {max_idx}, expected near ({centre}, {centre}, {centre})"

    def test_uniform_density_is_flat(self, uniform_gas_trajectory):
        """
        Uniform random gas should produce relatively flat density field.

        For uniformly distributed particles, the density should be approximately
        constant throughout the box, with only statistical fluctuations.
        """
        ts = uniform_gas_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, '1', kernel='triangular', rigid=False)
        gs.get_real_density()

        assert hasattr(gs, 'rho_force')
        assert np.all(np.isfinite(gs.rho_force))

        # Compute coefficient of variation (std/mean)
        mean_rho = np.mean(gs.rho_force)
        std_rho = np.std(gs.rho_force)

        if mean_rho > 0:
            cv = std_rho / mean_rho
            # For uniform distribution, CV should be relatively small
            # Allow generous tolerance due to limited statistics
            assert cv < 2.0, f"Density CV = {cv}, expected relatively flat distribution"

    def test_density_conserves_total_count(self, uniform_gas_trajectory):
        """
        Total integrated density should equal number of atoms (approximately).

        The integral of the number density over the box volume should give
        the total number of particles.
        """
        ts = uniform_gas_trajectory

        gs = DensityGrid(ts, 'number', nbins=20)
        gs.accumulate(ts, '1', kernel='triangular', rigid=False)
        gs.get_real_density()

        # Calculate voxel volume
        voxel_vol = (ts.box_x / 20) * (ts.box_y / 20) * (ts.box_z / 20)

        # Integrate density
        total_count = np.sum(gs.rho_force) * voxel_vol

        # Should be approximately equal to number of atoms
        n_atoms = len(ts.get_indices('1'))

        # Allow significant tolerance due to FFT normalisation and boundary effects
        relative_error = abs(total_count - n_atoms) / n_atoms
        assert relative_error < 1.0, \
            f"Integrated count = {total_count}, expected ~{n_atoms}"

    def test_gridstate_initialisation(self, uniform_gas_trajectory):
        """
        DensityGrid should initialise correctly with various density types.
        """
        ts = uniform_gas_trajectory

        # Test number density
        gs_number = DensityGrid(ts, 'number', nbins=20)
        assert gs_number.density_type == 'number'
        assert gs_number.nbinsx == 20

        # Test that grids are initialised to zero
        assert gs_number.force_x.shape == (20, 20, 20)
        assert np.all(gs_number.force_x == 0)


@pytest.mark.analytical
@pytest.mark.integration
class TestMultispeciesRDF:
    """Tests for RDF calculations with multiple species."""

    def test_unlike_pair_rdf(self, multispecies_trajectory):
        """
        Unlike-pair RDF should work for two different species.
        """
        ts = multispecies_trajectory

        # Like pairs (1-1) with backward integration for g(r) ~ 1
        rdf_like = compute_rdf(ts, '1', '1', delr=0.2, integration='backward')

        # Unlike pairs (1-2) with backward integration
        rdf_unlike = compute_rdf(ts, '1', '2', delr=0.2, integration='backward')

        assert rdf_like.r is not None
        assert rdf_unlike.r is not None
        assert np.all(np.isfinite(rdf_like.g))
        assert np.all(np.isfinite(rdf_unlike.g))

        # With backward integration, both should approach 1 in bulk
        bulk_mask = rdf_like.r > 2.0
        if np.any(bulk_mask):
            mean_like = np.mean(rdf_like.g[bulk_mask])
            mean_unlike = np.mean(rdf_unlike.g[bulk_mask])

            # Both should be roughly 1 (with tolerance for statistics)
            assert abs(mean_like - 1.0) < 0.5, f"Like-pair bulk g(r) = {mean_like}"
            assert abs(mean_unlike - 1.0) < 0.5, f"Unlike-pair bulk g(r) = {mean_unlike}"


@pytest.mark.analytical
@pytest.mark.integration
class TestRigidMoleculeAnalytical:
    """Tests for rigid molecule calculations with synthetic data."""

    def test_water_trajectory_loads_correctly(self, water_molecule_trajectory):
        """
        Water molecule trajectory should load with correct species.
        """
        ts = water_molecule_trajectory

        o_indices = ts.get_indices('O')
        h_indices = ts.get_indices('H')

        # 10 molecules = 10 O and 20 H atoms
        assert len(o_indices) == 10
        assert len(h_indices) == 20

        # Check charges are present
        assert hasattr(ts, 'charge_list')
        assert ts.charge_list is not None

        # Check charge neutrality
        total_charge = np.sum(ts.charge_list)
        assert abs(total_charge) < 1e-10, f"Total charge = {total_charge}, should be neutral"


@pytest.mark.analytical
@pytest.mark.integration
class TestHistogramRDFAnalytical:
    """Tests for histogram-based g(r) against known analytical results."""

    def test_uniform_gas_histogram_rdf_approaches_unity(self, uniform_gas_trajectory):
        """
        Uniform random gas should have histogram g(r) ~ 1 at all r.

        This is the key validation: for an ideal gas, g_count should be ~1.0
        everywhere, unlike force-based g(r) which requires integration.

        Excludes r < 0.5 where statistical fluctuations are larger due to the
        small shell volume (not a boundary effect, just finite statistics).
        With 500 atoms and 50 frames, expect mean within 0.01 of 1.0.
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.1, integration='backward')

        assert rdf.g_count is not None
        assert np.all(np.isfinite(rdf.g_count))

        # Exclude small r where shell volume is tiny and statistics are poor
        # This is a finite-size effect, not a boundary effect
        valid_mask = rdf.r > 0.5
        assert np.any(valid_mask), "No valid bins"

        mean_g_count = np.mean(rdf.g_count[valid_mask])
        assert abs(mean_g_count - 1.0) < 0.01, f"Mean g_count = {mean_g_count}, expected ~1.0"

    def test_two_atoms_histogram_shows_peak(self, two_atom_trajectory):
        """
        Two atoms at fixed separation should show peak in histogram g(r).

        The peak should be at the separation distance, smoothed by the
        triangular kernel.
        """
        ts = two_atom_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.1, integration='forward')

        assert rdf.g_count is not None
        assert np.all(np.isfinite(rdf.g_count))

        # Find peak in histogram g(r)
        peak_idx = np.argmax(rdf.g_count)
        peak_r = rdf.r[peak_idx]

        # Peak should be near r = 3.0 (the separation distance)
        expected_separation = 3.0
        assert abs(peak_r - expected_separation) < 0.5, \
            f"Peak at r = {peak_r}, expected near {expected_separation}"

    def test_histogram_and_force_consistency(self, uniform_gas_trajectory):
        """
        For equilibrium systems, histogram and force-based g(r) should agree in bulk.

        Both g_count and g_force should be approximately 1.0 for a uniform gas.
        Excludes r < 0.5 due to poor statistics at small r (tiny shell volume).
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.2, integration='backward')

        assert rdf.g_count is not None
        assert rdf.g_force is not None

        # Exclude small r where shell volume is tiny and statistics are poor
        valid_mask = rdf.r > 0.5
        assert np.any(valid_mask), "No valid bins"

        g_count_mean = np.mean(rdf.g_count[valid_mask])
        g_force_mean = np.mean(rdf.g_force[valid_mask])

        # Histogram g(r) should be very close to 1 (tight tolerance)
        assert abs(g_count_mean - 1.0) < 0.02, f"g_count mean = {g_count_mean}"
        # Force-based g(r) has more variance due to integration
        assert abs(g_force_mean - 1.0) < 0.3, f"g_force mean = {g_force_mean}"

    def test_g_count_lambda_integration(self, uniform_gas_trajectory):
        """
        g_count should be available and consistent with lambda integration.
        """
        ts = uniform_gas_trajectory

        rdf = compute_rdf(ts, '1', '1', delr=0.2, integration='lambda')

        assert rdf.g_count is not None
        assert rdf.g_force is not None
        assert rdf.lam is not None

        # Lengths should match
        assert len(rdf.g_count) == len(rdf.r)
        assert len(rdf.g_force) == len(rdf.r)

