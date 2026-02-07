"""Tests for the RDF class API."""

import numpy as np
import pytest

from revelsMD.trajectories import NumpyTrajectory


@pytest.fixture
def water_trajectory():
    """Create a simple water-like trajectory for testing."""
    positions = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
        [[0.9, 1.9, 2.9], [3.9, 4.9, 5.9], [6.9, 7.9, 8.9]]
    ], dtype=float)
    forces = np.array([
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
        [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]],
    ], dtype=float)
    species = ["H", "O", "H"]

    return NumpyTrajectory(
        positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
    )


class TestRDFClassAPI:
    """Test the RDF class interface."""

    def test_rdf_constructor_stores_parameters(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, species_a='O', species_b='H', delr=0.02)
        assert rdf.species_a == 'O'
        assert rdf.species_b == 'H'
        assert rdf.delr == 0.02

    def test_rdf_accumulate_sets_progress(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        assert rdf.progress == 'initialized'
        rdf.accumulate(water_trajectory)
        assert rdf.progress == 'accumulated'

    def test_rdf_get_rdf_forward_sets_results(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')
        assert rdf.r is not None
        assert rdf.g is not None
        assert len(rdf.r) == len(rdf.g)

    def test_rdf_get_rdf_backward_sets_results(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='backward')
        assert rdf.r is not None
        assert rdf.g is not None

    def test_rdf_get_rdf_lambda_sets_lambda(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='lambda')
        assert rdf.lam is not None
        assert len(rdf.lam) == len(rdf.g)

    def test_rdf_lambda_none_for_non_lambda(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')
        assert rdf.lam is None

    def test_rdf_get_rdf_before_accumulate_raises(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        with pytest.raises(RuntimeError):
            rdf.get_rdf()

    def test_rdf_deposit_single_frame(self, water_trajectory):
        """Test low-level deposit method for user-controlled iteration."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        positions, forces = water_trajectory.get_frame(0)
        rdf.deposit(positions, forces)
        assert rdf.progress == 'accumulated'

    def test_rdf_deposit_multiple_frames(self, water_trajectory):
        """Test deposit can be called multiple times."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        for positions, forces in water_trajectory.iter_frames(stop=3):
            rdf.deposit(positions, forces)
        rdf.get_rdf(integration='forward')
        assert rdf.r is not None
        assert rdf.g is not None


class TestRDFResultsMatchLegacy:
    """Verify new class produces same results as legacy functions."""

    def test_forward_matches_run_rdf_from_zero(self, water_trajectory):
        from revelsMD.rdf import RDF
        from revelsMD.revels_rdf import run_rdf

        legacy_result = run_rdf(water_trajectory, 'O', 'H', from_zero=True)

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')

        np.testing.assert_allclose(rdf.r, legacy_result[0])
        np.testing.assert_allclose(rdf.g, legacy_result[1])

    def test_backward_matches_run_rdf_from_inf(self, water_trajectory):
        from revelsMD.rdf import RDF
        from revelsMD.revels_rdf import run_rdf

        legacy_result = run_rdf(water_trajectory, 'O', 'H', from_zero=False)

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='backward')

        np.testing.assert_allclose(rdf.r, legacy_result[0])
        np.testing.assert_allclose(rdf.g, legacy_result[1])

    def test_lambda_matches_run_rdf_lambda(self, water_trajectory):
        from revelsMD.rdf import RDF
        from revelsMD.revels_rdf import run_rdf_lambda

        legacy_result = run_rdf_lambda(water_trajectory, 'O', 'H')

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='lambda')

        np.testing.assert_allclose(rdf.r, legacy_result[:, 0])
        np.testing.assert_allclose(rdf.g, legacy_result[:, 1])
        np.testing.assert_allclose(rdf.lam, legacy_result[:, 2])

    def test_deposit_matches_accumulate(self, water_trajectory):
        """Verify deposit-based iteration matches accumulate convenience method."""
        from revelsMD.rdf import RDF

        # Using accumulate
        rdf1 = RDF(water_trajectory, 'O', 'H')
        rdf1.accumulate(water_trajectory)
        rdf1.get_rdf(integration='forward')

        # Using deposit
        rdf2 = RDF(water_trajectory, 'O', 'H')
        for positions, forces in water_trajectory.iter_frames():
            rdf2.deposit(positions, forces)
        rdf2.get_rdf(integration='forward')

        np.testing.assert_allclose(rdf1.r, rdf2.r)
        np.testing.assert_allclose(rdf1.g, rdf2.g)


class TestComputeRDFConvenience:
    """Test the compute_rdf convenience function."""

    def test_compute_rdf_returns_rdf_object(self, water_trajectory):
        from revelsMD.rdf import compute_rdf, RDF
        result = compute_rdf(water_trajectory, 'O', 'H')
        assert isinstance(result, RDF)

    def test_compute_rdf_has_results(self, water_trajectory):
        from revelsMD.rdf import compute_rdf
        rdf = compute_rdf(water_trajectory, 'O', 'H', integration='forward')
        assert rdf.r is not None
        assert rdf.g is not None
        assert rdf.progress == 'computed'

    def test_compute_rdf_matches_manual_workflow(self, water_trajectory):
        """Verify compute_rdf produces same results as manual RDF workflow."""
        from revelsMD.rdf import RDF, compute_rdf

        # Manual workflow
        rdf1 = RDF(water_trajectory, 'O', 'H')
        rdf1.accumulate(water_trajectory)
        rdf1.get_rdf(integration='forward')

        # Convenience function
        rdf2 = compute_rdf(water_trajectory, 'O', 'H', integration='forward')

        np.testing.assert_allclose(rdf1.r, rdf2.r)
        np.testing.assert_allclose(rdf1.g, rdf2.g)


class TestRDFSpeciesValidation:
    """Test validation of species atom counts."""

    def test_like_species_with_one_atom_raises(self):
        """Like-species RDF with only 1 atom should raise ValueError."""
        from revelsMD.rdf import RDF

        positions = np.array([[[1, 2, 3]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0]]], dtype=float)
        species = ["O"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="at least 2 atoms"):
            RDF(ts, 'O', 'O')

    def test_like_species_with_zero_atoms_raises(self):
        """Like-species RDF with 0 atoms should raise ValueError."""
        from revelsMD.rdf import RDF

        positions = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=float)
        species = ["H", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="No atoms found for species 'O'"):
            RDF(ts, 'O', 'O')

    def test_unlike_species_with_empty_first_raises(self):
        """Unlike-species RDF with no atoms for first species should raise ValueError."""
        from revelsMD.rdf import RDF

        positions = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=float)
        species = ["H", "H"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="No atoms found for species 'O'"):
            RDF(ts, 'O', 'H')

    def test_unlike_species_with_empty_second_raises(self):
        """Unlike-species RDF with no atoms for second species should raise ValueError."""
        from revelsMD.rdf import RDF

        positions = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=float)
        forces = np.array([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=float)
        species = ["O", "O"]

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        with pytest.raises(ValueError, match="No atoms found for species 'H'"):
            RDF(ts, 'O', 'H')


class TestRDFValidation:
    """Test input validation for RDF methods."""

    def test_get_rdf_invalid_integration_raises(self, water_trajectory):
        """get_rdf with invalid integration should raise ValueError."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)

        with pytest.raises(ValueError, match="integration must be"):
            rdf.get_rdf(integration='invalid')

    def test_compute_rdf_invalid_integration_raises(self, water_trajectory):
        """compute_rdf with invalid integration should raise ValueError."""
        from revelsMD.rdf import compute_rdf

        with pytest.raises(ValueError, match="integration must be"):
            compute_rdf(water_trajectory, 'O', 'H', integration='invalid')


class TestRDFAccumulatePeriod:
    """Test that accumulate period parameter affects frame sampling."""

    def test_accumulate_period_affects_frame_count(self, water_trajectory):
        """accumulate with period=2 should process half as many frames."""
        from revelsMD.rdf import RDF

        # Full trajectory (3 frames)
        rdf1 = RDF(water_trajectory, 'O', 'H')
        rdf1.accumulate(water_trajectory, period=1)

        # Every other frame (period=2, so frames 0, 2 = 2 frames)
        rdf2 = RDF(water_trajectory, 'O', 'H')
        rdf2.accumulate(water_trajectory, period=2)

        # Period=2 should have accumulated fewer frames
        assert rdf2._frame_count < rdf1._frame_count

    def test_accumulate_period_parameter_is_used(self, water_trajectory):
        """accumulate period parameter is passed to frame iteration."""
        from revelsMD.rdf import RDF

        # This test verifies that period is actually used by checking that
        # the _frame_count reflects the expected number of frames processed
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory, start=0, stop=3, period=2)

        # With period=2, frames 0 and 2 are processed (2 frames total)
        assert rdf._frame_count == 2


class TestRDFCountAccumulation:
    """Test histogram-based g(r) accumulation."""

    def test_counts_accumulator_initialised(self, water_trajectory):
        """RDF should have _counts array initialised to zeros."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')

        assert hasattr(rdf, '_counts')
        assert rdf._counts.shape == rdf._accumulated.shape
        np.testing.assert_array_equal(rdf._counts, 0)

    def test_counts_accumulated_after_deposit(self):
        """Counts should be non-zero after depositing frames."""
        from revelsMD.rdf import RDF

        # Create trajectory with atoms close enough to be within rmax
        # Two H atoms at positions (0,0,0) and (2,0,0) - distance 2.0
        positions = np.array([
            [[0, 0, 0], [2, 0, 0]],
        ], dtype=float)
        forces = np.array([
            [[0.1, 0, 0], [0.1, 0, 0]],
        ], dtype=float)
        species = ['H', 'H']

        ts = NumpyTrajectory(
            positions, forces, 10.0, 10.0, 10.0, species, temperature=300.0, units="real"
        )

        rdf = RDF(ts, 'H', 'H')
        pos, frc = ts.get_frame(0)
        rdf.deposit(pos, frc)

        # Distance is 2.0, rmax is 5.0, delr is 0.01, so should be in bins
        assert np.sum(rdf._counts) > 0

    def test_g_count_property_available_after_get_rdf(self, water_trajectory):
        """g_count property available after get_rdf."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')

        assert rdf.g_count is not None
        assert len(rdf.g_count) > 0

    def test_g_force_property_available_after_get_rdf(self, water_trajectory):
        """g_force property available (alias for g)."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')

        assert rdf.g_force is not None
        # g_force should be same as g
        np.testing.assert_array_equal(rdf.g_force, rdf.g)

    def test_g_count_same_length_as_g_force(self, water_trajectory):
        """g_count and g_force have the same length (both at bin edges)."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')

        assert len(rdf.g_count) == len(rdf.g_force)

    def test_g_count_uses_same_r_as_g_force(self, water_trajectory):
        """g_count corresponds to the same r values as g_force."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')

        # Both should have length matching r
        assert len(rdf.g_count) == len(rdf.r)
        assert len(rdf.g_force) == len(rdf.r)

    @pytest.mark.parametrize("attr", ["g_count", "g_force"])
    def test_g_property_is_none_before_get_rdf(self, water_trajectory, attr):
        """g_count/g_force is None before calling get_rdf."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)

        assert getattr(rdf, attr) is None

    def test_g_count_lambda_integration(self, water_trajectory):
        """g_count should work with lambda integration."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='lambda')

        assert rdf.g_count is not None
        # For lambda, r is trimmed to bins[1:], so g_count should match
        assert len(rdf.g_count) == len(rdf.r)

    def test_g_count_backward_integration(self, water_trajectory):
        """g_count should work with backward integration."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='backward')

        assert rdf.g_count is not None
        assert len(rdf.g_count) == len(rdf.r)


class TestRDFBackendSelection:
    """Test that RDF class uses backend-selected functions."""

    def test_rdf_uses_numba_backend_when_available(self, water_trajectory):
        """RDF should use Numba functions when numba is available."""
        pytest.importorskip('numba')
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')

        # Check that the instance methods are from the numba module
        assert 'numba' in rdf._compute_pairwise.__module__
        assert 'numba' in rdf._accumulate_binned.__module__
        assert 'numba' in rdf._accumulate_triangular.__module__

    def test_rdf_stores_backend_functions_as_instance_methods(self, water_trajectory):
        """RDF should store backend functions as instance methods."""
        from revelsMD.rdf import RDF

        rdf = RDF(water_trajectory, 'O', 'H')

        # These should be callable
        assert callable(rdf._compute_pairwise)
        assert callable(rdf._accumulate_binned)
        assert callable(rdf._accumulate_triangular)
