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

    def test_rdf_class_importable(self):
        from revelsMD.rdf import RDF
        assert RDF is not None

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

    def test_compute_rdf_importable(self):
        from revelsMD.rdf import compute_rdf
        assert callable(compute_rdf)

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
