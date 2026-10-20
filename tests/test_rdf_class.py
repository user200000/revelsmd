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

    def test_rdf_default_rmax_is_half_box(self, water_trajectory):
        """Default rmax should be half the shortest box dimension."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        assert rdf.rmax == pytest.approx(5.0)  # box is 10x10x10

    def test_rdf_custom_rmax(self, water_trajectory):
        """Custom rmax should override the default."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H', rmax=3.0)
        assert rdf.rmax == 3.0
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')
        assert rdf.r[-1] == pytest.approx(3.0)

    def test_standard_r_reaches_rmax(self, water_trajectory):
        """Standard integration r grid should include rmax."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H', rmax=5.0, delr=0.1)
        rdf.accumulate(water_trajectory)
        rdf.get_rdf(integration='forward')
        assert rdf.r[-1] == pytest.approx(5.0)
        assert rdf.r[0] == pytest.approx(0.0)

    def test_rmax_beyond_the_inscribed_sphere_raises(self):
        from revelsMD.rdf import RDF
        traj = _ideal_gas(10.0, n_frames=1, n_atoms=4)
        with pytest.raises(ValueError, match=r"rmax=5\.5 is larger than .* \(5\)"):
            RDF(traj, 'A', 'A', rmax=5.5)

    @pytest.mark.parametrize("rmax", [5.0, 5.0 * (1 + 1e-12)])
    def test_rmax_at_the_inscribed_sphere_is_accepted(self, rmax):
        # Including a value that differs from the radius only by rounding.
        from revelsMD.rdf import RDF
        traj = _ideal_gas(10.0, n_frames=1, n_atoms=4)
        assert RDF(traj, 'A', 'A', rmax=rmax).rmax == rmax

    @pytest.mark.parametrize("box, delr, last", [
        (10.0, 0.3, 4.8),      # rmax 5.0 is 16.67 bins: the grid stops at 16
        (14.227, 0.02, 7.10),  # rmax 7.1135 is 355.7 bins
    ])
    def test_r_grid_stops_at_the_last_whole_bin_within_rmax(self, box, delr, last):
        from revelsMD.rdf import RDF
        traj = _ideal_gas(box, n_frames=1, n_atoms=4)
        rdf = RDF(traj, 'A', 'A', delr=delr)
        rdf.accumulate(traj)
        rdf.get_rdf(integration='forward')
        assert rdf.r[-1] == pytest.approx(last)
        assert rdf.r[-1] <= rdf.rmax

    @pytest.mark.parametrize("box, rmax, delr, last", [
        # rmax / delr is a whole number, but in floating point falls just
        # below it: 2.9999999999999996, 6.999999999999999, and for the
        # inscribed radius of a 7 A cube 34.99999999999998.
        (10.0, 0.3, 0.1, 0.3),
        (10.0, 0.7, 0.1, 0.7),
        (7.0, None, 0.1, 3.5),
    ])
    def test_a_whole_number_of_bins_keeps_its_last_point(self, box, rmax, delr, last):
        from revelsMD.rdf import RDF
        traj = _ideal_gas(box, n_frames=1, n_atoms=4)
        rdf = RDF(traj, 'A', 'A', rmax=rmax, delr=delr)
        rdf.accumulate(traj)
        rdf.get_rdf(integration='forward')
        assert rdf.r[-1] == pytest.approx(last)

    def test_ideal_gas_counts_are_unbiased_at_the_last_point(self):
        # Past the inscribed radius a shell is only partly sampled, and an
        # ideal gas gives g_count < 1 there (0.93 at r = 5.1 in this box).
        from revelsMD.rdf import RDF
        traj = _ideal_gas(10.0, n_frames=200, n_atoms=200)
        rdf = RDF(traj, 'A', 'A', delr=0.3)
        rdf.accumulate(traj)
        rdf.get_rdf(integration='forward')
        assert rdf.g_count[-1] == pytest.approx(1.0, abs=0.03)

    def test_all_methods_return_same_grid(self, water_trajectory):
        """All integration methods should return the same r grid."""
        from revelsMD.rdf import RDF

        rdf_fwd = RDF(water_trajectory, 'O', 'H', delr=0.1)
        rdf_fwd.accumulate(water_trajectory)
        rdf_fwd.get_rdf(integration='forward')

        rdf_bwd = RDF(water_trajectory, 'O', 'H', delr=0.1)
        rdf_bwd.accumulate(water_trajectory)
        rdf_bwd.get_rdf(integration='backward')

        rdf_lam = RDF(water_trajectory, 'O', 'H', delr=0.1)
        rdf_lam.accumulate(water_trajectory)
        rdf_lam.get_rdf(integration='lambda')

        np.testing.assert_allclose(rdf_fwd.r, rdf_bwd.r)
        np.testing.assert_allclose(rdf_fwd.r, rdf_lam.r)
        assert len(rdf_fwd.g) == len(rdf_lam.g)
        assert len(rdf_fwd.g_count) == len(rdf_lam.g_count)

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


def _ideal_gas(box, n_frames, n_atoms, seed=1):
    """Uniformly random positions and random forces in a cubic box."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, box, (n_frames, n_atoms, 3))
    return NumpyTrajectory(positions, rng.normal(size=positions.shape), box, box, box,
                           ["A"] * n_atoms, temperature=1.0, units="lj")


def _water(box=10.0, temperature=300.0, order=None, seed=0, n_waters=4):
    """A small water-like trajectory; ``order`` permutes its atoms."""
    rng = np.random.default_rng(seed)
    species = np.array(["O", "H", "H"] * n_waters)
    positions = rng.uniform(0, box, (6, len(species), 3))
    forces = rng.normal(size=(6, len(species), 3))
    if order is not None:
        species, positions, forces = species[order], positions[:, order], forces[:, order]
    return NumpyTrajectory(positions, forces, box, box, box, list(species),
                           temperature=temperature, units="real")


class TestRDFAccumulateTrajectories:
    """accumulate() takes frames from any trajectory of the RDF's system."""

    @pytest.mark.parametrize("other, message", [
        (lambda: _water(box=12.0), "cell differs"),
        (lambda: _water(temperature=600.0), "temperature or units"),
        (lambda: _water(n_waters=3), "has 3 'O' and 6 'H' atoms, but the RDF was built for 4 and 8"),
    ], ids=["cell", "temperature", "species counts"])
    def test_another_system_raises(self, other, message):
        from revelsMD.rdf import RDF
        rdf = RDF(_water(), 'O', 'H', delr=0.1)
        with pytest.raises(ValueError, match=message):
            rdf.accumulate(other())

    def test_like_species_counts_differ_raises(self):
        from revelsMD.rdf import RDF
        rdf = RDF(_water(), 'O', 'O', delr=0.1)
        with pytest.raises(ValueError, match="has 3 'O' atoms, but the RDF was built for 4$"):
            rdf.accumulate(_water(n_waters=3))

    def test_a_rejected_trajectory_keeps_earlier_results(self):
        from revelsMD.rdf import RDF
        traj = _water()
        rdf = RDF(traj, 'O', 'H', delr=0.1)
        rdf.accumulate(traj)
        rdf.get_rdf(integration='lambda')
        g = rdf.g.copy()
        with pytest.raises(ValueError, match="cell differs"):
            rdf.accumulate(_water(box=12.0))
        assert rdf._frame_count == 6
        np.testing.assert_array_equal(rdf.g, g)

    @pytest.mark.parametrize("integration", ['forward', 'backward', 'lambda'])
    def test_atoms_are_found_by_name_in_each_trajectory(self, integration):
        # The same frames with the atoms listed in another order.
        from revelsMD.rdf import RDF
        order = np.random.default_rng(1).permutation(12)
        reference = RDF(_water(), 'O', 'H', delr=0.1)
        reference.accumulate(_water())
        reordered = RDF(_water(), 'O', 'H', delr=0.1)
        reordered.accumulate(_water(order=order))
        reference.get_rdf(integration=integration)
        reordered.get_rdf(integration=integration)
        np.testing.assert_allclose(reordered.g, reference.g, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("integration", ['forward', 'backward', 'lambda'])
    def test_accumulating_in_parts_matches_one_call(self, integration):
        from revelsMD.rdf import RDF
        traj = _water()
        whole = RDF(traj, 'O', 'H', delr=0.1)
        whole.accumulate(traj)
        parts = RDF(traj, 'O', 'H', delr=0.1)
        parts.accumulate(traj, stop=2)
        parts.accumulate(traj, start=2)
        whole.get_rdf(integration=integration)
        parts.get_rdf(integration=integration)
        assert parts._frame_count == whole._frame_count == 6
        np.testing.assert_allclose(parts.g, whole.g, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(parts.g_count, whole.g_count, rtol=1e-12, atol=1e-14)

    def test_accumulate_clears_earlier_results(self):
        from revelsMD.rdf import RDF
        traj = _water()
        rdf = RDF(traj, 'O', 'H', delr=0.1)
        rdf.accumulate(traj, stop=3)
        rdf.get_rdf(integration='lambda')
        rdf.accumulate(traj, start=3)
        assert rdf.g is None and rdf.g_count is None and rdf.lam is None and rdf.r is None


class TestForwardAndBackwardOnOneGrid:
    """g(r) from both estimators is reported at the r it belongs to."""

    def test_a_pair_enters_the_forward_estimator_at_its_distance(self):
        # One A-B pair at r = 2.05 in every frame, with bins 0.1 wide: forward
        # g(r) integrates over pairs closer than r, so it is 0 up to r = 2.0
        # and steps at r = 2.1; backward g(r) integrates over pairs further
        # than r, so it steps at the same place.
        from revelsMD.rdf import RDF
        positions = np.array([[[1.0, 1.0, 1.0], [3.05, 1.0, 1.0]]] * 4)
        forces = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 4)
        traj = NumpyTrajectory(positions, forces, 10.0, 10.0, 10.0, ["A", "B"],
                               temperature=1.0, units="lj")
        rdf = RDF(traj, "A", "B", delr=0.1)
        rdf.accumulate(traj)
        rdf.get_rdf("forward")
        below, above = np.flatnonzero(rdf.r < 2.05)[-1], np.flatnonzero(rdf.r > 2.05)[0]
        assert np.all(rdf.g[:below + 1] == 0.0)
        step = rdf.g[above]
        assert np.all(rdf.g[above:] == step) and step != 0.0
        rdf.get_rdf("backward")
        assert np.all(rdf.g[:below + 1] == rdf.g[0]) and np.all(rdf.g[above:] == 1.0)
        assert rdf.g[0] == pytest.approx(1.0 - step, rel=1e-12)

    def test_forward_minus_backward_is_the_same_at_every_r(self):
        # Both integrate the same binned force sums from opposite ends, so on
        # one grid they differ by the full integral minus 1 at every r.
        from revelsMD.rdf import RDF
        traj = _water()
        rdf = RDF(traj, "O", "H", delr=0.1)
        rdf.accumulate(traj)
        rdf.get_rdf("forward")
        forward = rdf.g.copy()
        rdf.get_rdf("backward")
        difference = forward - rdf.g
        np.testing.assert_allclose(difference, difference[0], rtol=0, atol=1e-12)


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

        with pytest.raises(ValueError, match="no atoms found for species 'O'"):
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

        with pytest.raises(ValueError, match="no atoms found for species 'O'"):
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

        with pytest.raises(ValueError, match="no atoms found for species 'H'"):
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


class TestRDFFrameBounds:

    def test_start_past_the_last_frame_raises(self, water_trajectory):
        # frame_range, tested in test_trajectories, checks the bounds.
        from revelsMD.rdf import RDF
        with pytest.raises(ValueError):
            RDF(water_trajectory, 'O', 'H').accumulate(water_trajectory, start=3)

    def test_negative_bounds_count_from_the_end(self, water_trajectory):
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        rdf.accumulate(water_trajectory, start=-3, stop=-1)
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

    def test_counts_sum_to_one_per_pair(self):
        """Counts hold one unit of weight per pair within rmax."""
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
        rdf.accumulate(ts)

        # One pair at distance 2.0 (within rmax 5.0); triangular weights sum to 1.
        assert np.sum(rdf._counts) == pytest.approx(1.0, abs=1e-12)

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


class TestRDFCellMatrix:
    """Test that RDF uses cell_matrix for volume and rmax."""

    def test_rdf_stores_cell_matrix(self, water_trajectory):
        """RDF should store cell_matrix from trajectory."""
        from revelsMD.rdf import RDF
        rdf = RDF(water_trajectory, 'O', 'H')
        np.testing.assert_allclose(rdf._cell_matrix, water_trajectory.cell_matrix)

    def test_rdf_rmax_from_inscribed_sphere(self):
        """Default rmax should use inscribed_sphere_radius for triclinic cells."""
        from revelsMD.rdf import RDF

        # Triclinic cell
        cell_matrix = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 1, 2]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 1.1, 2.1]],
        ], dtype=float)
        forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        ], dtype=float)
        ts = NumpyTrajectory(
            positions, forces, species_list=["H", "O", "H"],
            temperature=300.0, units="real", cell_matrix=cell_matrix,
        )

        rdf = RDF(ts, 'O', 'H')

        from revelsMD.cell import inscribed_sphere_radius
        expected_rmax = inscribed_sphere_radius(cell_matrix)
        assert abs(rdf.rmax - expected_rmax) < 1e-10

    def test_rdf_volume_from_cell_determinant(self):
        """RDF prefactor should use det(cell_matrix) for volume."""
        from revelsMD.rdf import RDF

        cell_matrix = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 1, 2]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 1.1, 2.1]],
        ], dtype=float)
        forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        ], dtype=float)
        ts = NumpyTrajectory(
            positions, forces, species_list=["H", "O", "H"],
            temperature=300.0, units="real", cell_matrix=cell_matrix,
        )

        rdf = RDF(ts, 'O', 'H')

        # Volume should be det(cell_matrix) = 10 * 9 * 8 = 720
        expected_volume = abs(np.linalg.det(cell_matrix))
        # Prefactor includes volume / (n_a * n_b) / 2 for unlike species
        # n_O = 1, n_H = 2
        expected_prefactor = expected_volume / (2 * 1) / 2
        assert abs(rdf._prefactor - expected_prefactor) < 1e-10

    def test_rdf_triclinic_accumulate_produces_results(self):
        """RDF accumulate and get_rdf should work for triclinic cells."""
        from revelsMD.rdf import RDF

        cell_matrix = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        positions = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 1, 2]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 1.1, 2.1]],
        ], dtype=float)
        forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        ], dtype=float)
        ts = NumpyTrajectory(
            positions, forces, species_list=["H", "O", "H"],
            temperature=300.0, units="real", cell_matrix=cell_matrix,
        )

        rdf = RDF(ts, 'O', 'H')
        rdf.accumulate(ts)
        rdf.get_rdf(integration='forward')

        assert rdf.r is not None
        assert rdf.g is not None
        assert len(rdf.r) == len(rdf.g)
