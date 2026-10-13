"""Species selection must not depend on how a file orders or numbers its atoms."""

import numpy as np
import pytest

from revelsMD.density import DensityGrid
from revelsMD.rdf import compute_rdf


@pytest.mark.integration
class TestShuffledDumpEquivalence:

    def test_frames_identical_to_sorted_dump(self, example1_trajectory, example1_shuffled_trajectory):
        for sorted_frame, shuffled_frame in zip(
            example1_trajectory.iter_frames(), example1_shuffled_trajectory.iter_frames(), strict=True
        ):
            np.testing.assert_array_equal(shuffled_frame.positions, sorted_frame.positions)
            np.testing.assert_array_equal(shuffled_frame.forces, sorted_frame.forces)

    @pytest.mark.parametrize("species_b", ["1", "2"])
    def test_rdf_identical_to_sorted_dump(self, example1_trajectory, example1_shuffled_trajectory, species_b):
        reference = compute_rdf(example1_trajectory, "1", species_b, integration="forward", delr=0.05)
        result = compute_rdf(example1_shuffled_trajectory, "1", species_b, integration="forward", delr=0.05)
        np.testing.assert_array_equal(result.r, reference.r)
        np.testing.assert_array_equal(result.g, reference.g)
        np.testing.assert_array_equal(result.g_count, reference.g_count)


@pytest.mark.integration
class TestTopologyNumberingIndependence:

    def test_indices_are_positional(self, example4_gro_trajectory):
        universe = example4_gro_trajectory.mdanalysis_universe
        expected = np.flatnonzero(universe.atoms.names == "Ow")
        np.testing.assert_array_equal(example4_gro_trajectory.get_indices("Ow"), expected)

    def test_indices_match_tpr_topology(self, example4_trajectory, example4_gro_trajectory):
        for name in ("Ow", "Hw1", "Hw2"):
            np.testing.assert_array_equal(
                example4_gro_trajectory.get_indices(name),
                example4_trajectory.get_indices(name),
            )

    def test_density_identical_across_topologies(self, example4_trajectory, example4_gro_trajectory):
        grids = []
        for traj in (example4_trajectory, example4_gro_trajectory):
            grid = DensityGrid(traj, density_type="number", nbins=20)
            grid.accumulate(traj, atom_names="Ow", stop=1)
            grids.append(grid)
        np.testing.assert_array_equal(grids[1].rho_count, grids[0].rho_count)
        np.testing.assert_array_equal(grids[1].rho_force, grids[0].rho_force)
