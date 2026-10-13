"""Row order in a LAMMPS dump must not affect any result."""

import numpy as np
import pytest

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
