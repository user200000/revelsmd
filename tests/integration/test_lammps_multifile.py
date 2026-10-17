"""A dump split into overlapping restart files reads as the single file."""

from pathlib import Path

import numpy as np
import pytest

from revelsMD.rdf import compute_rdf
from revelsMD.trajectories import LammpsTrajectory

TEST_DATA_DIR = Path(__file__).parents[1] / "data"
FRAME_LINES = 9 + 2880  # header lines + atom rows per frame of the Example 1 dump


@pytest.fixture(scope="module")
def split_trajectory(tmp_path_factory):
    source = TEST_DATA_DIR / "example_1_LJ" / "dump.nh.lammps"
    lines = source.read_text().splitlines(keepends=True)
    directory = tmp_path_factory.mktemp("split")
    first = directory / "run1.dump"
    second = directory / "run2.dump"
    first.write_text("".join(lines[: 7 * FRAME_LINES]))   # frames 0-6
    second.write_text("".join(lines[4 * FRAME_LINES:]))   # frames 4-9, restarting 3 frames back
    return LammpsTrajectory(
        [str(first), str(second)],
        str(TEST_DATA_DIR / "example_1_LJ" / "data.fin.nh.data"),
        temperature=1.35, units="lj", atom_style="id resid type charge x y z ix iy iz",
    )


@pytest.mark.integration
@pytest.mark.requires_example1
class TestSplitDump:

    def test_frames_identical_to_single_file(self, example1_trajectory, split_trajectory):
        assert split_trajectory.frames == example1_trajectory.frames == 10
        for single, split in zip(example1_trajectory.iter_frames(), split_trajectory.iter_frames(), strict=True):
            np.testing.assert_array_equal(split.positions, single.positions)
            np.testing.assert_array_equal(split.forces, single.forces)

    @pytest.mark.parametrize("species_b", ["1", "2"])
    def test_rdf_identical_to_single_file(self, example1_trajectory, split_trajectory, species_b):
        reference = compute_rdf(example1_trajectory, "1", species_b, integration="forward", delr=0.05)
        result = compute_rdf(split_trajectory, "1", species_b, integration="forward", delr=0.05)
        np.testing.assert_array_equal(result.g, reference.g)
        np.testing.assert_array_equal(result.g_count, reference.g_count)
