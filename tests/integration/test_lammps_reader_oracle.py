"""LAMMPS frames read by LammpsTrajectory against an independent dump parser."""

from pathlib import Path

import numpy as np
import pytest

from .conftest import parse_dump

TEST_DATA_DIR = Path(__file__).parents[1] / "data"
EXAMPLE1 = TEST_DATA_DIR / "example_1_LJ"


@pytest.fixture(scope="module")
def oracle_frames():
    return parse_dump(EXAMPLE1 / "dump.nh.lammps")


@pytest.mark.integration
@pytest.mark.requires_example1
class TestExample1AgainstAPlainParser:

    def test_unshuffled_dump(self, example1_trajectory, oracle_frames):
        assert len(oracle_frames) == example1_trajectory.frames == 10
        for got, (positions, forces) in zip(
            example1_trajectory.iter_frames(), oracle_frames, strict=True
        ):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    def test_shuffled_dump(self, example1_shuffled_trajectory, oracle_frames):
        shuffled = parse_dump(EXAMPLE1 / "dump.nh.shuffled.lammps")
        for (p_shuffled, f_shuffled), (positions, forces) in zip(
            shuffled, oracle_frames, strict=True
        ):
            np.testing.assert_array_equal(p_shuffled, positions)
            np.testing.assert_array_equal(f_shuffled, forces)
        for got, (positions, forces) in zip(
            example1_shuffled_trajectory.iter_frames(), oracle_frames, strict=True
        ):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)
