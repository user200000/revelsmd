"""LAMMPS frames read by LammpsTrajectory against an independent dump parser."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parents[1] / "data"
EXAMPLE1 = TEST_DATA_DIR / "example_1_LJ"


def parse_dump(path):
    """Positions and forces of every frame, rows in ascending id order.

    Plain Python: the file is split into frames at ``ITEM: TIMESTEP``, the
    columns are located from each frame's ``ITEM: ATOMS`` line and every
    value is converted with ``float()``.
    """
    text = Path(path).read_text()
    frames = []
    for block in text.split("ITEM: TIMESTEP\n")[1:]:
        lines = block.splitlines()
        n_atoms = int(lines[2])
        assert lines[7].startswith("ITEM: ATOMS")
        columns = lines[7].split()[2:]
        rows = [line.split() for line in lines[8:8 + n_atoms]]
        assert len(rows) == n_atoms and len(lines) == 8 + n_atoms
        rows.sort(key=lambda row: int(row[columns.index("id")]))
        positions = [[float(row[columns.index(c)]) for c in ("x", "y", "z")] for row in rows]
        forces = [[float(row[columns.index(c)]) for c in ("fx", "fy", "fz")] for row in rows]
        frames.append((np.array(positions), np.array(forces)))
    return frames


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
