"""Species selection on LAMMPS dumps whose rows are not in id order.

Uses the six-atom fixture in tests/data/lammps_small: ids {3, 7, 12, 20,
21, 40}, types alternating by id (3:1, 7:2, 12:1, 20:2, 21:1, 40:2), rows
written in a different order in each frame. Every coordinate encodes the
atom id, so a wrong row shows up as a wrong number.
"""

from pathlib import Path

import numpy as np
import pytest

from revelsMD.trajectories import LammpsTrajectory

FIXTURE_DIR = Path(__file__).parent / "data" / "lammps_small"
ATOM_STYLE = "id type x y z"
IDS_SORTED = np.array([3, 7, 12, 20, 21, 40])


@pytest.fixture(scope="module")
def small():
    return LammpsTrajectory(
        str(FIXTURE_DIR / "dump.lammps"),
        str(FIXTURE_DIR / "data.small.data"),
        temperature=1.0, units="lj", atom_style=ATOM_STYLE,
    )


def test_get_indices_returns_positions_in_id_order(small):
    np.testing.assert_array_equal(small.get_indices("1"), [0, 2, 4])
    np.testing.assert_array_equal(small.get_indices("2"), [1, 3, 5])


def test_frames_are_returned_in_ascending_id_order(small):
    frames = list(small.iter_frames())
    assert len(frames) == 2
    expected0 = np.column_stack([IDS_SORTED, IDS_SORTED + 0.5, IDS_SORTED + 0.25])
    expected1 = expected0 + 1.0
    np.testing.assert_array_equal(frames[0].positions, expected0)
    np.testing.assert_array_equal(frames[1].positions, expected1)
    np.testing.assert_array_equal(
        frames[1].forces, np.column_stack([-IDS_SORTED, np.ones(6), IDS_SORTED])
    )


def test_selected_positions_belong_to_the_selected_species(small):
    idx = small.get_indices("2")
    frame0, frame1 = small.iter_frames()
    np.testing.assert_array_equal(frame0.positions[idx, 0], [7.0, 20.0, 40.0])
    np.testing.assert_array_equal(frame1.positions[idx, 0], [8.0, 21.0, 41.0])


def test_get_frame_matches_iteration(small):
    frames = list(small.iter_frames())
    np.testing.assert_array_equal(small.get_frame(1).positions, frames[1].positions)
    np.testing.assert_array_equal(small.get_frame(1).forces, frames[1].forces)


def test_dump_without_id_column_is_rejected():
    with pytest.raises(RuntimeError, match="'id'"):
        LammpsTrajectory(
            str(FIXTURE_DIR / "dump_noid.lammps"),
            str(FIXTURE_DIR / "data.small.data"),
            temperature=1.0, units="lj", atom_style=ATOM_STYLE,
        )


def _make(dump_name):
    return LammpsTrajectory(
        str(FIXTURE_DIR / dump_name),
        str(FIXTURE_DIR / "data.small.data"),
        temperature=1.0, units="lj", atom_style=ATOM_STYLE,
    )


def test_frame_whose_ids_do_not_match_the_topology_is_rejected():
    traj = _make("dump_dupid.lammps")
    with pytest.raises(ValueError, match="frame 0"):
        list(traj.iter_frames())


def test_later_corrupt_frame_is_rejected_at_that_frame():
    traj = _make("dump_badframe.lammps")
    frames = traj.iter_frames()
    next(frames)
    with pytest.raises(ValueError, match="frame 1"):
        next(frames)
