import pytest
import numpy as np
from abc import ABC
from unittest.mock import MagicMock, patch

from revelsMD.trajectory_states import (
    TrajectoryState,
    MDATrajectoryState,
    NumpyTrajectoryState,
    LammpsTrajectoryState,
    VaspTrajectoryState,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_mdanalysis_universe():
    """Mock MDAnalysis Universe object with orthorhombic box."""
    mock = MagicMock()
    mock.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock.trajectory = [0, 1, 2]
    mock.select_atoms.return_value.ids = np.array([1, 2, 3])
    mock.select_atoms.return_value.charges = np.array([0.1, 0.2, 0.3])
    mock.select_atoms.return_value.masses = np.array([12.0, 1.0, 16.0])
    return mock


@pytest.fixture
def mock_vasprun():
    """Mock Vasprun object with structures and forces."""
    mock_lattice = MagicMock()
    mock_lattice.angles = np.array([90.0, 90.0, 90.0])
    mock_lattice.matrix = np.diag([5.0, 5.0, 5.0])

    mock_structure = MagicMock()
    mock_structure.lattice = mock_lattice
    mock_structure.frac_coords = np.array([[0.0, 0.0, 0.0]])
    mock_structure.cart_coords = np.array([[1.0, 1.0, 1.0]])

    mock = MagicMock()
    mock.structures = [mock_structure]
    mock.start = mock_structure
    mock.cart_coords = np.array([[[1.0, 1.0, 1.0]]])
    mock.forces = np.array([[[0.1, 0.2, 0.3]]])
    mock.start.indices_from_symbol.return_value = np.array([0])
    return mock


# -----------------------------------------------------------------------------
# MDATrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
def test_mda_initialization_and_accessors(mock_universe, mock_mdanalysis_universe):
    mock_universe.return_value = mock_mdanalysis_universe
    state = MDATrajectoryState("traj.xtc", "topol.pdb")

    assert state.variety == "mda"
    assert state.frames == 3
    assert np.isclose(state.box_x, 10.0)
    assert np.isclose(state.box_y, 10.0)
    assert np.isclose(state.box_z, 10.0)

    assert np.all(state.get_indices("H") == np.array([1, 2, 3]))
    assert np.allclose(state.get_charges("H"), [0.1, 0.2, 0.3])
    assert np.allclose(state.get_masses("H"), [12.0, 1.0, 16.0])

    # backward compatibility
    assert np.all(state.get_indicies("H") == np.array([1, 2, 3]))


@patch("revelsMD.trajectory_states.MD.Universe", side_effect=Exception("fail"))
def test_mda_raises_on_universe_failure(mock_universe):
    with pytest.raises(RuntimeError, match="Failed to load MDAnalysis Universe"):
        MDATrajectoryState("traj.xtc", "topol.pdb")


def test_mda_raises_no_topology():
    with pytest.raises(ValueError, match="topology file is required"):
        MDATrajectoryState("traj.xtc", "")


# -----------------------------------------------------------------------------
# NumpyTrajectoryState
# -----------------------------------------------------------------------------
def test_numpy_state_valid_and_accessors():
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]

    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)
    assert state.frames == 5
    assert state.variety == "numpy"
    assert np.allclose(state.get_indices("H"), [1, 2])

    # backward alias
    assert np.allclose(state.get_indicies("O"), [0])


def test_numpy_state_species_not_found():
    positions = np.zeros((1, 2, 3))
    forces = np.ones((1, 2, 3))
    species = ["O", "H"]
    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)
    with pytest.raises(ValueError, match="Species 'C' not found"):
        state.get_indices("C")


def test_numpy_state_invalid_shapes_and_box():
    with pytest.raises(ValueError, match="incommensurate"):
        NumpyTrajectoryState(np.zeros((1, 2, 3)), np.ones((1, 3, 3)), 10, 10, 10, ["O", "H"])

    with pytest.raises(ValueError, match="incommensurate"):
        NumpyTrajectoryState(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), 10, 10, 10, ["O"])

    with pytest.raises(ValueError, match="positive values"):
        NumpyTrajectoryState(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), -1, 10, 10, ["O", "H"])


# -----------------------------------------------------------------------------
# LammpsTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
@patch("revelsMD.trajectory_states.first_read", return_value=(10, 5, ["id", "x", "y", "z"], 9, np.zeros((3, 2))))
def test_lammps_state_valid(mock_first_read, mock_universe, mock_mdanalysis_universe):
    mock_universe.return_value = mock_mdanalysis_universe
    state = LammpsTrajectoryState("dump.lammpstrj", "data.lmp")
    assert state.variety == "lammps"
    assert np.isclose(state.box_x, 10.0)
    assert state.frames == 3


@patch("revelsMD.trajectory_states.MD.Universe", side_effect=Exception("bad universe"))
@patch("revelsMD.trajectory_states.first_read", return_value=(10, 5, [], 9, np.zeros((3, 2))))
def test_lammps_state_universe_error(mock_first_read, mock_universe):
    with pytest.raises(RuntimeError, match="Failed to load LAMMPS trajectory"):
        LammpsTrajectoryState("dump.lammpstrj", "data.lmp")


def test_lammps_state_requires_topology():
    with pytest.raises(ValueError, match="topology file is required"):
        LammpsTrajectoryState("dump.lammpstrj", None)


# -----------------------------------------------------------------------------
# VaspTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_state_valid(mock_vasprun):
    mock_vasprun_instance = mock_vasprun.return_value

    # ✅ Properly configure the mock Vasprun so lattice.matrix is 3×3
    from unittest.mock import MagicMock
    import numpy as np

    mock_vasprun_instance.structures = [MagicMock()]
    mock_vasprun_instance.structures[0].lattice.matrix = np.eye(3)
    mock_vasprun_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_vasprun_instance.structures[0].lattice = mock_vasprun_instance.structures[0].lattice
    mock_vasprun_instance.structures[0].frac_coords = np.array([[0.0, 0.0, 0.0]])
    mock_vasprun_instance.cart_coords = np.zeros((1, 1, 3))
    mock_vasprun_instance.forces = np.zeros((1, 1, 3))
    mock_vasprun_instance.start = mock_vasprun_instance.structures[0]
    mock_vasprun_instance.start.indices_from_symbol.return_value = np.array([0])

    # Then run the test
    state = VaspTrajectoryState("vasprun.xml")
    assert state.variety == "vasp"
    assert np.isclose(state.box_x, 1.0)
    assert np.allclose(state.positions, np.zeros((1, 1, 3)))
    assert np.allclose(state.forces, np.zeros((1, 1, 3)))
    assert np.all(state.get_indices("H") == np.array([0]))



@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_state_raises_no_forces(mock_vasprun):
    mock = MagicMock()
    mock.structures = [MagicMock()]
    mock.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock.structures[0].lattice.matrix = np.diag([5.0, 5.0, 5.0])
    mock.forces = None
    mock.cart_coords = np.zeros((1, 1, 3))
    mock_vasprun.return_value = mock
    with pytest.raises(ValueError, match="No forces found"):
        VaspTrajectoryState("vasprun.xml")


@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_state_invalid_angles(mock_vasprun):
    mock = MagicMock()
    mock.structures = [MagicMock()]
    mock.structures[0].lattice.angles = [90.0, 95.0, 90.0]
    mock.structures[0].lattice.matrix = np.diag([5.0, 5.0, 5.0])
    mock.forces = np.zeros((1, 1, 3))
    mock.cart_coords = np.zeros((1, 1, 3))
    mock_vasprun.return_value = mock
    with pytest.raises(ValueError, match="orthorhombic"):
        VaspTrajectoryState("vasprun.xml")


# -----------------------------------------------------------------------------
# TrajectoryState ABC
# -----------------------------------------------------------------------------
def test_trajectory_state_is_abstract():
    """TrajectoryState should be an abstract base class."""
    assert issubclass(TrajectoryState, ABC)


def test_trajectory_state_cannot_be_instantiated():
    """TrajectoryState should not be directly instantiable."""
    with pytest.raises(TypeError, match="abstract"):
        TrajectoryState()


def test_concrete_classes_are_subclasses():
    """All concrete trajectory classes should inherit from TrajectoryState."""
    assert issubclass(MDATrajectoryState, TrajectoryState)
    assert issubclass(NumpyTrajectoryState, TrajectoryState)
    assert issubclass(LammpsTrajectoryState, TrajectoryState)
    assert issubclass(VaspTrajectoryState, TrajectoryState)


# -----------------------------------------------------------------------------
# iter_frames - NumpyTrajectoryState
# -----------------------------------------------------------------------------
def test_numpy_iter_frames_yields_all_frames():
    """iter_frames should yield positions and forces for each frame."""
    n_frames, n_atoms = 5, 3
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)
    species = ["O", "H", "H"]

    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

    frames = list(state.iter_frames())
    assert len(frames) == n_frames

    for i, (pos, frc) in enumerate(frames):
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


def test_numpy_iter_frames_with_start_stop_stride():
    """iter_frames should respect start, stop, and stride parameters."""
    n_frames, n_atoms = 10, 2
    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.zeros((n_frames, n_atoms, 3))
    species = ["A", "B"]

    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

    # Test start=2, stop=8, stride=2 -> frames 2, 4, 6
    frames = list(state.iter_frames(start=2, stop=8, stride=2))
    assert len(frames) == 3

    expected_indices = [2, 4, 6]
    for idx, (pos, _) in zip(expected_indices, frames):
        np.testing.assert_array_equal(pos, positions[idx])


# -----------------------------------------------------------------------------
# iter_frames - VaspTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_iter_frames_yields_all_frames(mock_vasprun):
    """iter_frames should yield positions and forces for each frame."""
    n_frames, n_atoms = 3, 2
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)

    mock_instance = mock_vasprun.return_value
    mock_instance.structures = [MagicMock() for _ in range(n_frames)]
    mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
    mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_instance.start = mock_instance.structures[0]
    mock_instance.start.indices_from_symbol.return_value = np.array([0])
    mock_instance.cart_coords = positions
    mock_instance.forces = forces

    state = VaspTrajectoryState("vasprun.xml")

    frames_list = list(state.iter_frames())
    assert len(frames_list) == n_frames

    for i, (pos, frc) in enumerate(frames_list):
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_iter_frames_with_start_stop_stride(mock_vasprun):
    """iter_frames should respect start, stop, and stride parameters."""
    n_frames, n_atoms = 10, 2
    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.zeros((n_frames, n_atoms, 3))

    mock_instance = mock_vasprun.return_value
    mock_instance.structures = [MagicMock() for _ in range(n_frames)]
    mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
    mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_instance.start = mock_instance.structures[0]
    mock_instance.start.indices_from_symbol.return_value = np.array([0])
    mock_instance.cart_coords = positions
    mock_instance.forces = forces

    state = VaspTrajectoryState("vasprun.xml")

    # Test start=2, stop=8, stride=2 -> frames 2, 4, 6
    frames_list = list(state.iter_frames(start=2, stop=8, stride=2))
    assert len(frames_list) == 3

    expected_indices = [2, 4, 6]
    for idx, (pos, _) in zip(expected_indices, frames_list):
        np.testing.assert_array_equal(pos, positions[idx])


# -----------------------------------------------------------------------------
# iter_frames - MDATrajectoryState
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# iter_frames - LammpsTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
@patch("revelsMD.trajectory_states.first_read")
def test_lammps_iter_frames_yields_positions_and_forces(mock_first_read, mock_universe):
    """iter_frames should yield positions and forces for each frame."""
    n_frames, n_atoms = 3, 4

    # Mock first_read to return metadata
    # dic simulates: ITEM: ATOMS id type x y z fx fy fz
    mock_dic = ["ITEM:", "ATOMS", "id", "type", "x", "y", "z", "fx", "fy", "fz"]
    mock_first_read.return_value = (n_frames, n_atoms, mock_dic, 9, np.zeros((3, 2)))

    # Mock MDAnalysis universe
    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_uni.trajectory = mock_trajectory
    mock_universe.return_value = mock_uni

    # Create test data - positions and forces for each frame
    positions = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]
    forces = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]

    # Mock get_a_frame to return combined position+force data
    frame_idx = [0]
    def mock_get_a_frame(f, num_ats, header_length, strngdex):
        idx = frame_idx[0]
        frame_idx[0] += 1
        # Return combined [x, y, z, fx, fy, fz] array
        return np.hstack([positions[idx], forces[idx]])

    with patch("revelsMD.trajectory_states.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectory_states.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("builtins.open", MagicMock()):
                state = LammpsTrajectoryState("dump.lammpstrj", "data.lmp")

                frames_list = list(state.iter_frames())
                assert len(frames_list) == n_frames

                for i, (pos, frc) in enumerate(frames_list):
                    np.testing.assert_array_equal(pos, positions[i])
                    np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectory_states.MD.Universe")
@patch("revelsMD.trajectory_states.first_read")
def test_lammps_iter_frames_with_start_stop_stride(mock_first_read, mock_universe):
    """iter_frames should respect start, stop, and stride parameters."""
    n_frames, n_atoms = 10, 2

    mock_dic = ["ITEM:", "ATOMS", "id", "type", "x", "y", "z", "fx", "fy", "fz"]
    mock_first_read.return_value = (n_frames, n_atoms, mock_dic, 9, np.zeros((3, 2)))

    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_uni.trajectory = mock_trajectory
    mock_universe.return_value = mock_uni

    # Create predictable test data
    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float) + 1000

    # Track which frames get_a_frame is called for
    frames_read = []
    def mock_get_a_frame(f, num_ats, header_length, strngdex):
        # We need to track the actual frame index being read
        idx = len(frames_read)
        frames_read.append(idx)
        return np.hstack([positions[idx], forces[idx]])

    # Track frame_skip calls
    skip_calls = []
    def mock_frame_skip(f, num_ats, num_skip, header_length):
        skip_calls.append(num_skip)

    with patch("revelsMD.trajectory_states.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectory_states.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectory_states.frame_skip", side_effect=mock_frame_skip):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectoryState("dump.lammpstrj", "data.lmp")

                    # Test start=2, stop=8, stride=2 -> should yield frames at indices 2, 4, 6
                    frames_list = list(state.iter_frames(start=2, stop=8, stride=2))
                    assert len(frames_list) == 3


# -----------------------------------------------------------------------------
# iter_frames - MDATrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
def test_mda_iter_frames_yields_positions_and_forces(mock_universe):
    """iter_frames should yield positions and forces for each frame."""
    n_frames, n_atoms = 3, 4
    positions = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]
    forces = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]

    # Create mock timestep objects with positions and forces
    mock_timesteps = []
    for i in range(n_frames):
        ts = MagicMock()
        ts.positions = positions[i]
        ts.forces = forces[i]
        mock_timesteps.append(ts)

    # Create mock trajectory that supports slicing
    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_trajectory.__getitem__ = MagicMock(
        side_effect=lambda s: mock_timesteps[s] if isinstance(s, int) else mock_timesteps[s.start:s.stop:s.step]
    )

    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_uni.trajectory = mock_trajectory
    mock_uni.select_atoms.return_value.ids = np.array([1, 2, 3])
    mock_universe.return_value = mock_uni

    state = MDATrajectoryState("traj.xtc", "topol.pdb")

    frames_list = list(state.iter_frames())
    assert len(frames_list) == n_frames

    for i, (pos, frc) in enumerate(frames_list):
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - NumpyTrajectoryState
# -----------------------------------------------------------------------------
def test_numpy_get_frame_returns_correct_data():
    """get_frame should return positions and forces for the specified index."""
    n_frames, n_atoms = 5, 3
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)
    species = ["O", "H", "H"]

    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

    for i in range(n_frames):
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


def test_numpy_get_frame_random_access():
    """get_frame should support random access in any order."""
    n_frames, n_atoms = 10, 2
    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.zeros((n_frames, n_atoms, 3))
    species = ["A", "B"]

    state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

    # Access frames in non-sequential order
    for i in [7, 2, 9, 0, 5]:
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])


# -----------------------------------------------------------------------------
# get_frame - VaspTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.Vasprun")
def test_vasp_get_frame_returns_correct_data(mock_vasprun):
    """get_frame should return positions and forces for the specified index."""
    n_frames, n_atoms = 5, 2
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)

    mock_instance = mock_vasprun.return_value
    mock_instance.structures = [MagicMock() for _ in range(n_frames)]
    mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
    mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_instance.start = mock_instance.structures[0]
    mock_instance.start.indices_from_symbol.return_value = np.array([0])
    mock_instance.cart_coords = positions
    mock_instance.forces = forces

    state = VaspTrajectoryState("vasprun.xml")

    for i in range(n_frames):
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - MDATrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
def test_mda_get_frame_returns_correct_data(mock_universe):
    """get_frame should return positions and forces for the specified index."""
    n_frames, n_atoms = 5, 3
    positions = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]
    forces = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]

    mock_timesteps = []
    for i in range(n_frames):
        ts = MagicMock()
        ts.positions = positions[i]
        ts.forces = forces[i]
        mock_timesteps.append(ts)

    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_trajectory.__getitem__ = MagicMock(side_effect=lambda idx: mock_timesteps[idx])

    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_uni.trajectory = mock_trajectory
    mock_uni.select_atoms.return_value.ids = np.array([1, 2, 3])
    mock_universe.return_value = mock_uni

    state = MDATrajectoryState("traj.xtc", "topol.pdb")

    for i in range(n_frames):
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - LammpsTrajectoryState
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectory_states.MD.Universe")
@patch("revelsMD.trajectory_states.first_read")
def test_lammps_get_frame_returns_correct_data(mock_first_read, mock_universe):
    """get_frame should return positions and forces for the specified index."""
    n_frames, n_atoms = 5, 3

    mock_dic = ["ITEM:", "ATOMS", "id", "type", "x", "y", "z", "fx", "fy", "fz"]
    mock_first_read.return_value = (n_frames, n_atoms, mock_dic, 9, np.zeros((3, 2)))

    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_uni.trajectory = mock_trajectory
    mock_universe.return_value = mock_uni

    positions = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]
    forces = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]

    def mock_get_a_frame(f, num_ats, header_length, strngdex):
        idx = mock_get_a_frame.call_count
        mock_get_a_frame.call_count += 1
        return np.hstack([positions[idx], forces[idx]])
    mock_get_a_frame.call_count = 0

    with patch("revelsMD.trajectory_states.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectory_states.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectory_states.frame_skip"):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectoryState("dump.lammpstrj", "data.lmp")

                    for i in range(n_frames):
                        pos, frc = state.get_frame(i)
                        np.testing.assert_array_equal(pos, positions[i])
                        np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectory_states.MD.Universe")
@patch("revelsMD.trajectory_states.first_read")
def test_lammps_get_frame_random_access(mock_first_read, mock_universe):
    """get_frame should support random access after caching."""
    n_frames, n_atoms = 10, 2

    mock_dic = ["ITEM:", "ATOMS", "id", "type", "x", "y", "z", "fx", "fy", "fz"]
    mock_first_read.return_value = (n_frames, n_atoms, mock_dic, 9, np.zeros((3, 2)))

    mock_uni = MagicMock()
    mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    mock_trajectory = MagicMock()
    mock_trajectory.__len__ = MagicMock(return_value=n_frames)
    mock_uni.trajectory = mock_trajectory
    mock_universe.return_value = mock_uni

    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float) + 1000

    def mock_get_a_frame(f, num_ats, header_length, strngdex):
        idx = mock_get_a_frame.call_count
        mock_get_a_frame.call_count += 1
        return np.hstack([positions[idx], forces[idx]])
    mock_get_a_frame.call_count = 0

    with patch("revelsMD.trajectory_states.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectory_states.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectory_states.frame_skip"):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectoryState("dump.lammpstrj", "data.lmp")

                    # Random access in any order
                    for i in [7, 2, 9, 0, 5]:
                        pos, frc = state.get_frame(i)
                        np.testing.assert_array_equal(pos, positions[i])
                        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# iter_frames - Negative index handling (Pythonic behaviour)
# -----------------------------------------------------------------------------
class TestIterFramesNegativeIndices:
    """Test that iter_frames handles negative indices like Python slices."""

    def test_numpy_negative_stop_excludes_last_frame(self):
        """stop=-1 should iterate up to but not including the last frame."""
        n_frames, n_atoms = 5, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))
        species = ["A", "B"]

        state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

        # stop=-1 means "all but last" -> frames 0, 1, 2, 3
        frames = list(state.iter_frames(stop=-1))
        assert len(frames) == 4

        for i, (pos, _) in enumerate(frames):
            np.testing.assert_array_equal(pos, positions[i])

    def test_numpy_negative_start(self):
        """start=-3 should start 3 frames from the end."""
        n_frames, n_atoms = 10, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))
        species = ["A", "B"]

        state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

        # start=-3 means start at frame 7 (10-3=7)
        frames = list(state.iter_frames(start=-3))
        assert len(frames) == 3

        expected_indices = [7, 8, 9]
        for i, (pos, _) in enumerate(frames):
            np.testing.assert_array_equal(pos, positions[expected_indices[i]])

    def test_numpy_negative_start_and_stop(self):
        """Both negative start and stop should work together."""
        n_frames, n_atoms = 10, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))
        species = ["A", "B"]

        state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

        # start=-5 (frame 5), stop=-2 (frame 8) -> frames 5, 6, 7
        frames = list(state.iter_frames(start=-5, stop=-2))
        assert len(frames) == 3

        expected_indices = [5, 6, 7]
        for i, (pos, _) in enumerate(frames):
            np.testing.assert_array_equal(pos, positions[expected_indices[i]])

    def test_numpy_stop_none_means_all_frames(self):
        """stop=None should iterate through all frames."""
        n_frames, n_atoms = 5, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))
        species = ["A", "B"]

        state = NumpyTrajectoryState(positions, forces, 10, 10, 10, species)

        frames = list(state.iter_frames(stop=None))
        assert len(frames) == n_frames

    @patch("revelsMD.trajectory_states.Vasprun")
    def test_vasp_negative_stop(self, mock_vasprun):
        """VaspTrajectoryState should handle negative stop index."""
        n_frames, n_atoms = 5, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))

        mock_instance = mock_vasprun.return_value
        mock_instance.structures = [MagicMock() for _ in range(n_frames)]
        mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
        mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
        mock_instance.start = mock_instance.structures[0]
        mock_instance.start.indices_from_symbol.return_value = np.array([0])
        mock_instance.cart_coords = positions
        mock_instance.forces = forces

        state = VaspTrajectoryState("vasprun.xml")

        # stop=-1 means all but last -> frames 0, 1, 2, 3
        frames = list(state.iter_frames(stop=-1))
        assert len(frames) == 4

    @patch("revelsMD.trajectory_states.MD.Universe")
    def test_mda_negative_stop(self, mock_universe):
        """MDATrajectoryState should handle negative stop index."""
        n_frames, n_atoms = 5, 3
        positions = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]
        forces = [np.random.rand(n_atoms, 3) for _ in range(n_frames)]

        mock_timesteps = []
        for i in range(n_frames):
            ts = MagicMock()
            ts.positions = positions[i]
            ts.forces = forces[i]
            mock_timesteps.append(ts)

        mock_trajectory = MagicMock()
        mock_trajectory.__len__ = MagicMock(return_value=n_frames)
        # Simulate slicing behaviour
        mock_trajectory.__getitem__ = MagicMock(
            side_effect=lambda s: mock_timesteps[s.start:s.stop:s.step] if isinstance(s, slice) else mock_timesteps[s]
        )

        mock_uni = MagicMock()
        mock_uni.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        mock_uni.trajectory = mock_trajectory
        mock_uni.select_atoms.return_value.ids = np.array([1, 2, 3])
        mock_universe.return_value = mock_uni

        state = MDATrajectoryState("traj.xtc", "topol.pdb")

        # stop=-1 normalized to 4, so frames 0, 1, 2, 3
        frames = list(state.iter_frames(stop=-1))
        assert len(frames) == 4
