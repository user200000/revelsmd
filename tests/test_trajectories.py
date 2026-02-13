import pytest
import numpy as np
import scipy.constants as constants
from abc import ABC
from unittest.mock import MagicMock, patch

from revelsMD.trajectories import (
    MDATrajectory,
    NumpyTrajectory,
    LammpsTrajectory,
    VaspTrajectory,
    DataUnavailableError,
)
from revelsMD.trajectories._base import Trajectory, compute_beta


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
# MDATrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
def test_mda_initialization_and_accessors(mock_universe, mock_mdanalysis_universe):
    mock_universe.return_value = mock_mdanalysis_universe
    state = MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)

    assert state.frames == 3
    assert np.isclose(state.box_x, 10.0)
    assert np.isclose(state.box_y, 10.0)
    assert np.isclose(state.box_z, 10.0)

    assert np.all(state.get_indices("H") == np.array([1, 2, 3]))
    assert np.allclose(state.get_charges("H"), [0.1, 0.2, 0.3])
    assert np.allclose(state.get_masses("H"), [12.0, 1.0, 16.0])

    # backward compatibility
    assert np.all(state.get_indicies("H") == np.array([1, 2, 3]))


@patch("revelsMD.trajectories.mda.MD.Universe", side_effect=Exception("fail"))
def test_mda_raises_on_universe_failure(mock_universe):
    with pytest.raises(RuntimeError, match="Failed to load MDAnalysis Universe"):
        MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)


def test_mda_raises_no_topology():
    with pytest.raises(ValueError, match="topology file is required"):
        MDATrajectory("traj.xtc", "", temperature=300.0)


# -----------------------------------------------------------------------------
# NumpyTrajectory
# -----------------------------------------------------------------------------
def test_numpy_state_valid_and_accessors():
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)
    assert state.frames == 5
    assert np.allclose(state.get_indices("H"), [1, 2])

    # backward alias
    assert np.allclose(state.get_indicies("O"), [0])


def test_numpy_state_species_not_found():
    positions = np.zeros((1, 2, 3))
    forces = np.ones((1, 2, 3))
    species = ["O", "H"]
    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)
    with pytest.raises(ValueError, match="Species 'C' not found"):
        state.get_indices("C")


def test_numpy_state_invalid_shapes_and_box():
    with pytest.raises(ValueError, match="incommensurate"):
        NumpyTrajectory(np.zeros((1, 2, 3)), np.ones((1, 3, 3)), 10, 10, 10, ["O", "H"], temperature=300.0)

    with pytest.raises(ValueError, match="incommensurate"):
        NumpyTrajectory(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), 10, 10, 10, ["O"], temperature=300.0)

    with pytest.raises(ValueError, match="positive values"):
        NumpyTrajectory(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), -1, 10, 10, ["O", "H"], temperature=300.0)


# -----------------------------------------------------------------------------
# LammpsTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
@patch("revelsMD.trajectories.lammps.first_read", return_value=(10, 5, ["id", "x", "y", "z"], 9, np.zeros((3, 2))))
def test_lammps_state_valid(mock_first_read, mock_universe, mock_mdanalysis_universe):
    mock_universe.return_value = mock_mdanalysis_universe
    state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)
    assert np.isclose(state.box_x, 10.0)
    assert state.frames == 3


@patch("revelsMD.trajectories.mda.MD.Universe", side_effect=Exception("bad universe"))
@patch("revelsMD.trajectories.lammps.first_read", return_value=(10, 5, [], 9, np.zeros((3, 2))))
def test_lammps_state_universe_error(mock_first_read, mock_universe):
    with pytest.raises(RuntimeError, match="Failed to load LAMMPS trajectory"):
        LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)


def test_lammps_state_requires_topology():
    with pytest.raises(ValueError, match="topology file is required"):
        LammpsTrajectory("dump.lammpstrj", None, temperature=300.0)


# -----------------------------------------------------------------------------
# VaspTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.vasp.Vasprun")
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
    state = VaspTrajectory("vasprun.xml", temperature=300.0)
    assert np.isclose(state.box_x, 1.0)
    assert np.allclose(state.positions, np.zeros((1, 1, 3)))
    assert np.allclose(state.forces, np.zeros((1, 1, 3)))
    assert np.all(state.get_indices("H") == np.array([0]))



@patch("revelsMD.trajectories.vasp.Vasprun")
def test_vasp_state_raises_no_forces(mock_vasprun):
    mock = MagicMock()
    mock.structures = [MagicMock()]
    mock.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock.structures[0].lattice.matrix = np.diag([5.0, 5.0, 5.0])
    mock.forces = None
    mock.cart_coords = np.zeros((1, 1, 3))
    mock_vasprun.return_value = mock
    with pytest.raises(ValueError, match="No forces found"):
        VaspTrajectory("vasprun.xml", temperature=300.0)


@patch("revelsMD.trajectories.vasp.Vasprun")
def test_vasp_state_invalid_angles(mock_vasprun):
    mock = MagicMock()
    mock.structures = [MagicMock()]
    mock.structures[0].lattice.angles = [90.0, 95.0, 90.0]
    mock.structures[0].lattice.matrix = np.diag([5.0, 5.0, 5.0])
    mock.forces = np.zeros((1, 1, 3))
    mock.cart_coords = np.zeros((1, 1, 3))
    mock_vasprun.return_value = mock
    with pytest.raises(ValueError, match="orthorhombic"):
        VaspTrajectory("vasprun.xml", temperature=300.0)


# -----------------------------------------------------------------------------
# Trajectory ABC
# -----------------------------------------------------------------------------
def test_trajectory_state_is_abstract():
    """Trajectory should be an abstract base class."""
    assert issubclass(Trajectory, ABC)


def test_trajectory_state_cannot_be_instantiated():
    """Trajectory should not be directly instantiable."""
    with pytest.raises(TypeError, match="abstract"):
        Trajectory()


def test_concrete_classes_are_subclasses():
    """All concrete trajectory classes should inherit from Trajectory."""
    assert issubclass(MDATrajectory, Trajectory)
    assert issubclass(NumpyTrajectory, Trajectory)
    assert issubclass(LammpsTrajectory, Trajectory)
    assert issubclass(VaspTrajectory, Trajectory)


# -----------------------------------------------------------------------------
# Trajectory ABC - Shared validation helpers
# -----------------------------------------------------------------------------
class TestValidateOrthorhombic:
    """Tests for _validate_orthorhombic shared helper."""

    def test_valid_orthorhombic_angles(self):
        """Should not raise for 90 degree angles."""
        # No exception expected
        Trajectory._validate_orthorhombic([90.0, 90.0, 90.0])

    def test_valid_orthorhombic_angles_within_tolerance(self):
        """Should accept angles within tolerance of 90 degrees."""
        Trajectory._validate_orthorhombic([90.0001, 89.9999, 90.0])

    def test_invalid_non_orthorhombic_angles(self):
        """Should raise ValueError for non-orthorhombic angles."""
        with pytest.raises(ValueError, match="orthorhombic"):
            Trajectory._validate_orthorhombic([90.0, 95.0, 90.0])

    def test_invalid_triclinic_angles(self):
        """Should raise ValueError for triclinic cell angles."""
        with pytest.raises(ValueError, match="orthorhombic"):
            Trajectory._validate_orthorhombic([80.0, 85.0, 70.0])


class TestValidateBoxDimensions:
    """Tests for _validate_box_dimensions shared helper."""

    def test_valid_positive_dimensions(self):
        """Should return dimensions for valid positive values."""
        lx, ly, lz = Trajectory._validate_box_dimensions(10.0, 20.0, 30.0)
        assert lx == 10.0
        assert ly == 20.0
        assert lz == 30.0

    def test_invalid_zero_dimension(self):
        """Should raise ValueError if any dimension is zero."""
        with pytest.raises(ValueError, match="positive"):
            Trajectory._validate_box_dimensions(10.0, 0.0, 30.0)

    def test_invalid_negative_dimension(self):
        """Should raise ValueError if any dimension is negative."""
        with pytest.raises(ValueError, match="positive"):
            Trajectory._validate_box_dimensions(10.0, -5.0, 30.0)

    def test_invalid_non_finite_dimension(self):
        """Should raise ValueError if any dimension is non-finite."""
        with pytest.raises(ValueError, match="finite"):
            Trajectory._validate_box_dimensions(10.0, np.inf, 30.0)

    def test_invalid_nan_dimension(self):
        """Should raise ValueError if any dimension is NaN."""
        with pytest.raises(ValueError, match="finite"):
            Trajectory._validate_box_dimensions(np.nan, 20.0, 30.0)


# -----------------------------------------------------------------------------
# iter_frames - NumpyTrajectory
# -----------------------------------------------------------------------------
def test_numpy_iter_frames_yields_all_frames():
    """iter_frames should yield positions and forces for each frame."""
    n_frames, n_atoms = 5, 3
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

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

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    # Test start=2, stop=8, stride=2 -> frames 2, 4, 6
    frames = list(state.iter_frames(start=2, stop=8, stride=2))
    assert len(frames) == 3

    expected_indices = [2, 4, 6]
    for idx, (pos, _) in zip(expected_indices, frames):
        np.testing.assert_array_equal(pos, positions[idx])


# -----------------------------------------------------------------------------
# iter_frames - VaspTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.vasp.Vasprun")
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

    state = VaspTrajectory("vasprun.xml", temperature=300.0)

    frames_list = list(state.iter_frames())
    assert len(frames_list) == n_frames

    for i, (pos, frc) in enumerate(frames_list):
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectories.vasp.Vasprun")
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

    state = VaspTrajectory("vasprun.xml", temperature=300.0)

    # Test start=2, stop=8, stride=2 -> frames 2, 4, 6
    frames_list = list(state.iter_frames(start=2, stop=8, stride=2))
    assert len(frames_list) == 3

    expected_indices = [2, 4, 6]
    for idx, (pos, _) in zip(expected_indices, frames_list):
        np.testing.assert_array_equal(pos, positions[idx])


# -----------------------------------------------------------------------------
# iter_frames - MDATrajectory
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# iter_frames - LammpsTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
@patch("revelsMD.trajectories.lammps.first_read")
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

    with patch("revelsMD.trajectories.lammps.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectories.lammps.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("builtins.open", MagicMock()):
                state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)

                frames_list = list(state.iter_frames())
                assert len(frames_list) == n_frames

                for i, (pos, frc) in enumerate(frames_list):
                    np.testing.assert_array_equal(pos, positions[i])
                    np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectories.mda.MD.Universe")
@patch("revelsMD.trajectories.lammps.first_read")
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

    with patch("revelsMD.trajectories.lammps.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectories.lammps.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectories.lammps.frame_skip", side_effect=mock_frame_skip):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)

                    # Test start=2, stop=8, stride=2 -> should yield frames at indices 2, 4, 6
                    frames_list = list(state.iter_frames(start=2, stop=8, stride=2))
                    assert len(frames_list) == 3


# -----------------------------------------------------------------------------
# iter_frames - MDATrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
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

    state = MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)

    frames_list = list(state.iter_frames())
    assert len(frames_list) == n_frames

    for i, (pos, frc) in enumerate(frames_list):
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - NumpyTrajectory
# -----------------------------------------------------------------------------
def test_numpy_get_frame_returns_correct_data():
    """get_frame should return positions and forces for the specified index."""
    n_frames, n_atoms = 5, 3
    positions = np.random.rand(n_frames, n_atoms, 3)
    forces = np.random.rand(n_frames, n_atoms, 3)
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

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

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    # Access frames in non-sequential order
    for i in [7, 2, 9, 0, 5]:
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])


# -----------------------------------------------------------------------------
# get_frame - VaspTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.vasp.Vasprun")
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

    state = VaspTrajectory("vasprun.xml", temperature=300.0)

    for i in range(n_frames):
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - MDATrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
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

    state = MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)

    for i in range(n_frames):
        pos, frc = state.get_frame(i)
        np.testing.assert_array_equal(pos, positions[i])
        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_frame - LammpsTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.mda.MD.Universe")
@patch("revelsMD.trajectories.lammps.first_read")
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

    with patch("revelsMD.trajectories.lammps.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectories.lammps.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectories.lammps.frame_skip"):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)

                    for i in range(n_frames):
                        pos, frc = state.get_frame(i)
                        np.testing.assert_array_equal(pos, positions[i])
                        np.testing.assert_array_equal(frc, forces[i])


@patch("revelsMD.trajectories.mda.MD.Universe")
@patch("revelsMD.trajectories.lammps.first_read")
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

    with patch("revelsMD.trajectories.lammps.get_a_frame", side_effect=mock_get_a_frame):
        with patch("revelsMD.trajectories.lammps.define_strngdex", return_value=[2, 3, 4, 5, 6, 7]):
            with patch("revelsMD.trajectories.lammps.frame_skip"):
                with patch("builtins.open", MagicMock()):
                    state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=300.0)

                    # Random access in any order
                    for i in [7, 2, 9, 0, 5]:
                        pos, frc = state.get_frame(i)
                        np.testing.assert_array_equal(pos, positions[i])
                        np.testing.assert_array_equal(frc, forces[i])


# -----------------------------------------------------------------------------
# get_charges / get_masses - NumpyTrajectory
# -----------------------------------------------------------------------------
def test_numpy_get_charges_returns_correct_values():
    """get_charges should return charges for atoms of the specified species."""
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]
    charges = np.array([0.5, -0.25, -0.25])
    masses = np.array([16.0, 1.0, 1.0])

    state = NumpyTrajectory(
        positions, forces, 10, 10, 10, species,
        temperature=300.0, charge_list=charges, mass_list=masses
    )

    np.testing.assert_array_equal(state.get_charges("O"), [0.5])
    np.testing.assert_array_equal(state.get_charges("H"), [-0.25, -0.25])


def test_numpy_get_masses_returns_correct_values():
    """get_masses should return masses for atoms of the specified species."""
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]
    charges = np.array([0.5, -0.25, -0.25])
    masses = np.array([16.0, 1.0, 1.0])

    state = NumpyTrajectory(
        positions, forces, 10, 10, 10, species,
        temperature=300.0, charge_list=charges, mass_list=masses
    )

    np.testing.assert_array_equal(state.get_masses("O"), [16.0])
    np.testing.assert_array_equal(state.get_masses("H"), [1.0, 1.0])


def test_numpy_get_charges_raises_without_charge_data():
    """get_charges should raise DataUnavailableError when charge data is not available."""
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    with pytest.raises(DataUnavailableError, match="Charge data not available"):
        state.get_charges("O")


def test_numpy_get_masses_raises_without_mass_data():
    """get_masses should raise DataUnavailableError when mass data is not available."""
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    with pytest.raises(DataUnavailableError, match="Mass data not available"):
        state.get_masses("O")


# -----------------------------------------------------------------------------
# get_charges / get_masses - VaspTrajectory
# -----------------------------------------------------------------------------
@patch("revelsMD.trajectories.vasp.Vasprun")
def test_vasp_get_charges_raises_error(mock_vasprun):
    """get_charges should raise DataUnavailableError for VASP trajectories."""
    mock_instance = mock_vasprun.return_value
    mock_instance.structures = [MagicMock()]
    mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
    mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_instance.start = mock_instance.structures[0]
    mock_instance.start.indices_from_symbol.return_value = np.array([0])
    mock_instance.cart_coords = np.zeros((1, 1, 3))
    mock_instance.forces = np.zeros((1, 1, 3))

    state = VaspTrajectory("vasprun.xml", temperature=300.0)

    with pytest.raises(DataUnavailableError, match="Charge data not available"):
        state.get_charges("H")


@patch("revelsMD.trajectories.vasp.Vasprun")
def test_vasp_get_masses_raises_error(mock_vasprun):
    """get_masses should raise DataUnavailableError for VASP trajectories."""
    mock_instance = mock_vasprun.return_value
    mock_instance.structures = [MagicMock()]
    mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
    mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
    mock_instance.start = mock_instance.structures[0]
    mock_instance.start.indices_from_symbol.return_value = np.array([0])
    mock_instance.cart_coords = np.zeros((1, 1, 3))
    mock_instance.forces = np.zeros((1, 1, 3))

    state = VaspTrajectory("vasprun.xml", temperature=300.0)

    with pytest.raises(DataUnavailableError, match="Mass data not available"):
        state.get_masses("H")


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

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

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

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

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

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

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

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

        frames = list(state.iter_frames(stop=None))
        assert len(frames) == n_frames

    @patch("revelsMD.trajectories.vasp.Vasprun")
    def test_vasp_negative_stop(self, mock_vasprun):
        """VaspTrajectory should handle negative stop index."""
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

        state = VaspTrajectory("vasprun.xml", temperature=300.0)

        # stop=-1 means all but last -> frames 0, 1, 2, 3
        frames = list(state.iter_frames(stop=-1))
        assert len(frames) == 4

    @patch("revelsMD.trajectories.mda.MD.Universe")
    def test_mda_negative_stop(self, mock_universe):
        """MDATrajectory should handle negative stop index."""
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

        state = MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)

        # stop=-1 normalized to 4, so frames 0, 1, 2, 3
        frames = list(state.iter_frames(stop=-1))
        assert len(frames) == 4


# -----------------------------------------------------------------------------
# compute_beta - Unit conversion function
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Trajectory temperature and beta attributes
# -----------------------------------------------------------------------------
class TestTrajectoryBetaAttribute:
    """Tests for temperature and beta attributes on trajectory classes."""

    def test_numpy_trajectory_requires_temperature(self):
        """NumpyTrajectory should require temperature as a keyword argument."""
        positions = np.zeros((5, 3, 3))
        forces = np.ones((5, 3, 3))
        species = ["O", "H", "H"]

        with pytest.raises(TypeError):
            NumpyTrajectory(positions, forces, 10, 10, 10, species)

    def test_numpy_trajectory_stores_temperature(self):
        """NumpyTrajectory should store the temperature attribute."""
        positions = np.zeros((5, 3, 3))
        forces = np.ones((5, 3, 3))
        species = ["O", "H", "H"]

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)
        assert state.temperature == 300.0

    def test_numpy_trajectory_computes_beta(self):
        """NumpyTrajectory should compute beta from temperature and units."""
        positions = np.zeros((5, 3, 3))
        forces = np.ones((5, 3, 3))
        species = ["O", "H", "H"]

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0, units='real')
        expected_beta = compute_beta('real', 300.0)
        assert pytest.approx(state.beta, rel=1e-12) == expected_beta

    def test_numpy_trajectory_beta_with_lj_units(self):
        """In LJ units at T=1, beta should be 1.0."""
        positions = np.zeros((5, 3, 3))
        forces = np.ones((5, 3, 3))
        species = ["O", "H", "H"]

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=1.0, units='lj')
        assert state.beta == 1.0

    @patch("revelsMD.trajectories.mda.MD.Universe")
    def test_mda_trajectory_requires_temperature(self, mock_universe, mock_mdanalysis_universe):
        """MDATrajectory should require temperature as a keyword argument."""
        mock_universe.return_value = mock_mdanalysis_universe

        with pytest.raises(TypeError):
            MDATrajectory("traj.xtc", "topol.pdb")

    @patch("revelsMD.trajectories.mda.MD.Universe")
    def test_mda_trajectory_stores_temperature_and_beta(self, mock_universe, mock_mdanalysis_universe):
        """MDATrajectory should store temperature and compute beta."""
        mock_universe.return_value = mock_mdanalysis_universe

        state = MDATrajectory("traj.xtc", "topol.pdb", temperature=300.0)
        assert state.temperature == 300.0
        expected_beta = compute_beta('mda', 300.0)
        assert pytest.approx(state.beta, rel=1e-12) == expected_beta

    @patch("revelsMD.trajectories.vasp.Vasprun")
    def test_vasp_trajectory_requires_temperature(self, mock_vasprun):
        """VaspTrajectory should require temperature as a keyword argument."""
        mock_instance = mock_vasprun.return_value
        mock_instance.structures = [MagicMock()]
        mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
        mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
        mock_instance.start = mock_instance.structures[0]
        mock_instance.cart_coords = np.zeros((1, 1, 3))
        mock_instance.forces = np.zeros((1, 1, 3))

        with pytest.raises(TypeError):
            VaspTrajectory("vasprun.xml")

    @patch("revelsMD.trajectories.vasp.Vasprun")
    def test_vasp_trajectory_stores_temperature_and_beta(self, mock_vasprun):
        """VaspTrajectory should store temperature and compute beta in metal units."""
        mock_instance = mock_vasprun.return_value
        mock_instance.structures = [MagicMock()]
        mock_instance.structures[0].lattice.matrix = np.eye(3) * 10.0
        mock_instance.structures[0].lattice.angles = [90.0, 90.0, 90.0]
        mock_instance.start = mock_instance.structures[0]
        mock_instance.start.indices_from_symbol.return_value = np.array([0])
        mock_instance.cart_coords = np.zeros((1, 1, 3))
        mock_instance.forces = np.zeros((1, 1, 3))

        state = VaspTrajectory("vasprun.xml", temperature=500.0)
        assert state.temperature == 500.0
        expected_beta = compute_beta('metal', 500.0)
        assert pytest.approx(state.beta, rel=1e-12) == expected_beta

    @patch("revelsMD.trajectories.mda.MD.Universe")
    @patch("revelsMD.trajectories.lammps.first_read", return_value=(10, 5, ["id", "x", "y", "z"], 9, np.zeros((3, 2))))
    def test_lammps_trajectory_requires_temperature(self, mock_first_read, mock_universe, mock_mdanalysis_universe):
        """LammpsTrajectory should require temperature as a keyword argument."""
        mock_universe.return_value = mock_mdanalysis_universe

        with pytest.raises(TypeError):
            LammpsTrajectory("dump.lammpstrj", "data.lmp")

    @patch("revelsMD.trajectories.mda.MD.Universe")
    @patch("revelsMD.trajectories.lammps.first_read", return_value=(10, 5, ["id", "x", "y", "z"], 9, np.zeros((3, 2))))
    def test_lammps_trajectory_stores_temperature_and_beta(self, mock_first_read, mock_universe, mock_mdanalysis_universe):
        """LammpsTrajectory should store temperature and compute beta."""
        mock_universe.return_value = mock_mdanalysis_universe

        state = LammpsTrajectory("dump.lammpstrj", "data.lmp", temperature=350.0, units='real')
        assert state.temperature == 350.0
        expected_beta = compute_beta('real', 350.0)
        assert pytest.approx(state.beta, rel=1e-12) == expected_beta


class TestComputeBeta:
    """Tests for the compute_beta() function."""

    def test_compute_beta_lj_at_unit_temperature(self):
        """In LJ units at T=1, beta should be 1.0."""
        assert compute_beta('lj', 1.0) == 1.0

    def test_compute_beta_lj_at_higher_temperature(self):
        """In LJ units, beta = 1/T."""
        assert compute_beta('lj', 2.0) == 0.5
        assert compute_beta('lj', 0.5) == 2.0

    def test_compute_beta_real_units(self):
        """Verify beta in 'real' units (LAMMPS kcal/mol)."""
        # kB in real units = R / (calorie * 1000) ≈ 0.001987 kcal/mol/K
        k_real = constants.physical_constants['molar gas constant'][0] / constants.calorie / 1000
        temperature = 300.0
        expected = 1.0 / (k_real * temperature)
        assert pytest.approx(compute_beta('real', temperature), rel=1e-12) == expected

    def test_compute_beta_metal_units(self):
        """Verify beta in 'metal' units (LAMMPS eV)."""
        # kB in metal units ≈ 8.617e-5 eV/K
        k_metal = constants.physical_constants['Boltzmann constant in eV/K'][0]
        temperature = 300.0
        expected = 1.0 / (k_metal * temperature)
        assert pytest.approx(compute_beta('metal', temperature), rel=1e-12) == expected

    def test_compute_beta_mda_units(self):
        """Verify beta in 'mda' units (MDAnalysis kJ/mol)."""
        # kB in mda units = R / 1000 ≈ 0.008314 kJ/mol/K
        k_mda = constants.physical_constants['molar gas constant'][0] / 1000
        temperature = 300.0
        expected = 1.0 / (k_mda * temperature)
        assert pytest.approx(compute_beta('mda', temperature), rel=1e-12) == expected

    def test_compute_beta_case_insensitive(self):
        """Unit system should be case-insensitive."""
        assert compute_beta('LJ', 1.0) == compute_beta('lj', 1.0)
        assert compute_beta('REAL', 300.0) == compute_beta('real', 300.0)
        assert compute_beta('Metal', 300.0) == compute_beta('metal', 300.0)
        assert compute_beta('MDA', 300.0) == compute_beta('mda', 300.0)

    def test_compute_beta_strips_whitespace(self):
        """Unit system should ignore leading/trailing whitespace."""
        assert compute_beta('  lj  ', 1.0) == compute_beta('lj', 1.0)

    def test_compute_beta_invalid_unit_raises(self):
        """Unsupported unit system should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported unit system"):
            compute_beta('quantum-donut', 300.0)

    def test_compute_beta_zero_temperature_raises(self):
        """Zero temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            compute_beta('real', 0.0)

    def test_compute_beta_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            compute_beta('real', -100.0)

    def test_compute_beta_infinite_temperature_raises(self):
        """Infinite temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature must be finite"):
            compute_beta('real', float('inf'))

    def test_compute_beta_nan_temperature_raises(self):
        """NaN temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature must be finite"):
            compute_beta('real', float('nan'))


# -----------------------------------------------------------------------------
# Cell matrix validation
# -----------------------------------------------------------------------------
class TestValidateCellMatrix:
    """Tests for Trajectory._validate_cell_matrix."""

    def test_valid_orthorhombic_cell(self):
        cell = np.diag([10.0, 8.0, 6.0])
        Trajectory._validate_cell_matrix(cell)  # should not raise

    def test_valid_triclinic_cell(self):
        cell = np.array([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 8.0]])
        Trajectory._validate_cell_matrix(cell)  # should not raise

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Trajectory._validate_cell_matrix(np.eye(4))

    def test_non_finite_raises(self):
        cell = np.diag([10.0, np.inf, 6.0])
        with pytest.raises(ValueError, match="finite"):
            Trajectory._validate_cell_matrix(cell)

    def test_nan_raises(self):
        cell = np.diag([10.0, np.nan, 6.0])
        with pytest.raises(ValueError, match="finite"):
            Trajectory._validate_cell_matrix(cell)

    def test_zero_volume_raises(self):
        cell = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="volume"):
            Trajectory._validate_cell_matrix(cell)
