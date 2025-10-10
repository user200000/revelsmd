import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from revelsMD.trajectory_states import (
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

