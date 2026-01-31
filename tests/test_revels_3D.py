import pytest
import numpy as np
from pathlib import Path
from revelsMD.revels_3D import Revels3D
from revelsMD.density import Selection, GridState
from ase import Atoms


class TSMock:
    """Minimal trajectory-state mock with required attributes for testing."""
    def __init__(self, temperature: float = 300.0, units: str = "real"):
        self.box_x = 10.0
        self.box_y = 10.0
        self.box_z = 10.0
        self.units = units
        self.temperature = temperature
        self.frames = 2

        # Compute beta from temperature and units
        from revelsMD.trajectories._base import compute_beta
        self.beta = compute_beta(units, temperature)

        # Two atoms, 2 frames
        self.positions = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[2, 3, 4], [5, 6, 7]],
        ])
        self.forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.1, 0.1, 0.0], [0.0, 0.0, 0.1]],
        ])

        self.species = ["H", "O"]
        self._ids = {"H": np.array([0]), "O": np.array([1])}
        self._charges = {"H": np.array([0.1]), "O": np.array([-0.1])}
        self._masses = {"H": np.array([1.0]), "O": np.array([16.0])}

    def get_indices(self, atype):
        return self._ids[atype]

    def get_charges(self, atype):
        return self._charges[atype]

    def get_masses(self, atype):
        return self._masses[atype]

    def iter_frames(self, start=0, stop=None, stride=1):
        if stop is None:
            stop = self.frames
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index):
        return self.positions[index], self.forces[index]


@pytest.fixture
def ts():
    """Fixture providing a basic test trajectory."""
    return TSMock()


# ---------------------------
# GridState Initialization
# ---------------------------

def test_gridstate_initialization(ts):
    gs = GridState(ts, density_type="number", nbins=4)
    assert gs.nbinsx == 4
    assert gs.lx == pytest.approx(ts.box_x / 4)
    assert gs.voxel_volume > 0
    assert np.all(gs.forceX == 0)
    assert gs.grid_progress == "Generated"


def test_gridstate_uses_trajectory_beta(ts):
    """GridState should use beta from the trajectory object."""
    gs = GridState(ts, density_type="number", nbins=4)
    assert gs.beta == ts.beta


def test_invalid_box(ts):
    ts.box_x = -10.0
    with pytest.raises(ValueError):
        GridState(ts, "number")


def test_invalid_bins(ts):
    with pytest.raises(ValueError):
        GridState(ts, "number", nbinsx=0, nbinsy=4, nbinsz=4)


# ---------------------------
# k-vectors and FFT utilities
# ---------------------------

def test_kvectors_ksquared_shapes(ts):
    gs = GridState(ts, "number", nbins=4)
    kx, ky, kz = gs.get_kvectors()
    assert kx.shape[0] == gs.nbinsx
    ks = gs.get_ksquared()
    assert ks.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)
    assert np.all(ks >= 0)


# ---------------------------
# GridState._process_frame: Box & Triangular kernels
# ---------------------------

@pytest.mark.parametrize("kernel", ["box", "triangular"])
def test_process_frame_kernels(ts, kernel):
    """_process_frame deposits positions/forces to grid using specified kernel."""
    gs = GridState(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    gs._process_frame(pos, frc, weight=1.0, kernel=kernel)
    assert np.any(gs.forceX != 0)
    assert np.any(gs.counter != 0)


def test_process_frame_increments_count(ts):
    """_process_frame increments the frame count."""
    gs = GridState(ts, "number", nbins=4)
    assert gs.count == 0
    gs._process_frame(np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 0.0, 0.0]]))
    assert gs.count == 1
    gs._process_frame(np.array([[2.0, 3.0, 4.0]]), np.array([[0.0, 0.5, 0.0]]))
    assert gs.count == 2


def test_process_frame_invalid_kernel(ts):
    """_process_frame raises ValueError for unknown kernel."""
    gs = GridState(ts, "number", nbins=4)
    with pytest.raises(ValueError, match="Unsupported kernel"):
        gs._process_frame(np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 0.0, 0.0]]), kernel="invalid")


# ---------------------------
# GridState.deposit_to_grid
# ---------------------------

def test_deposit_to_grid_single_array(ts):
    """deposit_to_grid with single array deposits once."""
    gs = GridState(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    gs.deposit_to_grid(pos, frc, weights=1.0, kernel="triangular")

    assert gs.count == 1
    assert np.any(gs.counter != 0)
    assert np.any(gs.forceX != 0)


def test_deposit_to_grid_list_of_arrays(ts):
    """deposit_to_grid with list of arrays deposits each separately."""
    gs = GridState(ts, "number", nbins=4)
    pos_list = [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])]
    frc_list = [np.array([[0.5, 0.0, 0.0]]), np.array([[0.0, 0.5, 0.0]])]
    gs.deposit_to_grid(pos_list, frc_list, weights=1.0, kernel="box")

    assert gs.count == 2
    assert np.any(gs.counter != 0)


def test_deposit_to_grid_broadcasts_scalar_weight(ts):
    """deposit_to_grid broadcasts scalar weight to all position arrays."""
    gs = GridState(ts, "number", nbins=4)
    pos_list = [np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])]
    frc_list = [np.array([[0.5, 0.0, 0.0]]), np.array([[0.0, 0.5, 0.0]])]
    gs.deposit_to_grid(pos_list, frc_list, weights=1.0, kernel="triangular")

    # Both depositions should have been made
    assert gs.count == 2
    assert np.any(gs.counter != 0)


# ---------------------------
# Selection
# ---------------------------

def test_selectionstate_single(ts):
    ss = Selection(ts, "H", centre_location=True)
    assert ss.single_species
    assert isinstance(ss.indices, np.ndarray)


def test_selectionstate_single_with_charges(ts):
    ss = Selection(ts, "H", centre_location=True, density_type='charge')
    assert np.all(ss.charges == 0.1)


def test_selectionstate_single_with_polarisation(ts):
    ss = Selection(ts, "H", centre_location=True, density_type='polarisation')
    assert np.all(ss.charges == 0.1)
    assert np.all(ss.masses == 1.0)


def test_selectionstate_rigid(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True)
    assert not ss.single_species
    assert isinstance(ss.indices, list)
    assert len(ss.indices) == 2


def test_selectionstate_rigid_with_polarisation(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True, density_type='polarisation')
    assert len(ss.masses) == 2
    assert len(ss.charges) == 2


def test_selectionstate_badcentre(ts):
    with pytest.raises(ValueError):
        Selection(ts, ["H", "O"], centre_location="invalid")


def test_position_centre_valid(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True)
    ss.position_centre(1)
    assert ss.species_number == 1


def test_position_centre_out_of_range(ts):
    ss = Selection(ts, ["H", "O"], centre_location=True)
    with pytest.raises(ValueError):
        ss.position_centre(10)


def test_selectionstate_rigid_water():
    """Rigid molecules require unique labels for each atom in the molecule.

    Since topology data is not used, atoms are identified by unique labels
    (e.g. Ow, Hw1, Hw2 for water) rather than element symbols.
    """
    class WaterTSMock:
        box_x = box_y = box_z = 10.0
        units = "real"
        species = ["Ow", "Hw1", "Hw2"]
        # 2 water molecules: Ow, Hw1, Hw2 each = 2 atoms per species
        _ids = {"Ow": np.array([0, 3]), "Hw1": np.array([1, 4]), "Hw2": np.array([2, 5])}
        _charges = {"Ow": np.array([-0.8, -0.8]), "Hw1": np.array([0.4, 0.4]), "Hw2": np.array([0.4, 0.4])}
        _masses = {"Ow": np.array([16.0, 16.0]), "Hw1": np.array([1.0, 1.0]), "Hw2": np.array([1.0, 1.0])}

        def get_indices(self, atype):
            return self._ids[atype]

        def get_charges(self, atype):
            return self._charges[atype]

        def get_masses(self, atype):
            return self._masses[atype]

    ts_water = WaterTSMock()
    ss = Selection(ts_water, ["Ow", "Hw1", "Hw2"], centre_location=True, rigid=True)
    assert len(ss.indices) == 3
    assert len(ss.indices[0]) == 2  # 2 Ow atoms
    assert len(ss.indices[1]) == 2  # 2 Hw1 atoms
    assert len(ss.indices[2]) == 2  # 2 Hw2 atoms


# ---------------------------
# Full pipeline
# ---------------------------

def test_full_number_density_pipeline(tmp_path, ts):
    gs = GridState(ts, "number", nbins=4)
    gs.make_force_grid(ts, atom_names="H", rigid=False)
    assert gs.grid_progress == "Allocated"

    gs.get_real_density()
    assert hasattr(gs, "rho")
    assert gs.rho.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)

    cube_file = tmp_path / "density.cube"
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    gs.write_to_cube(atoms, gs.rho, cube_file)
    assert cube_file.exists()


def test_get_lambda_basic(ts):
    """Test basic get_lambda functionality."""
    gs = GridState(ts, "number", nbins=4)
    gs.make_force_grid(ts, atom_names="H", rigid=False)
    gs.get_real_density()
    gs2 = gs.get_lambda(ts, sections=1)
    assert gs2.grid_progress == "Lambda"
    assert hasattr(gs2, "optimal_density")
    assert gs2.optimal_density.shape == gs2.expected_rho.shape
