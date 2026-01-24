import pytest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from revelsMD.revels_3D import Revels3D
from ase import Atoms


class TSMock:
    """Minimal trajectory-state mock with required attributes for testing."""
    def __init__(self):
        self.box_x = 10.0
        self.box_y = 10.0
        self.box_z = 10.0
        self.units = "real"
        self.frames = 2
        self.variety = "numpy"
        self.charge_and_mass = True

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


@pytest.fixture
def ts():
    """Fixture providing a basic test trajectory."""
    return TSMock()


# ---------------------------
# GridState Initialization
# ---------------------------

def test_gridstate_initialization(ts):
    gs = Revels3D.GridState(ts, density_type="number", temperature=300, nbins=4)
    assert gs.nbinsx == 4
    assert gs.lx == pytest.approx(ts.box_x / 4)
    assert gs.voxel_volume > 0
    assert np.all(gs.forceX == 0)
    assert gs.grid_progress == "Generated"


def test_invalid_box(ts):
    ts.box_x = -10.0
    with pytest.raises(ValueError):
        Revels3D.GridState(ts, "number", 300)


def test_invalid_bins(ts):
    with pytest.raises(ValueError):
        Revels3D.GridState(ts, "number", 300, nbinsx=0, nbinsy=4, nbinsz=4)


# ---------------------------
# k-vectors and FFT utilities
# ---------------------------

def test_kvectors_ksquared_shapes(ts):
    gs = Revels3D.GridState(ts, "number", 300, nbins=4)
    kx, ky, kz = gs.get_kvectors()
    assert kx.shape[0] == gs.nbinsx
    ks = gs.get_ksquared()
    assert ks.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)
    assert np.all(ks >= 0)


# ---------------------------
# HelperFunctions: Box & Triangular kernels
# ---------------------------

@pytest.mark.parametrize("kernel", ["box", "triangular"])
def test_helper_process_frame_kernels(ts, kernel):
    gs = Revels3D.GridState(ts, "number", 300, nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    Revels3D.HelperFunctions.process_frame(ts, gs, pos, frc, a=1.0, kernel=kernel)
    assert np.any(gs.forceX != 0)
    assert np.any(gs.counter != 0)


# ---------------------------
# SelectionState
# ---------------------------

def test_selectionstate_single(ts):
    ss = Revels3D.SelectionState(ts, "H", centre_location=True)
    assert ss.indistinguishable_set
    assert isinstance(ss.indices, np.ndarray)
    assert np.all(ss.charges == 0.1)
    assert np.all(ss.masses == 1.0)


def test_selectionstate_rigid(ts):
    ss = Revels3D.SelectionState(ts, ["H", "O"], centre_location=True)
    assert not ss.indistinguishable_set
    assert isinstance(ss.indices, list)
    assert len(ss.indices) == 2
    assert len(ss.masses) == 2


def test_selectionstate_badcentre(ts):
    with pytest.raises(ValueError):
        Revels3D.SelectionState(ts, ["H", "O"], centre_location="invalid")


def test_position_centre_valid(ts):
    ss = Revels3D.SelectionState(ts, ["H", "O"], centre_location=True)
    ss.position_centre(1)
    assert ss.species_number == 1


def test_position_centre_out_of_range(ts):
    ss = Revels3D.SelectionState(ts, ["H", "O"], centre_location=True)
    with pytest.raises(ValueError):
        ss.position_centre(10)


# ---------------------------
# Full pipeline
# ---------------------------

def test_full_number_density_pipeline(tmp_path, ts):
    gs = Revels3D.GridState(ts, "number", temperature=300, nbins=4)
    gs.make_force_grid(ts, atom_names="H", rigid=False)
    assert gs.grid_progress == "Allocated"

    gs.get_real_density()
    assert hasattr(gs, "rho")
    assert gs.rho.shape == (gs.nbinsx, gs.nbinsy, gs.nbinsz)

    cube_file = tmp_path / "density.cube"
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    gs.write_to_cube(atoms, gs.rho, cube_file, convert_pmg=False)
    assert cube_file.exists()


def test_get_lambda_basic(ts):
    gs = Revels3D.GridState(ts, "number", 300, nbins=4)
    gs.make_force_grid(ts, atom_names="H", rigid=False)
    gs.get_real_density()
    gs2 = gs.get_lambda(ts, sections=1)
    assert gs2.grid_progress == "Lambda"
    assert hasattr(gs2, "optimal_density")
    assert gs2.optimal_density.shape == gs2.expected_rho.shape


# ---------------------------
# find_coms with calc_dipoles
# ---------------------------

def test_find_coms_dipole_known_value():
    """
    Test find_coms dipole calculation against a known analytical result.

    Uses a simple 3-atom linear molecule with equal masses:
      A (-1.0 charge) at x=0, B (+0.5) at x=1, C (+0.5) at x=2

    COM = (0 + 1 + 2) / 3 = 1.0  (equal masses)

    Dipole = sum_i(q_i * (r_i - COM))
           = -1.0*(0-1) + 0.5*(1-1) + 0.5*(2-1)
           = 1.0 + 0 + 0.5 = 1.5 in x-direction
    """
    class LinearMoleculeMock:
        def __init__(self):
            self.box_x = self.box_y = self.box_z = 20.0
            self.charge_and_mass = True

        def get_indices(self, atype):
            return {"A": np.array([0]), "B": np.array([1]), "C": np.array([2])}[atype]

        def get_charges(self, atype):
            return {"A": np.array([-1.0]), "B": np.array([0.5]), "C": np.array([0.5])}[atype]

        def get_masses(self, atype):
            return {"A": np.array([1.0]), "B": np.array([1.0]), "C": np.array([1.0])}[atype]

    ts = LinearMoleculeMock()
    positions = np.array([[0, 5, 5], [1, 5, 5], [2, 5, 5]], dtype=float)

    ss = Revels3D.SelectionState(ts, ["A", "B", "C"], centre_location=True, rigid=True)
    gs = SimpleNamespace(SS=ss)

    coms, dipoles = Revels3D.HelperFunctions.find_coms(positions, ts, gs, ss, calc_dipoles=True)

    assert coms.shape == (1, 3)
    assert dipoles.shape == (1, 3)
    assert np.isclose(coms[0, 0], 1.0), f"COM should be at x=1.0, got {coms[0, 0]}"
    assert np.isclose(dipoles[0, 0], 1.5), f"Dipole x-component should be 1.5, got {dipoles[0, 0]}"
    assert np.isclose(dipoles[0, 1], 0.0), f"Dipole y-component should be 0, got {dipoles[0, 1]}"
    assert np.isclose(dipoles[0, 2], 0.0), f"Dipole z-component should be 0, got {dipoles[0, 2]}"


# ---------------------------
# triangular_allocation tests
# ---------------------------

class GridStateMock:
    """Minimal GridState mock for testing triangular_allocation."""
    def __init__(self, nbins=4, box=10.0):
        self.nbinsx = self.nbinsy = self.nbinsz = nbins
        self.lx = self.ly = self.lz = box / nbins
        self.binsx = np.linspace(0, box, nbins + 1)
        self.binsy = np.linspace(0, box, nbins + 1)
        self.binsz = np.linspace(0, box, nbins + 1)
        self.forceX = np.zeros((nbins, nbins, nbins))
        self.forceY = np.zeros((nbins, nbins, nbins))
        self.forceZ = np.zeros((nbins, nbins, nbins))
        self.counter = np.zeros((nbins, nbins, nbins))


def test_triangular_weights_sum_to_one():
    """Trilinear weights from all 8 voxels should sum to 1 for any position."""
    gs = GridStateMock(nbins=4, box=10.0)

    # Arbitrary position inside the box
    homeX = np.array([3.7])
    homeY = np.array([6.2])
    homeZ = np.array([1.8])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 1.0), f"Weights should sum to 1, got {gs.counter.sum()}"


def test_triangular_particle_at_voxel_centre():
    """Particle at voxel centre should distribute weight to surrounding vertices."""
    gs = GridStateMock(nbins=4, box=10.0)

    # Centre of voxel [1,1,1] is at (3.75, 3.75, 3.75) for box=10, nbins=4
    # Voxel edges are at 2.5 and 5.0
    homeX = np.array([3.75])
    homeY = np.array([3.75])
    homeZ = np.array([3.75])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    # At centre, frac = 0.5 in each dimension, so each of 8 corners gets 0.125
    assert np.isclose(gs.counter.sum(), 1.0)
    assert np.count_nonzero(gs.counter) == 8, "Should deposit to exactly 8 voxels"
    assert np.allclose(gs.counter[gs.counter > 0], 0.125), "Each corner should get 1/8"


def test_triangular_particle_at_corner():
    """Particle at voxel corner should go entirely to one voxel."""
    gs = GridStateMock(nbins=4, box=10.0)

    # Corner at (2.5, 2.5, 2.5) - edge between voxels
    homeX = np.array([2.5])
    homeY = np.array([2.5])
    homeZ = np.array([2.5])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    # At corner frac=0 or 1, weight goes to single voxel
    assert np.isclose(gs.counter.sum(), 1.0)
    assert np.isclose(gs.counter.max(), 1.0), "All weight should go to one voxel"


def test_triangular_periodic_boundary():
    """Particle near box edge should wrap indices correctly."""
    gs = GridStateMock(nbins=4, box=10.0)

    # Near upper boundary - should wrap to voxel 0
    homeX = np.array([9.5])
    homeY = np.array([5.0])
    homeZ = np.array([5.0])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    # Should still sum to 1 (no out-of-bounds)
    assert np.isclose(gs.counter.sum(), 1.0)
    # Should have deposits in both voxel 3 and voxel 0 (wrapped)
    assert gs.counter[3, :, :].sum() > 0, "Should deposit to voxel 3"
    assert gs.counter[0, :, :].sum() > 0, "Should deposit to wrapped voxel 0"


def test_triangular_force_direction_preserved():
    """Force vector components should deposit with correct sign and magnitude."""
    gs = GridStateMock(nbins=4, box=10.0)

    # At voxel corner so all weight goes to one voxel
    homeX = np.array([2.5])
    homeY = np.array([2.5])
    homeZ = np.array([2.5])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([1.5]), foy=np.array([-2.0]), foz=np.array([0.5]),
        a=1.0
    )

    assert np.isclose(gs.forceX.sum(), 1.5), f"forceX should be 1.5, got {gs.forceX.sum()}"
    assert np.isclose(gs.forceY.sum(), -2.0), f"forceY should be -2.0, got {gs.forceY.sum()}"
    assert np.isclose(gs.forceZ.sum(), 0.5), f"forceZ should be 0.5, got {gs.forceZ.sum()}"


def test_triangular_multiple_particles():
    """Total counter sum should equal number of particles times weight."""
    gs = GridStateMock(nbins=4, box=10.0)

    # 3 particles at different positions
    homeX = np.array([1.0, 5.0, 8.0])
    homeY = np.array([2.0, 5.0, 7.0])
    homeZ = np.array([3.0, 5.0, 6.0])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    Revels3D.HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([1.0, 1.0, 1.0]),
        foy=np.array([0.0, 0.0, 0.0]),
        foz=np.array([0.0, 0.0, 0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 3.0), f"Total count should be 3, got {gs.counter.sum()}"
    assert np.isclose(gs.forceX.sum(), 3.0), f"Total forceX should be 3, got {gs.forceX.sum()}"

