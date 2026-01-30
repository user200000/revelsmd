import pytest
import numpy as np
from pathlib import Path
from revelsMD.revels_3D import Revels3D
from revelsMD.density import SelectionState, HelperFunctions, Estimators, GridState
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
# HelperFunctions: Box & Triangular kernels
# ---------------------------

@pytest.mark.parametrize("kernel", ["box", "triangular"])
def test_helper_process_frame_kernels(ts, kernel):
    gs = GridState(ts, "number", nbins=4)
    pos = np.array([[1.0, 2.0, 3.0]])
    frc = np.array([[0.5, 0.0, 0.0]])
    HelperFunctions.process_frame(ts, gs, pos, frc, a=1.0, kernel=kernel)
    assert np.any(gs.forceX != 0)
    assert np.any(gs.counter != 0)


# ---------------------------
# SelectionState
# ---------------------------

def test_selectionstate_single(ts):
    ss = SelectionState(ts, "H", centre_location=True)
    assert ss.indistinguishable_set
    assert isinstance(ss.indices, np.ndarray)
    assert np.all(ss.charges == 0.1)
    assert np.all(ss.masses == 1.0)


def test_selectionstate_rigid(ts):
    ss = SelectionState(ts, ["H", "O"], centre_location=True)
    assert not ss.indistinguishable_set
    assert isinstance(ss.indices, list)
    assert len(ss.indices) == 2
    assert len(ss.masses) == 2


def test_selectionstate_badcentre(ts):
    with pytest.raises(ValueError):
        SelectionState(ts, ["H", "O"], centre_location="invalid")


def test_position_centre_valid(ts):
    ss = SelectionState(ts, ["H", "O"], centre_location=True)
    ss.position_centre(1)
    assert ss.species_number == 1


def test_position_centre_out_of_range(ts):
    ss = SelectionState(ts, ["H", "O"], centre_location=True)
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
    ss = SelectionState(ts_water, ["Ow", "Hw1", "Hw2"], centre_location=True, rigid=True)
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

        def get_indices(self, atype):
            return {"A": np.array([0]), "B": np.array([1]), "C": np.array([2])}[atype]

        def get_charges(self, atype):
            return {"A": np.array([-1.0]), "B": np.array([0.5]), "C": np.array([0.5])}[atype]

        def get_masses(self, atype):
            return {"A": np.array([1.0]), "B": np.array([1.0]), "C": np.array([1.0])}[atype]

    ts = LinearMoleculeMock()
    positions = np.array([[0, 5, 5], [1, 5, 5], [2, 5, 5]], dtype=float)

    ss = SelectionState(ts, ["A", "B", "C"], centre_location=True, rigid=True)

    coms, dipoles = HelperFunctions.find_coms(positions, ts, None, ss, calc_dipoles=True)

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

    HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 1.0), f"Weights should sum to 1, got {gs.counter.sum()}"


def test_triangular_arbitrary_position_weights():
    """Verify trilinear weights match analytical formula at non-special position."""
    gs = GridStateMock(nbins=4, box=10.0)
    lx = 2.5  # voxel size

    # Position (3.0, 4.0, 8.0)
    homeX = np.array([3.0])
    homeY = np.array([4.0])
    homeZ = np.array([8.0])

    x = np.digitize(homeX, gs.binsx)  # returns 2
    y = np.digitize(homeY, gs.binsy)  # returns 2
    z = np.digitize(homeZ, gs.binsz)  # returns 4

    # Code computes: frac = 1 + (home - x*lx) / lx
    # fracx = 1 + (3.0 - 2*2.5) / 2.5 = 1 + (-2.0)/2.5 = 0.2
    # fracy = 1 + (4.0 - 2*2.5) / 2.5 = 1 + (-1.0)/2.5 = 0.6
    # fracz = 1 + (8.0 - 4*2.5) / 2.5 = 1 + (-2.0)/2.5 = 0.2
    fracx, fracy, fracz = 0.2, 0.6, 0.2

    # Expected weights from trilinear formula
    expected = {
        (0, 0, 0): (1 - fracx) * (1 - fracy) * (1 - fracz),  # f_000
        (0, 0, 1): (1 - fracx) * (1 - fracy) * fracz,        # f_001
        (0, 1, 0): (1 - fracx) * fracy * (1 - fracz),        # f_010
        (1, 0, 0): fracx * (1 - fracy) * (1 - fracz),        # f_100
        (1, 0, 1): fracx * (1 - fracy) * fracz,              # f_101
        (0, 1, 1): (1 - fracx) * fracy * fracz,              # f_011
        (1, 1, 0): fracx * fracy * (1 - fracz),              # f_110
        (1, 1, 1): fracx * fracy * fracz,                    # f_111
    }

    HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([0.0]), foy=np.array([0.0]), foz=np.array([0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 1.0)

    # Voxel indices: gx = ((x-1) % 4, x % 4) = (1, 2), gy = (1, 2), gz = (3, 0)
    # Note: z wraps because digitize returns 4 for position 8.0
    gx = (1, 2)
    gy = (1, 2)
    gz = (3, 0)

    for (dx, dy, dz), expected_weight in expected.items():
        actual = gs.counter[gx[dx], gy[dy], gz[dz]]
        assert np.isclose(actual, expected_weight), \
            f"Weight at offset ({dx},{dy},{dz}) should be {expected_weight:.4f}, got {actual:.4f}"


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

    HelperFunctions.triangular_allocation(
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

    HelperFunctions.triangular_allocation(
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

    HelperFunctions.triangular_allocation(
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

    HelperFunctions.triangular_allocation(
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

    HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([1.0, 1.0, 1.0]),
        foy=np.array([0.0, 0.0, 0.0]),
        foz=np.array([0.0, 0.0, 0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 3.0), f"Total count should be 3, got {gs.counter.sum()}"
    assert np.isclose(gs.forceX.sum(), 3.0), f"Total forceX should be 3, got {gs.forceX.sum()}"


def test_triangular_overlapping_particles():
    """Two particles at same position should accumulate, not overwrite.

    This test verifies that the grid allocation correctly handles overlapping
    particles. The fix uses np.add.at() (NumPy) or explicit loops (Numba)
    instead of fancy indexing with +=.
    """
    gs = GridStateMock(nbins=4, box=10.0)

    # Two particles at IDENTICAL positions
    homeX = np.array([5.0, 5.0])
    homeY = np.array([5.0, 5.0])
    homeZ = np.array([5.0, 5.0])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    # Different forces: particle 1 has [1,0,0], particle 2 has [2,0,0]
    HelperFunctions.triangular_allocation(
        gs, x, y, z, homeX, homeY, homeZ,
        fox=np.array([1.0, 2.0]),
        foy=np.array([0.0, 0.0]),
        foz=np.array([0.0, 0.0]),
        a=1.0
    )

    # Expected: total forceX = 1 + 2 = 3, counter = 2
    # Bug: only the last particle's contribution is kept (forceX = 2, counter = 1)
    assert np.isclose(gs.counter.sum(), 2.0), f"Total count should be 2, got {gs.counter.sum()}"
    assert np.isclose(gs.forceX.sum(), 3.0), f"Total forceX should be 3 (1+2), got {gs.forceX.sum()}"


# ---------------------------
# sum_forces tests
# ---------------------------

def test_sum_forces_known_value():
    """Sum forces should add force vectors across rigid body components."""
    # 2 molecules, each with 3 atoms (A, B, C)
    # Molecule 0: atoms 0, 2, 4
    # Molecule 1: atoms 1, 3, 5
    class SSMock:
        indices = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    forces = np.array([
        [1.0, 0.0, 0.0],  # atom 0 (mol 0, species A)
        [0.0, 1.0, 0.0],  # atom 1 (mol 1, species A)
        [2.0, 0.0, 0.0],  # atom 2 (mol 0, species B)
        [0.0, 2.0, 0.0],  # atom 3 (mol 1, species B)
        [3.0, 0.0, 0.0],  # atom 4 (mol 0, species C)
        [0.0, 3.0, 0.0],  # atom 5 (mol 1, species C)
    ])

    result = HelperFunctions.sum_forces(SSMock(), forces)

    assert result.shape == (2, 3)
    # Molecule 0: [1,0,0] + [2,0,0] + [3,0,0] = [6,0,0]
    assert np.allclose(result[0], [6.0, 0.0, 0.0])
    # Molecule 1: [0,1,0] + [0,2,0] + [0,3,0] = [0,6,0]
    assert np.allclose(result[1], [0.0, 6.0, 0.0])


# ---------------------------
# find_coms (COM only) tests
# ---------------------------

def test_find_coms_equal_masses():
    """COM with equal masses should be geometric centre."""
    class TSMock:
        box_x = box_y = box_z = 20.0

    class SSMock:
        indices = [np.array([0]), np.array([1]), np.array([2])]
        masses = [np.array([1.0]), np.array([1.0]), np.array([1.0])]

    # 3 atoms in a line: x=0, x=3, x=6
    positions = np.array([[0, 5, 5], [3, 5, 5], [6, 5, 5]], dtype=float)

    coms = HelperFunctions.find_coms(positions, TSMock(), None, SSMock())

    assert coms.shape == (1, 3)
    # COM = (0 + 3 + 6) / 3 = 3.0
    assert np.isclose(coms[0, 0], 3.0)
    assert np.isclose(coms[0, 1], 5.0)
    assert np.isclose(coms[0, 2], 5.0)


def test_find_coms_unequal_masses():
    """COM with unequal masses should be mass-weighted."""
    class TSMock:
        box_x = box_y = box_z = 20.0

    class SSMock:
        indices = [np.array([0]), np.array([1])]
        masses = [np.array([1.0]), np.array([3.0])]

    # 2 atoms: light at x=0, heavy at x=4
    positions = np.array([[0, 5, 5], [4, 5, 5]], dtype=float)

    coms = HelperFunctions.find_coms(positions, TSMock(), None, SSMock())

    # COM = (1*0 + 3*4) / (1+3) = 12/4 = 3.0
    assert np.isclose(coms[0, 0], 3.0)


# ---------------------------
# box_allocation tests
# ---------------------------

def test_box_allocation_single_particle():
    """Box kernel deposits entirely to one voxel."""
    gs = GridStateMock(nbins=4, box=10.0)

    homeX = np.array([3.7])
    homeY = np.array([6.2])
    homeZ = np.array([1.8])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    HelperFunctions.box_allocation(
        gs, x, y, z,
        fox=np.array([1.5]), foy=np.array([-0.5]), foz=np.array([2.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 1.0)
    assert np.count_nonzero(gs.counter) == 1, "Should deposit to exactly 1 voxel"
    assert np.isclose(gs.forceX.sum(), 1.5)
    assert np.isclose(gs.forceY.sum(), -0.5)
    assert np.isclose(gs.forceZ.sum(), 2.0)


def test_box_allocation_multiple_particles():
    """Box kernel accumulates correctly for multiple particles."""
    gs = GridStateMock(nbins=4, box=10.0)

    homeX = np.array([1.0, 5.0, 8.0])
    homeY = np.array([1.0, 5.0, 8.0])
    homeZ = np.array([1.0, 5.0, 8.0])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    HelperFunctions.box_allocation(
        gs, x, y, z,
        fox=np.array([1.0, 2.0, 3.0]),
        foy=np.array([0.0, 0.0, 0.0]),
        foz=np.array([0.0, 0.0, 0.0]),
        a=1.0
    )

    assert np.isclose(gs.counter.sum(), 3.0)
    assert np.isclose(gs.forceX.sum(), 6.0)  # 1 + 2 + 3


def test_box_allocation_overlapping_particles():
    """Two particles in same voxel should accumulate, not overwrite.

    This test verifies that the box allocation correctly handles overlapping
    particles. The fix uses np.add.at() (NumPy) or explicit loops (Numba)
    instead of fancy indexing with +=.
    """
    gs = GridStateMock(nbins=10, box=10.0)

    # Two particles in the SAME voxel
    homeX = np.array([5.0, 5.0])
    homeY = np.array([5.0, 5.0])
    homeZ = np.array([5.0, 5.0])

    x = np.digitize(homeX, gs.binsx)
    y = np.digitize(homeY, gs.binsy)
    z = np.digitize(homeZ, gs.binsz)

    HelperFunctions.box_allocation(
        gs, x, y, z,
        fox=np.array([1.0, 2.0]),
        foy=np.array([0.0, 0.0]),
        foz=np.array([0.0, 0.0]),
        a=1.0
    )

    # Expected: counter = 2, forceX = 3 (not just the last value)
    assert np.isclose(gs.counter.sum(), 2.0), f"Total count should be 2, got {gs.counter.sum()}"
    assert np.isclose(gs.forceX.sum(), 3.0), f"Total forceX should be 3 (1+2), got {gs.forceX.sum()}"


# ---------------------------
# Estimator selection tests
# ---------------------------

class TestEstimatorSelection:
    """
    Test that make_force_grid selects the correct estimator function
    based on density_type, rigid, and centre_location parameters.
    """

    @pytest.fixture
    def ts_single_species(self):
        """Mock with single species (indistinguishable_set=True)."""
        ts = TSMock()
        ts._ids = {"H": np.array([0, 1])}
        ts._charges = {"H": np.array([0.1, 0.1])}
        ts._masses = {"H": np.array([1.0, 1.0])}
        return ts

    @pytest.fixture
    def ts_multi_species(self):
        """Mock with multiple species (indistinguishable_set=False).

        One H atom and one O atom per molecule (1 molecule total).
        Rigid molecule validation requires equal atom counts per species.
        """
        ts = TSMock()
        # One H atom (index 0) and one O atom (index 1)
        # Use float arrays to avoid dtype casting issues in find_coms
        ts.positions = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ])
        ts.forces = np.array([
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.1, 0.1, 0.0], [0.0, 0.0, 0.1]],
        ])
        ts._ids = {"H": np.array([0]), "O": np.array([1])}
        ts._charges = {"H": np.array([0.1]), "O": np.array([-0.2])}
        ts._masses = {"H": np.array([1.0]), "O": np.array([16.0])}
        return ts

    # --- Number density tests ---

    def test_number_single_species(self, ts_single_species):
        """Single species number density uses single_frame_number_single_grid."""
        gs = GridState(ts_single_species, "number", nbins=4)
        gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)
        assert gs.single_frame_function == Estimators.single_frame_number_single_grid

    def test_number_multi_species_not_rigid(self, ts_multi_species):
        """Multi-species, non-rigid number density uses single_frame_number_many_grid."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)
        assert gs.single_frame_function == Estimators.single_frame_number_many_grid

    def test_number_rigid_com(self, ts_multi_species):
        """Rigid number density at COM uses single_frame_rigid_number_com_grid."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True)
        assert gs.single_frame_function == Estimators.single_frame_rigid_number_com_grid

    def test_number_rigid_atom(self, ts_multi_species):
        """Rigid number density at specific atom uses single_frame_rigid_number_atom_grid."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0)
        assert gs.single_frame_function == Estimators.single_frame_rigid_number_atom_grid

    # --- Charge density tests ---

    def test_charge_single_species(self, ts_single_species):
        """Single species charge density uses single_frame_number_single_grid."""
        gs = GridState(ts_single_species, "charge", nbins=4)
        gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)
        # Note: single species charge uses number_single_grid (per current implementation)
        assert gs.single_frame_function == Estimators.single_frame_number_single_grid

    def test_charge_multi_species_not_rigid(self, ts_multi_species):
        """Multi-species, non-rigid charge density uses single_frame_charge_many_grid."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)
        assert gs.single_frame_function == Estimators.single_frame_charge_many_grid

    @pytest.mark.xfail(reason="Bug: SS.charges is list of arrays, not summed (see issue #11)")
    def test_charge_rigid_com(self, ts_multi_species):
        """Rigid charge density at COM uses single_frame_rigid_charge_com_grid."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True)
        assert gs.single_frame_function == Estimators.single_frame_rigid_charge_com_grid

    def test_charge_rigid_atom(self, ts_multi_species):
        """Rigid charge density at specific atom uses single_frame_rigid_charge_atom_grid."""
        gs = GridState(ts_multi_species, "charge", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0)
        assert gs.single_frame_function == Estimators.single_frame_rigid_charge_atom_grid

    # --- Polarisation density tests ---

    def test_polarisation_rigid_com(self, ts_multi_species):
        """Rigid polarisation density at COM uses single_frame_rigid_polarisation_com_grid."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=True, polarisation_axis=0)
        assert gs.single_frame_function == Estimators.single_frame_rigid_polarisation_com_grid
        assert gs.selection_state.polarisation_axis == 0

    def test_polarisation_rigid_atom(self, ts_multi_species):
        """Rigid polarisation density at specific atom uses single_frame_rigid_polarisation_atom_grid."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location=0, polarisation_axis=1)
        assert gs.single_frame_function == Estimators.single_frame_rigid_polarisation_atom_grid
        assert gs.selection_state.polarisation_axis == 1

    # --- Error cases ---

    def test_polarisation_not_rigid_raises(self, ts_multi_species):
        """Polarisation without rigid=True raises ValueError."""
        gs = GridState(ts_multi_species, "polarisation", nbins=4)
        with pytest.raises(ValueError, match="rigid molecules"):
            gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=False)

    def test_polarisation_single_species_raises(self, ts_single_species):
        """Polarisation with single species raises ValueError."""
        gs = GridState(ts_single_species, "polarisation", nbins=4)
        with pytest.raises(ValueError, match="single atom"):
            gs.make_force_grid(ts_single_species, atom_names="H", rigid=True, centre_location=True)

    def test_invalid_density_type_raises(self, ts_single_species):
        """Invalid density type raises ValueError."""
        gs = GridState(ts_single_species, "number", nbins=4)
        gs.density_type = "invalid"
        with pytest.raises(ValueError, match="Supported densities"):
            gs.make_force_grid(ts_single_species, atom_names="H", rigid=False)

    def test_rigid_invalid_centre_location_raises(self, ts_multi_species):
        """Rigid with invalid centre_location raises ValueError."""
        gs = GridState(ts_multi_species, "number", nbins=4)
        with pytest.raises(ValueError, match="centre_location"):
            gs.make_force_grid(ts_multi_species, atom_names=["H", "O"], rigid=True, centre_location="invalid")
