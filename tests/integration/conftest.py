"""Shared fixtures and configuration for integration tests."""

import pytest
import numpy as np
from pathlib import Path
from typing import Optional


# Path to the examples directory
EXAMPLES_DIR = Path(__file__).parents[2] / "examples"
REFERENCE_DATA_DIR = Path(__file__).parent.parent / "reference_data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "analytical: tests against known analytical results")
    config.addinivalue_line("markers", "regression: regression tests against stored reference data")
    config.addinivalue_line("markers", "requires_example1: requires Example 1 data (~12MB)")
    config.addinivalue_line("markers", "requires_example2: requires Example 2 data (~500MB)")
    config.addinivalue_line("markers", "requires_example4: requires Example 4 data (~1.7GB)")
    config.addinivalue_line("markers", "requires_vasp: requires VASP vasprun.xml data")


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def assert_arrays_close(actual, expected, rtol=1e-6, atol=1e-8, context=""):
    """Assert arrays are close with informative error messages."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        max_diff = np.max(np.abs(actual - expected))
        max_rel = np.max(np.abs((actual - expected) / (np.abs(expected) + 1e-15)))
        raise AssertionError(
            f"{context}\nMax absolute diff: {max_diff}\nMax relative diff: {max_rel}\n{e}"
        )


def load_reference_data(subdir: str, filename: str) -> Optional[dict]:
    """Load reference data from .npz file if it exists."""
    ref_path = REFERENCE_DATA_DIR / subdir / filename
    if not ref_path.exists():
        return None
    return dict(np.load(ref_path))


# ---------------------------------------------------------------------------
# Trajectory fixtures - LAMMPS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example1_trajectory():
    """
    Load Example 1 LJ trajectory for RDF tests.

    Example 1: Lennard-Jones fluid, ~12MB
    - 2880 atoms (2 types)
    - 50 frames
    - 14.227 cubic LJ box
    """
    from revelsMD.trajectories import LammpsTrajectory

    dump_file = EXAMPLES_DIR / "example_1_LJ" / "dump.nh.lammps"
    data_file = EXAMPLES_DIR / "example_1_LJ" / "data.fin.nh.data"

    if not dump_file.exists():
        pytest.skip("Example 1 data not available")

    return LammpsTrajectory(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
    )


@pytest.fixture(scope="module")
def example2_trajectory():
    """
    Load Example 2 LJ trajectory for 3D density tests.

    Example 2: Lennard-Jones 3D density, ~500MB
    - 2880 atoms
    - 2500 frames
    - Frozen central particle + solvating LJ spheres
    """
    from revelsMD.trajectories import LammpsTrajectory

    dump_file = EXAMPLES_DIR / "example_2_LJ_3D" / "dump.nh.lammps"
    data_file = EXAMPLES_DIR / "example_2_LJ_3D" / "data.fin.nh.data"

    if not dump_file.exists():
        pytest.skip("Example 2 data not available")

    return LammpsTrajectory(
        str(dump_file),
        str(data_file),
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
    )


# ---------------------------------------------------------------------------
# Trajectory fixtures - GROMACS/MDAnalysis
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example4_trajectory():
    """
    Load Example 4 rigid water trajectory (subset for faster tests).

    Uses a 100-frame subset (~23MB) rather than the full 8000-frame
    trajectory (~1.7GB) for reasonable test times.

    Example 4: SPC/E water
    - 6339 atoms (2113 water molecules)
    - 100 frames (subset)
    - GROMACS trr/tpr format
    """
    from revelsMD.trajectories import MDATrajectory

    # Use subset trajectory for faster tests
    TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "example_4_subset"
    trr_file = TEST_DATA_DIR / "prod_100frames.trr"
    tpr_file = TEST_DATA_DIR / "prod.tpr"

    # Fall back to full trajectory if subset not available
    if not trr_file.exists():
        trr_file = EXAMPLES_DIR / "example_4_rigid_water" / "prod.trr"
        tpr_file = EXAMPLES_DIR / "example_4_rigid_water" / "prod.tpr"

    if not trr_file.exists():
        pytest.skip("Example 4 data not available")

    return MDATrajectory(str(trr_file), str(tpr_file))


# ---------------------------------------------------------------------------
# Trajectory fixtures - VASP
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vasp_trajectory():
    """
    Load VASP trajectory from vasprun.xml (subset for faster tests).

    Uses a 50-frame subset (~2.5MB) from Example 3: BaSnF4 solid electrolyte
    - 324 atoms (Ba, Sn, F)
    - 50 frames (subset from 3001 total in r1)
    - Temperature: 600K
    """
    from revelsMD.trajectories import VaspTrajectory

    # Use subset trajectory for faster tests
    TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "example_3_vasp_subset"
    vasprun_file = TEST_DATA_DIR / "vasprun.xml"

    if not vasprun_file.exists():
        pytest.skip("VASP test data not available")

    return VaspTrajectory(str(vasprun_file))


# ---------------------------------------------------------------------------
# Trajectory fixtures - NumPy (synthetic)
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_gas_trajectory():
    """
    Generate trajectory with uniform random positions and random forces.

    The force-sampling method requires non-zero forces to work correctly.
    With random forces, this should produce g(r) ~ 1 for all r (within statistical noise).
    """
    from revelsMD.trajectories import NumpyTrajectory

    np.random.seed(42)
    n_atoms = 500
    n_frames = 10
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    # Use random forces - required for force-sampling method to work
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    return NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj'
    )


@pytest.fixture
def two_atom_trajectory():
    """
    Generate trajectory with two atoms at fixed separation.

    Atom 0 at origin, Atom 1 at (3.0, 0, 0).
    g(r) should show a peak at r = 3.0.
    """
    from revelsMD.trajectories import NumpyTrajectory

    n_frames = 5
    box = 10.0
    separation = 3.0

    positions = np.zeros((n_frames, 2, 3))
    positions[:, 0, :] = [box / 2, box / 2, box / 2]  # Centre of box
    positions[:, 1, :] = [box / 2 + separation, box / 2, box / 2]

    # Random forces for force-sampling method
    np.random.seed(43)
    forces = np.random.randn(n_frames, 2, 3) * 0.1
    species = ['1', '1']

    return NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj'
    )


@pytest.fixture
def single_atom_trajectory():
    """
    Generate trajectory with single atom at known position.

    Atom at (5.0, 5.0, 5.0) in a 10x10x10 box.
    3D density should show a peak at this location.
    """
    from revelsMD.trajectories import NumpyTrajectory

    n_frames = 5
    box = 10.0

    positions = np.zeros((n_frames, 1, 3))
    positions[:, 0, :] = [5.0, 5.0, 5.0]

    # Random forces for force-sampling method
    np.random.seed(44)
    forces = np.random.randn(n_frames, 1, 3) * 0.1
    species = ['1']

    return NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj'
    )


@pytest.fixture
def cubic_lattice_trajectory():
    """
    Generate simple cubic lattice trajectory.

    4x4x4 = 64 atoms on a simple cubic lattice with spacing 2.5
    in a 10x10x10 box. g(r) should show peaks at r = 2.5, 3.54 (sqrt(2)*2.5), etc.
    """
    from revelsMD.trajectories import NumpyTrajectory

    n_frames = 5
    box = 10.0
    spacing = 2.5
    n_per_dim = 4

    # Generate lattice positions
    lattice_positions = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                lattice_positions.append([i * spacing, j * spacing, k * spacing])

    lattice_positions = np.array(lattice_positions)
    n_atoms = len(lattice_positions)

    positions = np.zeros((n_frames, n_atoms, 3))
    for frame in range(n_frames):
        positions[frame] = lattice_positions

    # Random forces for force-sampling method
    np.random.seed(45)
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    return NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj'
    )


@pytest.fixture
def water_molecule_trajectory():
    """
    Generate trajectory with rigid water molecules for testing dipole/polarisation.

    Creates 10 water molecules with known geometry and charges.
    """
    from revelsMD.trajectories import NumpyTrajectory

    n_frames = 5
    n_molecules = 10
    box = 15.0

    # Water geometry (approximate SPC/E)
    # O at origin, H1 at (0.816, 0.577, 0), H2 at (-0.816, 0.577, 0)
    h_distance = 1.0  # O-H distance
    h_angle = 109.47 * np.pi / 180  # H-O-H angle

    # Build molecular positions
    n_atoms = n_molecules * 3  # O, H, H per molecule
    positions = np.zeros((n_frames, n_atoms, 3))
    forces = np.zeros((n_frames, n_atoms, 3))

    np.random.seed(42)
    mol_centres = np.random.uniform(2, box - 2, (n_molecules, 3))

    for mol_idx in range(n_molecules):
        o_idx = mol_idx * 3
        h1_idx = mol_idx * 3 + 1
        h2_idx = mol_idx * 3 + 2

        centre = mol_centres[mol_idx]

        for frame in range(n_frames):
            # O at centre
            positions[frame, o_idx] = centre
            # H atoms
            positions[frame, h1_idx] = centre + [h_distance * np.sin(h_angle / 2), h_distance * np.cos(h_angle / 2), 0]
            positions[frame, h2_idx] = centre + [-h_distance * np.sin(h_angle / 2), h_distance * np.cos(h_angle / 2), 0]

    species = ['O', 'H', 'H'] * n_molecules

    # SPC/E charges: O = -0.8476, H = +0.4238
    charges = np.array([-0.8476, 0.4238, 0.4238] * n_molecules)
    masses = np.array([15.999, 1.008, 1.008] * n_molecules)

    return NumpyTrajectory(
        positions, forces, box, box, box, species,
        units='real', charge_list=charges, mass_list=masses
    )


@pytest.fixture
def multispecies_trajectory():
    """
    Generate trajectory with multiple species for unlike-pair RDF testing.

    200 atoms of type '1', 100 atoms of type '2' in a 10x10x10 box.
    """
    from revelsMD.trajectories import NumpyTrajectory

    np.random.seed(42)
    n_type1 = 200
    n_type2 = 100
    n_atoms = n_type1 + n_type2
    n_frames = 5
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    # Random forces for force-sampling method
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_type1 + ['2'] * n_type2

    return NumpyTrajectory(
        positions, forces, box, box, box, species, units='lj'
    )


# ---------------------------------------------------------------------------
# Conversion helpers for cross-backend tests
# ---------------------------------------------------------------------------

def lammps_to_numpy(lammps_ts, start=0, stop=None, stride=1):
    """
    Convert a LammpsTrajectory to NumpyTrajectory.

    Parameters
    ----------
    lammps_ts : LammpsTrajectory
        Source trajectory
    start, stop, stride : int
        Frame selection parameters

    Returns
    -------
    NumpyTrajectory
        Equivalent trajectory with data loaded into numpy arrays
    """
    from revelsMD.trajectories import NumpyTrajectory

    universe = lammps_ts.mdanalysis_universe

    if stop is None or stop == -1:
        stop = len(universe.trajectory)

    frames_to_load = range(start, stop, stride)
    n_frames = len(frames_to_load)
    n_atoms = len(universe.atoms)

    positions = np.zeros((n_frames, n_atoms, 3))
    forces = np.zeros((n_frames, n_atoms, 3))

    for i, frame_idx in enumerate(frames_to_load):
        universe.trajectory[frame_idx]
        positions[i] = universe.atoms.positions
        forces[i] = universe.atoms.forces

    # Build species list from atom types
    species = [str(atom.type) for atom in universe.atoms]

    return NumpyTrajectory(
        positions, forces,
        lammps_ts.box_x, lammps_ts.box_y, lammps_ts.box_z,
        species, units=lammps_ts.units
    )


def mda_to_numpy(mda_ts, start=0, stop=None, stride=1):
    """
    Convert an MDATrajectory to NumpyTrajectory.

    Parameters
    ----------
    mda_ts : MDATrajectory
        Source trajectory
    start, stop, stride : int
        Frame selection parameters

    Returns
    -------
    NumpyTrajectory
        Equivalent trajectory with data loaded into numpy arrays
    """
    from revelsMD.trajectories import NumpyTrajectory

    universe = mda_ts.mdanalysis_universe

    if stop is None or stop == -1:
        stop = len(universe.trajectory)

    frames_to_load = range(start, stop, stride)
    n_frames = len(frames_to_load)
    n_atoms = len(universe.atoms)

    positions = np.zeros((n_frames, n_atoms, 3))
    forces = np.zeros((n_frames, n_atoms, 3))

    for i, frame_idx in enumerate(frames_to_load):
        universe.trajectory[frame_idx]
        positions[i] = universe.atoms.positions
        forces[i] = universe.atoms.forces

    # Build species list from atom names
    species = [atom.name for atom in universe.atoms]

    # Get charges and masses if available
    try:
        charges = universe.atoms.charges
        masses = universe.atoms.masses
    except Exception:
        charges = None
        masses = None

    return NumpyTrajectory(
        positions, forces,
        mda_ts.box_x, mda_ts.box_y, mda_ts.box_z,
        species, units=mda_ts.units,
        charge_list=charges, mass_list=masses
    )
