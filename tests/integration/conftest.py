"""Shared fixtures and configuration for integration tests."""

import pytest
import numpy as np
from pathlib import Path


# Canonical committed test data (trimmed trajectory subsets). Full-length
# trajectories live outside the repository and are used only by
# scripts/validate_*.py.
TEST_DATA_DIR = Path(__file__).parents[1] / "data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "analytical: tests against known analytical results")
    config.addinivalue_line("markers", "regression: regression tests against stored reference data")
    config.addinivalue_line("markers", "requires_example1: uses the committed Example 1 subset in tests/data")
    config.addinivalue_line("markers", "requires_example2: uses the committed Example 2 subset in tests/data")
    config.addinivalue_line("markers", "requires_example4: uses the committed Example 4 subset in tests/data")
    config.addinivalue_line("markers", "requires_vasp: uses the committed VASP subset in tests/data")


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


# ---------------------------------------------------------------------------
# Trajectory fixtures - LAMMPS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example1_trajectory():
    """
    Load Example 1 LJ trajectory for RDF tests.

    Example 1: Lennard-Jones fluid (committed 10-frame subset)
    - 2880 atoms (2 types)
    - 10 frames
    - 14.227 cubic LJ box
    """
    from revelsMD.trajectories import LammpsTrajectory

    dump_file = TEST_DATA_DIR / "example_1_LJ" / "dump.nh.lammps"
    data_file = TEST_DATA_DIR / "example_1_LJ" / "data.fin.nh.data"

    return LammpsTrajectory(
        str(dump_file),
        str(data_file),
        temperature=1.35,  # LJ reduced temperature from simulation
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
    )


@pytest.fixture(scope="module")
def example2_trajectory():
    """
    Load Example 2 LJ trajectory for 3D density tests.

    Example 2: Lennard-Jones 3D density (committed 10-frame subset)
    - 2880 atoms
    - 10 frames
    - Frozen central particle + solvating LJ spheres
    """
    from revelsMD.trajectories import LammpsTrajectory

    dump_file = TEST_DATA_DIR / "example_2_LJ_3D" / "dump.nh.lammps"
    data_file = TEST_DATA_DIR / "example_2_LJ_3D" / "data.fin.nh.data"

    return LammpsTrajectory(
        str(dump_file),
        str(data_file),
        temperature=1.35,  # LJ reduced temperature from simulation
        units='lj',
        atom_style="id resid type q x y z ix iy iz",
    )


# ---------------------------------------------------------------------------
# Trajectory fixtures - GROMACS/MDAnalysis
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example4_trajectory():
    """
    Load Example 4 rigid water trajectory (committed 10-frame subset).

    Example 4: SPC/E water
    - 6339 atoms (2113 water molecules)
    - 10 frames
    - GROMACS trr/tpr format
    """
    from revelsMD.trajectories import MDATrajectory

    trr_file = TEST_DATA_DIR / "example_4_water" / "prod.trr"
    tpr_file = TEST_DATA_DIR / "example_4_water" / "prod.tpr"

    return MDATrajectory(str(trr_file), str(tpr_file), temperature=300.0)


# ---------------------------------------------------------------------------
# Trajectory fixtures - VASP
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vasp_trajectory():
    """
    Load VASP trajectory from vasprun.xml (committed 10-step subset).

    Example 3: BaSnF4 solid electrolyte
    - 324 atoms (Ba, Sn, F)
    - 10 calculation steps (subset from 3001 total in r1)
    - Temperature: 600K
    """
    from revelsMD.trajectories import VaspTrajectory

    vasprun_file = TEST_DATA_DIR / "example_3_vasp" / "vasprun.xml"

    return VaspTrajectory(str(vasprun_file), temperature=600.0)  # BaSnF4 at 600K


# ---------------------------------------------------------------------------
# Trajectory fixtures - NumPy (synthetic)
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_gas_trajectory():
    """
    Generate trajectory with uniform random positions and random forces.

    The force-sampling method requires non-zero forces to work correctly.
    With random forces, this should produce g(r) ~ 1 for all r (within statistical noise).

    Uses 500 atoms and 50 frames for good convergence of histogram g(r) to unity.
    """
    from revelsMD.trajectories import NumpyTrajectory

    np.random.seed(42)
    n_atoms = 500
    n_frames = 50
    box = 10.0

    positions = np.random.uniform(0, box, (n_frames, n_atoms, 3))
    # Use random forces - required for force-sampling method to work
    forces = np.random.randn(n_frames, n_atoms, 3) * 0.1
    species = ['1'] * n_atoms

    return NumpyTrajectory(
        positions, forces, box, box, box, species, temperature=1.0, units='lj'
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
        positions, forces, box, box, box, species, temperature=1.0, units='lj'
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
        positions, forces, box, box, box, species, temperature=1.0, units='lj'
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
        positions, forces, box, box, box, species, temperature=1.0, units='lj'
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
        species, temperature=lammps_ts.temperature, units=lammps_ts.units
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
        species, temperature=mda_ts.temperature, units=mda_ts.units,
        charge_list=charges, mass_list=masses
    )
