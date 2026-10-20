import pytest
import numpy as np
import scipy.constants as constants
from abc import ABC

from revelsMD.trajectories import (
    MDATrajectory,
    NumpyTrajectory,
    LammpsTrajectory,
    VaspTrajectory,
    DataUnavailableError,
)
from revelsMD.trajectories._base import Trajectory, compute_beta, frame_range, normalize_bounds


# -----------------------------------------------------------------------------
# MDATrajectory
# -----------------------------------------------------------------------------
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


def test_numpy_state_species_not_found():
    positions = np.zeros((1, 2, 3))
    forces = np.ones((1, 2, 3))
    species = ["O", "H"]
    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)
    with pytest.raises(ValueError, match="species 'C' not found"):
        state.get_indices("C")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_numpy_state_non_finite_box_raises(bad):
    with pytest.raises(ValueError, match="must all be finite"):
        NumpyTrajectory(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), 10, bad, 10, ["O", "H"],
                        temperature=300.0)


@pytest.mark.parametrize("box", [(-1, 10, 10), (10, -1, 10), (10, 10, -1)])
def test_numpy_state_negative_box_raises(box):
    with pytest.raises(ValueError, match="positive"):
        NumpyTrajectory(np.zeros((1, 2, 3)), np.ones((1, 2, 3)), *box, ["O", "H"], temperature=300.0)


def test_numpy_state_get_frame_out_of_range_raises():
    state = NumpyTrajectory(np.zeros((3, 2, 3)), np.ones((3, 2, 3)), 10, 10, 10, ["O", "H"],
                            temperature=300.0)
    for index in (3, -4):
        with pytest.raises(IndexError, match=f"frame {index} is out of range for 3 frames"):
            state.get_frame(index)


def test_warnings_point_at_the_caller_with_a_relative_import_path(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    work = tmp_path / "notebooks"
    work.mkdir()
    from lammps_dumps import frame
    (work / "a.dump").write_text("ITEM: TIMESTEP\n")
    (work / "b.dump").write_text(frame(0))
    script = work / "run.py"
    script.write_text(
        "import sys, warnings\n"
        f"sys.path.insert(0, {str(Path('..') / Path(__import__('os').path.relpath(repo, tmp_path)))!r})\n"
        "from revelsMD.trajectories.lammps import combine_frames, index_dump\n"
        "with warnings.catch_warnings(record=True) as record:\n"
        "    warnings.simplefilter('always')\n"
        "    combine_frames([index_dump('a.dump'), index_dump('b.dump')])\n"
        "import revelsMD\n"
        "print(revelsMD.__file__)\n"
        "print(record[0].filename)\n"
    )
    out = subprocess.run([sys.executable, "run.py"], cwd=work, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert ".." in out[0]
    assert out[1].endswith("run.py")


def test_numpy_state_get_frame_needs_an_integer():
    state = NumpyTrajectory(np.zeros((3, 2, 3)), np.ones((3, 2, 3)), 10, 10, 10, ["O", "H"],
                            temperature=300.0)
    with pytest.raises(TypeError):
        state.get_frame(1.0)
    np.testing.assert_array_equal(state.get_frame(np.int64(-1)).forces, np.ones((2, 3)))


def test_numpy_state_non_finite_frame_raises_when_read():
    positions = np.zeros((3, 2, 3))
    forces = np.ones((3, 2, 3))
    forces[1, 0, 2] = np.inf
    state = NumpyTrajectory(positions, forces, 10, 10, 10, ["O", "H"], temperature=300.0)
    frames = state.iter_frames()
    next(frames)
    with pytest.raises(ValueError, match="frame 1: the forces contain NaN or infinite"):
        next(frames)
    with pytest.raises(ValueError, match="frame 1: the forces contain NaN or infinite"):
        state.get_frame(-2)


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
def test_lammps_state_requires_topology():
    with pytest.raises(ValueError, match="topology file is required"):
        LammpsTrajectory("dump.lammpstrj", None, temperature=300.0)


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

    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame.positions, positions[i])
        np.testing.assert_array_equal(frame.forces, forces[i])


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
    for idx, frame in zip(expected_indices, frames):
        np.testing.assert_array_equal(frame.positions, positions[idx])


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
        frame = state.get_frame(i)
        np.testing.assert_array_equal(frame.positions, positions[i])
        np.testing.assert_array_equal(frame.forces, forces[i])


def test_numpy_get_frame_random_access():
    """get_frame should support random access in any order."""
    n_frames, n_atoms = 10, 2
    positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
    forces = np.zeros((n_frames, n_atoms, 3))
    species = ["A", "B"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    # Access frames in non-sequential order
    for i in [7, 2, 9, 0, 5]:
        frame = state.get_frame(i)
        np.testing.assert_array_equal(frame.positions, positions[i])


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

    with pytest.raises(DataUnavailableError, match="charge data not available"):
        state.get_charges("O")


def test_numpy_get_masses_raises_without_mass_data():
    """get_masses should raise DataUnavailableError when mass data is not available."""
    positions = np.zeros((5, 3, 3))
    forces = np.ones((5, 3, 3))
    species = ["O", "H", "H"]

    state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

    with pytest.raises(DataUnavailableError, match="mass data not available"):
        state.get_masses("O")


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

        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.positions, positions[i])

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
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.positions, positions[expected_indices[i]])

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
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.positions, positions[expected_indices[i]])

    def test_numpy_stop_none_means_all_frames(self):
        """stop=None should iterate through all frames."""
        n_frames, n_atoms = 5, 2
        positions = np.arange(n_frames * n_atoms * 3).reshape(n_frames, n_atoms, 3).astype(float)
        forces = np.zeros((n_frames, n_atoms, 3))
        species = ["A", "B"]

        state = NumpyTrajectory(positions, forces, 10, 10, 10, species, temperature=300.0)

        frames = list(state.iter_frames(stop=None))
        assert len(frames) == n_frames


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


class TestNumpyTrajectoryShapes:
    """NumpyTrajectory checks its arrays when it is built, not at the first frame read."""

    def _make(self, **kwargs):
        from revelsMD.trajectories.numpy import NumpyTrajectory
        args = dict(positions=np.zeros((2, 4, 3)), forces=np.zeros((2, 4, 3)), box_x=10.0, box_y=10.0,
                    box_z=10.0, species_list=["A"] * 4, temperature=1.0, units="lj")
        args.update(kwargs)
        return NumpyTrajectory(**args)

    def test_positions_without_a_frame_axis_raise(self):
        with pytest.raises(ValueError, match=r"positions must have shape \(frames, atoms, 3\)"):
            self._make(positions=np.zeros((4, 3)), forces=np.zeros((4, 3)))

    @pytest.mark.parametrize("name", ["charge_list", "mass_list"])
    @pytest.mark.parametrize("length", [2, 6])
    def test_charge_and_mass_lists_need_one_value_per_atom(self, name, length):
        with pytest.raises(ValueError, match=rf"{name} has {length} values for 4 atoms"):
            self._make(**{name: np.ones(length)})


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

    @pytest.mark.parametrize("temperature", [300, np.float32(300.0), np.float64(300.0)])
    def test_compute_beta_is_the_same_for_any_numeric_type(self, temperature):
        """Equal temperatures give a bit-identical float beta, whatever their type."""
        beta = compute_beta('real', temperature)
        assert type(beta) is float
        assert beta == compute_beta('real', 300.0)

    def test_compute_beta_strips_whitespace(self):
        """Unit system should ignore leading/trailing whitespace."""
        assert compute_beta('  lj  ', 1.0) == compute_beta('lj', 1.0)

    def test_compute_beta_invalid_unit_raises(self):
        """Unsupported unit system should raise ValueError."""
        with pytest.raises(ValueError, match="unsupported unit system"):
            compute_beta('quantum-donut', 300.0)

    def test_compute_beta_zero_temperature_raises(self):
        """Zero temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            compute_beta('real', 0.0)

    def test_compute_beta_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            compute_beta('real', -100.0)

    def test_compute_beta_infinite_temperature_raises(self):
        """Infinite temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be finite"):
            compute_beta('real', float('inf'))

    def test_compute_beta_nan_temperature_raises(self):
        """NaN temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature must be finite"):
            compute_beta('real', float('nan'))


# -----------------------------------------------------------------------------
# Cell matrix validation
# -----------------------------------------------------------------------------
class TestValidateCellMatrixContext:

    def test_where_prefixes_the_message(self):
        with pytest.raises(ValueError, match="^a.dump, frame 0: cell matrix must have positive volume"):
            Trajectory._validate_cell_matrix(np.zeros((3, 3)), "a.dump, frame 0")


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


# -----------------------------------------------------------------------------
# Cell matrix attribute and properties on NumpyTrajectory
# -----------------------------------------------------------------------------
class TestCellMatrixProperties:
    """Tests for cell_matrix, cell_volume, is_orthorhombic, box_x/y/z."""

    @pytest.fixture
    def orthorhombic_traj(self):
        """Orthorhombic NumpyTrajectory with box (10, 8, 6)."""
        return NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            box_x=10.0, box_y=8.0, box_z=6.0,
            species_list=["A", "A", "A"],
            temperature=300.0, units="real",
        )

    def test_cell_matrix_is_set(self, orthorhombic_traj):
        cell = orthorhombic_traj.cell_matrix
        expected = np.diag([10.0, 8.0, 6.0])
        np.testing.assert_allclose(cell, expected)

    def test_cell_volume(self, orthorhombic_traj):
        assert orthorhombic_traj.cell_volume == pytest.approx(10.0 * 8.0 * 6.0)

    def test_is_orthorhombic(self, orthorhombic_traj):
        assert orthorhombic_traj.is_orthorhombic is True

    def test_box_x_y_z_for_orthorhombic(self, orthorhombic_traj):
        assert orthorhombic_traj.box_x == pytest.approx(10.0)
        assert orthorhombic_traj.box_y == pytest.approx(8.0)
        assert orthorhombic_traj.box_z == pytest.approx(6.0)


# -----------------------------------------------------------------------------
# NumpyTrajectory cell_matrix constructor parameter
# -----------------------------------------------------------------------------
class TestNumpyTrajectoryCellMatrix:
    """Tests for constructing NumpyTrajectory with cell_matrix parameter."""

    def test_construction_with_orthorhombic_cell_matrix(self):
        cell = np.diag([10.0, 8.0, 6.0])
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            cell_matrix=cell,
            species_list=["A", "A", "A"],
            temperature=300.0, units="real",
        )
        np.testing.assert_allclose(traj.cell_matrix, cell)
        assert traj.is_orthorhombic is True
        assert traj.box_x == pytest.approx(10.0)

    def test_construction_with_triclinic_cell_matrix(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            cell_matrix=cell,
            species_list=["A", "A", "A"],
            temperature=300.0, units="real",
        )
        np.testing.assert_allclose(traj.cell_matrix, cell)
        assert traj.is_orthorhombic is False

    def test_box_x_raises_for_triclinic_cell(self):
        cell = np.array([
            [10.0, 0.0, 0.0],
            [3.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ])
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            cell_matrix=cell,
            species_list=["A", "A", "A"],
            temperature=300.0, units="real",
        )
        with pytest.raises(AttributeError, match="box_x is not defined"):
            _ = traj.box_x
        with pytest.raises(AttributeError, match="box_y is not defined"):
            _ = traj.box_y
        with pytest.raises(AttributeError, match="box_z is not defined"):
            _ = traj.box_z

    def test_construction_with_box_x_y_z_still_works(self):
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            box_x=10.0, box_y=8.0, box_z=6.0,
            species_list=["A", "A", "A"],
            temperature=300.0, units="real",
        )
        np.testing.assert_allclose(traj.cell_matrix, np.diag([10.0, 8.0, 6.0]))

    def test_both_cell_matrix_and_box_raises(self):
        with pytest.raises(ValueError, match="cannot specify both"):
            NumpyTrajectory(
                positions=np.zeros((2, 3, 3)),
                forces=np.zeros((2, 3, 3)),
                box_x=10.0, box_y=8.0, box_z=6.0,
                cell_matrix=np.diag([10.0, 8.0, 6.0]),
                species_list=["A", "A", "A"],
                temperature=300.0, units="real",
            )

    def test_neither_cell_matrix_nor_box_raises(self):
        with pytest.raises(ValueError, match="must specify either"):
            NumpyTrajectory(
                positions=np.zeros((2, 3, 3)),
                forces=np.zeros((2, 3, 3)),
                species_list=["A", "A", "A"],
                temperature=300.0, units="real",
            )

    def test_invalid_cell_matrix_raises(self):
        with pytest.raises(ValueError, match="shape"):
            NumpyTrajectory(
                positions=np.zeros((2, 3, 3)),
                forces=np.zeros((2, 3, 3)),
                cell_matrix=np.eye(4),
                species_list=["A", "A", "A"],
                temperature=300.0, units="real",
            )

    def test_partial_box_dimensions_raises(self):
        """Providing only some of box_x/y/z should raise a clear error."""
        with pytest.raises(ValueError, match="all three"):
            NumpyTrajectory(
                positions=np.zeros((2, 3, 3)),
                forces=np.zeros((2, 3, 3)),
                box_x=10.0,
                species_list=["A", "A", "A"],
                temperature=300.0, units="real",
            )

    def test_construction_without_species_list(self):
        """Construction without species_list should work."""
        cell = np.diag([10.0, 8.0, 6.0])
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            cell_matrix=cell,
            temperature=300.0, units="real",
        )
        assert traj.species_string is None

    def test_get_indices_without_species_list_raises(self):
        """Calling get_indices when species_list was not provided should raise."""
        cell = np.diag([10.0, 8.0, 6.0])
        traj = NumpyTrajectory(
            positions=np.zeros((2, 3, 3)),
            forces=np.zeros((2, 3, 3)),
            cell_matrix=cell,
            temperature=300.0, units="real",
        )
        with pytest.raises(ValueError, match="[Ss]pecies list"):
            traj.get_indices("A")


# -----------------------------------------------------------------------------
# normalize_bounds
# -----------------------------------------------------------------------------
class TestNormalizeBounds:
    """Tests for the module-level normalize_bounds function."""

    def test_defaults_stop_to_n_frames(self):
        assert normalize_bounds(10, 0, None, 1) == (0, 10, 1)

    def test_negative_start(self):
        assert normalize_bounds(10, -3, None, 1) == (7, 10, 1)

    def test_negative_stop(self):
        assert normalize_bounds(10, 0, -2, 1) == (0, 8, 1)

    def test_negative_start_and_stop(self):
        assert normalize_bounds(10, -5, -1, 1) == (5, 9, 1)

    def test_start_clamped_to_n_frames(self):
        assert normalize_bounds(10, 15, None, 1) == (10, 10, 1)

    def test_stop_clamped_to_n_frames(self):
        assert normalize_bounds(10, 0, 20, 1) == (0, 10, 1)

    def test_large_negative_start_clamped_to_zero(self):
        assert normalize_bounds(10, -100, None, 1) == (0, 10, 1)

    def test_large_negative_stop_clamped_to_zero(self):
        assert normalize_bounds(10, 0, -100, 1) == (0, 0, 1)

    def test_stride_passed_through(self):
        assert normalize_bounds(10, 0, None, 3) == (0, 10, 3)

    def test_zero_stride_raises(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            normalize_bounds(10, 0, None, 0)

    def test_negative_stride_raises(self):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            normalize_bounds(10, 0, None, -1)

    def test_zero_frames(self):
        assert normalize_bounds(0, 0, None, 1) == (0, 0, 1)


class TestCheckFixedCell:
    """check_fixed_cell accepts only an identical cell."""

    def test_identical_cell_passes(self):
        from revelsMD.trajectories._base import check_fixed_cell
        cell = np.array([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.0, 8.0]])
        check_fixed_cell(cell.copy(), cell, "f.dump, timestep 0")

    def test_any_difference_raises_naming_the_frame(self):
        from revelsMD.trajectories._base import check_fixed_cell
        cell = np.diag([10.0, 10.0, 10.0])
        changed = cell.copy()
        changed[0, 0] = np.nextafter(10.0, 11.0)
        with pytest.raises(ValueError, match="f.dump, timestep 100: the cell differs.*fixed cell"):
            check_fixed_cell(changed, cell, "f.dump, timestep 100")


class TestAsPaths:
    """as_paths normalises file arguments to a list of strings, in the order given."""

    def test_single_string(self):
        from revelsMD.trajectories._base import as_paths
        assert as_paths("a.dump") == ["a.dump"]

    def test_single_path(self, tmp_path):
        from revelsMD.trajectories._base import as_paths
        assert as_paths(tmp_path / "a.dump") == [str(tmp_path / "a.dump")]

    def test_sequence_of_mixed(self, tmp_path):
        from revelsMD.trajectories._base import as_paths
        assert as_paths(("a.dump", tmp_path / "b.dump")) == ["a.dump", str(tmp_path / "b.dump")]

    def test_bytes_paths(self):
        from revelsMD.trajectories._base import as_paths
        assert as_paths(b"a.dump") == ["a.dump"]
        assert as_paths([b"a.dump", "b.dump"]) == ["a.dump", "b.dump"]

    def test_array_of_paths(self, tmp_path):
        from revelsMD.trajectories._base import as_paths
        assert as_paths(np.array(["a.dump", "b.dump"])) == ["a.dump", "b.dump"]

    def test_empty_sequence_is_rejected(self):
        from revelsMD.trajectories._base import as_paths
        with pytest.raises(ValueError, match="at least one trajectory file"):
            as_paths([])

class TestReadingWithMDAnalysis:
    """Errors raised while MDAnalysis parses files become RuntimeError naming them."""

    @pytest.mark.parametrize("error", [StopIteration(), EOFError(), OSError("XDR"), KeyError("x")])
    def test_parse_errors_are_relabelled(self, error):
        from revelsMD.trajectories._base import reading_with_mdanalysis
        with pytest.raises(RuntimeError, match=f"MDAnalysis cannot read a.trr: {type(error).__name__}"):
            with reading_with_mdanalysis("a.trr"):
                raise error

    @pytest.mark.parametrize("error", [
        FileNotFoundError(), PermissionError(), IsADirectoryError(), NotADirectoryError(), ImportError(),
        MemoryError(), RecursionError(), UserWarning("w"), KeyboardInterrupt(),
    ])
    def test_other_errors_pass_through(self, error):
        from revelsMD.trajectories._base import reading_with_mdanalysis
        with pytest.raises(type(error)):
            with reading_with_mdanalysis("a.trr"):
                raise error


class TestCheckFinite:
    """check_finite rejects NaN and infinity in either array, naming the frame."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_raise(self, bad):
        from revelsMD.trajectories._base import check_finite
        good = np.zeros((2, 3))
        broken = good.copy()
        broken[1, 2] = bad
        check_finite(good, good, "f, frame 0")
        with pytest.raises(ValueError, match="f, frame 3: the positions contain NaN or infinite"):
            check_finite(broken, good, "f, frame 3")
        with pytest.raises(ValueError, match="f, frame 3: the forces contain NaN or infinite"):
            check_finite(good, broken, "f, frame 3")

    def test_frame_found_by_a_strided_read_is_named_by_its_index(self):
        forces = np.ones((10, 2, 3))
        forces[5, 0, 0] = np.nan
        traj = NumpyTrajectory(np.zeros((10, 2, 3)), forces, 10, 10, 10, ["O", "H"], temperature=300.0)
        with pytest.raises(ValueError, match=r"^frame 5: the forces"):
            list(traj.iter_frames(start=1, stride=2))


class TestReadFramesOneFileAtATime:
    """The shared reader closes each file when it moves on and when it stops."""

    def _files(self, tmp_path):
        paths = []
        for name in ("a", "b"):
            path = tmp_path / name
            path.write_bytes(name.encode())
            paths.append(str(path))
        return paths

    def test_files_are_closed_on_switch_and_at_the_end(self, tmp_path):
        from revelsMD.trajectories._base import read_frames_one_file_at_a_time
        paths = self._files(tmp_path)
        handles = []

        def read(handle, index):
            handles.append(handle)
            assert all(h.closed for h in handles if h is not handle)
            return handle.read(1)

        frames = read_frames_one_file_at_a_time([0, 1], lambda i: paths[i], read)
        assert list(frames) == [b"a", b"b"]
        assert len(handles) == 2 and all(h.closed for h in handles)

    def test_each_file_is_opened_once_per_run_of_frames(self, tmp_path, monkeypatch):
        import builtins
        from revelsMD.trajectories import _base
        paths = self._files(tmp_path)
        opened = []

        def counting_open(path, mode):
            opened.append(path)
            return builtins.open(path, mode)

        monkeypatch.setattr(_base, "open", counting_open, raising=False)
        frames = _base.read_frames_one_file_at_a_time(
            [0, 0, 1, 1], lambda i: paths[i], lambda handle, index: index)
        assert list(frames) == [0, 0, 1, 1]
        assert opened == paths

    def test_file_is_closed_when_reading_raises(self, tmp_path):
        from revelsMD.trajectories._base import read_frames_one_file_at_a_time
        paths = self._files(tmp_path)
        handles = []

        def read(handle, index):
            handles.append(handle)
            raise ValueError("bad frame")

        with pytest.raises(ValueError, match="bad frame"):
            list(read_frames_one_file_at_a_time([0], lambda i: paths[i], read))
        assert handles[0].closed

    def test_file_is_closed_when_iteration_is_abandoned(self, tmp_path):
        from revelsMD.trajectories._base import read_frames_one_file_at_a_time
        paths = self._files(tmp_path)
        handles = []

        def read(handle, index):
            handles.append(handle)
            return index

        frames = read_frames_one_file_at_a_time([0, 0], lambda i: paths[i], read)
        assert next(frames) == 0
        frames.close()
        assert handles[0].closed


class TestFrameRange:

    @pytest.mark.parametrize("start, stop, period, expected", [
        (0, None, 1, range(5)),
        (-2, None, 1, range(3, 5)),
        (1, -1, 2, range(1, 4, 2)),
        (-5, 5, 1, range(5)),
    ])
    def test_selected_frames(self, start, stop, period, expected):
        assert frame_range(5, start, stop, period) == expected

    @pytest.mark.parametrize("start, stop, period", [
        (5, None, 1), (6, None, 1), (-6, None, 1), (0, 6, 1), (0, -6, 1), (3, 2, 1), (0, None, 0),
    ])
    def test_outside_the_trajectory_or_empty_raises(self, start, stop, period):
        with pytest.raises(ValueError):
            frame_range(5, start, stop, period)
