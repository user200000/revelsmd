"""MDATrajectory on the committed water trajectory and variants written from it."""

import warnings
from pathlib import Path

import MDAnalysis as MD
import h5py
import numpy as np
import pytest

from revelsMD.trajectories import MDATrajectory
from revelsMD.trajectories._base import compute_beta

DATA = Path(__file__).parent / "data"
WATER = DATA / "example_4_water"
TPR = str(WATER / "prod.tpr")
TRR = str(WATER / "prod.trr")
LAMMPS_DUMP = DATA / "lammps_small" / "dump.lammps"


def write_variant(path, edit=None, n_frames=3):
    """Write the first frames of the water trajectory, applying edit(ts) to each."""
    universe = MD.Universe(TPR, TRR)
    with MD.Writer(str(path), n_atoms=universe.atoms.n_atoms) as writer:
        for ts in universe.trajectory[:n_frames]:
            if edit is not None:
                edit(ts)
            writer.write(universe.atoms)
    return str(path)


def write_frames(path, frames, edit=None):
    """Write the given frames of the water trajectory, applying edit(ts) to each."""
    universe = MD.Universe(TPR, TRR)
    with MD.Writer(str(path), n_atoms=universe.atoms.n_atoms) as writer:
        for ts in universe.trajectory[list(frames)]:
            if edit is not None:
                edit(ts)
            writer.write(universe.atoms)
    return str(path)


def shift_time(offset):
    def edit(ts):
        ts.time = ts.time + offset
    return edit


@pytest.fixture(scope="module")
def traj():
    return MDATrajectory(TRR, TPR, temperature=300.0)


@pytest.fixture(scope="module")
def universe():
    return MD.Universe(TPR, TRR)


class TestReading:

    def test_frames_cell_and_selections(self, traj, universe):
        assert traj.frames == 10
        np.testing.assert_array_equal(traj.cell_matrix, np.diag([40.0, 40.0, 40.0]))
        names = universe.atoms.names
        np.testing.assert_array_equal(traj.get_indices("Ow"), np.flatnonzero(names == "Ow"))
        selection = universe.select_atoms("name Ow")
        np.testing.assert_array_equal(traj.get_charges("Ow"), selection.charges)
        np.testing.assert_array_equal(traj.get_masses("Ow"), selection.masses)

    def test_iteration_and_slicing(self, traj, universe):
        frames = list(traj.iter_frames())
        assert len(frames) == 10
        for k, frame in enumerate(frames):
            ts = universe.trajectory[k]
            np.testing.assert_array_equal(frame.positions, ts.positions)
            np.testing.assert_array_equal(frame.forces, ts.forces)
        assert len(list(traj.iter_frames(stop=-1))) == 9

    @pytest.mark.parametrize("start, expected", [(2, (2, 4, 6)), (3, (3, 5, 7))])
    def test_sliced_iteration_returns_the_selected_frames(self, traj, universe, start, expected):
        frames = list(traj.iter_frames(start=start, stop=8, stride=2))
        assert len(frames) == 3
        for frame, k in zip(frames, expected):
            ts = universe.trajectory[k]
            np.testing.assert_array_equal(frame.positions, ts.positions)
            np.testing.assert_array_equal(frame.forces, ts.forces)

    def test_get_frame_in_any_order(self, traj, universe):
        for k in (7, 2, 9, 0):
            np.testing.assert_array_equal(traj.get_frame(k).forces, universe.trajectory[k].forces)

    def test_temperature_and_beta(self, traj):
        assert traj.temperature == 300.0
        assert traj.beta == pytest.approx(compute_beta("mda", 300.0), rel=1e-12)

    def test_temperature_is_required(self):
        with pytest.raises(TypeError):
            MDATrajectory(TRR, TPR)

    def test_triclinic_box(self, tmp_path):
        dims = np.array([40.0, 38.0, 36.0, 80.0, 85.0, 70.0])
        path = write_variant(tmp_path / "tri.trr", lambda ts: setattr(ts, "dimensions", dims))
        traj = MDATrajectory(path, TPR, temperature=300.0)
        # a along x, b in the xy plane, c completing the angles alpha, beta, gamma.
        a, b, c = 40.0, 38.0, 36.0
        alpha, beta, gamma = np.radians([80.0, 85.0, 70.0])
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        expected = np.array([
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
            [cx, cy, np.sqrt(c**2 - cx**2 - cy**2)],
        ])
        np.testing.assert_allclose(traj.cell_matrix, expected, rtol=1e-6, atol=1e-5)
        assert traj.is_orthorhombic is False
        assert len(list(traj.iter_frames())) == 3


class TestValidation:

    def test_trajectory_without_forces_raises_at_construction(self, tmp_path):
        path = write_variant(tmp_path / "positions.xtc")
        with pytest.raises(ValueError, match="positions.xtc: the trajectory has no forces"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_frame_without_forces_raises_when_read(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        path = write_variant(tmp_path / "partial.trr", drop_forces)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        frames = traj.iter_frames()
        next(frames)
        with pytest.raises(ValueError, match="partial.trr, frame 1: no forces"):
            next(frames)
        with pytest.raises(ValueError, match="frame 1: no forces"):
            traj.get_frame(1)
        with pytest.raises(ValueError, match="partial.trr, frame 1: no forces"):
            next(traj.iter_frames(start=1))

    def test_trajectory_without_positions_raises_at_construction(self, tmp_path):
        path = write_variant(tmp_path / "forces.trr", lambda ts: setattr(ts, "has_positions", False))
        with pytest.raises(ValueError, match="forces.trr: the trajectory has no positions"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_frame_without_positions_raises_when_read(self, tmp_path):
        def drop_positions(ts):
            if ts.frame == 1:
                ts.has_positions = False
        path = write_variant(tmp_path / "partial.trr", drop_positions)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        frames = traj.iter_frames()
        next(frames)
        with pytest.raises(ValueError, match="partial.trr, frame 1: no positions"):
            next(frames)
        with pytest.raises(ValueError, match="frame 1: no positions"):
            traj.get_frame(1)

    def test_changing_box_raises_when_read(self, tmp_path):
        def grow(ts):
            if ts.frame == 2:
                ts.dimensions = ts.dimensions * np.array([1.01, 1, 1, 1, 1, 1], dtype=np.float32)
        path = write_variant(tmp_path / "npt.trr", grow)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="npt.trr, frame 2: the cell differs"):
            list(traj.iter_frames())
        with pytest.raises(ValueError, match="npt.trr, frame 2: the cell differs"):
            traj.get_frame(2)

    def test_changing_box_angle_raises_when_read(self, tmp_path):
        def shear(ts):
            if ts.frame == 1:
                ts.dimensions = np.array([40, 40, 40, 90, 90, 89.5], dtype=np.float32)
        path = write_variant(tmp_path / "sheared.trr", shear)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="sheared.trr, frame 1: the cell differs"):
            traj.get_frame(1)

    def test_frame_without_a_box_raises_when_read(self, tmp_path):
        def drop_box(ts):
            if ts.frame == 1:
                ts.dimensions = np.zeros(6)
        path = write_variant(tmp_path / "boxless.trr", drop_box)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="boxless.trr, frame 1: no simulation box"):
            traj.get_frame(1)

    def test_non_finite_forces_raise_when_read(self, tmp_path):
        def blow_up(ts):
            if ts.frame == 1:
                ts.forces[3, 1] = np.nan
        path = write_variant(tmp_path / "nan.trr", blow_up)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="nan.trr, frame 1: the forces contain NaN"):
            traj.get_frame(1)

    def test_degenerate_box_raises_naming_the_file(self, tmp_path):
        dims = np.array([1e-5, 1e-5, 1e-5, 90.0, 90.0, 90.0])
        path = write_variant(tmp_path / "flat.trr", lambda ts: setattr(ts, "dimensions", dims))
        with pytest.raises(ValueError, match=r"flat\.trr: Cell matrix must have positive volume"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_trajectory_without_a_box_raises_at_construction(self, tmp_path):
        path = write_variant(tmp_path / "nobox.trr", lambda ts: setattr(ts, "dimensions", np.zeros(6)))
        with pytest.raises(ValueError, match="nobox.trr: the trajectory has no simulation box"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_list_of_files_is_joined(self, tmp_path, universe):
        first = write_variant(tmp_path / "a.trr", shift_time(-1e6), n_frames=3)
        traj = MDATrajectory([first, TRR], TPR, temperature=300.0)
        assert traj.frames == 13
        np.testing.assert_array_equal(traj.get_frame(3).forces, universe.trajectory[0].forces)

    @pytest.mark.parametrize("first, second", [
        (range(0, 5), range(5, 10)),  # consecutive
        (range(0, 5), range(4, 10)),  # -noappend: the restart repeats frame 4
        (range(0, 8), range(4, 10)),  # a restart from frame 4 of a longer run
    ])
    def test_restart_files_are_joined_by_time(self, tmp_path, universe, first, second):
        a = write_frames(tmp_path / "a.trr", first)
        b = write_frames(tmp_path / "b.trr", second)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 10
        for k, frame in enumerate(traj.iter_frames()):
            np.testing.assert_array_equal(frame.forces, universe.trajectory[k].forces)
        for k in (3, 4, 5, -1):
            np.testing.assert_array_equal(traj.get_frame(k).forces, universe.trajectory[k].forces)
        got = list(traj.iter_frames(start=3, stop=9, stride=2))
        for frame, k in zip(got, (3, 5, 7), strict=True):
            np.testing.assert_array_equal(frame.forces, universe.trajectory[k].forces)

    def test_restart_ending_before_the_frames_it_replaces_warns(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(0, 8))
        b = write_frames(tmp_path / "b.trr", range(3, 6))
        with pytest.warns(UserWarning, match=r"b\.trr replaces the frames of .*a\.trr .* 2 frames are dropped"):
            traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 6

    def test_file_starting_before_the_previous_one_raises(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(3, 6))
        b = write_frames(tmp_path / "b.trr", range(0, 5))
        with pytest.raises(ValueError, match=r"b\.trr starts at time .* at or before the start of .*a\.trr"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_errors_in_a_list_name_the_file_and_its_frame(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        def later_without_forces(ts):
            drop_forces(ts)
            ts.time = ts.time + 1e6
        partial = write_variant(tmp_path / "partial.trr", later_without_forces)
        traj = MDATrajectory([TRR, partial], TPR, temperature=300.0)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            traj.get_frame(11)
        assert traj.get_frame(10).forces.shape == traj.get_frame(9).forces.shape
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            traj.get_frame(-2)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            list(traj.iter_frames(start=10))

    def test_get_frame_needs_an_integer(self, traj):
        with pytest.raises(TypeError):
            traj.get_frame(1.0)

    def test_same_file_twice_raises(self):
        with pytest.raises(ValueError, match="is given twice"):
            MDATrajectory([TRR, TRR], TPR, temperature=300.0)

    def test_unordered_files_raise(self):
        with pytest.raises(TypeError, match="not set"):
            MDATrajectory({TRR}, TPR, temperature=300.0)

    # MDAnalysis's TRR reader fails while closing a file it never opened.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing.tpr"):
            MDATrajectory(TRR, str(tmp_path / "missing.tpr"), temperature=300.0)
        with pytest.raises(OSError, match="missing.trr"):
            MDATrajectory(str(tmp_path / "missing.trr"), TPR, temperature=300.0)

    def test_files_mdanalysis_cannot_parse(self, tmp_path):
        dump = tmp_path / "small.lammpsdump"
        dump.write_bytes(LAMMPS_DUMP.read_bytes())
        with pytest.raises(RuntimeError, match="Failed to load MDAnalysis Universe.*same number of atoms"):
            MDATrajectory(str(dump), TPR, temperature=300.0)

    def test_unknown_atom_name_raises(self, traj):
        with pytest.raises(ValueError, match="no atoms named 'Na'"):
            traj.get_indices("Na")


class TestUnits:
    """Only formats whose length and force units MDAnalysis knows are accepted."""

    @pytest.mark.filterwarnings("ignore:Guessed all Masses")
    def test_lammps_dump_is_rejected(self, tmp_path):
        dump = tmp_path / "small.lammpsdump"
        dump.write_bytes(LAMMPS_DUMP.read_bytes())
        with pytest.raises(ValueError, match="small.lammpsdump: MDAnalysis does not know the length and force units.*LammpsTrajectory"):
            MDATrajectory(str(dump), str(dump), temperature=300.0)

    @pytest.mark.filterwarnings("ignore:Unknown masses")
    def test_dl_poly_history_is_rejected(self, tmp_path):
        # Two argon atoms in two frames, positions, velocities and forces
        # (keytrj 2, imcon 1); MDAnalysis gives DL_POLY no force unit.
        frames = []
        for step in (0, 10):
            atoms = "".join(
                f"Ar {i} 39.948 0.0\n {x} {x} {x}\n 0.0 0.0 0.0\n 100.0 -50.0 25.0\n"
                for i, x in ((1, 1.0), (2, 5.0))
            )
            frames.append(
                f"timestep {step} 2 2 1 0.001\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\n"
                + atoms
            )
        history = tmp_path / "run.history"
        history.write_text("test history\n 2 1 2\n" + "".join(frames))
        with pytest.raises(ValueError, match="run.history: MDAnalysis does not know the length and force units") as e:
            MDATrajectory(str(history), str(history), temperature=300.0)
        assert "LammpsTrajectory" not in str(e.value)

    def test_h5md_with_units_is_read(self, tmp_path, universe):
        path = write_variant(tmp_path / "water.h5md")
        traj = MDATrajectory(path, TPR, temperature=300.0)
        frame = traj.get_frame(2)
        np.testing.assert_allclose(frame.forces, universe.trajectory[2].forces, rtol=1e-6)

    def test_h5md_without_a_force_unit_is_rejected(self, tmp_path):
        path = write_variant(tmp_path / "unitless.h5md")
        with h5py.File(path, "r+") as f:
            del f["particles/trajectory/force/value"].attrs["unit"]
        # MDAnalysis refuses to convert units it cannot read.
        with pytest.raises(RuntimeError, match="H5MD file must have readable units"):
            MDATrajectory(path, TPR, temperature=300.0)
        # The unit check rejects the reader MDAnalysis opens without conversion.
        from revelsMD.trajectories.mda import _check_reader
        reader = MD.Universe(TPR, path, convert_units=False).trajectory
        with pytest.raises(ValueError, match="unitless.h5md: MDAnalysis does not know the length and force units"):
            _check_reader(reader)

    def test_every_file_of_a_list_is_checked(self, tmp_path):
        good = write_variant(tmp_path / "good.h5md", n_frames=4)
        bad = write_variant(tmp_path / "bad.h5md", n_frames=4)
        with h5py.File(bad, "r+") as f:
            group = f["particles/trajectory/force"]
            steps = group["step"][()]
            del group["step"]
            group["step"] = steps * 2
        with pytest.raises(ValueError, match="bad.h5md: the forces are saved at different steps"):
            MDATrajectory([good, bad], TPR, temperature=300.0)

    def test_h5md_without_force_steps_cannot_be_checked(self, tmp_path):
        path = write_variant(tmp_path / "nosteps.h5md")
        with h5py.File(path, "r+") as f:
            del f["particles/trajectory/force/step"]
        with pytest.raises(RuntimeError, match="nosteps.h5md: cannot check"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_h5md_step_check_fails_closed(self, tmp_path):
        from revelsMD.trajectories.mda import _check_reader
        reader = MD.Universe(TPR, write_variant(tmp_path / "water.h5md")).trajectory
        _check_reader(reader)
        del reader._particle_group
        with pytest.raises(RuntimeError, match="cannot check that this H5MD file"):
            _check_reader(reader)

    def test_h5md_with_forces_at_other_steps_is_rejected(self, tmp_path):
        path = write_variant(tmp_path / "steps.h5md", n_frames=4)
        with h5py.File(path, "r+") as f:
            # The writer links one step dataset into every group.
            group = f["particles/trajectory/force"]
            steps = group["step"][()]
            del group["step"]
            group["step"] = steps * 2
        with pytest.raises(ValueError, match="steps.h5md: the forces are saved at different steps"):
            MDATrajectory(path, TPR, temperature=300.0)

