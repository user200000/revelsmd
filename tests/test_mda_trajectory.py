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


def write_frames(path, frames, edit=None, **writer_options):
    """Write the given frames of the water trajectory, applying edit(ts) to each."""
    universe = MD.Universe(TPR, TRR)
    with MD.Writer(str(path), n_atoms=universe.atoms.n_atoms, **writer_options) as writer:
        for ts in universe.trajectory[list(frames)]:
            if edit is not None:
                edit(ts)
            writer.write(universe.atoms)
    return str(path)


def shift_time(offset):
    """Move frames by ``offset`` ps (1000 steps of the water run per ps)."""
    def edit(ts):
        ts.time = ts.time + offset
        ts.data["step"] = ts.data["step"] + int(offset * 1000)
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
        first = write_variant(tmp_path / "a.trr", shift_time(-1e3), n_frames=3)
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

    def test_restart_starting_at_the_previous_files_start_raises(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(3, 6))
        b = write_frames(tmp_path / "b.trr", range(3, 8))
        with pytest.raises(ValueError, match=r"b\.trr starts at step 1500, at or before the start of .*a\.trr"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_restarts_are_joined_by_step_when_times_coincide(self, tmp_path, universe):
        # Late in a long run, single-precision times of neighbouring frames
        # are equal; the steps still tell them apart.
        def late(ts):
            ts.time = ts.time + 1e7
            ts.data["step"] = ts.data["step"] + 1_000_000
        a = write_frames(tmp_path / "a.trr", range(0, 8), late)
        b = write_frames(tmp_path / "b.trr", range(5, 10), late)
        assert MD.Universe(TPR, a).trajectory[5].time == MD.Universe(TPR, a).trajectory[4].time
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 10
        for k, frame in enumerate(traj.iter_frames()):
            np.testing.assert_array_equal(frame.forces, universe.trajectory[k].forces)

    @pytest.mark.filterwarnings("ignore:Could not find netCDF4")
    def test_files_without_steps_are_joined_by_time(self, tmp_path, universe):
        a = write_frames(tmp_path / "a.ncdf", range(0, 5), forces=True)
        b = write_frames(tmp_path / "b.ncdf", range(4, 10), forces=True)
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 10
        for k, frame in enumerate(traj.iter_frames()):
            np.testing.assert_allclose(frame.forces, universe.trajectory[k].forces, rtol=1e-5)
        with pytest.raises(ValueError, match=r"c\.ncdf starts at time 0\.0 ps, at or before the start of .*b\.ncdf \(time 2\.0 ps\); times must continue from file to file, as in a run and its restarts\. Load independent runs as separate trajectories"):
            MDATrajectory([b, write_frames(tmp_path / "c.ncdf", range(0, 3), forces=True)], TPR,
                          temperature=300.0)

    @staticmethod
    def _dropped(record):
        return [str(w.message) for w in record if "outside every joined file's range" in str(w.message)]

    def test_replaced_frames_in_a_gap_between_later_files_warn(self, tmp_path):
        files = [write_frames(tmp_path / f"{name}.trr", frames)
                 for name, frames in (("a", range(0, 10)), ("b", range(3, 5)), ("c", range(7, 10)))]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            traj = MDATrajectory(files, TPR, temperature=300.0)
        dropped = self._dropped(record)
        assert len(dropped) == 1
        assert "b.trr replaces the frames of" in dropped[0]
        assert "2 of those frames, between steps 2500 and 3000, lie" in dropped[0]
        assert traj.frames == 8

    @pytest.mark.parametrize("frames, replacing, lost", [
        # b leaves a's last two frames uncovered; c covers what it replaces of b.
        ((range(0, 10), range(5, 7), range(6, 8)), "b.trr", "2 of those frames, between steps 4000 and 4500"),
        # b covers what it replaces of a; c leaves b's last three frames uncovered.
        ((range(0, 5), range(3, 10), range(5, 7)), "c.trr", "3 of those frames, between steps 3500 and 4500"),
    ])
    def test_each_restart_is_judged_on_its_own_replaced_frames(self, tmp_path, frames, replacing, lost):
        files = [write_frames(tmp_path / f"{name}.trr", f) for name, f in zip("abc", frames)]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            MDATrajectory(files, TPR, temperature=300.0)
        dropped = self._dropped(record)
        assert len(dropped) == 1
        assert f"{replacing} replaces" in dropped[0] and lost in dropped[0]

    @pytest.mark.parametrize("steps, bad", [
        ((0, 2500, 4500, 1000), 3),
        ((0, 500, 500, 1000), 2),  # a repeated step
        ((0, 500, 1000, 1500, 2000, 2500, 1500, 2000, 2500, 3000), 6),  # cat of a run and its restart
    ])
    def test_steps_must_increase_within_a_file(self, tmp_path, steps, bad):
        def renumber(ts):
            ts.data["step"] = steps[ts.frame]
        path = write_frames(tmp_path / "c.trr", range(len(steps)), renumber)
        with pytest.raises(ValueError, match=fr"c\.trr, frame {bad}: steps do not increase within the file \(step {steps[bad - 1]} at frame {bad - 1}, step {steps[bad]} at frame {bad}\)"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_replaced_frames_are_checked_too(self, tmp_path):
        # Steps out of order in frames a restart replaces are still an error.
        steps = {0: 0, 1: 500, 2: 2000, 3: 1000}
        def renumber(ts):
            ts.data["step"] = steps[ts.frame]
        a = write_frames(tmp_path / "a.trr", range(4), renumber)
        b = write_frames(tmp_path / "b.trr", range(3, 6))
        with pytest.raises(ValueError, match=r"a\.trr, frame 3: steps do not increase"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_steps_that_wrap_past_32_bits_are_unwrapped(self, tmp_path, universe):
        # GROMACS writes TRR steps as signed 32-bit integers.
        top = 2**31
        def near_wrap(base):
            def renumber(ts):
                step = base + 500 * ts.frame
                ts.data["step"] = step - 2**32 if step >= top else step
            return renumber
        base = top - 2000
        a = write_frames(tmp_path / "a.trr", range(0, 8), near_wrap(base))
        b = write_frames(tmp_path / "b.trr", range(6, 10), near_wrap(base))
        assert MD.Universe(TPR, a).trajectory[5].data["step"] < 0
        assert MD.Universe(TPR, b).trajectory[0].data["step"] < 0
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 10
        for k, frame in enumerate(traj.iter_frames()):
            np.testing.assert_array_equal(frame.forces, universe.trajectory[k].forces)

    def test_cut_short_final_frame_is_dropped_with_a_warning(self, tmp_path, universe):
        path = write_frames(tmp_path / "killed.trr", range(5))
        data = Path(path).read_bytes()
        Path(path).write_bytes(data[:-1000])
        with pytest.warns(UserWarning, match=r"killed\.trr: dropping the incomplete final frame \(frame 4\)"):
            traj = MDATrajectory(path, TPR, temperature=300.0)
        assert traj.frames == 4
        np.testing.assert_array_equal(traj.get_frame(-1).forces, universe.trajectory[3].forces)

    def test_frames_mdanalysis_cannot_read_are_named(self, monkeypatch):
        from MDAnalysis.coordinates.TRR import TRRReader
        traj = MDATrajectory(TRR, TPR, temperature=300.0)
        read_frame = TRRReader._read_frame

        def failing(self, frame):
            if frame == 3:
                raise OSError("TRR read error = float")
            return read_frame(self, frame)

        monkeypatch.setattr(TRRReader, "_read_frame", failing)
        with pytest.raises(ValueError, match=r"MDAnalysis cannot read .*prod\.trr, frame 3: OSError: TRR read error = float"):
            traj.get_frame(3)
        with pytest.raises(ValueError, match=r"prod\.trr, frame 3: OSError"):
            list(traj.iter_frames())

    @pytest.mark.filterwarnings("ignore:Could not find netCDF4")
    def test_trr_and_netcdf_are_joined_by_time(self, tmp_path, universe):
        a = write_frames(tmp_path / "a.trr", range(0, 5))
        b = write_frames(tmp_path / "b.ncdf", range(4, 10), forces=True)
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 10
        for k, frame in enumerate(traj.iter_frames()):
            np.testing.assert_allclose(frame.forces, universe.trajectory[k].forces, rtol=1e-5)

    def test_readers_must_match_the_files(self, tmp_path, monkeypatch):
        from revelsMD.trajectories import mda
        a = write_frames(tmp_path / "a.trr", range(0, 5))
        b = write_frames(tmp_path / "b.trr", range(5, 10))
        monkeypatch.setattr(mda, "_readers", lambda trajectory: [trajectory])
        with pytest.raises(RuntimeError, match="MDAnalysis opened 1 readers for 2 files"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_joined_trajectory_is_rewound(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(0, 8))
        b = write_frames(tmp_path / "b.trr", range(4, 10))
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.mdanalysis_universe.trajectory.frame == 0

    def test_restart_ending_before_the_frames_it_replaces_warns(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(0, 8))
        b = write_frames(tmp_path / "b.trr", range(3, 6))
        with pytest.warns(UserWarning, match=r"b\.trr replaces the frames of .*a\.trr from step 1500 on; 2 of those frames, between steps 3000 and 3500, lie outside"):
            traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 6

    def test_file_starting_before_the_previous_one_raises(self, tmp_path):
        a = write_frames(tmp_path / "a.trr", range(3, 6))
        b = write_frames(tmp_path / "b.trr", range(0, 5))
        with pytest.raises(ValueError, match=r"b\.trr starts at step 0, at or before the start of .*a\.trr \(step 1500\); steps must continue from file to file, as in a run and its restarts\. Load independent runs as separate trajectories"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_errors_in_a_list_name_the_file_and_its_frame(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        def later_without_forces(ts):
            drop_forces(ts)
            shift_time(1e3)(ts)
        partial = write_variant(tmp_path / "partial.trr", later_without_forces)
        traj = MDATrajectory([TRR, partial], TPR, temperature=300.0)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            traj.get_frame(11)
        assert traj.get_frame(10).forces.shape == traj.get_frame(9).forces.shape
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            traj.get_frame(-2)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            list(traj.iter_frames(start=10))

    def test_errors_at_the_first_frame_of_a_later_file_name_it(self, tmp_path):
        def first_without_forces(ts):
            if ts.frame == 0:
                ts.has_forces = False
            shift_time(1e3)(ts)
        partial = write_variant(tmp_path / "partial.trr", first_without_forces)
        traj = MDATrajectory([TRR, partial], TPR, temperature=300.0)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 0: no forces"):
            traj.get_frame(10)

    def test_get_frame_needs_an_integer(self, traj):
        with pytest.raises(TypeError):
            traj.get_frame(1.0)

    def test_same_file_twice_raises(self):
        with pytest.raises(ValueError, match="is given twice"):
            MDATrajectory([TRR, TRR], TPR, temperature=300.0)

    def test_unordered_files_raise(self):
        with pytest.raises(TypeError, match="not set"):
            MDATrajectory({TRR}, TPR, temperature=300.0)

    def test_topology_cut_short_raises_runtime_error(self, tmp_path):
        # MDAnalysis raises StopIteration for this file; it must not escape,
        # where it would silently end a caller's loop.
        cut = tmp_path / "cut.gro"
        cut.write_bytes((WATER / "prod.gro").read_bytes()[:6])
        with pytest.raises(RuntimeError, match=r"MDAnalysis cannot read .*cut\.gro with .*prod\.trr: StopIteration"):
            MDATrajectory(TRR, str(cut), temperature=300.0)
        loaded = list(map(lambda top: MDATrajectory(TRR, top, temperature=300.0), [TPR]))
        assert len(loaded) == 1
        with pytest.raises(RuntimeError):
            list(map(lambda top: MDATrajectory(TRR, top, temperature=300.0), [TPR, str(cut), TPR]))

    @pytest.mark.parametrize("name, data, raised", [
        ("empty.tpr", b"", "EOFError"),
        ("cut.tpr", (WATER / "prod.tpr").read_bytes()[:200], "EOFError"),
    ])
    def test_topologies_mdanalysis_cannot_parse(self, tmp_path, name, data, raised):
        path = tmp_path / name
        path.write_bytes(data)
        with pytest.raises(RuntimeError, match=fr"MDAnalysis cannot read .*{name} with .*: {raised}"):
            MDATrajectory(TRR, str(path), temperature=300.0)

    # MDAnalysis's readers fail while closing a file they could not open.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_trajectory_mdanalysis_cannot_parse_is_named_in_a_list(self, tmp_path):
        good = write_frames(tmp_path / "good.trr", range(0, 5))
        bad = tmp_path / "bad.trr"
        bad.write_bytes(Path(TRR).read_bytes()[:40])
        with pytest.raises(RuntimeError, match=r"MDAnalysis cannot read .*prod\.tpr with .*bad\.trr: OSError"):
            MDATrajectory([good, str(bad)], TPR, temperature=300.0)

    def test_unreadable_and_directory_files_raise_the_os_error(self, tmp_path):
        import os
        directory = tmp_path / "x.trr"
        directory.mkdir()
        with pytest.raises(IsADirectoryError, match="x.trr"):
            MDATrajectory(str(directory), TPR, temperature=300.0)
        if os.name == "posix" and os.geteuid() != 0:
            locked = tmp_path / "locked.trr"
            locked.write_bytes(Path(TRR).read_bytes())
            locked.chmod(0)
            try:
                with pytest.raises(PermissionError, match="locked.trr"):
                    MDATrajectory(str(locked), TPR, temperature=300.0)
            finally:
                locked.chmod(0o600)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_tng_without_pytng_raises_import_error(self, tmp_path):
        import importlib.util
        if importlib.util.find_spec("pytng") is not None:
            pytest.skip("pytng is installed")
        junk = tmp_path / "x.tng"
        junk.write_bytes(b"not a tng file")
        with pytest.raises(ImportError):
            MDATrajectory(str(junk), TPR, temperature=300.0)

    # MDAnalysis wraps the stream in another NamedStream, and warns.
    @pytest.mark.filterwarnings("ignore:Constructed NamedStream from a NamedStream")
    def test_bytes_and_stream_topologies(self, universe):
        from io import StringIO
        from MDAnalysis.lib.util import NamedStream
        assert MDATrajectory(TRR, TPR.encode(), temperature=300.0).frames == 10
        stream = NamedStream(StringIO((WATER / "prod.gro").read_text()), "prod.gro")
        assert MDATrajectory(TRR, stream, temperature=300.0).frames == 10

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing.tpr"):
            MDATrajectory(TRR, str(tmp_path / "missing.tpr"), temperature=300.0)
        with pytest.raises(FileNotFoundError, match="missing.trr"):
            MDATrajectory([TRR, str(tmp_path / "missing.trr")], TPR, temperature=300.0)

    def test_files_mdanalysis_cannot_parse(self, tmp_path):
        dump = tmp_path / "small.lammpsdump"
        dump.write_bytes(LAMMPS_DUMP.read_bytes())
        with pytest.raises(RuntimeError, match=r"MDAnalysis cannot read .*prod\.tpr with .*small\.lammpsdump: ValueError: .*same number of atoms"):
            MDATrajectory(str(dump), TPR, temperature=300.0)

    def test_unknown_atom_name_raises(self, traj):
        with pytest.raises(ValueError, match="no atoms named 'Na'"):
            traj.get_indices("Na")


class TestJoinRestarts:
    """join_restarts on step arrays, against a plain list join."""

    def test_random_joins_match_a_plain_list_join(self):
        import random
        from revelsMD.trajectories._mda_steps import join_restarts

        def plain_join(files):
            joined, previous, replacements = [], None, []
            for path, steps in files:
                if previous is not None and steps[0] <= previous[1][0]:
                    return None, []
                replaced = [t for p, t in joined if t >= steps[0]]
                if replaced:
                    replacements.append((path, previous[0], steps[0], replaced))
                joined = [(p, t) for p, t in joined if t < steps[0]] + [(path, t) for t in steps]
                previous = (path, steps)
            return joined, replacements

        rng = random.Random(20261016)
        for case in range(400):
            files, start = [], 0
            for k in range(rng.randint(1, 5)):
                start = max(0, start + rng.choices([-100, 0, 100, 200, 300, 700], [1, 1, 3, 3, 3, 2])[0])
                interval = rng.choice([10, 50, 100, 200])
                files.append((f"f{k}", [start + interval * j for j in range(rng.randint(1, 8))]))
            expected, replacements = plain_join(files)
            arrays = [np.array(steps, dtype=np.int64) for _, steps in files]
            paths = [path for path, _ in files]
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                if expected is None:
                    with pytest.raises(ValueError, match="at or before the start of"):
                        join_restarts(arrays, paths, "step")
                    continue
                kept = join_restarts(arrays, paths, "step")
            got = [(path, int(t)) for (path, _), array, n in zip(files, arrays, kept) for t in array[:n]]
            assert got == expected
            spans = {}
            for path, t in expected:
                lo, hi = spans.get(path, (t, t))
                spans[path] = (min(lo, t), max(hi, t))
            predicted = []
            for path, replaced_path, first, replaced in replacements:
                lost = [t for t in replaced if not any(lo <= t <= hi for lo, hi in spans.values())]
                if lost:
                    predicted.append((path, replaced_path, first, len(lost), lost[0], lost[-1]))
            reported = []
            for w in record:
                message = str(w.message)
                path, replaced_path = message.split(" replaces the frames of ")[0], message.split(" replaces the frames of ")[1].split(" from step ")[0]
                first = int(message.split(" from step ")[1].split(" on;")[0])
                if "the frame at step" in message:
                    t = int(message.split("the frame at step ")[1].split(" lies")[0])
                    reported.append((path, replaced_path, first, 1, t, t))
                else:
                    count = int(message.split(" on; ")[1].split(" of those frames")[0])
                    lo = int(message.split("between steps ")[1].split(" and ")[0])
                    hi = int(message.split(" and ")[1].split(", lie")[0])
                    reported.append((path, replaced_path, first, count, lo, hi))
            assert reported == predicted

    def test_files_without_steps_or_times_cannot_be_joined(self):
        from revelsMD.trajectories._mda_steps import FrameKeys, order_keys
        keys = [FrameKeys(None, None, 3), FrameKeys(None, np.array([0.0, 1.0]), 2)]
        with pytest.raises(ValueError, match="a.nc: the file records neither steps nor times"):
            order_keys(keys, ["a.nc", "b.nc"])
        unit, arrays = order_keys(keys[:1], ["a.nc"])
        assert unit == "time" and arrays[0].tolist() == [0, 1, 2]

    def test_equal_single_precision_times_name_the_cause(self):
        from revelsMD.trajectories._mda_steps import join_restarts
        times = np.array([3e5, 3e5 + 0.02, 3e5 + 0.04], dtype=np.float32).astype(np.float64)
        with pytest.raises(ValueError, match="times do not increase within the file .*single precision"):
            join_restarts([times], ["a.ncdf"], "time")

    def test_mixed_formats_order_by_time(self):
        from revelsMD.trajectories._mda_steps import FrameKeys, order_keys
        keys = [FrameKeys(np.array([0, 500]), np.array([0.0, 0.5]), 2),
                FrameKeys(None, np.array([1.0, 1.5]), 2)]
        unit, arrays = order_keys(keys, ["a.trr", "b.ncdf"])
        assert unit == "time" and arrays[0].tolist() == [0.0, 0.5]


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

    def test_h5md_steps_that_cannot_be_read_cannot_be_checked(self):
        from MDAnalysis.coordinates.H5MD import H5MDReader
        from revelsMD.trajectories.mda import _check_reader

        class Particles(dict):
            def __getitem__(self, key):
                raise OSError("unreadable dataset")

        class Reader(H5MDReader):
            def __init__(self):
                pass

            def __del__(self):
                pass

        reader = Reader()
        reader.filename = "broken.h5md"
        reader.units = {"length": "Angstrom", "force": "kJ/(mol*Angstrom)"}
        reader._particle_group = Particles(position=None, force=None)
        with pytest.raises(RuntimeError, match="broken.h5md: cannot check .*unreadable dataset"):
            _check_reader(reader)

    def test_h5md_steps_stored_as_a_group_cannot_be_checked(self, tmp_path):
        path = write_variant(tmp_path / "groupsteps.h5md")
        with h5py.File(path, "r+") as f:
            group = f["particles/trajectory/force"]
            del group["step"]
            group.create_group("step")
        with pytest.raises(RuntimeError, match="groupsteps.h5md: cannot check"):
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

