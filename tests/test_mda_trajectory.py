"""MDATrajectory on the committed water trajectory and variants written from it."""

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


def write_frames(path, frames=range(3), edit=None, **writer_options):
    """Write frames of the water trajectory (the first three by default), applying edit(ts) to each."""
    universe = MD.Universe(TPR, TRR)
    with MD.Writer(str(path), n_atoms=universe.atoms.n_atoms, **writer_options) as writer:
        for ts in universe.trajectory[list(frames)]:
            if edit is not None:
                edit(ts)
            writer.write(universe.atoms)
    return str(path)


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
        path = write_frames(tmp_path / "tri.trr", edit=lambda ts: setattr(ts, "dimensions", dims))
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
        path = write_frames(tmp_path / "positions.xtc")
        with pytest.raises(ValueError, match="positions.xtc: the trajectory has no forces"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_frame_without_forces_raises_when_read(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        path = write_frames(tmp_path / "partial.trr", edit=drop_forces)
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
        path = write_frames(tmp_path / "forces.trr", edit=lambda ts: setattr(ts, "has_positions", False))
        with pytest.raises(ValueError, match="forces.trr: the trajectory has no positions"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_frame_without_positions_raises_when_read(self, tmp_path):
        def drop_positions(ts):
            if ts.frame == 1:
                ts.has_positions = False
        path = write_frames(tmp_path / "partial.trr", edit=drop_positions)
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
        path = write_frames(tmp_path / "npt.trr", edit=grow)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="npt.trr, frame 2: the cell differs"):
            list(traj.iter_frames())
        with pytest.raises(ValueError, match="npt.trr, frame 2: the cell differs"):
            traj.get_frame(2)

    def test_changing_box_angle_raises_when_read(self, tmp_path):
        def shear(ts):
            if ts.frame == 1:
                ts.dimensions = np.array([40, 40, 40, 90, 90, 89.5], dtype=np.float32)
        path = write_frames(tmp_path / "sheared.trr", edit=shear)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="sheared.trr, frame 1: the cell differs"):
            traj.get_frame(1)

    def test_frame_without_a_box_raises_when_read(self, tmp_path):
        def drop_box(ts):
            if ts.frame == 1:
                ts.dimensions = np.zeros(6)
        path = write_frames(tmp_path / "boxless.trr", edit=drop_box)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="boxless.trr, frame 1: no simulation box"):
            traj.get_frame(1)

    def test_non_finite_forces_raise_when_read(self, tmp_path):
        def blow_up(ts):
            if ts.frame == 1:
                ts.forces[3, 1] = np.nan
        path = write_frames(tmp_path / "nan.trr", edit=blow_up)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="nan.trr, frame 1: the forces contain NaN"):
            traj.get_frame(1)

    def test_degenerate_box_raises_naming_the_file(self, tmp_path):
        dims = np.array([1e-5, 1e-5, 1e-5, 90.0, 90.0, 90.0])
        path = write_frames(tmp_path / "flat.trr", edit=lambda ts: setattr(ts, "dimensions", dims))
        with pytest.raises(ValueError, match=r"flat\.trr: Cell matrix must have positive volume"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_trajectory_without_a_box_raises_at_construction(self, tmp_path):
        path = write_frames(tmp_path / "nobox.trr", edit=lambda ts: setattr(ts, "dimensions", np.zeros(6)))
        with pytest.raises(ValueError, match="nobox.trr: the trajectory has no simulation box"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_files_are_concatenated_as_given(self, tmp_path, universe):
        # Frames the files share, as -noappend restarts repeat, are kept in each.
        a = write_frames(tmp_path / "a.trr", range(0, 5))
        b = write_frames(tmp_path / "b.trr", range(4, 10))
        traj = MDATrajectory([a, b], TPR, temperature=300.0)
        assert traj.frames == 11
        expected = list(range(0, 5)) + list(range(4, 10))
        for frame, k in zip(traj.iter_frames(), expected, strict=True):
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

    def test_readers_must_match_the_files(self, tmp_path, monkeypatch):
        from revelsMD.trajectories import mda
        a = write_frames(tmp_path / "a.trr", range(0, 5))
        b = write_frames(tmp_path / "b.trr", range(5, 10))
        monkeypatch.setattr(mda, "_readers", lambda trajectory: [trajectory])
        with pytest.raises(RuntimeError, match="MDAnalysis opened 1 readers for 2 files"):
            MDATrajectory([a, b], TPR, temperature=300.0)

    def test_errors_in_a_list_name_the_file_and_its_frame(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        partial = write_frames(tmp_path / "partial.trr", edit=drop_forces)
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
        partial = write_frames(tmp_path / "partial.trr", edit=first_without_forces)
        traj = MDATrajectory([TRR, partial], TPR, temperature=300.0)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 0: no forces"):
            traj.get_frame(10)

    # Whatever MDAnalysis raises for a file it cannot parse becomes a
    # RuntimeError naming the file. StopIteration in particular must not
    # escape, where it would silently end a caller's loop.
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    @pytest.mark.parametrize("name, source, length, which, raised", [
        ("cut.gro", "prod.gro", 6, "topology", "StopIteration"),
        ("empty.tpr", "prod.tpr", 0, "topology", "EOFError"),
        ("cut.tpr", "prod.tpr", 200, "topology", "EOFError"),
        ("cut.trr", "prod.trr", 40, "trajectory", "OSError"),
    ])
    def test_files_mdanalysis_cannot_parse_are_named(self, tmp_path, name, source, length, which, raised):
        cut = tmp_path / name
        cut.write_bytes((WATER / source).read_bytes()[:length])
        trajectory, topology = (TRR, str(cut)) if which == "topology" else (str(cut), TPR)
        with pytest.raises(RuntimeError, match=fr"MDAnalysis cannot read .*{name}.*: {raised}"):
            MDATrajectory(trajectory, topology, temperature=300.0)

    def test_stop_iteration_does_not_end_a_callers_loop(self, tmp_path):
        cut = tmp_path / "cut.gro"
        cut.write_bytes((WATER / "prod.gro").read_bytes()[:6])
        with pytest.raises(RuntimeError):
            list(map(lambda top: MDATrajectory(TRR, top, temperature=300.0), [TPR, str(cut), TPR]))

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
        path = write_frames(tmp_path / "water.h5md")
        traj = MDATrajectory(path, TPR, temperature=300.0)
        frame = traj.get_frame(2)
        np.testing.assert_allclose(frame.forces, universe.trajectory[2].forces, rtol=1e-6)

    def test_h5md_without_a_force_unit_is_rejected(self, tmp_path):
        path = write_frames(tmp_path / "unitless.h5md")
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
        good = write_frames(tmp_path / "good.h5md", range(4))
        bad = write_frames(tmp_path / "bad.h5md", range(4))
        with h5py.File(bad, "r+") as f:
            group = f["particles/trajectory/force"]
            steps = group["step"][()]
            del group["step"]
            group["step"] = steps * 2
        with pytest.raises(ValueError, match="bad.h5md: the forces are saved at different steps"):
            MDATrajectory([good, bad], TPR, temperature=300.0)

    def test_h5md_without_force_steps_cannot_be_checked(self, tmp_path):
        path = write_frames(tmp_path / "nosteps.h5md")
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
        path = write_frames(tmp_path / "groupsteps.h5md")
        with h5py.File(path, "r+") as f:
            group = f["particles/trajectory/force"]
            del group["step"]
            group.create_group("step")
        with pytest.raises(RuntimeError, match="groupsteps.h5md: cannot check"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_h5md_step_check_fails_closed(self, tmp_path):
        from revelsMD.trajectories.mda import _check_reader
        reader = MD.Universe(TPR, write_frames(tmp_path / "water.h5md")).trajectory
        _check_reader(reader)
        del reader._particle_group
        with pytest.raises(RuntimeError, match="cannot check that this H5MD file"):
            _check_reader(reader)

    def test_h5md_with_forces_at_other_steps_is_rejected(self, tmp_path):
        path = write_frames(tmp_path / "steps.h5md", range(4))
        with h5py.File(path, "r+") as f:
            # The writer links one step dataset into every group.
            group = f["particles/trajectory/force"]
            steps = group["step"][()]
            del group["step"]
            group["step"] = steps * 2
        with pytest.raises(ValueError, match="steps.h5md: the forces are saved at different steps"):
            MDATrajectory(path, TPR, temperature=300.0)

