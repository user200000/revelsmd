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


def write_variant(path, edit=None, n_frames=3):
    """Write the first frames of the water trajectory, applying edit(ts) to each."""
    universe = MD.Universe(TPR, TRR)
    with MD.Writer(str(path), n_atoms=universe.atoms.n_atoms) as writer:
        for ts in universe.trajectory[:n_frames]:
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

    def test_sliced_iteration_returns_the_selected_frames(self, traj, universe):
        frames = list(traj.iter_frames(start=2, stop=8, stride=2))
        assert len(frames) == 3
        for frame, k in zip(frames, (2, 4, 6)):
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

    def test_trajectory_without_a_box_raises_at_construction(self, tmp_path):
        path = write_variant(tmp_path / "nobox.trr", lambda ts: setattr(ts, "dimensions", np.zeros(6)))
        with pytest.raises(ValueError, match="nobox.trr: the trajectory has no simulation box"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_list_of_files_is_joined(self, tmp_path, universe):
        first = write_variant(tmp_path / "a.trr", n_frames=3)
        traj = MDATrajectory([first, TRR], TPR, temperature=300.0)
        assert traj.frames == 13
        np.testing.assert_array_equal(traj.get_frame(3).forces, universe.trajectory[0].forces)

    def test_errors_in_a_list_name_the_file_and_its_frame(self, tmp_path):
        def drop_forces(ts):
            if ts.frame == 1:
                ts.has_forces = False
        partial = write_variant(tmp_path / "partial.trr", drop_forces)
        traj = MDATrajectory([TRR, partial], TPR, temperature=300.0)
        with pytest.raises(ValueError, match=r"partial\.trr, frame 1: no forces"):
            traj.get_frame(11)
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

    def test_unreadable_files(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to load MDAnalysis Universe"):
            MDATrajectory(TRR, str(tmp_path / "missing.tpr"), temperature=300.0)

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

    def test_h5md_with_units_is_read(self, tmp_path, universe):
        path = write_variant(tmp_path / "water.h5md")
        traj = MDATrajectory(path, TPR, temperature=300.0)
        frame = traj.get_frame(2)
        np.testing.assert_allclose(frame.forces, universe.trajectory[2].forces, rtol=1e-6)

    def test_h5md_without_a_force_unit_is_rejected(self, tmp_path):
        path = write_variant(tmp_path / "unitless.h5md")
        with h5py.File(path, "r+") as f:
            del f["particles/trajectory/force/value"].attrs["unit"]
        with pytest.raises(RuntimeError, match="H5MD file must have readable units"):
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

