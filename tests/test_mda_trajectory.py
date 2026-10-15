"""MDATrajectory on the committed water trajectory and variants written from it."""

from pathlib import Path

import MDAnalysis as MD
import numpy as np
import pytest
from MDAnalysis.lib.mdamath import triclinic_vectors

from revelsMD.trajectories import MDATrajectory
from revelsMD.trajectories._base import compute_beta

WATER = Path(__file__).parent / "data" / "example_4_water"
TPR = str(WATER / "prod.tpr")
TRR = str(WATER / "prod.trr")


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
        np.testing.assert_allclose(traj.cell_matrix, triclinic_vectors(universe.dimensions))
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
        assert len(list(traj.iter_frames(start=2, stop=8, stride=2))) == 3
        assert len(list(traj.iter_frames(stop=-1))) == 9

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
        np.testing.assert_allclose(traj.cell_matrix, triclinic_vectors(dims.astype(np.float32)))
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

    def test_changing_box_raises_when_read(self, tmp_path):
        def grow(ts):
            if ts.frame == 2:
                ts.dimensions = ts.dimensions * np.array([1.01, 1, 1, 1, 1, 1], dtype=np.float32)
        path = write_variant(tmp_path / "npt.trr", grow)
        traj = MDATrajectory(path, TPR, temperature=300.0)
        with pytest.raises(ValueError, match="npt.trr, frame 2: the cell differs"):
            list(traj.iter_frames())

    def test_trajectory_without_a_box_raises_at_construction(self, tmp_path):
        path = write_variant(tmp_path / "nobox.trr", lambda ts: setattr(ts, "dimensions", np.zeros(6)))
        with pytest.raises(ValueError, match="nobox.trr: the trajectory has no simulation box"):
            MDATrajectory(path, TPR, temperature=300.0)

    def test_unreadable_files(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to load MDAnalysis Universe"):
            MDATrajectory(TRR, str(tmp_path / "missing.tpr"), temperature=300.0)
