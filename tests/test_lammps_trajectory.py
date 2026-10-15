"""LammpsTrajectory on small dump files written for each case."""

import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from lammps_dumps import IDS_SORTED, forces, frame, positions, rows, write
from revelsMD.trajectories import LammpsTrajectory
from revelsMD.trajectories._base import compute_beta

SMALL = Path(__file__).parent / "data" / "lammps_small"
DATA = str(SMALL / "data.small.data")


@contextmanager
def open_file_limit(limit):
    """Lower the soft open-file limit for the duration of the block."""
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def load(dumps, **kwargs):
    kwargs.setdefault("atom_style", "id type x y z")
    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("units", "lj")
    return LammpsTrajectory(dumps, DATA, **kwargs)


@pytest.fixture
def six_frames(tmp_path):
    return write(tmp_path / "six.dump", *(frame(100 * k, float(k)) for k in range(6)))


class TestReading:

    def test_frames_and_cell(self, six_frames):
        traj = load(six_frames)
        assert traj.frames == 6
        np.testing.assert_array_equal(traj.cell_matrix, np.diag([10.0, 10.0, 10.0]))

    def test_start_stop_stride(self, six_frames):
        got = list(load(six_frames).iter_frames(start=1, stop=5, stride=2))
        assert len(got) == 2
        np.testing.assert_array_equal(got[0].positions, positions(1.0))
        np.testing.assert_array_equal(got[1].positions, positions(3.0))
        np.testing.assert_array_equal(got[1].forces, forces(300))

    def test_negative_stop(self, six_frames):
        assert len(list(load(six_frames).iter_frames(stop=-1))) == 5

    def test_get_frame_in_any_order_matches_iteration(self, six_frames):
        traj = load(six_frames)
        frames = list(traj.iter_frames())
        for k in (4, 0, 5, 2, -1):
            np.testing.assert_array_equal(traj.get_frame(k).positions, frames[k].positions)
            np.testing.assert_array_equal(traj.get_frame(k).forces, frames[k].forces)

    def test_get_indices(self, six_frames):
        traj = load(six_frames)
        np.testing.assert_array_equal(traj.get_indices("1"), [0, 2, 4])
        np.testing.assert_array_equal(traj.get_indices("2"), [1, 3, 5])

    def test_temperature_and_beta(self, six_frames):
        traj = load(six_frames, temperature=350.0, units="real")
        assert traj.temperature == 350.0
        assert traj.beta == pytest.approx(compute_beta("real", 350.0), rel=1e-12)

    def test_temperature_is_required(self, six_frames):
        with pytest.raises(TypeError):
            LammpsTrajectory(six_frames, DATA)

    def test_path_objects_and_tuples(self, tmp_path):
        a = Path(write(tmp_path / "a.dump", frame(0)))
        b = Path(write(tmp_path / "b.dump", frame(100, 1.0)))
        traj = LammpsTrajectory((a, b), Path(DATA), temperature=1.0, units="lj")
        assert traj.frames == 2

    def test_unreadable_topology(self, six_frames, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to load LAMMPS topology"):
            LammpsTrajectory(six_frames, str(tmp_path / "missing.data"),
                             temperature=1.0, units="lj", atom_style="id type x y z")


class TestMultipleFiles:

    def test_overlap_takes_the_later_file(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0), frame(200, 2.0))
        b = write(tmp_path / "b.dump", frame(100, 50.0), frame(300, 52.0))
        traj = load([a, b])
        assert traj.frames == 3
        got = list(traj.iter_frames())
        for g, shift in zip(got, (0.0, 50.0, 52.0)):
            np.testing.assert_array_equal(g.positions, positions(shift))
        np.testing.assert_array_equal(traj.get_frame(1).positions, positions(50.0))

    def test_boundary_frame_counted_once(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", frame(100, 1.0), frame(200, 2.0))
        assert load([a, b]).frames == 3

    def test_killed_run_final_frame_is_dropped_with_a_warning(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0), frame(200, 2.0)[:-25])
        with pytest.warns(UserWarning, match="incomplete final frame"):
            traj = load(a)
        assert traj.frames == 2
        np.testing.assert_array_equal(traj.get_frame(1).positions, positions(1.0))

    def test_missing_final_newline_drops_the_final_frame(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0).rstrip("\n"))
        with pytest.warns(UserWarning, match="incomplete final frame"):
            assert load(a).frames == 1


class TestOpenFiles:

    def test_one_file_per_frame_does_not_exhaust_file_handles(self, tmp_path):
        n_files = 300
        dumps = [write(tmp_path / f"d{k:03d}.dump", frame(100 * k, float(k % 7)))
                 for k in range(n_files)]
        traj = load(dumps)
        with open_file_limit(64):
            got = list(traj.iter_frames())
        assert len(got) == n_files
        np.testing.assert_array_equal(got[-1].positions, positions(float((n_files - 1) % 7)))


class TestValidation:

    def test_changing_box_raises(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0),
                  frame(100, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n"))
        with pytest.raises(ValueError, match="a.dump, timestep 100: the cell differs"):
            load(a)

    def test_non_periodic_boundaries_raise(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0, box_item="ff pp pp"))
        with pytest.raises(ValueError, match="periodic"):
            load(a)

    def test_missing_force_columns_raise_at_construction(self, tmp_path):
        atom_rows = [" ".join(r.split()[:5]) for r in rows(0)]
        a = write(tmp_path / "a.dump", frame(0, columns="id type x y z", atom_rows=atom_rows))
        with pytest.raises(ValueError, match="lack fx fy fz"):
            load(a)

    def test_atom_count_must_match_the_topology(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0, atom_rows=rows(0)[:5], n_atoms=5))
        with pytest.raises(ValueError, match="5 atoms, but the topology has 6"):
            load(a)


class TestCoordinateColumns:

    def test_unwrapped_coordinates(self, tmp_path):
        images = {20: (1, 0, -1), 3: (0, 2, 0), 40: (-1, -1, 3), 12: (0, 0, 0), 21: (2, 0, 0), 7: (0, -3, 1)}
        atom_rows = []
        for r in rows(0):
            i, t, x, y, z, *f = r.split()
            ix, iy, iz = images[int(i)]
            atom_rows.append(" ".join([i, t, repr(float(x) + 10 * ix), repr(float(y) + 10 * iy),
                                       repr(float(z) + 10 * iz)] + f))
        a = write(tmp_path / "u.dump", frame(0, columns="id type xu yu zu fx fy fz", atom_rows=atom_rows))
        got = next(load(a).iter_frames()).positions
        shifts = np.array([images[i] for i in IDS_SORTED]) * 10.0
        np.testing.assert_array_equal(got, positions() + shifts)
        np.testing.assert_allclose(np.mod(got, 10.0), np.mod(positions(), 10.0), atol=1e-12)

    def test_scaled_coordinates(self, tmp_path):
        atom_rows = []
        for r in rows(0):
            i, t, x, y, z, *f = r.split()
            atom_rows.append(" ".join([i, t, repr(float(x) / 10), repr(float(y) / 10),
                                       repr(float(z) / 10)] + f))
        a = write(tmp_path / "s.dump", frame(0, columns="id type xs ys zs fx fy fz", atom_rows=atom_rows))
        got = next(load(a).iter_frames()).positions
        np.testing.assert_allclose(got, positions(), rtol=1e-12, atol=1e-12)


class TestTopology:

    def test_atom_style_inferred_matches_explicit(self, six_frames):
        inferred = load(six_frames, atom_style=None)
        explicit = load(six_frames)
        named = load(six_frames, atom_style="atomic")
        for traj in (inferred, named):
            np.testing.assert_array_equal(
                traj.mdanalysis_universe.atoms.types, explicit.mdanalysis_universe.atoms.types)
            np.testing.assert_array_equal(traj.get_indices("2"), explicit.get_indices("2"))

    def test_topology_without_a_recognised_extension(self, six_frames, tmp_path):
        renamed = tmp_path / "data_small"
        shutil.copy(DATA, renamed)
        traj = LammpsTrajectory(six_frames, str(renamed), temperature=1.0, units="lj")
        np.testing.assert_array_equal(traj.get_indices("1"), [0, 2, 4])

    def test_full_style_data_file(self):
        lj = Path(__file__).parent / "data" / "example_1_LJ"
        inferred = LammpsTrajectory(str(lj / "dump.nh.lammps"), str(lj / "data.fin.nh.data"),
                                    temperature=1.35, units="lj")
        explicit = LammpsTrajectory(str(lj / "dump.nh.lammps"), str(lj / "data.fin.nh.data"),
                                    temperature=1.35, units="lj",
                                    atom_style="id resid type q x y z ix iy iz")
        np.testing.assert_array_equal(
            inferred.mdanalysis_universe.atoms.types, explicit.mdanalysis_universe.atoms.types)
        np.testing.assert_array_equal(
            inferred.mdanalysis_universe.atoms.ids, explicit.mdanalysis_universe.atoms.ids)
