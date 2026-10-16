"""LammpsTrajectory on small dump files written for each case."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from file_limits import open_descriptors, open_file_limit
from lammps_dumps import IDS_SORTED, atom_type, forces, frame, positions, rows, write
from traced_memory import bytes_per_extra_item
from revelsMD.trajectories import LammpsTrajectory
from revelsMD.trajectories._base import compute_beta

SMALL = Path(__file__).parent / "data" / "lammps_small"
DATA = str(SMALL / "data.small.data")


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

    def test_get_frame_names_negative_frames_by_position(self, tmp_path):
        bad_ids = [r.replace("40 2", "41 2", 1) for r in rows(200)]
        a = write(tmp_path / "a.dump", frame(0), frame(100), frame(200, atom_rows=bad_ids))
        with pytest.raises(ValueError, match=r"a\.dump, frame 2 \(timestep 200\)"):
            load(a).get_frame(-1)
        with pytest.raises(IndexError, match="frame -4 is out of range for 3 frames"):
            load(a).get_frame(-4)

    def test_get_frame_needs_an_integer(self, six_frames):
        traj = load(six_frames)
        with pytest.raises(TypeError):
            traj.get_frame(2.0)
        np.testing.assert_array_equal(traj.get_frame(np.int64(2)).positions, positions(2.0))

    def test_unknown_type_raises(self, six_frames):
        with pytest.raises(ValueError, match="no atoms of type '3'"):
            load(six_frames).get_indices("3")

    @pytest.mark.parametrize("atom_style", ["id type x y z", None])
    def test_missing_topology(self, six_frames, tmp_path, atom_style):
        with pytest.raises(FileNotFoundError, match="missing.data"):
            LammpsTrajectory(six_frames, str(tmp_path / "missing.data"),
                             temperature=1.0, units="lj", atom_style=atom_style)

    def test_repeated_topology_ids_raise(self, six_frames, tmp_path):
        repeated = tmp_path / "repeated.data"
        repeated.write_text(Path(DATA).read_text().replace("\n21 1 21.0", "\n20 1 21.0"))
        with pytest.raises(RuntimeError, match=r"repeated\.data: atom ids repeat"):
            LammpsTrajectory(six_frames, str(repeated), temperature=1.0, units="lj")

    def test_binary_topology_without_a_style_raises(self, six_frames, tmp_path):
        binary = tmp_path / "binary.data"
        binary.write_bytes(b"title\n\xff\xfe\x00\x01 atoms\n")
        with pytest.raises(ValueError, match=r"binary\.data: no 'Atoms # <style>' comment"):
            LammpsTrajectory(six_frames, str(binary), temperature=1.0, units="lj")

    def test_dump_units_must_match(self, tmp_path):
        a = write(tmp_path / "a.dump", "ITEM: UNITS\nmetal\n", frame(0))
        with pytest.raises(ValueError, match=r"a\.dump: the dump was written with ITEM: UNITS metal, but units='lj'; pass units='metal'"):
            load(a)
        assert load(a, units="metal").units == "metal"

    def test_non_ascii_atom_style_names_the_file(self, six_frames, tmp_path):
        data = tmp_path / "odd.data"
        data.write_bytes(Path(DATA).read_bytes().replace(b"Atoms # atomic", "Atoms # \u00e5tomic".encode()))
        with pytest.raises(ValueError, match=r"odd\.data: atom style '\\xc3\\xa5tomic' has no known column layout"):
            LammpsTrajectory(six_frames, str(data), temperature=1.0, units="lj")

    def test_bytes_topology(self, six_frames):
        assert LammpsTrajectory(six_frames, DATA.encode(), temperature=1.0, units="lj").frames == 6

    def test_unparseable_topology(self, six_frames, tmp_path):
        broken = tmp_path / "broken.data"
        broken.write_text("title\n\n6 atoms\n\nAtoms # atomic\n\n1 x\n")
        with pytest.raises(RuntimeError, match=r"MDAnalysis cannot read .*broken\.data: "):
            LammpsTrajectory(six_frames, str(broken), temperature=1.0, units="lj")


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

    def test_negative_get_frame_across_a_file_boundary(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0), frame(200, 2.0))
        b = write(tmp_path / "b.dump", frame(300, 3.0), frame(400, 4.0))
        traj = load([a, b])
        frames = list(traj.iter_frames())
        assert len(frames) == 5
        for k in range(1, 6):
            np.testing.assert_array_equal(traj.get_frame(-k).positions, frames[-k].positions)
            np.testing.assert_array_equal(traj.get_frame(-k).forces, frames[-k].forces)
        np.testing.assert_array_equal(traj.get_frame(-1).positions, positions(4.0))
        np.testing.assert_array_equal(traj.get_frame(-3).positions, positions(2.0))

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

    def test_warnings_are_attributed_to_the_caller(self, tmp_path):
        # One file with a cut final frame, then a restart ending before the frames it replaces.
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0), frame(200), frame(300)[:-25])
        b = write(tmp_path / "b.dump", frame(100), frame(150))
        for files in (a, [a, b]):
            with pytest.warns(UserWarning) as record:
                LammpsTrajectory(files, DATA, temperature=1.0, units="lj", atom_style="atomic")
            assert [w.filename for w in record] == [__file__] * len(record)

    def test_file_with_only_a_partial_frame_first(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0)[:-25])
        b = write(tmp_path / "b.dump", frame(100, 1.0), frame(200, 2.0))
        with pytest.warns(UserWarning, match=r"a\.dump: dropping the incomplete final frame \(timestep 0\)"):
            traj = load([a, b])
        assert traj.frames == 2
        np.testing.assert_array_equal(traj.get_frame(0).positions, positions(1.0))

    def test_empty_file_from_a_job_that_wrote_nothing(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", "")
        with pytest.warns(UserWarning, match=r"b\.dump: dropping the incomplete final frame"):
            assert load([a, b]).frames == 2

    def test_files_with_different_column_orders(self, tmp_path):
        reordered = [" ".join(r.split()[:2] + r.split()[5:] + r.split()[2:5]) for r in rows(200)]
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", frame(200, columns="id type fx fy fz x y z", atom_rows=reordered))
        traj = load([a, b])
        for got in (traj.get_frame(2), list(traj.iter_frames())[2]):
            np.testing.assert_array_equal(got.positions, positions())
            np.testing.assert_array_equal(got.forces, forces(200))

    def test_restart_replacing_frames_after_an_atom_was_lost(self, tmp_path):
        lost = frame(300, atom_rows=rows(300)[:5], n_atoms=5)
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0), frame(200, 2.0), lost)
        b = write(tmp_path / "b.dump", frame(200, 20.0), frame(300, 30.0))
        traj = load([a, b])
        assert traj.frames == 4
        np.testing.assert_array_equal(traj.get_frame(3).positions, positions(30.0))


class TestOpenFiles:

    def test_one_file_per_frame_does_not_exhaust_file_handles(self, tmp_path):
        n_files = 300
        dumps = [write(tmp_path / f"d{k:03d}.dump", frame(100 * k, float(k % 7)))
                 for k in range(n_files)]
        traj = load(dumps)
        with open_file_limit(open_descriptors() + 5):
            got = list(traj.iter_frames())
        assert len(got) == n_files
        np.testing.assert_array_equal(got[-1].positions, positions(float((n_files - 1) % 7)))


class TestIndexMemory:

    def test_index_memory_per_frame_is_small(self, tmp_path):
        n = 2000
        paths = {m: write(tmp_path / f"{m}.dump", *(frame(k) for k in range(m)))
                 for m in (n, 2 * n)}
        per_frame = bytes_per_extra_item(lambda m: load(paths[m]), n)
        assert per_frame < 40, f"the index keeps {per_frame:.0f} bytes per frame"


class TestValidation:

    def test_changing_box_raises(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0),
                  frame(100, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n"))
        with pytest.raises(ValueError, match="a.dump, timestep 100: the cell differs"):
            load(a)

    def test_box_changing_between_files_raises(self, tmp_path):
        bounds = "0.0 10.5\n0.0 10.0\n0.0 10.0\n"
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", frame(200, 2.0, bounds=bounds), frame(300, 3.0, bounds=bounds))
        with pytest.raises(ValueError, match=r"b\.dump, timestep 200: the cell differs"):
            load([a, b])

    def test_box_changing_only_in_a_later_files_first_frame_names_that_frame(self, tmp_path):
        bounds = "0.0 10.5\n0.0 10.0\n0.0 10.0\n"
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", frame(200, 2.0, bounds=bounds), frame(300, 3.0))
        with pytest.raises(ValueError, match=r"b\.dump, timestep 200: the cell differs"):
            load([a, b])

    def test_non_periodic_boundaries_raise(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0, box_item="ff pp pp"))
        with pytest.raises(ValueError, match="periodic"):
            load(a)

    def test_missing_force_columns_raise_at_construction(self, tmp_path):
        atom_rows = [" ".join(r.split()[:5]) for r in rows(0)]
        a = write(tmp_path / "a.dump", frame(0, columns="id type x y z", atom_rows=atom_rows))
        with pytest.raises(ValueError, match="lack fx fy fz"):
            load(a)

    def test_zero_volume_box_raises_naming_the_file(self, tmp_path):
        z = write(tmp_path / "z.dump", frame(0, bounds="0.0 10.0\n0.0 10.0\n5.0 5.0\n"))
        with pytest.raises(ValueError, match=r"z\.dump, frame 0 \(timestep 0\): Cell matrix must have positive volume"):
            load(z)

    def test_atom_count_must_match_the_topology(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0, atom_rows=rows(0)[:5], n_atoms=5))
        with pytest.raises(ValueError, match="5 atoms, but the topology has 6"):
            load(a)

    def test_atom_count_of_a_later_file_must_match_the_topology(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", *(frame(t, atom_rows=rows(t)[:5], n_atoms=5) for t in (200, 300)))
        with pytest.raises(ValueError, match=r"b\.dump, timestep 200: 5 atoms, but the topology has 6"):
            load([a, b])

    def test_first_frame_with_the_wrong_atom_count_is_named(self, tmp_path):
        a = write(tmp_path / "a.dump", frame(0, atom_rows=rows(0)[:5], n_atoms=5), frame(100), frame(200))
        with pytest.raises(ValueError, match=r"a\.dump, timestep 0: 5 atoms, but the topology has 6"):
            load(a)

    def test_read_errors_name_the_file_and_frame(self, tmp_path):
        bad_ids = [r.replace("40 2", "41 2", 1) for r in rows(300)]
        a = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        b = write(tmp_path / "b.dump", frame(200, 2.0), frame(300, atom_rows=bad_ids))
        traj = load([a, b])
        with pytest.raises(ValueError, match=r"b\.dump, frame 3 \(timestep 300\): frame atom ids do not match"):
            traj.get_frame(3)
        with pytest.raises(ValueError, match=r"b\.dump, frame 3 \(timestep 300\)"):
            list(traj.iter_frames())

    def test_data_file_with_other_types_raises_when_read(self, six_frames, tmp_path):
        swapped = Path(DATA).read_text().replace("12 1 12.0", "12 2 12.0").replace("40 2 40.0", "40 1 40.0")
        data = tmp_path / "swapped.data"
        data.write_text(swapped)
        traj = LammpsTrajectory(six_frames, str(data), temperature=1.0, units="lj")
        with pytest.raises(ValueError, match=r"six\.dump, frame 0 \(timestep 0\): .*atom id 12 has type 1 in the dump and 2"):
            traj.get_frame(0)

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_non_finite_values_raise_when_read(self, tmp_path, value):
        broken = [r.replace(" 100 ", f" {value} ") for r in rows(100)]
        a = write(tmp_path / "a.dump", frame(0), frame(100, atom_rows=broken))
        traj = load(a)
        with pytest.raises(ValueError, match=r"a\.dump, frame 1 \(timestep 100\): the forces contain NaN"):
            traj.get_frame(1)


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

    @staticmethod
    def scaled_dump(tmp_path):
        atom_rows = []
        for r in rows(0):
            i, t, x, y, z, *f = r.split()
            atom_rows.append(" ".join([i, t, repr(float(x) / 10), repr(float(y) / 10),
                                       repr(float(z) / 10)] + f))
        return write(tmp_path / "s.dump", frame(0, columns="id type xs ys zs fx fy fz", atom_rows=atom_rows))

    def test_scaled_coordinates(self, tmp_path):
        got = next(load(self.scaled_dump(tmp_path)).iter_frames()).positions
        np.testing.assert_allclose(got, positions(), rtol=1e-12, atol=1e-12)

    def test_changing_cell_matrix_does_not_change_scaled_frames(self, tmp_path):
        traj = load(self.scaled_dump(tmp_path))
        before = traj.get_frame(0).positions
        traj.cell_matrix *= 2.0
        np.testing.assert_array_equal(traj.get_frame(0).positions, before)


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

    def test_charges_of_a_full_style_data_file(self, six_frames, tmp_path):
        # The lammps_small atoms in the full layout (id resid type charge x y z),
        # with charge id / 10.
        text = Path(DATA).read_text()
        head, _, atoms = text.partition("Atoms # atomic\n")
        full_rows = []
        for line in atoms.splitlines():
            if line.strip():
                i, t, x, y, z = line.split()
                full_rows.append(f"{i} 1 {t} {int(i) / 10} {x} {y} {z}\n")
        path = tmp_path / "data.full"
        path.write_text(head + "Atoms # full\n\n" + "".join(full_rows))

        # The charges as written, read back with plain Python and ordered by id.
        written = path.read_text().partition("Atoms # full\n")[2].split("\n")
        table = sorted((int(w[0]), w[2], float(w[3])) for w in map(str.split, written) if w)
        for style in (None, "full"):
            traj = LammpsTrajectory(six_frames, str(path), temperature=1.0, units="lj",
                                    atom_style=style)
            for atype in ("1", "2"):
                expected = [q for _, t, q in table if t == atype]
                assert len(expected) == 3 and all(expected)
                # MDAnalysis stores charges in single precision.
                np.testing.assert_array_equal(
                    traj.get_charges(atype), np.array(expected, dtype=np.float32))

    def test_full_style_data_file(self):
        lj = Path(__file__).parent / "data" / "example_1_LJ"
        inferred = LammpsTrajectory(str(lj / "dump.nh.lammps"), str(lj / "data.fin.nh.data"),
                                    temperature=1.35, units="lj")
        explicit = LammpsTrajectory(str(lj / "dump.nh.lammps"), str(lj / "data.fin.nh.data"),
                                    temperature=1.35, units="lj",
                                    atom_style="id resid type charge x y z ix iy iz")
        np.testing.assert_array_equal(
            inferred.mdanalysis_universe.atoms.types, explicit.mdanalysis_universe.atoms.types)
        np.testing.assert_array_equal(
            inferred.mdanalysis_universe.atoms.ids, explicit.mdanalysis_universe.atoms.ids)


TRICLINIC = str(SMALL / "dump_triclinic.lammps")
TRICLINIC_CELL = np.array([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, -1.5, 10.0]])


class TestTriclinic:

    def test_cell_and_positions(self):
        traj = load(TRICLINIC)
        np.testing.assert_array_equal(traj.cell_matrix, TRICLINIC_CELL)
        first, second = traj.iter_frames()
        np.testing.assert_array_equal(first.positions, positions())
        np.testing.assert_array_equal(second.positions, positions(1.0))

    def test_cell_matches_ase(self):
        from ase.io import read
        atoms = read(TRICLINIC, format="lammps-dump-text", index=0)
        np.testing.assert_allclose(load(TRICLINIC).cell_matrix, atoms.cell.array, atol=1e-12)

    def test_cell_matches_mdanalysis(self):
        import MDAnalysis as MD
        from MDAnalysis.lib.mdamath import triclinic_vectors
        universe = MD.Universe(DATA, TRICLINIC, topology_format="DATA",
                               format="LAMMPSDump", atom_style="id type x y z")
        np.testing.assert_allclose(
            load(TRICLINIC).cell_matrix, triclinic_vectors(universe.dimensions), atol=1e-5)

    def test_positions_match_ase(self):
        from ase.io import read
        for k, got in enumerate(load(TRICLINIC).iter_frames()):
            atoms = read(TRICLINIC, format="lammps-dump-text", index=k)
            np.testing.assert_array_equal(got.positions, atoms.positions)

    def test_scaled_positions_with_a_box_origin_match_ase(self, tmp_path):
        from ase.io import read
        # Bounds for the triclinic cell with origin (5, -3, 2).
        bounds = "5.0 18.0 2.0\n-4.5 7.0 1.0\n2.0 12.0 -1.5\n"
        rng = np.random.default_rng(3)
        atom_rows = [f"{i} {atom_type(i)} " + " ".join(repr(float(v)) for v in rng.uniform(-0.2, 1.2, 3))
                     + " 0 0 0" for i in (20, 3, 40, 12, 21, 7)]
        path = write(tmp_path / "scaled.dump", frame(
            0, box_item="xy xz yz pp pp pp", bounds=bounds,
            columns="id type xs ys zs fx fy fz", atom_rows=atom_rows))
        atoms = read(path, format="lammps-dump-text", index=0)
        # ASE returns scaled positions relative to the box origin.
        expected = atoms.positions + atoms.get_celldisp().ravel()
        np.testing.assert_allclose(load(path).get_frame(0).positions, expected, rtol=0, atol=1e-13)
