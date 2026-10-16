"""VaspTrajectory on the committed vasprun.xml and variants of it."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from file_limits import open_descriptors, open_file_limit
from traced_memory import bytes_per_extra_item
from vasprun_variants import (
    SOURCE, independent_frames as reference_frames, join, split, truncated, with_basis,
    without_forces, write,
)
from revelsMD.trajectories import DataUnavailableError, VaspTrajectory
from revelsMD.trajectories._base import compute_beta
from revelsMD.trajectories.vasp import index_vasprun


@pytest.fixture(scope="module")
def traj():
    return VaspTrajectory(str(SOURCE), temperature=600.0)


class TestReading:

    def test_frames_and_cell(self, traj):
        assert traj.frames == 10
        np.testing.assert_allclose(traj.cell_matrix, np.eye(3) * 18.351083)

    def test_iteration_matches_the_reader(self, traj):
        for got, (positions, forces) in zip(traj.iter_frames(), reference_frames(), strict=True):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    def test_start_stop_stride_and_negative_stop(self, traj):
        reference = reference_frames()
        got = list(traj.iter_frames(start=2, stop=8, stride=2))
        assert len(got) == 3
        for frame, k in zip(got, (2, 4, 6)):
            np.testing.assert_array_equal(frame.positions, reference[k][0])
        assert len(list(traj.iter_frames(stop=-1))) == 9

    def test_get_frame_in_any_order(self, traj):
        reference = reference_frames()
        for k in (7, 2, 9, 0, 5, -1):
            np.testing.assert_array_equal(traj.get_frame(k).positions, reference[k][0])
            np.testing.assert_array_equal(traj.get_frame(k).forces, reference[k][1])

    def test_get_indices(self, traj):
        names = np.array(index_vasprun(str(SOURCE)).atom_names)
        for symbol in ("Ba", "Sn", "F"):
            np.testing.assert_array_equal(traj.get_indices(symbol), np.flatnonzero(names == symbol))

    def test_charges_and_masses_are_unavailable(self, traj):
        with pytest.raises(DataUnavailableError, match=r"vasprun\.xml records no charges.*for F with charges="):
            traj.get_charges("F")
        with pytest.raises(DataUnavailableError, match=r"vasprun\.xml records no masses.*for F with masses="):
            traj.get_masses("F")

    def test_charges_and_masses_given_by_element(self, traj):
        given = VaspTrajectory(str(SOURCE), temperature=600.0, masses={"F": 18.998}, charges={"F": -1})
        n = len(traj.get_indices("F"))
        np.testing.assert_array_equal(given.get_masses("F"), [18.998] * n)
        np.testing.assert_array_equal(given.get_charges("F"), [-1.0] * n)
        with pytest.raises(DataUnavailableError, match="for Ba with masses="):
            given.get_masses("Ba")
        with pytest.raises(ValueError, match="masses= names species that no atom has: O"):
            VaspTrajectory(str(SOURCE), temperature=600.0, masses={"O": 16.0})

    def test_temperature_and_beta(self):
        traj = VaspTrajectory(str(SOURCE), temperature=500.0)
        assert traj.temperature == 500.0
        assert traj.beta == pytest.approx(compute_beta("metal", 500.0), rel=1e-12)

    def test_temperature_is_required(self):
        with pytest.raises(TypeError):
            VaspTrajectory(str(SOURCE))

    def test_unknown_species_raises(self, traj):
        with pytest.raises(ValueError, match="no atoms of species 'Zr'; the species are Ba, F, Sn"):
            traj.get_indices("Zr")

    def test_get_frame_needs_an_integer(self, traj):
        with pytest.raises(TypeError):
            traj.get_frame(2.0)

    def test_no_whole_trajectory_arrays(self, traj):
        assert not hasattr(traj, "positions")
        assert not hasattr(traj, "forces")
        assert not hasattr(traj, "Vasprun")


class TestFiles:

    def test_split_files_equal_the_single_file(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        first = write(tmp_path / "1.xml", join(head, calculations[:5], tail))
        second = write(tmp_path / "2.xml", join(head, calculations[5:], tail))
        traj = VaspTrajectory([Path(first), Path(second)], temperature=600.0)
        assert traj.frames == 10
        for got, (positions, forces) in zip(traj.iter_frames(), reference_frames(), strict=True):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    def _split(self, tmp_path, edit_second=lambda c: c):
        head, calculations, tail = split(SOURCE.read_bytes())
        first = write(tmp_path / "1.xml", join(head, calculations[:5], tail))
        second = write(tmp_path / "2.xml", join(head, [edit_second(c) for c in calculations[5:]], tail))
        return first, second

    def test_second_file_with_a_reformatted_basis(self, tmp_path):
        first, second = self._split(tmp_path, lambda c: with_basis(c, lambda b: b))
        traj = VaspTrajectory([first, second], temperature=600.0)
        for got, (positions, forces) in zip(traj.iter_frames(), reference_frames(), strict=True):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    def test_negative_get_frame_across_files(self, tmp_path):
        traj = VaspTrajectory(list(self._split(tmp_path)), temperature=600.0)
        reference = reference_frames()
        for k in range(1, 11):
            np.testing.assert_array_equal(traj.get_frame(-k).positions, reference[-k][0])
            np.testing.assert_array_equal(traj.get_frame(-k).forces, reference[-k][1])

    def test_read_errors_name_the_file_and_calculation(self, tmp_path):
        def drop_a_force_row(calculation):
            start = calculation.find(b"<v>", calculation.find(b'<varray name="forces"'))
            return calculation[:start] + calculation[calculation.find(b"</v>", start) + 4:]
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[7] = drop_a_force_row(calculations[7])
        first = write(tmp_path / "1.xml", join(head, calculations[:5], tail))
        second = write(tmp_path / "2.xml", join(head, calculations[5:], tail))
        traj = VaspTrajectory([first, second], temperature=600.0)
        with pytest.raises(ValueError, match=r"2\.xml, calculation 2: expected positions and forces for 324"):
            traj.get_frame(7)
        with pytest.raises(ValueError, match=r"2\.xml, calculation 2:"):
            list(traj.iter_frames())

    def test_lattice_differing_from_the_indexed_one_raises_when_read(self, tmp_path):
        # The index takes the first bare <structure> in the bytes, here one
        # nested in another element with a scaled lattice; the reader takes
        # the step's own, so the two lattices differ.
        def nest_a_scaled_copy(calculation):
            own = calculation[calculation.find(b"<structure>"):
                              calculation.find(b"</structure>") + len(b"</structure>")]
            nested = b"<extra>" + with_basis(own, lambda b: b * 1.1) + b"</extra>\n"
            return calculation.replace(b"<structure>", nested + b"<structure>", 1)
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations = [nest_a_scaled_copy(c) for c in calculations]
        traj = VaspTrajectory(write(tmp_path / "v.xml", join(head, calculations, tail)),
                              temperature=600.0)
        with pytest.raises(ValueError, match=r"v\.xml, calculation 0: the lattice of the step's own <structure> differs"):
            traj.get_frame(0)

    def test_malformed_xml_raises_when_read(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        start = calculations[2].find(b"<v>", calculations[2].find(b'<varray name="forces"'))
        calculations[2] = calculations[2][:start] + b"<v" + calculations[2][start + 3:]
        traj = VaspTrajectory(write(tmp_path / "v.xml", join(head, calculations, tail)),
                              temperature=600.0)
        with pytest.raises(ValueError, match=r"v\.xml, calculation 2: malformed XML"):
            traj.get_frame(2)

    def test_non_finite_forces_raise_when_read(self, tmp_path):
        def nan_force(calculation):
            start = calculation.find(b"<v>", calculation.find(b'<varray name="forces"')) + 3
            return calculation[:start] + b" NaN 0.0 0.0 " + calculation[calculation.find(b"</v>", start):]
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[3] = nan_force(calculations[3])
        traj = VaspTrajectory(write(tmp_path / "v.xml", join(head, calculations, tail)), temperature=600.0)
        with pytest.raises(ValueError, match=r"v\.xml, calculation 3: the forces contain NaN"):
            traj.get_frame(3)

    def test_zero_volume_lattice_raises_naming_the_file(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        flat = [with_basis(c, lambda b: b * np.array([[1.0], [1.0], [0.0]])) for c in calculations]
        path = write(tmp_path / "flat.xml", join(head, flat, tail))
        with pytest.raises(ValueError, match=r"flat\.xml, calculation 0: Cell matrix must have positive volume"):
            VaspTrajectory(path, temperature=600.0)

    def test_files_with_different_atoms_raise(self, tmp_path):
        text = SOURCE.read_bytes()
        atominfo = text.find(b"<atominfo>")
        changed = text[:atominfo] + text[atominfo:].replace(b"<c>Ba</c>", b"<c>Sr</c>", 1)
        other = write(tmp_path / "other.xml", changed)
        with pytest.raises(ValueError, match="atoms differ"):
            VaspTrajectory([str(SOURCE), other], temperature=600.0)

    def test_killed_run(self, tmp_path):
        path = truncated(tmp_path / "vasprun.xml")
        with pytest.warns(UserWarning, match="incomplete final ionic step") as record:
            traj = VaspTrajectory(path, temperature=600.0)
        assert record[0].filename == __file__
        assert traj.frames == 9
        for got, (positions, forces) in zip(traj.iter_frames(), reference_frames()[:9], strict=True):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    def test_files_without_a_complete_step(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        killed = write(tmp_path / "1.xml", head + calculations[0][:-200])
        # Alone, it raises without warning first.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"no complete ionic steps in .*1\.xml"):
                VaspTrajectory(killed, temperature=600.0)
        # Before its restart, it is dropped with a warning.
        with pytest.warns(UserWarning, match=r"1\.xml: dropping the incomplete final ionic step; .*\(calculation 0 has no closing tag\)"):
            traj = VaspTrajectory([killed, str(SOURCE)], temperature=600.0)
        assert traj.frames == len(reference_frames())

    def test_killed_run_followed_by_its_restart(self, tmp_path):
        killed = truncated(tmp_path / "1.xml")
        head, calculations, tail = split(SOURCE.read_bytes())
        restart = write(tmp_path / "2.xml", join(head, calculations[9:], tail))
        with pytest.warns(UserWarning, match=r"1\.xml: dropping the incomplete final ionic step"):
            traj = VaspTrajectory([killed, restart], temperature=600.0)
        assert traj.frames == 10
        for got, (positions, forces) in zip(traj.iter_frames(), reference_frames(), strict=True):
            np.testing.assert_array_equal(got.positions, positions)
            np.testing.assert_array_equal(got.forces, forces)

    @pytest.mark.parametrize("k", [4, 9])
    def test_missing_forces_raise_at_construction(self, tmp_path, k):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[k] = without_forces(calculations[k])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match=f"calculation {k}: no forces"):
            VaspTrajectory(path, temperature=600.0)

    def test_changed_lattice_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = with_basis(calculations[4], lambda b: b * 1.01)
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 4: the cell differs"):
            VaspTrajectory(path, temperature=600.0)

    def test_lattice_changing_between_files_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        scaled = [with_basis(c, lambda b: b * 1.01) for c in calculations[5:]]
        first = write(tmp_path / "1.xml", join(head, calculations[:5], tail))
        second = write(tmp_path / "2.xml", join(head, scaled, tail))
        with pytest.raises(ValueError, match=r"2\.xml, calculation 0: the cell differs"):
            VaspTrajectory([first, second], temperature=600.0)

    def test_lattice_changing_only_in_a_later_files_first_step_names_that_step(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        changed = [with_basis(calculations[5], lambda b: b * 1.01)] + calculations[6:]
        first = write(tmp_path / "1.xml", join(head, calculations[:5], tail))
        second = write(tmp_path / "2.xml", join(head, changed, tail))
        with pytest.raises(ValueError, match=r"2\.xml, calculation 0: the cell differs"):
            VaspTrajectory([first, second], temperature=600.0)

    def test_triclinic_cell(self, tmp_path):
        shear = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.1, 0.2, 1.0]])
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations = [with_basis(c, lambda b: shear @ b) for c in calculations]
        traj = VaspTrajectory(write(tmp_path / "vasprun.xml", join(head, calculations, tail)),
                              temperature=600.0)
        np.testing.assert_allclose(traj.cell_matrix, shear @ (np.eye(3) * 18.351083))
        assert traj.is_orthorhombic is False
        assert traj.cell_volume > 0


class TestOpenFiles:

    def test_one_file_per_step_keeps_one_file_open(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        paths = [write(tmp_path / f"{k:02d}.xml", join(head, [c], tail))
                 for k, c in enumerate(calculations)]
        traj = VaspTrajectory(paths, temperature=600.0)
        with open_file_limit(open_descriptors() + 5):
            got = list(traj.iter_frames())
        assert len(got) == 10
        for frame, (positions, forces) in zip(got, reference_frames(), strict=True):
            np.testing.assert_array_equal(frame.positions, positions)
            np.testing.assert_array_equal(frame.forces, forces)


def skeleton_step(calculation):
    """A calculation cut down to its lattice and empty positions and forces arrays.

    It is enough to index, though not to read.
    """
    start = calculation.find(b"<structure>")
    end = calculation.find(b"</crystal>", start) + len(b"</crystal>")
    return (b"<calculation>\n" + calculation[start:end]
            + b'\n<varray name="positions" ></varray>\n</structure>\n'
            b'<varray name="forces" ></varray>\n</calculation>')


class TestIndexMemory:

    def test_index_memory_per_step_is_small(self, tmp_path):
        n = 2000
        head, calculations, tail = split(SOURCE.read_bytes())
        step = skeleton_step(calculations[0])
        paths = {m: write(tmp_path / f"{m}.xml", join(head, [step] * m, tail)) for m in (n, 2 * n)}
        per_step = bytes_per_extra_item(lambda m: VaspTrajectory(paths[m], temperature=600.0), n)
        assert per_step < 40, f"the index keeps {per_step:.0f} bytes per ionic step"
