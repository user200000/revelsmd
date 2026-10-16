"""Unit tests for the vasprun.xml index and frame reader."""

import io

import numpy as np
import pytest

from vasprun_variants import (
    SOURCE,
    join,
    split,
    truncated,
    with_basis,
    without_basis,
    without_forces,
    without_structure,
    write,
    independent_frames as _independent_frames,
)
from revelsMD.trajectories.vasp import index_vasprun, parse_basis, read_vasp_frame

SHEAR = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.1, 0.2, 1.0]])


class TestIndexVasprun:

    def test_complete_file(self):
        index = index_vasprun(str(SOURCE))
        assert len(index.starts) == len(index.ends) == 10
        # Step k is the k-th <calculation> element, in file order.
        text = SOURCE.read_bytes()
        _, calculations, _ = split(text)
        assert [text[a:b] for a, b in zip(index.starts, index.ends)] == calculations
        assert len(index.atom_names) == 324
        assert sorted(set(index.atom_names)) == ["Ba", "F", "Sn"]
        np.testing.assert_allclose(parse_basis(index.basis), np.eye(3) * 18.351083)

    def test_truncated_file_keeps_complete_steps_with_a_warning(self, tmp_path):
        path = truncated(tmp_path / "vasprun.xml")
        with pytest.warns(UserWarning, match=r"dropping the incomplete final ionic step \(calculation 9\); the file appears to have been cut short"):
            index = index_vasprun(path)
        assert len(index.starts) == len(index.ends) == 9

    def test_missing_forces_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = without_forces(calculations[4])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 4: no forces"):
            index_vasprun(path)

    def test_missing_structure_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = without_structure(calculations[4])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 4: no structure"):
            index_vasprun(path)

    def test_missing_basis_in_a_middle_step_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = without_basis(calculations[4])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 4: the structure has no lattice basis"):
            index_vasprun(path)

    def test_changed_basis_raises_naming_the_calculation(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = with_basis(calculations[4], lambda b: b * 1.01)
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match=r"vasprun\.xml, calculation 4: the cell differs from the first frame's"):
            index_vasprun(path)

    def test_first_step_is_checked_against_a_reference_basis(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        reference = index_vasprun(str(SOURCE))
        changed = [with_basis(calculations[0], lambda b: b * 1.01)] + calculations[1:]
        path = write(tmp_path / "vasprun.xml", join(head, changed, tail))
        with pytest.raises(ValueError, match=r"vasprun\.xml, calculation 0: the cell differs from the first frame's"):
            index_vasprun(path, reference)

    def test_basis_written_differently_with_the_same_values_is_accepted(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        original = calculations[4]
        calculations[4] = with_basis(original, lambda b: b)
        assert calculations[4] != original
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        index_vasprun(path)

    def test_unclosed_calculation_before_the_end_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[3] = calculations[3].replace(b"</calculation>", b"")
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 3: no closing"):
            index_vasprun(path)

    def test_no_atominfo_raises(self, tmp_path):
        text = SOURCE.read_bytes().replace(b"<atominfo>", b"<atominf>").replace(b"</atominfo>", b"</atominf>")
        with pytest.raises(ValueError, match="no <atominfo>"):
            index_vasprun(write(tmp_path / "vasprun.xml", text))

    def test_no_atom_list_raises(self, tmp_path):
        text = SOURCE.read_bytes().replace(b'<array name="atoms" >', b'<array name="atomz" >', 1)
        assert text != SOURCE.read_bytes()
        with pytest.raises(ValueError, match="no atom list in <atominfo>"):
            index_vasprun(write(tmp_path / "vasprun.xml", text))

    def test_empty_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="the file is empty"):
            index_vasprun(write(tmp_path / "vasprun.xml", b""))

    def test_run_killed_during_its_first_step_raises(self, tmp_path):
        head, calculations, _ = split(SOURCE.read_bytes())
        path = write(tmp_path / "vasprun.xml", head + calculations[0][:-200])
        with pytest.warns(UserWarning, match=r"calculation 0\)"):
            with pytest.raises(ValueError, match="no complete ionic steps"):
                index_vasprun(path)

    @pytest.mark.parametrize("edit, missing", [
        (without_forces, "no forces"),
        (without_structure, "no structure"),
        (without_basis, "the structure has no lattice basis"),
    ])
    def test_last_step_missing_data_raises(self, tmp_path, edit, missing):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[-1] = edit(calculations[-1])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match=f"calculation 9: {missing}"):
            index_vasprun(path)

    def test_missing_positions_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[2] = calculations[2].replace(b'<varray name="positions"', b'<varray name="positionz"', 1)
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 2: the structure has no positions"):
            index_vasprun(path)

    def test_malformed_basis_names_the_calculation(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[3] = with_basis(calculations[3], lambda b: b).replace(b"18.351083", b"*********", 1)
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 3: malformed lattice basis"):
            index_vasprun(path)

    def test_malformed_basis_in_every_step_names_the_first(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations = [with_basis(c, lambda b: b).replace(b"18.351083", b"*********", 1)
                        for c in calculations]
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 0: malformed lattice basis"):
            index_vasprun(path)

    def test_concatenated_files_raise(self, tmp_path):
        text = SOURCE.read_bytes()
        with pytest.raises(ValueError, match="a second <\\?xml at byte .*more than one vasprun.xml"):
            index_vasprun(write(tmp_path / "vasprun.xml", text + text))

    def test_atoms_are_compared_before_the_lattice(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        atominfo = head.find(b"<atominfo>")
        head = head[:atominfo] + head[atominfo:].replace(b"<c>Ba</c>", b"<c>Sr</c>", 1)
        scaled = [with_basis(c, lambda b: b * 1.01) for c in calculations]
        path = write(tmp_path / "other.xml", join(head, scaled, tail))
        with pytest.raises(ValueError, match="other.xml: the atoms differ from those in"):
            index_vasprun(path, index_vasprun(str(SOURCE)))


def index_cell(index):
    return parse_basis(index.basis)


@pytest.fixture(params=["committed", "sheared"])
def vasprun_path(request, tmp_path):
    """The committed file, and a copy whose basis is sheared so it is not symmetric."""
    if request.param == "committed":
        return str(SOURCE)
    head, calculations, tail = split(SOURCE.read_bytes())
    calculations = [with_basis(c, lambda b: SHEAR @ b) for c in calculations]
    return write(tmp_path / "vasprun.xml", join(head, calculations, tail))


class TestReadVaspFrame:

    def test_frames_match_an_independent_parse(self, vasprun_path):
        index = index_vasprun(vasprun_path)
        expected = _independent_frames(vasprun_path)
        with open(vasprun_path, "rb") as f:
            steps = zip(index.starts, index.ends, expected, strict=True)
            for start, end, (expected_positions, expected_forces) in steps:
                positions, forces = read_vasp_frame(f, start, end, 324)
                np.testing.assert_array_equal(positions, expected_positions)
                np.testing.assert_array_equal(forces, expected_forces)

    def test_one_atom_in_a_sheared_cell(self):
        # Fractional (0.5, 0.25, 0.5) in the lattice a = (2, 0, 0),
        # b = (1, 3, 0), c = (0.5, 0.5, 4) is 0.5 a + 0.25 b + 0.5 c.
        calculation = b"""<calculation>
 <structure>
  <crystal>
   <varray name="basis" >
    <v>       2.0       0.0       0.0 </v>
    <v>       1.0       3.0       0.0 </v>
    <v>       0.5       0.5       4.0 </v>
   </varray>
  </crystal>
  <varray name="positions" >
   <v>       0.5      0.25       0.5 </v>
  </varray>
 </structure>
 <varray name="forces" >
  <v>       0.1      -0.2       0.3 </v>
 </varray>
</calculation>"""
        positions, forces = read_vasp_frame(io.BytesIO(calculation), 0, len(calculation), 1)
        np.testing.assert_array_equal(positions, [[1.5, 1.0, 2.0]])
        np.testing.assert_array_equal(forces, [[0.1, -0.2, 0.3]])

    def test_structure_with_attributes_is_not_read(self, tmp_path):
        # Only the attribute-free <structure> is the ionic step's; another
        # placed before it must not be used.
        head, calculations, tail = split(SOURCE.read_bytes())
        _, original_forces = _independent_frames(str(SOURCE))[0]
        structure = calculations[0][calculations[0].find(b"<structure>"):
                                    calculations[0].find(b"</structure>") + len(b"</structure>")]
        extra = with_basis(structure, lambda b: b * 1.1).replace(b"<structure>", b'<structure name="extra">', 1)
        calculations[0] = calculations[0].replace(b"<structure>", extra + b"\n<structure>", 1)
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        index = index_vasprun(path)
        expected_positions, _ = _independent_frames(str(SOURCE))[0]
        with open(path, "rb") as f:
            positions, forces = read_vasp_frame(f, index.starts[0], index.ends[0], 324, index_cell(index))
        np.testing.assert_array_equal(positions, expected_positions)
        np.testing.assert_array_equal(forces, original_forces)

    def test_lattice_differing_from_the_cell_raises(self):
        index = index_vasprun(str(SOURCE))
        with open(SOURCE, "rb") as f, pytest.raises(ValueError, match="the lattice of the step's own <structure> differs"):
            read_vasp_frame(f, index.starts[0], index.ends[0], 324, index_cell(index) * 1.1)

    def test_wrong_atom_count_raises(self):
        index = index_vasprun(str(SOURCE))
        with open(SOURCE, "rb") as f, pytest.raises(ValueError, match="for 323 atoms"):
            read_vasp_frame(f, index.starts[0], index.ends[0], 323)

    def test_triclinic_basis(self, tmp_path):
        shear = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.1, 0.2, 1.0]])
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations = [with_basis(c, lambda b: shear @ b) for c in calculations]
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        index = index_vasprun(path)
        expected = shear @ (np.eye(3) * 18.351083)
        np.testing.assert_allclose(parse_basis(index.basis), expected)
