"""Unit tests for the vasprun.xml index and frame reader."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from lxml import etree

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
)
from revelsMD.trajectories.vasp import index_vasprun, parse_basis, parse_varray, read_vasp_frame

SHEAR = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.1, 0.2, 1.0]])


def test_parse_varray_types():
    assert np.allclose(parse_varray(etree.fromstring("<varray><v>1.0 2.0 3.0</v></varray>")), [[1, 2, 3]])
    assert parse_varray(etree.fromstring('<varray type="int"><v>1 2 3</v></varray>')) == [[1, 2, 3]]
    assert parse_varray(etree.fromstring('<varray type="logical"><v>T F T</v></varray>')) == [[True, False, True]]


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
        with pytest.warns(UserWarning, match="incomplete final ionic step.*9 complete steps"):
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
        reference = index_vasprun(str(SOURCE)).basis
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


def _floats(elements):
    """Rows of numbers from <v> elements, converted with float()."""
    return np.array([[float(x) for x in v.text.split()] for v in elements])


def _independent_frames(path):
    """Positions (fractional @ basis) and forces of each ionic step, read with ElementTree."""
    frames = []
    for calculation in ET.parse(path).getroot().findall("calculation"):
        structure = calculation.find("structure")
        basis = _floats(structure.find("crystal").findall("varray[@name='basis']/v"))
        fractional = _floats(structure.findall("varray[@name='positions']/v"))
        forces = _floats(calculation.findall("varray[@name='forces']/v"))
        frames.append((fractional @ basis, forces))
    return frames


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
