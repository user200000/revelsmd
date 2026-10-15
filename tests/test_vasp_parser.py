"""Unit tests for the vasprun.xml index and frame reader."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from lxml import etree

from vasprun_variants import SOURCE, join, split, truncated, with_basis, without_forces, write
from revelsMD.trajectories.vasp import index_vasprun, parse_basis, parse_varray, read_vasp_frame


def test_parse_varray_types():
    assert np.allclose(parse_varray(etree.fromstring("<varray><v>1.0 2.0 3.0</v></varray>")), [[1, 2, 3]])
    assert parse_varray(etree.fromstring('<varray type="int"><v>1 2 3</v></varray>')) == [[1, 2, 3]]
    assert parse_varray(etree.fromstring('<varray type="logical"><v>T F T</v></varray>')) == [[True, False, True]]


class TestIndexVasprun:

    def test_complete_file(self):
        index = index_vasprun(str(SOURCE))
        assert len(index.frames) == 10
        assert [f.calculation for f in index.frames] == list(range(10))
        assert len(index.atom_names) == 324
        assert sorted(set(index.atom_names)) == ["Ba", "F", "Sn"]
        np.testing.assert_allclose(parse_basis(index.frames[0].basis), np.eye(3) * 18.351083)

    def test_truncated_file_keeps_complete_steps_with_a_warning(self, tmp_path):
        path = truncated(tmp_path / "vasprun.xml")
        with pytest.warns(UserWarning, match="incomplete final ionic step.*9 complete steps"):
            index = index_vasprun(path)
        assert len(index.frames) == 9

    def test_missing_forces_raises(self, tmp_path):
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations[4] = without_forces(calculations[4])
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        with pytest.raises(ValueError, match="calculation 4: no forces"):
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


class TestReadVaspFrame:

    def test_frames_match_an_independent_parse(self):
        index = index_vasprun(str(SOURCE))
        root = ET.parse(SOURCE).getroot()
        calculations = root.findall("calculation")
        with open(SOURCE, "rb") as f:
            for frame, calculation in zip(index.frames, calculations, strict=True):
                positions, forces = read_vasp_frame(f, frame, 324)
                structure = calculation.find("structure")
                basis = _floats(structure.find("crystal").findall("varray[@name='basis']/v"))
                fractional = _floats(structure.findall("varray[@name='positions']/v"))
                expected_forces = _floats(calculation.findall("varray[@name='forces']/v"))
                np.testing.assert_array_equal(positions, fractional @ basis)
                np.testing.assert_array_equal(forces, expected_forces)

    def test_wrong_atom_count_raises(self):
        index = index_vasprun(str(SOURCE))
        with open(SOURCE, "rb") as f, pytest.raises(ValueError, match="for 323 atoms"):
            read_vasp_frame(f, index.frames[0], 323)

    def test_triclinic_basis(self, tmp_path):
        shear = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [0.1, 0.2, 1.0]])
        head, calculations, tail = split(SOURCE.read_bytes())
        calculations = [with_basis(c, lambda b: shear @ b) for c in calculations]
        path = write(tmp_path / "vasprun.xml", join(head, calculations, tail))
        index = index_vasprun(path)
        expected = shear @ (np.eye(3) * 18.351083)
        np.testing.assert_allclose(parse_basis(index.frames[0].basis), expected)
