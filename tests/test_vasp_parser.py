"""
pytest test suite for py

These tests create minimal vasprun.xml-like files to validate parsing logic,
without needing full VASP outputs.
"""

import pytest
import numpy as np
from lxml import etree
from pymatgen.core import Structure
from revelsMD.trajectories.vasp import parse_varray, parse_structure, structure_from_structure_data, Vasprun


@pytest.fixture
def minimal_vasprun_xml(tmp_path):
    """Create a small valid vasprun.xml file for testing."""
    xml_content = """
<modeling>
  <atominfo>
    <array name="atoms">
      <set>
        <rc><c>H</c></rc>
        <rc><c>O</c></rc>
      </set>
    </array>
  </atominfo>

  <calculation>
    <structure>
      <crystal>
        <varray name="basis">
          <v>1.0 0.0 0.0</v>
          <v>0.0 1.0 0.0</v>
          <v>0.0 0.0 1.0</v>
        </varray>
      </crystal>
      <varray name="positions">
        <v>0.0 0.0 0.0</v>
        <v>0.5 0.5 0.5</v>
      </varray>
    </structure>
    <varray name="forces" type="float">
      <v>0.1 0.0 0.0</v>
      <v>-0.1 0.0 0.0</v>
    </varray>
  </calculation>
</modeling>
"""
    xml_path = tmp_path / "vasprun.xml"
    xml_path.write_text(xml_content)
    return str(xml_path)


# -----------------------------------------------------------------------------
# Unit tests for helper functions
# -----------------------------------------------------------------------------
def test_parse_varray_float():
    xml = etree.fromstring('<varray><v>1.0 2.0 3.0</v></varray>')
    arr = parse_varray(xml)
    assert np.allclose(arr, [[1.0, 2.0, 3.0]])


def test_parse_varray_int():
    xml = etree.fromstring('<varray type="int"><v>1 2 3</v></varray>')
    arr = parse_varray(xml)
    assert arr == [[1, 2, 3]]


def test_parse_varray_logical():
    xml = etree.fromstring('<varray type="logical"><v>T F T</v></varray>')
    arr = parse_varray(xml)
    assert arr == [[True, False, True]]


def test_parse_structure_returns_expected_dict():
    xml = etree.fromstring("""
    <structure>
      <crystal>
        <varray>
          <v>1 0 0</v>
          <v>0 1 0</v>
          <v>0 0 1</v>
        </varray>
      </crystal>
      <varray>
        <v>0 0 0</v>
        <v>0.5 0.5 0.5</v>
      </varray>
    </structure>
    """)
    parsed = parse_structure(xml)
    assert "lattice" in parsed
    assert "frac_coords" in parsed
    assert parsed["selective_dynamics"] is None
    assert np.shape(parsed["lattice"]) == (3, 3)


def test_structure_from_structure_data_creates_structure():
    lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    names = ["H", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    struct = structure_from_structure_data(lattice, names, coords)
    assert isinstance(struct, Structure)
    assert len(struct) == 2


# -----------------------------------------------------------------------------
# Tests for Vasprun class
# -----------------------------------------------------------------------------
def test_vasprun_parses_atom_names(minimal_vasprun_xml):
    vr = Vasprun(minimal_vasprun_xml)
    assert vr.atom_names == ["H", "O"]


def test_vasprun_parses_structures(minimal_vasprun_xml):
    vr = Vasprun(minimal_vasprun_xml)
    structs = vr.structures
    assert len(structs) == 1
    s = structs[0]
    assert np.allclose(s.lattice.matrix, np.eye(3))
    assert np.allclose(s.frac_coords, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def test_vasprun_frac_and_cart_coords_are_numpy(minimal_vasprun_xml):
    vr = Vasprun(minimal_vasprun_xml)
    assert isinstance(vr.frac_coords, np.ndarray)
    assert vr.frac_coords.shape == (1, 2, 3)
    assert isinstance(vr.cart_coords, np.ndarray)


def test_vasprun_forces_parsed(minimal_vasprun_xml):
    vr = Vasprun(minimal_vasprun_xml)
    forces = vr.forces
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (1, 2, 3)
    assert np.allclose(forces[0][0], [0.1, 0.0, 0.0])


def test_vasprun_raises_if_no_forces(tmp_path):
    """If there are no forces, parser should raise ValueError."""
    xml_no_forces = """
<modeling>
  <atominfo>
    <array name="atoms"><set><rc><c>H</c></rc></set></array>
  </atominfo>
  <calculation>
    <structure>
      <crystal><varray><v>1 0 0</v><v>0 1 0</v><v>0 0 1</v></varray></crystal>
      <varray><v>0 0 0</v></varray>
    </structure>
  </calculation>
</modeling>
"""
    path = tmp_path / "vasprun_no_forces.xml"
    path.write_text(xml_no_forces)
    vr = Vasprun(str(path))
    with pytest.raises(ValueError, match="No forces"):
        _ = vr.forces

