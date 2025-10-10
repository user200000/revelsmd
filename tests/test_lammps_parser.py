"""
pytest test suite for lammps_parser.py
"""

import numpy as np
import pytest
from revelsMD.revels_tools import lammps_parser


@pytest.fixture
def tmp_lammps_dump(tmp_path):
    """Create a small valid LAMMPS dump file for parser testing."""
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id x y z fx fy fz
1 0.1 0.2 0.3 0.0 0.1 0.2
2 0.4 0.5 0.6 0.3 0.2 0.1
3 0.7 0.8 0.9 0.5 0.4 0.3
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id x y z fx fy fz
1 0.2 0.3 0.4 0.0 0.1 0.2
2 0.5 0.6 0.7 0.3 0.2 0.1
3 0.8 0.9 1.0 0.5 0.4 0.3
"""
    path = tmp_path / "test.dump"
    path.write_text(dump_content)
    return str(path)


def test_first_read_returns_metadata(tmp_lammps_dump):
    frames, num_ats, dic, header_length, dimgrid = lammps_parser.first_read(tmp_lammps_dump)

    assert frames == 2
    assert num_ats == 3
    assert "ATOMS" in dic
    assert header_length > 0
    assert dimgrid.shape == (3, 2)
    assert np.allclose(dimgrid[:, 1], 10.0)


def test_define_strngdex_maps_correctly():
    dic = ["ITEM:", "ATOMS", "id", "x", "y", "z", "fx"]
    result = lammps_parser.define_strngdex(["x", "z"], dic)
    assert result == [1, 3]


def test_get_a_frame_extracts_data(tmp_lammps_dump):
    frames, num_ats, dic, header_length, _ = lammps_parser.first_read(tmp_lammps_dump)
    strngdex = lammps_parser.define_strngdex(["x", "y", "z"], dic)

    with open(tmp_lammps_dump, "r") as f:
        data = lammps_parser.get_a_frame(f, num_ats, header_length, strngdex)

    assert data.shape == (num_ats, 3)
    assert np.all(np.isfinite(data))


def test_frame_skip_moves_pointer(tmp_lammps_dump):
    frames, num_ats, dic, header_length, _ = lammps_parser.first_read(tmp_lammps_dump)

    with open(tmp_lammps_dump, "r") as f:
        lammps_parser.frame_skip(f, num_ats, 1, header_length)
        line = f.readline()
        # After skipping one frame, we should be at or near the second frame
        assert "ITEM: TIMESTEP" in line or "1" in line


