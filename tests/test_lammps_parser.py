"""Unit tests for the LAMMPS dump helpers."""

import numpy as np
import pytest

from revelsMD.trajectories.lammps import ColumnLayout, parse_box, resolve_columns

TRICLINIC = "ITEM: BOX BOUNDS xy xz yz pp pp pp"


class TestParseBox:

    def test_orthogonal_box(self):
        cell, origin, box = parse_box(
            "ITEM: BOX BOUNDS pp pp pp", ["0.0 10.0", "-1.0 9.0", "2.0 14.0"]
        )
        np.testing.assert_array_equal(cell, np.diag([10.0, 10.0, 12.0]))
        np.testing.assert_array_equal(origin, [0.0, -1.0, 2.0])
        np.testing.assert_array_equal(box, [0, 10, -1, 9, 2, 14, 0, 0, 0])

    def test_triclinic_box_follows_lammps_conventions(self):
        # Box xlo=0 xhi=10, ylo=0 yhi=10, zlo=0 zhi=10, xy=2 xz=1 yz=-1.5,
        # written as its bounding box (LAMMPS Howto triclinic).
        cell, origin, box = parse_box(
            TRICLINIC, ["0.0 13.0 2.0", "-1.5 10.0 1.0", "0.0 10.0 -1.5"]
        )
        np.testing.assert_array_equal(cell, [[10, 0, 0], [2, 10, 0], [1, -1.5, 10]])
        np.testing.assert_array_equal(origin, [0, 0, 0])
        np.testing.assert_array_equal(box, [0, 13, -1.5, 10, 0, 10, 2, 1, -1.5])

    def test_negative_tilts(self):
        # xy=-2 xz=-1 yz=1.5 on the same 10x10x10 box.
        cell, origin, _ = parse_box(
            TRICLINIC, ["-3.0 10.0 -2.0", "0.0 11.5 -1.0", "0.0 10.0 1.5"]
        )
        np.testing.assert_array_equal(cell, [[10, 0, 0], [-2, 10, 0], [-1, 1.5, 10]])
        np.testing.assert_array_equal(origin, [0, 0, 0])

    @pytest.mark.parametrize("item, bounds", [
        ("ITEM: BOX BOUNDS ff pp pp", ["0 10"] * 3),
        ("ITEM: BOX BOUNDS pp pp sm", ["0 10"] * 3),
        ("ITEM: BOX BOUNDS xy xz yz pp fs pp", ["0 10 0"] * 3),
        ("ITEM: BOX BOUNDS", ["0 10"] * 3),
    ])
    def test_non_periodic_or_unflagged_boundaries_are_rejected(self, item, bounds):
        with pytest.raises(ValueError, match="periodic"):
            parse_box(item, bounds)

    def test_general_triclinic_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            parse_box("ITEM: BOX BOUNDS abc origin pp pp pp", ["10 0 0 0"] * 3)

    def test_wrong_number_of_bound_values_is_rejected(self):
        with pytest.raises(ValueError, match="box bounds"):
            parse_box(TRICLINIC, ["0 10", "0 10", "0 10"])


class TestResolveColumns:

    def test_plain_coordinates_are_preferred(self):
        columns = ("id", "type", "x", "y", "z", "xu", "yu", "zu", "fx", "fy", "fz")
        assert resolve_columns(columns, "a.dump") == ColumnLayout(
            11, 0, (2, 3, 4), (8, 9, 10), False
        )

    def test_unwrapped(self):
        layout = resolve_columns(("fx", "fy", "fz", "xu", "yu", "zu", "id"), "a.dump")
        assert layout == ColumnLayout(7, 6, (3, 4, 5), (0, 1, 2), False)

    @pytest.mark.parametrize("names", [("xs", "ys", "zs"), ("xsu", "ysu", "zsu")])
    def test_scaled(self, names):
        layout = resolve_columns(("id",) + names + ("fx", "fy", "fz"), "a.dump")
        assert layout == ColumnLayout(7, 0, (1, 2, 3), (4, 5, 6), True)

    def test_missing_forces(self):
        with pytest.raises(ValueError, match=r"a\.dump: .*lack fx fy fz"):
            resolve_columns(("id", "type", "x", "y", "z"), "a.dump")

    def test_missing_coordinates(self):
        with pytest.raises(ValueError, match="one of x y z"):
            resolve_columns(("id", "fx", "fy", "fz"), "a.dump")

    def test_missing_id(self):
        with pytest.raises(ValueError, match="lack id"):
            resolve_columns(("x", "y", "z", "fx", "fy", "fz"), "a.dump")
