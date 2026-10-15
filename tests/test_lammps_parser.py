"""Unit tests for the LAMMPS dump helpers."""

import numpy as np
import pytest

from lammps_dumps import frame, rows, write
from revelsMD.trajectories.lammps import (
    ColumnLayout,
    DumpFrame,
    PartialFrame,
    index_dump,
    parse_box,
    resolve_columns,
)

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


def read_block(path, dump_frame):
    """The atom rows of an indexed frame, read back from the file."""
    with open(path, "rb") as f:
        f.seek(dump_frame.data_start)
        return f.read(dump_frame.data_end - dump_frame.data_start).decode()


class TestIndexDump:

    def test_complete_frames(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        index = index_dump(path)
        assert index.path == path
        assert index.partial is None
        assert [f.timestep for f in index.frames] == [0, 100]
        first = index.frames[0]
        assert first.n_atoms == 6
        assert first.columns == ("id", "type", "x", "y", "z", "fx", "fy", "fz")
        np.testing.assert_array_equal(first.cell, np.diag([10.0, 10.0, 10.0]))
        with open(path, "rb") as f:
            f.seek(first.data_start)
            block = f.read(first.data_end - first.data_start).decode()
        assert block == "".join(r + "\n" for r in rows(0))

    def test_cut_short_final_frame_is_partial(self, tmp_path):
        text = frame(0) + frame(100)
        path = write(tmp_path / "a.dump", text[: len(text) - 30])
        index = index_dump(path)
        assert [f.timestep for f in index.frames] == [0]
        assert index.partial == PartialFrame(path, 100)

    def test_header_cut_before_the_timestep_value(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0) + "ITEM: TIMESTEP\n")
        assert index_dump(path).partial == PartialFrame(path, None)

    def test_missing_final_newline_makes_the_final_frame_partial(self, tmp_path):
        text = frame(0) + frame(100)
        path = write(tmp_path / "a.dump", text.rstrip("\n"))
        index = index_dump(path)
        assert [f.timestep for f in index.frames] == [0]
        assert index.partial == PartialFrame(path, 100)

    def test_incomplete_frame_before_the_end_raises(self, tmp_path):
        short = frame(100, atom_rows=rows(100)[:4])
        path = write(tmp_path / "a.dump", frame(0), short, frame(200))
        with pytest.raises(ValueError, match="timestep 100 is incomplete but is not the last"):
            index_dump(path)

    def test_more_rows_than_the_header_states_raises(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0, n_atoms=5))
        with pytest.raises(ValueError, match="timestep 0: the atom rows do not match"):
            index_dump(path)

    def test_time_and_units_items_are_skipped(self, tmp_path):
        # dump_modify units yes / time yes write these before ITEM: TIMESTEP.
        path = write(
            tmp_path / "a.dump",
            "ITEM: UNITS\nlj\nITEM: TIME\n0\n", frame(0),
            "ITEM: TIME\n0.5\n", frame(100, 1.0),
            "ITEM: TIME\n1.0\n",
        )
        index = index_dump(path)
        assert [f.timestep for f in index.frames] == [0, 100]
        assert index.partial is None
        first, second = index.frames
        assert first.data_end <= second.data_start
        assert read_block(path, first) == "".join(r + "\n" for r in rows(0))
        assert read_block(path, second) == "".join(r + "\n" for r in rows(100, 1.0))

    @pytest.mark.parametrize("tail", ["I", "IT", "ITE", "ITEM"])
    def test_cut_off_next_header_is_dropped(self, tmp_path, tail):
        path = write(tmp_path / "a.dump", frame(0) + tail)
        index = index_dump(path)
        assert [f.timestep for f in index.frames] == [0]
        assert index.partial == PartialFrame(path, None)
        assert read_block(path, index.frames[0]) == "".join(r + "\n" for r in rows(0))

    def test_cut_off_next_header_after_a_frame_already_partial(self, tmp_path):
        # The rows of frame 100 are cut short, so that partial frame is reported.
        text = frame(0) + frame(100, atom_rows=rows(100)[:5]) + "ITE"
        path = write(tmp_path / "a.dump", text)
        index = index_dump(path)
        assert [f.timestep for f in index.frames] == [0]
        assert index.partial == PartialFrame(path, 100)

    def test_other_unterminated_tail_raises(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0) + "garbage")
        with pytest.raises(ValueError, match="the atom rows do not match"):
            index_dump(path)

    def test_timesteps_must_increase(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(100), frame(100, 1.0))
        with pytest.raises(ValueError, match="timesteps do not increase"):
            index_dump(path)

    def test_columns_must_not_change_within_a_file(self, tmp_path):
        reordered = frame(
            100, columns="type id x y z fx fy fz",
            atom_rows=[" ".join([r.split()[1], r.split()[0]] + r.split()[2:]) for r in rows(100)],
        )
        path = write(tmp_path / "a.dump", frame(0), reordered)
        with pytest.raises(ValueError, match="columns change"):
            index_dump(path)

    def test_box_errors_name_the_file_and_timestep(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0, box_item="ff pp pp"))
        with pytest.raises(ValueError, match="a.dump, timestep 0: box boundaries must be periodic"):
            index_dump(path)

    def test_empty_and_non_dump_files_raise(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            index_dump(write(tmp_path / "empty.dump", ""))
        with pytest.raises(ValueError, match="not a LAMMPS dump"):
            index_dump(write(tmp_path / "other.txt", "hello\n"))
