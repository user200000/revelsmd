"""Unit tests for the LAMMPS dump helpers."""

import random
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from lammps_dumps import (
    IDS_SORTED,
    forces as expected_forces,
    frame,
    positions as expected_positions,
    rows,
    write,
)
from revelsMD.trajectories.lammps import (
    ColumnLayout,
    HeaderMismatch,
    PartialFrame,
    combine_frames,
    index_dump,
    parse_box,
    read_dump_frame,
    resolve_atom_style,
    resolve_columns,
)

TRICLINIC = "ITEM: BOX BOUNDS xy xz yz pp pp pp"
DATA_DIR = Path(__file__).parent / "data"
SMALL_DATA = str(DATA_DIR / "lammps_small" / "data.small.data")
LJ_DATA = str(DATA_DIR / "example_1_LJ" / "data.fin.nh.data")


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

    def test_mixed_sign_tilts_with_single_tilt_extrema(self, tmp_path):
        # xy=2 xz=-1 yz=0.5 on the 10x10x10 box: the bounding box runs from
        # xlo + min(0, xy, xz, xy+xz) = -1 (xz alone) to
        # xhi + max(0, xy, xz, xy+xz) = 12 (xy alone), and y from 0 to 10.5.
        bounds = "-1.0 12.0 2.0\n0.0 10.5 -1.0\n0.0 10.0 0.5\n"
        expected = [[10, 0, 0], [2, 10, 0], [-1, 0.5, 10]]
        cell, origin, _ = parse_box(TRICLINIC, bounds.splitlines())
        np.testing.assert_array_equal(cell, expected)
        np.testing.assert_array_equal(origin, [0, 0, 0])

        from ase.io import read
        path = write(tmp_path / "mixed.dump", frame(0, box_item="xy xz yz pp pp pp", bounds=bounds))
        ase_cell = read(path, format="lammps-dump-text", index=0).cell.array
        np.testing.assert_allclose(cell, ase_cell, atol=1e-12)

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

    @pytest.mark.parametrize("bad", ["nan", "inf"])
    def test_non_finite_bounds_are_rejected(self, bad):
        with pytest.raises(ValueError, match="box bounds must be finite"):
            parse_box("ITEM: BOX BOUNDS pp pp pp", ["0.0 10.0", f"0.0 {bad}", "0.0 10.0"])

    def test_wrong_number_of_bound_values_is_rejected(self):
        with pytest.raises(ValueError, match="box bounds"):
            parse_box(TRICLINIC, ["0 10", "0 10", "0 10"])


class TestResolveColumns:

    def test_plain_coordinates_are_preferred(self):
        columns = ("id", "type", "x", "y", "z", "xu", "yu", "zu", "fx", "fy", "fz")
        assert resolve_columns(columns, "a.dump") == ColumnLayout(
            11, 0, (2, 3, 4), (8, 9, 10), False, 1
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


def read_block(path, index, k):
    """The atom rows of frame k of a dump index, read back from the file."""
    with open(path, "rb") as f:
        f.seek(index.starts[k])
        return f.read(index.ends[k] - index.starts[k]).decode()


class TestIndexDump:

    def test_complete_frames(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        index = index_dump(path)
        assert index.path == path
        assert index.partial is None
        assert index.timesteps.tolist() == [0, 100]
        assert index.header.n_atoms == 6
        assert index.header.columns == ("id", "type", "x", "y", "z", "fx", "fy", "fz")
        np.testing.assert_array_equal(index.header.cell, np.diag([10.0, 10.0, 10.0]))
        with open(path, "rb") as f:
            f.seek(index.starts[0])
            block = f.read(index.ends[0] - index.starts[0]).decode()
        assert block == "".join(r + "\n" for r in rows(0))

    def test_cut_short_final_frame_is_partial(self, tmp_path):
        text = frame(0) + frame(100)
        path = write(tmp_path / "a.dump", text[: len(text) - 30])
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert len(index.starts) == len(index.ends) == 1
        assert index.partial == PartialFrame(path, 100)

    def test_header_cut_before_the_timestep_value(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0) + "ITEM: TIMESTEP\n")
        assert index_dump(path).partial == PartialFrame(path, None)

    def test_missing_final_newline_makes_the_final_frame_partial(self, tmp_path):
        text = frame(0) + frame(100)
        path = write(tmp_path / "a.dump", text.rstrip("\n"))
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
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
        assert index.timesteps.tolist() == [0, 100]
        # The final ITEM: TIME begins a frame that was never written.
        assert index.partial == PartialFrame(path, None)
        assert index.ends[0] <= index.starts[1]
        assert read_block(path, index, 0) == "".join(r + "\n" for r in rows(0))
        assert read_block(path, index, 1) == "".join(r + "\n" for r in rows(100, 1.0))

    def test_complete_dump_with_time_and_units_has_no_partial_frame(self, tmp_path):
        path = write(
            tmp_path / "a.dump",
            "ITEM: UNITS\nlj\nITEM: TIME\n0\n", frame(0),
            "ITEM: TIME\n0.5\n", frame(100, 1.0),
        )
        index = index_dump(path)
        assert index.timesteps.tolist() == [0, 100]
        assert index.partial is None

    def test_blank_lines_within_the_preamble_are_accepted(self, tmp_path):
        path = write(tmp_path / "a.dump", "ITEM: UNITS\n\nlj\n\nITEM: TIME\n0\n\n", frame(0))
        assert index_dump(path).timesteps.tolist() == [0]

    def test_cut_preamble_after_the_last_frame_is_a_cut_header(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), "ITEM: UNITS\nlj\nITEM: U")
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert index.partial == PartialFrame(path, None)

    def test_units_are_recorded(self, tmp_path):
        path = write(tmp_path / "a.dump", "ITEM: UNITS\nmetal\n", frame(0), "ITEM: UNITS\nelectron\n", frame(100))
        assert index_dump(path).units == ("metal", "electron")
        assert index_dump(write(tmp_path / "b.dump", frame(0))).units == ()

    @pytest.mark.parametrize("extra", ["hello\n", "\0" * 50])
    def test_text_after_the_rows_names_its_byte(self, tmp_path, extra):
        path = write(tmp_path / "a.dump", frame(0), extra)
        offset = len(frame(0))
        with pytest.raises(ValueError, match=fr"timestep 0: the atom rows do not match the header's 6 atoms; unexpected text at byte {offset}: b'"):
            index_dump(path)

    def test_one_atom_frame_of_blank_rows_is_incomplete(self, tmp_path):
        blank = frame(100, atom_rows=["", ""], n_atoms=1)
        path = write(tmp_path / "a.dump", frame(0), blank, frame(200))
        with pytest.raises(ValueError, match="timestep 100 is incomplete but is not the last"):
            index_dump(path)

    def test_header_without_rows_before_another_frame_raises(self, tmp_path):
        empty = frame(100, atom_rows=[])
        path = write(tmp_path / "a.dump", frame(0), empty, frame(200))
        with pytest.raises(ValueError, match="timestep 100 is incomplete but is not the last"):
            index_dump(path)

    @pytest.mark.parametrize("text, timesteps", [
        ("hello world\n" + frame(0), [0]),
        ("ITEM: UNITS\nlj\nITEM: BONDS\n" + frame(0), [0]),
        (frame(0) + "ITEM: BONDS\n1 2 3\n" + frame(100), [0, 100]),
        (frame(0) + "ITEM: TIME\n" + frame(100), [0, 100]),
    ])
    def test_text_between_frames_is_skipped(self, tmp_path, text, timesteps):
        index = index_dump(write(tmp_path / "a.dump", text))
        assert index.timesteps.tolist() == timesteps
        assert index.partial is None

    def test_text_after_the_last_frame_is_a_cut_frame(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0) + "ITEM: BONDS\n1 2 3\n")
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert index.partial == PartialFrame(path, None)

    def test_value_cut_at_the_end_is_a_cut_header(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), "ITEM: TIME\n0.")
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert index.partial == PartialFrame(path, None)

    @pytest.mark.parametrize("tail", [
        "I", "IT", "ITE", "ITEM", "ITEM:", "ITEM: ", "ITEM: TI", "ITEM: TIMESTE",
        "ITEM: TIMESTEP", "ITEM: TIME",
    ])
    def test_cut_off_next_header_is_dropped(self, tmp_path, tail):
        path = write(tmp_path / "a.dump", frame(0) + tail)
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert index.partial == PartialFrame(path, None)
        assert read_block(path, index, 0) == "".join(r + "\n" for r in rows(0))

    def test_cut_off_next_header_after_a_frame_already_partial(self, tmp_path):
        # The rows of frame 100 are cut short, so that partial frame is reported.
        text = frame(0) + frame(100, atom_rows=rows(100)[:5]) + "ITE"
        path = write(tmp_path / "a.dump", text)
        index = index_dump(path)
        assert index.timesteps.tolist() == [0]
        assert index.partial == PartialFrame(path, 100)

    def test_other_unterminated_tail_raises(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0) + "garbage")
        with pytest.raises(ValueError, match="the atom rows do not match"):
            index_dump(path)

    def test_timesteps_must_increase(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(100), frame(100, 1.0))
        with pytest.raises(ValueError, match="timesteps do not increase"):
            index_dump(path)

    def test_timesteps_must_increase_from_frame_to_frame(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(200), frame(100))
        with pytest.raises(ValueError, match=r"timesteps do not increase \(200 is followed by 100\)"):
            index_dump(path)

    def test_columns_changing_within_a_file_are_recorded(self, tmp_path):
        reordered = frame(
            100, columns="type id x y z fx fy fz",
            atom_rows=[" ".join([r.split()[1], r.split()[0]] + r.split()[2:]) for r in rows(100)],
        )
        path = write(tmp_path / "a.dump", frame(0), reordered, frame(200))
        assert index_dump(path).mismatch == HeaderMismatch(
            1, f"{path}, timestep 100: the dump columns change within the file")

    def test_box_changing_within_a_file_is_recorded(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100),
                     frame(200, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n"), frame(300))
        index = index_dump(path)
        assert index.timesteps.tolist() == [0, 100, 200, 300]
        assert index.mismatch.frame == 2
        assert index.mismatch.message.startswith(
            f"{path}, timestep 200: the cell differs from the first frame's")

    @pytest.mark.parametrize("bounds", [
        "0.0 10.5\n0.0 10.0\n0.0 10.0\n",
        "0.0 10.0\n0.0 10.5\n0.0 10.0\n",
        "0.0 10.0\n0.0 10.0\n0.0 10.5\n",
        "-0.5 10.0\n0.0 10.0\n0.0 10.0\n",
    ])
    def test_any_bound_changing_within_a_file_is_recorded(self, tmp_path, bounds):
        path = write(tmp_path / "a.dump", frame(0), frame(100), frame(200, bounds=bounds))
        assert index_dump(path).mismatch.frame == 2

    def test_boundary_flags_changing_within_a_file_raise(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, box_item="pp pp ff"))
        with pytest.raises(ValueError, match=r"a\.dump, timestep 100: box boundaries must be periodic"):
            index_dump(path)

    def test_tilt_change_with_the_same_bounds_is_recorded(self, tmp_path):
        # Tilts (xy, xz) = (2, -1) and (-1, 2) give the same bounding box.
        first = frame(0, box_item="xy xz yz pp pp pp", bounds="-1.0 12.0 2.0\n0.0 10.5 -1.0\n0.0 10.0 0.0\n")
        second = frame(100, box_item="xy xz yz pp pp pp", bounds="-1.0 12.0 -1.0\n0.0 10.5 2.0\n0.0 10.0 0.0\n")
        path = write(tmp_path / "a.dump", first, second)
        index = index_dump(path)
        assert index.mismatch.frame == 1
        assert "timestep 100: the cell differs" in index.mismatch.message

    def test_atom_count_changing_within_a_file_is_recorded(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, atom_rows=rows(100)[:5], n_atoms=5))
        assert index_dump(path).mismatch == HeaderMismatch(
            1, f"{path}, timestep 100: 5 atoms, but the file's first frame has 6")

    def test_only_the_first_difference_is_recorded(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, atom_rows=rows(100)[:5], n_atoms=5),
                     frame(200, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n"))
        assert index_dump(path).mismatch.frame == 1

    def test_crlf_line_endings_read_as_lf(self, tmp_path):
        text = frame(0) + frame(100, 1.0)
        lf = write(tmp_path / "lf.dump", text)
        crlf = tmp_path / "crlf.dump"
        crlf.write_bytes(text.replace("\n", "\r\n").encode())
        assert index_dump(str(crlf)).timesteps.tolist() == [0, 100]
        for k in (0, 1):
            for got, expected in zip(_read(str(crlf), k), _read(lf, k)):
                np.testing.assert_array_equal(got, expected)

    def test_blank_lines_after_the_atom_rows_are_ignored(self, tmp_path):
        lf = write(tmp_path / "lf.dump", frame(0), frame(100, 1.0))
        blank = write(tmp_path / "blank.dump", frame(0), "\n  \n", frame(100, 1.0), "\n\n")
        index = index_dump(blank)
        assert index.timesteps.tolist() == [0, 100]
        assert index.partial is None
        for k in (0, 1):
            for got, expected in zip(_read(blank, k), _read(lf, k)):
                np.testing.assert_array_equal(got, expected)

    def test_malformed_header_numbers_name_the_file(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0).replace("\n6\n", "\nsix\n"))
        with pytest.raises(ValueError, match=r"a\.dump: malformed frame header at byte 0"):
            index_dump(path)

    def test_box_errors_name_the_file_and_timestep(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0, box_item="ff pp pp"))
        with pytest.raises(ValueError, match="a.dump, timestep 0: box boundaries must be periodic"):
            index_dump(path)

    @pytest.mark.parametrize("text", [
        "", "I", "ITEM: TI", "ITEM: TIMES", "ITEM: UNITS\nlj\n", "ITEM: UNITS\nl",
        "ITEM: TIME\n0.5\nITEM: TIM", "ITEM: UNITS\nlj\nITEM: TIME\n0.5\n",
        "ITEM: UNITS\r\nlj\r\n", "ITEM: UNITS\nlj\n ITEM: TI", "ITEM: TIME\n0.5",
        "ITEM: UNITS\nelec", "ITEM: TIME\n1.5e", "ITEM: UNITS\nlj\nITEM: U",
        "ITEM: NUMBER OF ATOMS\n6\n",
    ])
    def test_file_cut_before_its_first_frame_has_only_a_partial_frame(self, tmp_path, text):
        path = write(tmp_path / "a.dump", text)
        index = index_dump(path)
        assert index.header is None
        assert index.timesteps.tolist() == []
        assert index.partial == PartialFrame(path, None)

    @pytest.mark.parametrize("text", ["hello\n", "0 1 2\n3 4 5\n"])
    def test_non_dump_file_raises(self, tmp_path, text):
        with pytest.raises(ValueError, match="not a LAMMPS dump"):
            index_dump(write(tmp_path / "other.txt", text))

    def test_timestep_out_of_range_raises(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(2**63))
        with pytest.raises(ValueError, match=r"a\.dump: timestep 9223372036854775808 is out of range"):
            index_dump(path)

    def test_non_ascii_header_raises(self, tmp_path):
        path = tmp_path / "a.dump"
        path.write_bytes(frame(0).replace("NUMBER OF ATOMS", "NUMBER OF \u00c5TOMS").encode())
        with pytest.raises(ValueError, match=r"a\.dump: non-ASCII frame header at byte"):
            index_dump(str(path))

    def test_crlf_with_blank_lines_after_the_rows(self, tmp_path):
        lf = write(tmp_path / "lf.dump", frame(0), frame(100, 1.0))
        text = frame(0) + "\n \n" + frame(100, 1.0) + "\n"
        crlf = tmp_path / "crlf.dump"
        crlf.write_bytes(text.replace("\n", "\r\n").encode())
        index = index_dump(str(crlf))
        assert index.timesteps.tolist() == [0, 100]
        assert index.partial is None
        for k in (0, 1):
            for got, expected in zip(_read(str(crlf), k), _read(lf, k)):
                np.testing.assert_array_equal(got, expected)


def _combined(tmp_path, *files):
    """files: one list of frame strings per dump file.

    Returns the paths and the (path, timestep) of each joined frame, after
    checking that each frame's byte range is the one its file's index gives.
    """
    paths = [write(tmp_path / f"run{k}.dump", *frames) for k, frames in enumerate(files)]
    indices = {p: index_dump(p) for p in paths}
    table = combine_frames(list(indices.values()))
    frames = []
    for i in range(len(table)):
        index = indices[table.path(i)]
        k = index.timesteps.tolist().index(table.timesteps[i])
        assert (table.starts[i], table.ends[i]) == (index.starts[k], index.ends[k])
        frames.append((table.path(i), int(table.timesteps[i])))
    return paths, frames


class TestCombineFrames:

    def test_consecutive_files_are_joined(self, tmp_path):
        paths, frames = _combined(tmp_path, [frame(0), frame(100)], [frame(200), frame(300)])
        assert frames == [
            (paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[1], 300)]

    def test_overlap_is_replaced_by_the_later_file(self, tmp_path):
        paths, frames = _combined(
            tmp_path, [frame(0), frame(100), frame(200)], [frame(100, 50.0), frame(300, 52.0)]
            )
        assert frames == [
            (paths[0], 0), (paths[1], 100), (paths[1], 300)]

    def test_single_repeated_boundary_frame_is_kept_once(self, tmp_path):
        paths, frames = _combined(tmp_path, [frame(0), frame(100)], [frame(100), frame(200)])
        assert frames == [
            (paths[0], 0), (paths[1], 100), (paths[1], 200)]

    def test_restart_before_the_previous_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match=(
                "starts at timestep 0, at or before the start of .*timesteps must continue from "
                "file to file, as in a run and its restarts. Load independent runs as separate "
                "trajectories")):
            _combined(tmp_path, [frame(100), frame(200)], [frame(0), frame(100)])

    def test_restart_at_the_previous_file_start_raises(self, tmp_path):
        with pytest.raises(ValueError, match="starts at timestep 100, at or before the start of"):
            _combined(tmp_path, [frame(100), frame(200)], [frame(100), frame(300)])

    def test_partial_final_frame_warns_when_not_replaced(self, tmp_path):
        cut = frame(200)[:-30]
        with pytest.warns(UserWarning, match=r"run0\.dump: dropping the incomplete final frame \(timestep 200\)"):
            _, frames = _combined(tmp_path, [frame(0), frame(100), cut])
        assert [t for _, t in frames] == [0, 100]

    def test_partial_final_frame_replaced_by_the_next_file_still_warns(self, tmp_path):
        cut = frame(200)[:-30]
        with pytest.warns(UserWarning, match=r"run0\.dump: dropping the incomplete final frame \(timestep 200\)"):
            paths, frames = _combined(tmp_path, [frame(0), frame(100), cut], [frame(200), frame(300)])
        assert frames == [
            (paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[1], 300)]

    def test_partial_frame_followed_by_a_gap_warns(self, tmp_path):
        cut = frame(200)[:-30]
        with pytest.warns(UserWarning, match="incomplete final frame"):
            _, frames = _combined(tmp_path, [frame(0), frame(100), cut], [frame(300)])
        assert [t for _, t in frames] == [0, 100, 300]

    def test_middle_file_truncated_by_the_next_restart(self, tmp_path):
        paths, frames = _combined(
            tmp_path,
            [frame(0), frame(100), frame(200), frame(300)],
            [frame(200, 20.0), frame(300, 30.0), frame(400, 40.0)],
            [frame(300, 33.0), frame(500, 55.0)],
            )
        assert frames == [
            (paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[2], 300), (paths[2], 500)]

    def test_file_with_only_a_partial_frame_between_two_files(self, tmp_path):
        cut = frame(200)[:-30]
        with pytest.warns(UserWarning, match=r"run1\.dump: dropping the incomplete final frame \(timestep 200\)"):
            paths, frames = _combined(tmp_path, [frame(0), frame(100)], [cut], [frame(300), frame(400)])
        assert frames == [(paths[0], 0), (paths[0], 100), (paths[2], 300), (paths[2], 400)]

    def test_cut_header_replaced_by_a_restart_still_warns(self, tmp_path):
        with pytest.warns(UserWarning, match=r"run0\.dump: dropping the incomplete final frame;"):
            paths, frames = _combined(
                tmp_path, [frame(0), frame(100), "ITEM: TIM"], [frame(100, 1.0), frame(200)])
        assert frames == [(paths[0], 0), (paths[1], 100), (paths[1], 200)]

    def test_cut_header_not_replaced_by_the_next_file_warns(self, tmp_path):
        with pytest.warns(UserWarning, match=r"run0\.dump: dropping the incomplete final frame;"):
            paths, frames = _combined(
                tmp_path, [frame(0), frame(100), "ITEM: TIM"], [frame(200), frame(300)])
        assert frames == [(paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[1], 300)]

    def test_joined_arrays_are_read_only(self, tmp_path):
        paths = [write(tmp_path / "a.dump", frame(0)), write(tmp_path / "b.dump", frame(100))]
        table = combine_frames([index_dump(p) for p in paths])
        for values in (table.first, table.timesteps, table.starts, table.ends):
            with pytest.raises(ValueError, match="read-only"):
                values[0] = 1

    def test_no_complete_frames_raises(self, tmp_path):
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="no complete frames"):
                _combined(tmp_path, [frame(0)[:-30]])

    def test_restart_into_an_earlier_file_of_three_raises(self, tmp_path):
        # run2 starts after run0 but at or before run1's start, which would
        # discard the whole of run1.
        with pytest.raises(ValueError, match=r"run2\.dump starts at timestep 150, at or before the start of .*run1\.dump"):
            _combined(tmp_path, [frame(0), frame(100)], [frame(200), frame(300)], [frame(150), frame(400)])
        with pytest.raises(ValueError, match=r"run2\.dump starts at timestep 50"):
            _combined(tmp_path, [frame(0), frame(100)], [frame(200), frame(300)], [frame(50), frame(400)])

    def test_three_files_with_a_restart_into_the_second(self, tmp_path):
        paths, frames = _combined(
            tmp_path, [frame(0), frame(100)], [frame(200), frame(300)], [frame(250), frame(400)])
        assert frames == [
            (paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[2], 250), (paths[2], 400)]

    def test_cut_header_in_a_middle_file_replaced_by_a_restart_still_warns(self, tmp_path):
        with pytest.warns(UserWarning, match=r"run1\.dump: dropping the incomplete final frame;"):
            paths, frames = _combined(
                tmp_path, [frame(0), frame(100)], [frame(200), frame(300), "ITEM: TIM"],
                [frame(300, 1.0), frame(400)])
        assert frames == [(paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[2], 300), (paths[2], 400)]

    def test_empty_last_file_warns(self, tmp_path):
        with pytest.warns(UserWarning, match=r"run1\.dump: dropping the incomplete final frame"):
            paths, frames = _combined(tmp_path, [frame(0), frame(100)], [""])
        assert frames == [(paths[0], 0), (paths[0], 100)]

    def test_replaced_frames_are_dropped_silently(self, tmp_path):
        # A restart that ends before the frames it replaces, and one that
        # dumps less often: the earlier file's frames from the restart on go.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths, frames = _combined(
                tmp_path, [frame(t) for t in range(0, 401, 50)], [frame(200), frame(300)])
        assert frames == [(paths[0], t) for t in range(0, 151, 50)] + [
            (paths[1], 200), (paths[1], 300)]

    def test_restart_reaching_the_end_of_the_frames_it_replaces_is_silent(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, frames = _combined(
                tmp_path, [frame(0), frame(100), frame(200), frame(300)], [frame(200), frame(300)])
        assert [t for _, t in frames] == [0, 100, 200, 300]

    def test_header_change_in_frames_a_restart_replaces_is_accepted(self, tmp_path):
        lost_atom = frame(300, atom_rows=rows(300)[:5], n_atoms=5)
        grown_box = frame(300, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n")
        for changed in (lost_atom, grown_box):
            paths, frames = _combined(
                tmp_path, [frame(0), frame(100), frame(200), changed], [frame(200), frame(300), frame(400)])
            assert frames == [(paths[0], 0), (paths[0], 100), (paths[1], 200), (paths[1], 300), (paths[1], 400)]

    def test_header_change_in_kept_frames_raises(self, tmp_path):
        lost_atom = frame(300, atom_rows=rows(300)[:5], n_atoms=5)
        with pytest.raises(ValueError, match=r"run0\.dump, timestep 300: 5 atoms, but the file's first frame has 6"):
            _combined(tmp_path, [frame(0), frame(100), lost_atom], [frame(400)])
        grown_box = frame(300, bounds="0.0 10.5\n0.0 10.0\n0.0 10.0\n")
        with pytest.raises(ValueError, match=r"run1\.dump, timestep 300: the cell differs"):
            _combined(tmp_path, [frame(0)], [frame(200), grown_box])

    def test_box_of_a_later_files_first_frame_is_checked(self, tmp_path):
        bounds = "0.0 10.5\n0.0 10.0\n0.0 10.0\n"
        # The rest of run1 matches run0, so the first differing frame is run1's first.
        with pytest.raises(ValueError, match=r"run1\.dump, timestep 200: the cell differs"):
            _combined(tmp_path, [frame(0), frame(100)], [frame(200, bounds=bounds), frame(300)])

    def test_random_joins_match_a_plain_list_join(self, tmp_path):
        # A plain join of (path, timestep) lists: each file replaces every
        # earlier frame from its first timestep on, and may not start at or
        # before the first timestep of the file before it.
        def plain_join(files):
            joined, previous = [], None
            for path, timesteps, _ in files:
                if not timesteps:
                    continue
                if previous is not None and timesteps[0] <= previous[0]:
                    return None
                joined = [(p, t) for p, t in joined if t < timesteps[0]] + [(path, t) for t in timesteps]
                previous = timesteps
            return joined

        def reported_drops(messages):
            pattern = re.compile(r"(\S+): dropping the incomplete final frame(?: \(timestep (\d+)\))?;")
            return [(m.group(1), int(m.group(2)) if m.group(2) else None)
                    for m in map(pattern.match, messages) if m]

        rng = random.Random(20261015)
        for case in range(150):
            directory = tmp_path / str(case)
            directory.mkdir()
            # Each file: (path, timesteps, partial), where partial is None, or
            # (timestep or None,) for a final frame cut short.
            files, tails, start = [], [], 0
            for k in range(rng.randint(1, 4)):
                path = str(directory / f"run{k}.dump")
                if rng.random() < 0.08:  # a job that wrote nothing
                    files.append((path, [], (None,)))
                    tails.append("")
                    continue
                start = max(0, start + rng.choices([-100, 0, 100, 200, 300, 500], [1, 1, 3, 3, 3, 2])[0])
                interval = rng.choice([50, 100, 200])
                timesteps = [start + interval * j for j in range(rng.randint(1, 6))]
                kind = rng.choices(["none", "frame", "header"], [6, 2, 1])[0]
                if kind == "frame":  # killed while writing the next frame's rows
                    files.append((path, timesteps, (timesteps[-1] + interval,)))
                    tails.append(frame(timesteps[-1] + interval)[:-25])
                elif kind == "header":  # killed while writing its header
                    files.append((path, timesteps, (None,)))
                    tails.append("ITEM: TIM")
                else:
                    files.append((path, timesteps, None))
                    tails.append("")
            expected = plain_join(files)
            # Sometimes give one frame after a file's first a different box:
            # an error if and only if that frame is kept.
            changed = None
            candidates = [(path, t) for path, ts, _ in files for t in ts[1:]]
            replaced = [c for c in candidates if expected is not None and c not in expected]
            if replaced and rng.random() < 0.5:
                changed = rng.choice(replaced)
            elif candidates and rng.random() < 0.4:
                changed = rng.choice(candidates)
            grown = "0.0 10.5\n0.0 10.0\n0.0 10.0\n"
            texts = [[frame(t, bounds=grown) if (path, t) == changed else frame(t) for t in ts] + [tail]
                     for (path, ts, _), tail in zip(files, tails)]
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                if expected is None:
                    with pytest.raises(ValueError, match="at or before the start of"):
                        _combined(directory, *texts)
                    continue
                if not expected:
                    with pytest.raises(ValueError, match="no complete frames"):
                        _combined(directory, *texts)
                    continue
                if changed in expected:
                    with pytest.raises(ValueError, match=f"timestep {changed[1]}: the cell differs"):
                        _combined(directory, *texts)
                    continue
                _, got = _combined(directory, *texts)
            assert got == expected
            # Every incomplete final frame is dropped with a warning, and
            # nothing else warns.
            messages = [str(w.message) for w in record]
            dropped = [(path, partial[0]) for path, _, partial in files if partial is not None]
            assert reported_drops(messages) == dropped and len(messages) == len(dropped)
            steps = [t for _, t in got]
            assert steps == sorted(set(steps))
            last_with_frames = [(path, ts) for path, ts, _ in files if ts][-1]
            assert [t for p, t in got if p == last_with_frames[0]] == last_with_frames[1]

class TestResolveAtomStyle:

    def test_column_string_is_passed_through(self):
        assert resolve_atom_style("id type charge x y z", SMALL_DATA) == "id type charge x y z"

    @pytest.mark.parametrize("name, columns", [
        ("atomic", "id type x y z"),
        ("charge", "id type charge x y z"),
        ("full", "id resid type charge x y z"),
        ("molecular", "id resid type x y z"),
        ("bond", "id resid type x y z"),
        ("angle", "id resid type x y z"),
    ])
    def test_style_names(self, name, columns):
        assert resolve_atom_style(name, SMALL_DATA) == columns

    def test_style_read_from_the_data_file(self):
        assert resolve_atom_style(None, SMALL_DATA) == "id type x y z"   # Atoms # atomic
        assert resolve_atom_style(None, LJ_DATA) == "id resid type charge x y z"  # Atoms # full

    def test_title_line_is_not_read_as_the_atoms_header(self, tmp_path):
        # LAMMPS reads the first line of a data file as a free-text title.
        lines = Path(SMALL_DATA).read_text().splitlines(keepends=True)
        path = tmp_path / "data.title"
        path.write_text("Atoms # sphere title text\n" + "".join(lines[1:]))
        assert resolve_atom_style(None, str(path)) == "id type x y z"

    def test_data_file_without_a_style_comment(self, tmp_path):
        text = Path(SMALL_DATA).read_text().replace("Atoms # atomic", "Atoms")
        path = tmp_path / "data.nocomment"
        path.write_text(text)
        with pytest.raises(ValueError, match="no 'Atoms # <style>' comment.*column string"):
            resolve_atom_style(None, str(path))

    def test_unknown_style(self, tmp_path):
        text = Path(SMALL_DATA).read_text().replace("Atoms # atomic", "Atoms # sphere")
        path = tmp_path / "data.sphere"
        path.write_text(text)
        with pytest.raises(ValueError, match="'sphere' has no known column layout"):
            resolve_atom_style(None, str(path))
        with pytest.raises(ValueError, match="'sphere' has no known column layout"):
            resolve_atom_style("sphere", SMALL_DATA)


def _read(path, k=0):
    dump = index_dump(path)
    layout = resolve_columns(dump.header.columns, path)
    with open(path, "rb") as f:
        return read_dump_frame(f, dump.starts[k], dump.ends[k], dump.header, layout, IDS_SORTED)


class TestReadDumpFrame:

    def test_rows_are_returned_in_id_order(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0), frame(100, 1.0))
        positions, forces = _read(path, 1)
        np.testing.assert_array_equal(positions, expected_positions(1.0))
        np.testing.assert_array_equal(forces, expected_forces(100))
        assert positions.dtype == np.float64 and forces.dtype == np.float64

    def test_scaled_coordinates_are_converted(self, tmp_path):
        cell = np.array([[10.0, 0, 0], [2.0, 10.0, 0], [1.0, -1.5, 10.0]])
        bounds = "0.0 13.0 2.0\n-1.5 10.0 1.0\n0.0 10.0 -1.5\n"
        scaled = expected_positions() @ np.linalg.inv(cell)
        by_id = dict(zip(IDS_SORTED, scaled))
        atom_rows = [
            f"{i} 1 {float(by_id[i][0])!r} {float(by_id[i][1])!r} {float(by_id[i][2])!r} {-i} 0 {i}"
            for i in (20, 3, 40, 12, 21, 7)
        ]
        path = write(tmp_path / "s.dump", frame(
            0, box_item="xy xz yz pp pp pp", bounds=bounds,
            columns="id type xs ys zs fx fy fz", atom_rows=atom_rows,
        ))
        positions, _ = _read(path)
        np.testing.assert_allclose(positions, expected_positions(), rtol=1e-12, atol=1e-12)

    def test_scaled_coordinates_are_offset_by_the_box_origin(self, tmp_path):
        # The cell of the test above, with the box origin moved to (5, -3, 2).
        origin = np.array([5.0, -3.0, 2.0])
        cell = np.array([[10.0, 0, 0], [2.0, 10.0, 0], [1.0, -1.5, 10.0]])
        bounds = "5.0 18.0 2.0\n-4.5 7.0 1.0\n2.0 12.0 -1.5\n"
        scaled = (expected_positions() - origin) @ np.linalg.inv(cell)
        by_id = dict(zip(IDS_SORTED, scaled))
        atom_rows = [
            f"{i} 1 {float(by_id[i][0])!r} {float(by_id[i][1])!r} {float(by_id[i][2])!r} {-i} 0 {i}"
            for i in (20, 3, 40, 12, 21, 7)
        ]
        path = write(tmp_path / "o.dump", frame(
            0, box_item="xy xz yz pp pp pp", bounds=bounds,
            columns="id type xs ys zs fx fy fz", atom_rows=atom_rows,
        ))
        np.testing.assert_array_equal(index_dump(path).header.origin, origin)
        positions, _ = _read(path)
        np.testing.assert_allclose(positions, expected_positions(), rtol=1e-12, atol=1e-12)

    def test_scaled_coordinates_follow_the_lammps_formula(self, tmp_path):
        # Bounds for the cell a = (10, 0, 0), b = (2, 10, 0), c = (1, -1.5, 10)
        # with origin (5, -3, 2); position = origin + sa a + sb b + sc c.
        bounds = "5.0 18.0 2.0\n-4.5 7.0 1.0\n2.0 12.0 -1.5\n"
        cases = {
            3: ((0, 0, 0), (5.0, -3.0, 2.0)),
            7: ((1, 1, 1), (18.0, 5.5, 12.0)),
            12: ((0.25, 0.5, 0.75), (9.25, 0.875, 9.5)),
            20: ((0.5, 0, 0), (10.0, -3.0, 2.0)),
            21: ((0, 0.5, 0), (6.0, 2.0, 2.0)),
            40: ((0, 0, 0.5), (5.5, -3.75, 7.0)),
        }
        atom_rows = [f"{i} 1 {sa} {sb} {sc} 0 0 0" for i, ((sa, sb, sc), _) in cases.items()]
        path = write(tmp_path / "s.dump", frame(
            0, box_item="xy xz yz pp pp pp", bounds=bounds,
            columns="id type xs ys zs fx fy fz", atom_rows=atom_rows,
        ))
        positions, _ = _read(path)
        expected = [cases[i][1] for i in IDS_SORTED]
        np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-14)

    def test_types_not_matching_the_topology_raise(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0))
        dump = index_dump(path)
        layout = resolve_columns(dump.header.columns, path)
        types = np.array([b"1", b"2", b"1", b"2", b"2", b"2"])  # id 21 should be type 1
        with open(path, "rb") as f, pytest.raises(
                ValueError, match="atom id 21 has type 1 in the dump and 2 in the topology"):
            read_dump_frame(f, dump.starts[0], dump.ends[0], dump.header, layout, IDS_SORTED, types)

    def test_ids_not_matching_the_topology_raise(self, tmp_path):
        path = write(tmp_path / "a.dump", frame(0))
        dump = index_dump(path)
        layout = resolve_columns(dump.header.columns, path)
        wrong = np.array([3, 7, 12, 20, 21, 41])
        with open(path, "rb") as f, pytest.raises(ValueError, match="ids do not match the topology"):
            read_dump_frame(f, dump.starts[0], dump.ends[0], dump.header, layout, wrong)

    def test_rows_with_missing_values_raise(self, tmp_path):
        atom_rows = [r.rsplit(" ", 1)[0] if k == 2 else r for k, r in enumerate(rows(0))]
        path = write(tmp_path / "a.dump", frame(0, atom_rows=atom_rows))
        with pytest.raises(ValueError, match="do not all have the 8 columns"):
            _read(path)

    def test_numbers_parse_exactly_as_python_floats(self):
        # read_dump_frame converts byte strings with NumPy; it must agree
        # bit for bit with float() so results do not depend on the parser.
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(0, 1e3, 2000), rng.normal(0, 1e-7, 2000)])
        tokens = [repr(float(v)).encode() for v in values] + [f"{v:.6g}".encode() for v in values]
        parsed = np.array(tokens).astype(np.float64)
        np.testing.assert_array_equal(parsed, [float(t) for t in tokens])
