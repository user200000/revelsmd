"""LAMMPS backend: LammpsTrajectory and the helpers that index and read dump files."""

import array
import contextlib
import mmap
import os
import re
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import BinaryIO

import numpy as np

from revelsMD.cell import fractional_to_cartesian
from ._base import (
    TrajectoryFiles, TrajectoryPath,
    FileFrames, IndexedFileTrajectory, as_paths, check_fixed_cell,
    atom_values, given_values, open_universe, topology_values, warn_cut_short,
)


# Column layouts of the Atoms section for LAMMPS atom styles (read_data
# documentation), as MDAnalysis column strings.
STYLE_COLUMNS: dict[str, str] = {
    "atomic": "id type x y z",
    "charge": "id type charge x y z",
    "full": "id resid type charge x y z",
    "molecular": "id resid type x y z",
    "bond": "id resid type x y z",
    "angle": "id resid type x y z",
}

# Coordinate column sets in order of preference, and whether they are scaled
# (fractional) coordinates.
COORDINATE_SETS: tuple[tuple[tuple[str, str, str], bool], ...] = (
    (("x", "y", "z"), False),
    (("xu", "yu", "zu"), False),
    (("xs", "ys", "zs"), True),
    (("xsu", "ysu", "zsu"), True),
)
FORCE_COLUMNS = ("fx", "fy", "fz")


def parse_box(
    item_line: str, bound_lines: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an ``ITEM: BOX BOUNDS`` line and the three bound lines after it.

    Returns the cell (rows a, b, c), the origin (xlo, ylo, zlo), and a box
    array of the six bounds as written then the tilts xy, xz, yz (zero if
    orthogonal); equal box arrays mean an identical box. Raises
    ValueError for a box that is not fully periodic, is general triclinic
    (``abc origin``), or is malformed or not finite.
    """
    words = item_line.split()[3:]
    if words[:1] == ["abc"]:
        raise ValueError("general triclinic ('abc origin') dump boxes are not supported")
    triclinic = words[:3] == ["xy", "xz", "yz"]
    flags = words[3:] if triclinic else words
    if flags != ["pp", "pp", "pp"]:
        raise ValueError(
            "box boundaries must be periodic in all directions (pp pp pp); "
            f"got '{' '.join(flags)}'"
        )
    width = 3 if triclinic else 2
    values = np.array([line.split() for line in bound_lines], dtype=np.float64)
    if values.shape != (3, width):
        raise ValueError(f"malformed box bounds: {bound_lines}")
    if not np.isfinite(values).all():
        raise ValueError(f"box bounds must be finite: {bound_lines}")
    if triclinic:
        xy, xz, yz = (float(v) for v in values[:, 2])
    else:
        xy = xz = yz = 0.0
    xlo = values[0, 0] - min(0.0, xy, xz, xy + xz)
    xhi = values[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = values[1, 0] - min(0.0, yz)
    yhi = values[1, 1] - max(0.0, yz)
    zlo, zhi = values[2, 0], values[2, 1]
    cell = np.array([
        [xhi - xlo, 0.0, 0.0],
        [xy, yhi - ylo, 0.0],
        [xz, yz, zhi - zlo],
    ])
    origin = np.array([xlo, ylo, zlo])
    box = np.concatenate([values[:, :2].ravel(), [xy, xz, yz]])
    return cell, origin, box


@dataclass(frozen=True)
class ColumnLayout:
    """Where the id, coordinate and force columns are in a dump's rows."""

    n_columns: int
    id_column: int
    coordinate_columns: tuple[int, int, int]
    force_columns: tuple[int, int, int]
    scaled: bool
    type_column: int | None = None


def resolve_columns(columns: tuple[str, ...], path: str) -> ColumnLayout:
    """Locate the id, coordinate and force columns of a dump.

    Coordinates come from the first set in ``COORDINATE_SETS`` present.
    Raises ValueError if the id, the forces or every coordinate set is missing.
    """
    missing = []
    if "id" not in columns:
        missing.append("id")
    if not all(name in columns for name in FORCE_COLUMNS):
        missing.append("fx fy fz")
    coordinates = next(
        ((names, scaled) for names, scaled in COORDINATE_SETS
         if all(name in columns for name in names)),
        None,
    )
    if coordinates is None:
        missing.append("one of x y z, xu yu zu, xs ys zs, xsu ysu zsu")
    if missing or coordinates is None:
        raise ValueError(
            f"{path}: the dump columns ({' '.join(columns)}) lack "
            f"{'; '.join(missing)}; atom ids, positions and forces are required"
        )
    names, scaled = coordinates
    x, y, z = (columns.index(name) for name in names)
    fx, fy, fz = (columns.index(name) for name in FORCE_COLUMNS)
    type_column = columns.index("type") if "type" in columns else None
    return ColumnLayout(
        len(columns), columns.index("id"), (x, y, z), (fx, fy, fz), scaled, type_column
    )


_TIMESTEP = b"ITEM: TIMESTEP"
# The items that dump_modify units yes and time yes write before a frame's
# ITEM: TIMESTEP line, each followed by a line holding its value.
_UNITS = b"ITEM: UNITS"
_PREAMBLE_ITEMS = (_UNITS, b"ITEM: TIME")
_CONTENT = re.compile(rb"\S")


@dataclass(frozen=True, eq=False)
class DumpHeader:
    """What every complete frame of one dump file shares."""

    n_atoms: int
    columns: tuple[str, ...]
    cell: np.ndarray
    origin: np.ndarray
    box: np.ndarray


@dataclass(frozen=True)
class PartialFrame:
    """A dump file's unreadable final frame: its timestep, if known, and why."""

    path: str
    timestep: int | None
    reason: str = field(default="the file is empty", compare=False)


@dataclass(frozen=True)
class HeaderMismatch:
    """The first frame of a dump file whose header differs from the first frame's."""

    frame: int
    message: str


@dataclass(frozen=True, eq=False)
class DumpIndex:
    """The complete frames of one dump file, and its unreadable final frame.

    Frame ``k`` has timestep ``timesteps[k]`` (strictly increasing) and atom
    rows in bytes ``[starts[k], ends[k])``. ``header`` is the first complete
    frame's (None if there are none); frames before ``mismatch.frame``, or
    all if ``mismatch`` is None, share it.
    """

    path: str
    header: DumpHeader | None
    timesteps: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    partial: PartialFrame | None
    mismatch: HeaderMismatch | None = None
    units: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ScannedFrame:
    """A complete frame; ``header_lines`` are its stripped lines from NUMBER OF ATOMS to ATOMS."""

    timestep: int
    data_start: int
    data_end: int
    header: DumpHeader
    header_lines: list[str]
    units: tuple[str, ...]


@dataclass(frozen=True)
class _Unreadable:
    """A frame that cannot be read: why, where, and its timestep if known."""

    position: int
    timestep: int | None
    reason: str


def index_dump(path: str) -> DumpIndex:
    """Index the frames of one dump file without parsing the atom rows.

    An unreadable frame with no readable frame after it is recorded as
    ``partial``. The first frame whose columns, box or atom count differ
    from the file's first is recorded as ``mismatch``, not raised, since a
    later file may replace it. Raises ValueError if an unreadable frame is
    followed by a readable one, timesteps do not strictly increase, or a
    box is not supported.
    """
    timesteps, starts, ends = array.array("q"), array.array("q"), array.array("q")
    units: dict[str, None] = {}
    first: _ScannedFrame | None = None
    mismatch: HeaderMismatch | None = None
    partial: PartialFrame | None = None
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if size else contextlib.nullcontext(b"") as m:
            position = 0
            while _CONTENT.search(m, position) is not None:
                frame = _read_frame(m, path, position, first)
                if isinstance(frame, _Unreadable):
                    where = f"{frame.reason} at byte {frame.position}"
                    if _readable_frame_follows(m, path, frame.position, first):
                        raise ValueError(
                            f"{path}: {where}, but readable frames follow it; the file is "
                            "damaged, or was appended to after a job was killed "
                            "(dump_modify append yes), in which case split it at the restart"
                        )
                    partial = PartialFrame(path, frame.timestep, where)
                    break
                if first is None:
                    first = frame
                else:
                    if mismatch is None:
                        difference = _header_difference(frame, first.header, path)
                        if difference is not None:
                            mismatch = HeaderMismatch(len(timesteps), difference)
                    if frame.timestep <= timesteps[-1]:
                        raise ValueError(
                            f"{path}: timesteps do not increase "
                            f"({timesteps[-1]} is followed by {frame.timestep})"
                        )
                units.update(dict.fromkeys(frame.units))
                timesteps.append(frame.timestep)
                starts.append(frame.data_start)
                ends.append(frame.data_end)
                position = frame.data_end
    if first is None and partial is None:
        partial = PartialFrame(path, None)
    return DumpIndex(
        path, None if first is None else first.header,
        np.array(timesteps, dtype=np.int64), np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64), partial, mismatch, tuple(units),
    )


def _readable_frame_follows(
    m: "mmap.mmap | bytes", path: str, position: int, first: _ScannedFrame | None
) -> bool:
    """Whether a frame that can be read starts after byte ``position``."""
    following = m.find(_TIMESTEP, position + 1)
    while following != -1:
        if isinstance(_read_frame(m, path, following, first), _ScannedFrame):
            return True
        following = m.find(_TIMESTEP, following + 1)
    return False


def _next_line(m: "mmap.mmap | bytes", position: int) -> tuple[bytes, int] | None:
    """The stripped line at ``position`` and the position after it; None if it has no end."""
    newline = m.find(b"\n", position)
    if newline == -1:
        return None
    return m[position:newline].strip(), newline + 1


def _header_difference(entry: _ScannedFrame, first: DumpHeader, path: str) -> str | None:
    """How a frame's header differs from the file's first frame's, or None."""
    if entry.header is first:
        return None
    where = f"{path}, timestep {entry.timestep}"
    if entry.header.columns != first.columns:
        return f"{where}: the dump columns change within the file"
    try:
        check_fixed_cell(entry.header.box, first.box, where)
    except ValueError as e:
        return str(e)
    if entry.header.n_atoms != first.n_atoms:
        return (
            f"{where}: {entry.header.n_atoms} atoms, but the file's first frame "
            f"has {first.n_atoms}"
        )
    return None


def _read_frame(
    m: "mmap.mmap | bytes", path: str, start: int, first: _ScannedFrame | None
) -> "_ScannedFrame | _Unreadable":
    """Read the frame at byte ``start``, with any ITEM: UNITS and ITEM: TIME items before it.

    A frame whose header lines equal ``first``'s shares its header. The
    frame ends after its atom rows and any blank lines. A frame that cannot
    be read is returned as ``_Unreadable``; only an unsupported box in a
    complete header raises ValueError.
    """
    units: list[str] = []
    position = start
    while True:
        read = _next_line(m, position)
        if read is None:
            return _Unreadable(start, None, "a frame cut short")
        line, after = read
        if line == _TIMESTEP:
            break
        if line in _PREAMBLE_ITEMS:
            value = _next_line(m, after)
            while value is not None and not value[0]:
                value = _next_line(m, value[1])
            if value is None:
                return _Unreadable(start, None, "a frame cut short")
            if value[0].startswith(b"ITEM:"):
                return _Unreadable(position, None, f"{line.decode()} without a value")
            if line == _UNITS:
                units.append(value[0].decode("ascii", "backslashreplace"))
            after = value[1]
        elif line:
            return _Unreadable(position, None, f"unexpected text {line[:40]!r}")
        position = after
    frame_start = position
    lines: list[str] = []
    while len(lines) < 9:
        newline = m.find(b"\n", position)
        if newline == -1:
            return _Unreadable(frame_start, _header_timestep(lines), "a frame header cut short")
        try:
            lines.append(m[position:newline].decode("ascii").strip())
        except UnicodeDecodeError:
            return _Unreadable(position, _header_timestep(lines), "a non-ASCII frame header")
        position = newline + 1
    if (
        lines[2] != "ITEM: NUMBER OF ATOMS"
        or not lines[4].startswith("ITEM: BOX BOUNDS")
        or not lines[8].startswith("ITEM: ATOMS")
    ):
        return _Unreadable(frame_start, _header_timestep(lines), f"an unexpected frame header {lines}")
    try:
        timestep, n_atoms = int(lines[1]), int(lines[3])
    except ValueError:
        return _Unreadable(frame_start, None, "a malformed frame header")
    if not 0 <= timestep < 2**63:
        return _Unreadable(frame_start, None, f"an out-of-range timestep {timestep}")
    header_lines = lines[2:9]
    if first is not None and header_lines == first.header_lines:
        header = first.header
    else:
        try:
            cell, origin, box = parse_box(lines[4], lines[5:8])
        except ValueError as e:
            raise ValueError(f"{path}, timestep {timestep}: {e}") from e
        header = DumpHeader(n_atoms, tuple(lines[8].split()[2:]), cell, origin, box)
    next_item = m.find(b"ITEM:", position)
    rows_end = len(m) if next_item == -1 else next_item
    # Blank lines after the atom rows belong to the frame; a last row
    # without a newline may be cut short, so it is not complete.
    rows = m[position:rows_end]
    content_end = len(rows)
    while content_end and rows[content_end - 1] in b" \t\r\n":
        content_end -= 1
    terminated = rows.find(b"\n", content_end) != -1
    n_rows = rows.count(b"\n", 0, content_end) + (content_end > 0)
    complete_rows = n_rows if terminated or n_rows == 0 else n_rows - 1
    if complete_rows < n_atoms:
        return _Unreadable(frame_start, timestep, "atom rows cut short")
    data_end = rows_end
    if n_rows > n_atoms:  # text follows the rows; the next frame starts there
        data_end = position
        for _ in range(n_atoms):
            data_end = m.find(b"\n", data_end) + 1
    return _ScannedFrame(timestep, position, data_end, header, header_lines, tuple(units))


def _header_timestep(lines: list[str]) -> int | None:
    """The timestep of a cut-off header, if its value line was written."""
    try:
        return int(lines[1])
    except (IndexError, ValueError):
        return None


@dataclass(frozen=True, eq=False)
class FrameTable(FileFrames):
    """:class:`FileFrames` with each frame's timestep (strictly increasing) and each file's header.

    Only files that contribute frames are listed.
    """

    headers: list[DumpHeader]
    timesteps: np.ndarray


def combine_frames(indices: list[DumpIndex]) -> FrameTable:
    """Join the frames of dump files, given in run order, into one trajectory.

    Each file's frames replace the earlier files' frames from its own first
    timestep on, and incomplete final frames are dropped with a warning.
    Kept frames must match their file's first frame, and each file's first
    frame the first file's box; replaced frames are not compared. Raises
    ValueError if a file starts at or before the first timestep of the file
    before it, a kept frame's header differs, or there are no complete frames.
    """
    files: list[DumpIndex] = []
    headers: list[DumpHeader] = []
    for index in indices:
        if index.header is not None:
            files.append(index)
            headers.append(index.header)
    if not files:
        raise ValueError("no complete frames in " + "; ".join(
            index.path if index.partial is None else f"{index.path} ({index.partial.reason})"
            for index in indices
        ))
    for index in indices:
        if index.partial is not None:
            at = "" if index.partial.timestep is None else f" (timestep {index.partial.timestep})"
            warn_cut_short(index.partial.path, f"frame{at}", index.partial.reason)
    kept = [len(index.timesteps) for index in files]
    for k in range(1, len(files)):
        first, previous = int(files[k].timesteps[0]), files[k - 1]
        if first <= previous.timesteps[0]:
            raise ValueError(
                f"{files[k].path} starts at timestep {first}, at or before the start of "
                f"{previous.path} (timestep {int(previous.timesteps[0])}); timesteps must "
                "continue from file to file, as in a run and its restarts. Load "
                "independent runs as separate trajectories"
            )
        # Every earlier file's frames precede the previous file's first
        # timestep, so only the previous file's frames are replaced.
        kept[k - 1] = int(np.searchsorted(previous.timesteps, first))
    for index, header, n_kept in zip(files, headers, kept, strict=True):
        check_fixed_cell(
            header.box, headers[0].box, f"{index.path}, timestep {int(index.timesteps[0])}"
        )
        if index.mismatch is not None and index.mismatch.frame < n_kept:
            raise ValueError(index.mismatch.message)
    return FrameTable.join(
        [index.path for index in files],
        [index.starts[:k] for index, k in zip(files, kept)],
        [index.ends[:k] for index, k in zip(files, kept)],
        headers=headers,
        timesteps=np.concatenate([index.timesteps[:k] for index, k in zip(files, kept)]),
    )


def resolve_atom_style(atom_style: str | None, topology_file: str) -> str:
    """Return the MDAnalysis column string for a LAMMPS data file.

    ``atom_style`` is a column string (several words, returned as is), a
    style name in ``STYLE_COLUMNS``, or None for the data file's
    ``Atoms # <style>`` comment. Raises ValueError if there is no style or
    it has no known layout.
    """
    if atom_style is not None and len(atom_style.split()) > 1:
        return atom_style
    style = atom_style.strip() if atom_style is not None else _data_file_style(topology_file)
    if style is None:
        raise ValueError(
            f"{topology_file}: no 'Atoms # <style>' comment; pass atom_style "
            "as a column string, e.g. 'id type x y z'"
        )
    if style not in STYLE_COLUMNS:
        raise ValueError(
            f"{topology_file}: atom style '{style}' has no known column layout; pass "
            "atom_style as a column string, e.g. 'id type x y z'"
        )
    return STYLE_COLUMNS[style]


def _data_file_style(topology_file: str) -> str | None:
    """The style named by a data file's ``Atoms # <style>`` line, if any.

    The first line of a data file is a title and is not searched.
    """
    with open(topology_file, "rb") as f:
        next(f, None)
        for line in f:
            if line.split()[:1] == [b"Atoms"]:
                words = line.partition(b"#")[2].split()
                return words[0].decode("ascii", "backslashreplace") if words else None
    return None


def read_dump_frame(
    f: BinaryIO,
    start: int,
    end: int,
    header: DumpHeader,
    layout: ColumnLayout,
    expected_ids: np.ndarray,
    expected_types: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the positions and forces from the atom rows in bytes ``[start, end)`` of ``f``.

    Rows are returned in ascending atom-id order; scaled coordinates are
    converted to Cartesian from the box origin. ``expected_ids`` are the
    topology's ids, ascending, and ``expected_types`` its types in that
    order, as bytes, checked if the dump has a ``type`` column. Raises
    ValueError if a row lacks a column or the ids or types differ.
    """
    f.seek(start)
    tokens = f.read(end - start).split()
    n_columns = layout.n_columns
    if len(tokens) != header.n_atoms * n_columns:
        raise ValueError(
            f"the atom rows do not all have the {n_columns} columns named in the header"
        )
    # Only the needed columns are converted, each token with int() or
    # float(), which reject anything but a number (NUL bytes included).
    ids = np.array(tokens[layout.id_column::n_columns], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError(
            "frame atom ids do not match the topology: each topology id "
            "must appear exactly once"
        )
    if expected_types is not None and layout.type_column is not None:
        types = np.array(tokens[layout.type_column::n_columns], dtype=object)[order]
        differ = np.flatnonzero(types != expected_types)
        if differ.size:
            k = differ[0]
            raise ValueError(
                f"frame atom types do not match the topology: atom id "
                f"{expected_ids[k]} has type {types[k].decode()} in the dump and "
                f"{expected_types[k].decode()} in the topology"
            )
    columns = (*layout.coordinate_columns, *layout.force_columns)
    values = np.array([tokens[c::n_columns] for c in columns], dtype=np.float64)[:, order]
    positions = np.ascontiguousarray(values[:3].T)
    forces = np.ascontiguousarray(values[3:].T)
    if layout.scaled:
        positions = header.origin + fractional_to_cartesian(positions, header.cell)
    return positions, forces


# -----------------------------------------------------------------------------
# LAMMPS Trajectory Class
# -----------------------------------------------------------------------------

class LammpsTrajectory(IndexedFileTrajectory):
    """A trajectory read from LAMMPS custom dump files, with a LAMMPS data file as topology.

    Dump files are indexed on construction and frames are read from disk
    as needed. Frames have atoms in ascending id order and positions as
    written. Several dump files from a run and its restarts are joined in
    the order given, which must be run order: each file's frames replace
    the earlier files' frames from its own first timestep on.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        LAMMPS custom dump file(s), with ``id``, ``fx fy fz`` and one of
        ``x y z``, ``xu yu zu``, ``xs ys zs`` or ``xsu ysu zsu``.
    topology_file : str or path
        LAMMPS data file.
    temperature : float
        Temperature in Kelvin (reduced temperature for ``'lj'``).
    units : str, optional
        LAMMPS unit system (default ``'real'``).
    atom_style : str, optional
        Layout of the data file's Atoms section: an MDAnalysis column string
        such as ``'id type x y z'``, or a LAMMPS style name. By default it
        is read from the data file's ``Atoms # <style>`` comment.
    masses, charges : mapping, optional
        Masses or charges by atom type (e.g. ``{'1': 15.999}``), replacing
        the data file's for those types. Neither is guessed.

    Raises
    ------
    ValueError
        If a dump is damaged or lacks required columns, the box is not
        fixed and fully periodic, an atom count differs from the topology's,
        ``ITEM: UNITS`` does not match ``units``, a given mass or charge is
        not a finite number or names no atom's type, the atom style is unknown,
        the files are not in run order, or there are no complete frames.
    RuntimeError
        If MDAnalysis cannot read the topology, or its atom ids repeat.
    OSError
        If a file cannot be opened.

    Notes
    -----
    See :doc:`/tutorials/trajectory-formats` for the requirements on dump
    and data files, how restarts are joined, and how damaged or incomplete
    frames are handled.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        topology_file: TrajectoryPath | None = None,
        *,
        temperature: float,
        units: str = 'real',
        atom_style: str | None = None,
        masses: Mapping[str | int, float] | None = None,
        charges: Mapping[str | int, float] | None = None,
    ):
        super().__init__(units=units, temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        if topology_file is None:
            raise ValueError("A topology file is required for LAMMPS trajectories.")
        topology = os.fsdecode(topology_file)
        self._topology = topology
        paths = as_paths(trajectory_file)

        columns = resolve_atom_style(atom_style, topology)
        universe = open_universe(topology, topology_format="DATA", atom_style=columns)
        self.mdanalysis_universe = universe

        topology_ids = np.asarray(universe.atoms.ids)
        if not np.all(np.diff(topology_ids) > 0):
            raise RuntimeError(f"{topology}: atom ids repeat; each atom must have its own id")
        self._topology_ids = topology_ids
        self._topology_types = np.array([str(t).encode() for t in universe.atoms.types])
        def types() -> list[str]:
            return [str(t) for t in universe.atoms.types]
        self._masses = given_values(masses, types, "masses", "atom types")
        self._charges = given_values(charges, types, "charges", "atom types")

        indices = [index_dump(path) for path in paths]
        for index in indices:
            for style in index.units:
                if style.lower() != self.units:
                    raise ValueError(
                        f"{index.path}: the dump was written with ITEM: UNITS {style}, "
                        f"but units='{units}'; pass units='{style}'"
                    )
            # Every file's first frame is kept, so it must match the topology.
            if index.header is not None and index.header.n_atoms != len(topology_ids):
                raise ValueError(
                    f"{index.path}, timestep {int(index.timesteps[0])}: "
                    f"{index.header.n_atoms} atoms, but the topology has "
                    f"{len(topology_ids)}"
                )
        table = combine_frames(indices)
        self._layouts = [
            resolve_columns(header.columns, path)
            for path, header in zip(table.paths, table.headers)
        ]
        self._frames: FrameTable = table

        self.frames = len(table)
        self.cell_matrix = table.headers[0].cell.copy()
        self._validate_cell_matrix(self.cell_matrix, self._where(0))

    def get_indices(self, atype: str) -> np.ndarray:
        """Return the indices of the atoms of a LAMMPS type.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``).

        Returns
        -------
        np.ndarray
            Row positions in the frames, whose atoms are in ascending id order.

        Raises
        ------
        ValueError
            If no atom has that type.
        """
        indices = self.mdanalysis_universe.select_atoms(f'type {atype}').ix
        if len(indices) == 0:
            raise ValueError(f"{self._topology}: no atoms of type '{atype}'")
        return indices

    def get_charges(self, atype: str) -> np.ndarray:
        """Return the charges of the atoms of a LAMMPS type.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``).

        Raises
        ------
        DataUnavailableError
            If neither ``charges`` nor the data file gives them.
        """
        atoms = self.mdanalysis_universe.select_atoms(f'type {atype}')
        return atom_values(atoms.types, "charges", self._charges,
                           lambda: topology_values(atoms, "charges"), f"{self._topology}: the topology")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return the masses of the atoms of a LAMMPS type.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``).

        Raises
        ------
        DataUnavailableError
            If neither ``masses`` nor the data file gives them.
        """
        atoms = self.mdanalysis_universe.select_atoms(f'type {atype}')
        return atom_values(atoms.types, "masses", self._masses,
                           lambda: topology_values(atoms, "masses"), f"{self._topology}: the topology")

    def _where(self, index: int) -> str:
        """Name the file, the frame's position in it, and its timestep."""
        table = self._frames
        k, frame = table.locate(index)
        return f"{table.paths[k]}, frame {frame} (timestep {int(table.timesteps[index])})"

    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        table = self._frames
        k, _ = table.locate(index)
        return read_dump_frame(
            f, int(table.starts[index]), int(table.ends[index]),
            table.headers[k], self._layouts[k], self._topology_ids, self._topology_types,
        )
