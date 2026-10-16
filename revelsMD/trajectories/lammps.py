"""
LAMMPS trajectory backend for RevelsMD.

This module provides the LammpsTrajectory class for reading LAMMPS dump files,
and the helpers that index and read the dump format.
"""

import array
import mmap
import os
import re
from dataclasses import dataclass
from typing import BinaryIO

from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
import numpy as np

from revelsMD.cell import fractional_to_cartesian
from ._base import (
    TrajectoryFiles, TrajectoryPath,
    DataUnavailableError, FileFrames, IndexedFileTrajectory, as_paths, check_fixed_cell,
    open_universe, warn_cut_short,
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
    """Parse the ``ITEM: BOX BOUNDS`` header of a dump frame.

    Parameters
    ----------
    item_line : str
        The ``ITEM: BOX BOUNDS ...`` line.
    bound_lines : list of str
        The three lines that follow it.

    Returns
    -------
    cell : np.ndarray, shape (3, 3)
        Rows are the lattice vectors a, b, c.
    origin : np.ndarray, shape (3,)
        The corner (xlo, ylo, zlo).
    box : np.ndarray, shape (9,)
        The six bounds as written, then the tilts xy, xz, yz (zero for an
        orthogonal box). Equal arrays mean an identical box.

    Raises
    ------
    ValueError
        If any boundary is not periodic, the box is in the general
        triclinic ``abc origin`` format, or the bounds are malformed or
        not finite.
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

    Coordinates are taken from the first of ``x y z``, ``xu yu zu``,
    ``xs ys zs`` and ``xsu ysu zsu`` that is present.

    Raises
    ------
    ValueError
        If the id, the forces or every coordinate set is missing.
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
# The unit style that dump_modify units yes writes before a frame.
_UNITS = re.compile(rb"ITEM: UNITS[ \t\r]*\n[ \t]*(\S+)")


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
    """An incomplete final frame, as left by a job killed while writing."""

    path: str
    timestep: int | None


@dataclass(frozen=True)
class HeaderMismatch:
    """The first frame of a dump file whose header differs from the file's first frame's."""

    frame: int
    message: str


@dataclass(frozen=True, eq=False)
class DumpIndex:
    """The complete frames of one dump file, and its incomplete final frame.

    Frame ``k`` has timestep ``timesteps[k]`` (strictly increasing) and its
    atom rows occupy bytes ``[starts[k], ends[k])`` of the file. ``header``
    is the first complete frame's, and is None if there are none. Frames
    before ``mismatch.frame`` share it; if ``mismatch`` is None, all do.
    ``units`` holds the unit styles the file's ``ITEM: UNITS`` lines name.
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
    """One complete frame, as found by the scan of its file.

    ``header_lines`` are the stripped header lines from ``ITEM: NUMBER OF
    ATOMS`` to ``ITEM: ATOMS``.
    """

    timestep: int
    data_start: int
    data_end: int
    header: DumpHeader
    header_lines: list[str]


def index_dump(path: str) -> DumpIndex:
    """Index the frames of one dump file without parsing the atom rows.

    Every complete frame is compared with the file's first complete frame;
    the first whose columns, box or atom count differ is recorded in
    ``mismatch`` rather than raised, since a restart file may replace it.
    Text between frames, such as the ``ITEM: UNITS`` and ``ITEM: TIME``
    lines of ``dump_modify units yes`` and ``time yes``, is skipped; unit
    styles are recorded. An empty file, a file holding only the start of a
    frame header, and text after the last frame's atom rows are frames cut
    short.

    Raises
    ------
    ValueError
        If the file has no frames, if a frame other than the last is
        incomplete, if a frame has more rows than its header states, if a
        frame header is malformed, if timesteps do not strictly increase,
        or if a box is not supported.
    """
    no_frames = DumpIndex(
        path, None, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64),
        PartialFrame(path, None),
    )
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return no_frames
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            frame_starts = array.array("q")
            position = m.find(_TIMESTEP)
            while position != -1:
                frame_starts.append(position)
                position = m.find(_TIMESTEP, position + len(_TIMESTEP))
            if not frame_starts:
                head = m[:5]
                if head.startswith(b"ITEM:") or b"ITEM:".startswith(head):
                    return no_frames
                raise ValueError(f"{path}: no 'ITEM: TIMESTEP' line; not a LAMMPS dump file")
            units: dict[str, None] = {}
            _record_units(m[:frame_starts[0]], units)
            # A file cut off inside the next frame's header leaves part of an
            # "ITEM:" line after its last newline. Drop that tail.
            limit = len(m)
            dropped_header = False
            if m[-1:] != b"\n":
                last_newline = m.rfind(b"\n")
                tail = m[last_newline + 1:]
                if b"ITEM:".startswith(tail) or tail.startswith(b"ITEM:"):
                    limit = last_newline + 1
                    dropped_header = True
            n = len(frame_starts)
            timesteps = np.empty(n, dtype=np.int64)
            starts = np.empty(n, dtype=np.int64)
            ends = np.empty(n, dtype=np.int64)
            first: _ScannedFrame | None = None
            mismatch: HeaderMismatch | None = None
            partial: PartialFrame | None = None
            # Only the last frame may be incomplete, so complete frame k is
            # the file's k-th.
            for k in range(n):
                end = frame_starts[k + 1] if k + 1 < n else limit
                entry = _index_frame(m, path, frame_starts[k], end, first)
                if isinstance(entry, PartialFrame):
                    if k < n - 1:
                        raise ValueError(
                            f"{path}: the frame at timestep {entry.timestep} is "
                            "incomplete but is not the last frame in the file"
                        )
                    partial = entry
                    continue
                following = m[entry.data_end:end]
                _record_units(following, units)
                if k == n - 1 and following.strip():
                    dropped_header = True
                if first is None:
                    first = entry
                else:
                    if mismatch is None:
                        difference = _header_difference(entry, first.header, path)
                        if difference is not None:
                            mismatch = HeaderMismatch(k, difference)
                    previous = int(timesteps[k - 1])
                    if entry.timestep <= previous:
                        raise ValueError(
                            f"{path}: timesteps do not increase "
                            f"({previous} is followed by {entry.timestep})"
                        )
                timesteps[k] = entry.timestep
                starts[k] = entry.data_start
                ends[k] = entry.data_end
            count = n - (partial is not None)
            if partial is None and dropped_header:
                partial = PartialFrame(path, None)
    header = first.header if first is not None else None
    return DumpIndex(
        path, header, timesteps[:count], starts[:count], ends[:count], partial, mismatch,
        tuple(units),
    )


def _record_units(text: bytes, units: dict[str, None]) -> None:
    """Add the unit styles named by ``ITEM: UNITS`` lines in ``text`` to ``units``."""
    for match in _UNITS.finditer(text):
        units[match.group(1).decode("ascii", "backslashreplace")] = None


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


def _index_frame(
    m: mmap.mmap, path: str, start: int, end: int, first: _ScannedFrame | None
) -> "_ScannedFrame | PartialFrame":
    """Index one frame occupying bytes [start, end) of a mapped dump file.

    A frame whose header lines are those of the file's first complete
    frame, ``first``, shares its header rather than parsing a new one.
    """
    lines: list[str] = []
    position = start
    while len(lines) < 9:
        newline = m.find(b"\n", position, end)
        if newline == -1:
            return PartialFrame(path, _header_timestep(lines))
        try:
            lines.append(m[position:newline].decode("ascii").strip())
        except UnicodeDecodeError as e:
            raise ValueError(f"{path}: non-ASCII frame header at byte {position}") from e
        position = newline + 1
    if (
        lines[0] != "ITEM: TIMESTEP"
        or lines[2] != "ITEM: NUMBER OF ATOMS"
        or not lines[4].startswith("ITEM: BOX BOUNDS")
        or not lines[8].startswith("ITEM: ATOMS")
    ):
        raise ValueError(f"{path}: unexpected frame header at byte {start}: {lines}")
    header_lines = lines[2:9]
    try:
        timestep, n_atoms = int(lines[1]), int(lines[3])
    except ValueError as e:
        raise ValueError(f"{path}: malformed frame header at byte {start}: {e}") from e
    if not 0 <= timestep < 2**63:
        raise ValueError(f"{path}: timestep {timestep} is out of range")
    if first is not None and header_lines == first.header_lines:
        header = first.header
    else:
        try:
            cell, origin, box = parse_box(lines[4], lines[5:8])
        except ValueError as e:
            raise ValueError(f"{path}, timestep {timestep}: {e}") from e
        header = DumpHeader(n_atoms, tuple(lines[8].split()[2:]), cell, origin, box)
    next_item = m.find(b"ITEM:", position, end)
    rows_end = end if next_item == -1 else next_item
    # Blank lines after the atom rows are ignored; a last row without a
    # newline may be cut short, so it is not complete.
    rows = m[position:rows_end]
    content_end = len(rows)
    while content_end and rows[content_end - 1] in b" \t\r\n":
        content_end -= 1
    terminated = rows.find(b"\n", content_end) != -1
    n_rows = rows.count(b"\n", 0, content_end) + (content_end > 0)
    complete_rows = n_rows if terminated or n_rows == 0 else n_rows - 1
    if complete_rows < n_atoms:
        return PartialFrame(path, timestep)
    if n_rows > n_atoms:
        extra = 0
        for _ in range(n_atoms):
            extra = rows.index(b"\n", extra) + 1
        raise ValueError(
            f"{path}, timestep {timestep}: the atom rows do not match the "
            f"header's {n_atoms} atoms; unexpected text at byte {position + extra}: "
            f"{rows[extra:extra + 40]!r}"
        )
    return _ScannedFrame(timestep, position, rows_end, header, header_lines)


def _header_timestep(lines: list[str]) -> int | None:
    """The timestep of a cut-off header, if its value line was written."""
    try:
        return int(lines[1])
    except (IndexError, ValueError):
        return None


@dataclass(frozen=True, eq=False)
class FrameTable(FileFrames):
    """The frames of a trajectory joined from one or more dump files.

    Frame ``i`` is stored as :class:`FileFrames` describes, has timestep
    ``timesteps[i]`` (strictly increasing) and the header of its file,
    ``headers[k]``. Only files that contribute frames are listed.
    """

    headers: list[DumpHeader]
    timesteps: np.ndarray


def combine_frames(indices: list[DumpIndex]) -> FrameTable:
    """Join the frames of consecutive dump files into one trajectory.

    Files must be given in run order: the files of a run, then those of
    each restart. Each file's frames replace every frame of the files
    before it from its own first timestep on, as when a run restarts from
    a restart file. An incomplete final frame is dropped with a warning.
    Every kept frame must have the columns, box and atom count of its
    file's first frame, and every file's first frame the box of the first
    file's; frames that are replaced are not compared.

    Raises
    ------
    ValueError
        If a file starts at or before the first timestep of the file before
        it, if a kept frame's header differs as above, or if there are no
        complete frames.
    """
    for index in indices:
        if index.partial is not None:
            at = "" if index.partial.timestep is None else f" (timestep {index.partial.timestep})"
            warn_cut_short(index.partial.path, f"frame{at}")
    files = [index for index in indices if index.header is not None]
    if not files:
        raise ValueError(
            "no complete frames in " + ", ".join(index.path for index in indices)
        )
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
    headers = [index.header for index in files if index.header is not None]
    for index, header, n_kept in zip(files, headers, kept):
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

    Parameters
    ----------
    atom_style : str or None
        An MDAnalysis column string (several words), a LAMMPS atom style
        name from ``STYLE_COLUMNS``, or None to use the style named in the
        data file's ``Atoms # <style>`` comment.
    topology_file : str
        The LAMMPS data file.

    Raises
    ------
    ValueError
        If the style is not given and the data file names none, or if the
        style has no known column layout.
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
    """Read the positions and forces of one indexed frame.

    Parameters
    ----------
    f : binary file object
        The frame's dump file, opened in binary mode.
    start, end : int
        The byte range of the frame's atom rows.
    header : DumpHeader
        The header of the frame's file.
    layout : ColumnLayout
        The columns of the frame's file.
    expected_ids : np.ndarray
        The topology's atom ids in ascending order.
    expected_types : np.ndarray, optional
        The topology's atom types in the same order, as bytes. If given and
        the dump has a ``type`` column, the frame's types must equal them.

    Returns
    -------
    positions, forces : np.ndarray
        Arrays of shape ``(n_atoms, 3)``, rows in ascending atom-id order.
        Scaled coordinates are converted to Cartesian.

    Raises
    ------
    ValueError
        If the rows do not all have the header's columns, or if the frame's
        ids are not exactly ``expected_ids`` or its types not
        ``expected_types``.
    """
    f.seek(start)
    tokens = np.array(f.read(end - start).split())
    if tokens.size != header.n_atoms * layout.n_columns:
        raise ValueError(
            f"the atom rows do not all have the {layout.n_columns} columns "
            "named in the header"
        )
    table = tokens.reshape(header.n_atoms, layout.n_columns)
    ids = table[:, layout.id_column].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError(
            "frame atom ids do not match the topology: each topology id "
            "must appear exactly once"
        )
    table = table[order]
    if expected_types is not None and layout.type_column is not None:
        types = table[:, layout.type_column]
        differ = np.flatnonzero(types != expected_types)
        if differ.size:
            k = differ[0]
            raise ValueError(
                f"frame atom types do not match the topology: atom id "
                f"{expected_ids[k]} has type {types[k].decode()} in the dump and "
                f"{expected_types[k].decode()} in the topology"
            )
    positions = table[:, list(layout.coordinate_columns)].astype(np.float64)
    forces = table[:, list(layout.force_columns)].astype(np.float64)
    if layout.scaled:
        positions = header.origin + fractional_to_cartesian(positions, header.cell)
    return positions, forces


# -----------------------------------------------------------------------------
# LAMMPS Trajectory Class
# -----------------------------------------------------------------------------

class LammpsTrajectory(IndexedFileTrajectory):
    """
    Represents a molecular dynamics trajectory obtained from LAMMPS output.

    Dump files are indexed when the trajectory is constructed and frames
    are read from disk as they are needed; the index keeps a few tens of
    bytes per frame. MDAnalysis provides the topology only.

    Frames are returned with atoms in ascending id order, whatever order
    the dump was written in. Several dump files from a run and its restarts
    are joined in the order given, which must be run order: each file's
    frames replace every frame of the files before it from its own first
    timestep on. An incomplete final frame, as left by a job killed while
    writing, is dropped with a warning, as is an empty file or one holding
    only the start of a frame header. Frames that a later file replaces are
    not compared with their file's first frame, so a run whose box or atom
    count went wrong after the restart point can be joined with its
    restart.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        LAMMPS custom dump file(s). Each must have ``id``, ``fx fy fz`` and
        one of ``x y z``, ``xu yu zu``, ``xs ys zs`` or ``xsu ysu zsu``.
    topology_file : str or path
        LAMMPS data file.
    temperature : float
        Simulation temperature in Kelvin (reduced temperature for 'lj').
    units : str, optional
        LAMMPS unit system (default: ``'real'``).
    atom_style : str, optional
        Layout of the data file's Atoms section: an MDAnalysis column string
        such as ``'id type x y z'``, or one of the LAMMPS styles
        ``atomic``, ``charge``, ``full``, ``molecular``, ``bond`` or
        ``angle``. By default the style is read from the data file's
        ``Atoms # <style>`` comment.

    Attributes
    ----------
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.

    Raises
    ------
    ValueError
        If a dump file is malformed or lacks required columns, if a box is
        not fully periodic, if the box, the columns or the atom count change
        between kept frames of one file, if the box differs between files,
        if a frame's atom count differs from the topology's, if a dump's
        ``ITEM: UNITS`` names a unit style other than ``units``, if the atom
        style cannot be determined, if a file starts at or before the first
        timestep of the file before it, or if there are no complete frames.
    RuntimeError
        If MDAnalysis cannot parse the topology (the message names it), or
        if topology atom ids repeat.
    OSError
        If a dump file or the topology file cannot be opened
        (``FileNotFoundError`` if it does not exist).

    Notes
    -----
    Reading a frame raises ValueError if its atom ids are not exactly the
    topology's atom ids, if the dump has a ``type`` column whose types
    differ from the topology's, or if a position or force is NaN or
    infinite.

    Positions are returned as written, in the simulation's coordinates;
    scaled coordinates are converted with the box origin. MDAnalysis's own
    dump reader instead shifts positions so the box starts at the origin.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        topology_file: TrajectoryPath | None = None,
        *,
        temperature: float,
        units: str = 'real',
        atom_style: str | None = None,
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

        indices = [index_dump(path) for path in paths]
        for index in indices:
            for style in index.units:
                if style != units:
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
        """Return positional indices of the atoms of a given LAMMPS type.

        Frames yielded by this trajectory are in ascending atom-id order,
        and the returned indices are row positions in those frames.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``, ``'2'``).

        Returns
        -------
        np.ndarray
            Row indices into the per-frame position and force arrays.

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
        """Return atomic charges for a given LAMMPS atom type.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``, ``'2'``).

        Returns
        -------
        np.ndarray
            Array of atomic charges for the selected atoms.

        Raises
        ------
        DataUnavailableError
            If the atom style has no charge column.
        """
        try:
            return self.mdanalysis_universe.select_atoms(f'type {atype}').charges
        except NoDataError:
            raise DataUnavailableError("Charge data not available for this LAMMPS trajectory.")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for a given LAMMPS atom type.

        Parameters
        ----------
        atype : str
            LAMMPS atom type as a string (e.g. ``'1'``, ``'2'``).

        Returns
        -------
        np.ndarray
            Array of atomic masses for the selected atoms.

        Raises
        ------
        DataUnavailableError
            If mass data is absent from the topology file.
        """
        try:
            return self.mdanalysis_universe.select_atoms(f'type {atype}').masses
        except NoDataError:
            raise DataUnavailableError("Mass data not available for this LAMMPS trajectory.")

    def _where(self, index: int) -> str:
        """Name the file, frame and timestep of frame ``index``."""
        table = self._frames
        return f"{table.path(index)}, frame {index} (timestep {int(table.timesteps[index])})"

    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        table = self._frames
        k, _ = table.locate(index)
        return read_dump_frame(
            f, int(table.starts[index]), int(table.ends[index]),
            table.headers[k], self._layouts[k], self._topology_ids, self._topology_types,
        )
