"""
LAMMPS trajectory backend for RevelsMD.

This module provides the LammpsTrajectory class for reading LAMMPS dump files,
and the helpers that index and read the dump format.
"""

import mmap
import os
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError, as_paths, check_fixed_cell


# Column layouts of the Atoms section for LAMMPS atom styles (read_data
# documentation), as MDAnalysis column strings.
STYLE_COLUMNS: dict[str, str] = {
    "atomic": "id type x y z",
    "charge": "id type q x y z",
    "full": "id resid type q x y z",
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
        triclinic ``abc origin`` format, or the bounds are malformed.
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
    return ColumnLayout(len(columns), columns.index("id"), (x, y, z), (fx, fy, fz), scaled)


_TIMESTEP = b"ITEM: TIMESTEP"


@dataclass(frozen=True, eq=False)
class DumpFrame:
    """Where one complete dump frame lies in its file, with its header."""

    path: str
    timestep: int
    data_start: int
    data_end: int
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
class DumpIndex:
    """The complete frames of one dump file, and its incomplete final frame."""

    path: str
    frames: list[DumpFrame]
    partial: PartialFrame | None


def index_dump(path: str) -> DumpIndex:
    """Index the frames of one dump file without parsing the atom rows.

    Raises
    ------
    ValueError
        If the file is empty or has no frames, if a frame other than the
        last is incomplete, if a frame has more rows than its header
        states, if the columns change within the file, if timesteps do not
        strictly increase, or if a box is not supported.
    """
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            raise ValueError(f"{path}: the dump file is empty")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            starts = []
            position = m.find(_TIMESTEP)
            while position != -1:
                starts.append(position)
                position = m.find(_TIMESTEP, position + len(_TIMESTEP))
            if not starts:
                raise ValueError(f"{path}: no 'ITEM: TIMESTEP' line; not a LAMMPS dump file")
            # A file cut off inside the next frame's "ITEM:" header leaves a
            # prefix of "ITEM:" after its last newline. Drop that tail.
            limit = len(m)
            dropped_header = False
            if m[-1:] != b"\n":
                last_newline = m.rfind(b"\n")
                tail = m[last_newline + 1:]
                if tail and b"ITEM:".startswith(tail):
                    limit = last_newline + 1
                    dropped_header = True
            ends = starts[1:] + [limit]
            frames: list[DumpFrame] = []
            partial: PartialFrame | None = None
            for k, (start, end) in enumerate(zip(starts, ends)):
                entry = _index_frame(m, path, start, end)
                if isinstance(entry, DumpFrame):
                    frames.append(entry)
                elif k == len(starts) - 1:
                    partial = entry
                else:
                    raise ValueError(
                        f"{path}: the frame at timestep {entry.timestep} is "
                        "incomplete but is not the last frame in the file"
                    )
            if partial is None and dropped_header:
                partial = PartialFrame(path, None)
    for previous, current in zip(frames, frames[1:]):
        if current.columns != frames[0].columns:
            raise ValueError(
                f"{path}, timestep {current.timestep}: the dump columns change within the file"
            )
        if current.timestep <= previous.timestep:
            raise ValueError(
                f"{path}: timesteps do not increase "
                f"({previous.timestep} is followed by {current.timestep})"
            )
    return DumpIndex(path, frames, partial)


def _index_frame(m: mmap.mmap, path: str, start: int, end: int) -> "DumpFrame | PartialFrame":
    """Index one frame occupying bytes [start, end) of a mapped dump file."""
    lines: list[str] = []
    position = start
    while len(lines) < 9:
        newline = m.find(b"\n", position, end)
        if newline == -1:
            return PartialFrame(path, _header_timestep(lines))
        lines.append(m[position:newline].decode("ascii").strip())
        position = newline + 1
    if (
        lines[0] != "ITEM: TIMESTEP"
        or lines[2] != "ITEM: NUMBER OF ATOMS"
        or not lines[4].startswith("ITEM: BOX BOUNDS")
        or not lines[8].startswith("ITEM: ATOMS")
    ):
        raise ValueError(f"{path}: unexpected frame header at byte {start}: {lines}")
    timestep, n_atoms = int(lines[1]), int(lines[3])
    try:
        cell, origin, box = parse_box(lines[4], lines[5:8])
    except ValueError as e:
        raise ValueError(f"{path}, timestep {timestep}: {e}") from e
    next_item = m.find(b"ITEM:", position, end)
    rows_end = end if next_item == -1 else next_item
    n_rows = m[position:rows_end].count(b"\n")
    if n_rows < n_atoms:
        return PartialFrame(path, timestep)
    if n_rows > n_atoms or m[rows_end - 1:rows_end] != b"\n":
        raise ValueError(
            f"{path}, timestep {timestep}: the atom rows do not match the "
            f"header's {n_atoms} atoms"
        )
    return DumpFrame(
        path, timestep, position, rows_end, n_atoms,
        tuple(lines[8].split()[2:]), cell, origin, box,
    )


def _header_timestep(lines: list[str]) -> int | None:
    """The timestep of a cut-off header, if its value line was written."""
    try:
        return int(lines[1])
    except (IndexError, ValueError):
        return None


def combine_frames(indices: list[DumpIndex]) -> list[DumpFrame]:
    """Join the frames of consecutive dump files into one trajectory.

    Frames are kept in timestep order. When a file starts at a timestep
    that an earlier file already contains, the earlier frames from that
    timestep on are replaced by the later file's. An incomplete final frame
    is dropped, with a warning unless the next file's frames replace it.

    Raises
    ------
    ValueError
        If a file starts at or before the first timestep of the previous
        file, or if there are no complete frames.
    """
    combined: list[DumpFrame] = []
    previous: DumpIndex | None = None
    pending: PartialFrame | None = None
    for index in indices:
        if index.frames:
            first = index.frames[0].timestep
            if previous is not None and first <= previous.frames[0].timestep:
                raise ValueError(
                    f"{index.path} starts at timestep {first}, at or before the "
                    f"start of {previous.path} (timestep "
                    f"{previous.frames[0].timestep}); timesteps must continue "
                    "from one file to the next"
                )
            if pending is not None and not _superseded(pending, combined, first):
                _warn_dropped(pending)
            pending = None
            combined = [frame for frame in combined if frame.timestep < first]
            combined.extend(index.frames)
            previous = index
        if index.partial is not None:
            if pending is not None:
                _warn_dropped(pending)
            pending = index.partial
    if pending is not None:
        _warn_dropped(pending)
    if not combined:
        raise ValueError(
            "no complete frames in " + ", ".join(index.path for index in indices)
        )
    return combined


def _superseded(partial: PartialFrame, combined: list[DumpFrame], first: int) -> bool:
    """Whether a file starting at ``first`` replaces an incomplete frame."""
    if partial.timestep is not None:
        return first <= partial.timestep
    return bool(combined) and first <= combined[-1].timestep


def _warn_dropped(partial: PartialFrame) -> None:
    at = "" if partial.timestep is None else f" (timestep {partial.timestep})"
    warnings.warn(
        f"{partial.path}: dropping the incomplete final frame{at}; "
        "the file appears to have been cut short",
        UserWarning,
        stacklevel=3,
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
            f"atom style '{style}' has no known column layout; pass atom_style "
            "as a column string, e.g. 'id type x y z'"
        )
    return STYLE_COLUMNS[style]


def _data_file_style(topology_file: str) -> str | None:
    """The style named by a data file's ``Atoms # <style>`` line, if any."""
    with open(topology_file) as f:
        for line in f:
            if line.split()[:1] == ["Atoms"]:
                words = line.partition("#")[2].split()
                return words[0] if words else None
    return None


def read_dump_frame(
    f: BinaryIO,
    frame: DumpFrame,
    layout: ColumnLayout,
    expected_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the positions and forces of one indexed frame.

    Parameters
    ----------
    f : binary file object
        The frame's dump file, opened in binary mode.
    frame : DumpFrame
        The frame's index entry.
    layout : ColumnLayout
        The columns of the frame's file.
    expected_ids : np.ndarray
        The topology's atom ids in ascending order.

    Returns
    -------
    positions, forces : np.ndarray
        Arrays of shape ``(n_atoms, 3)``, rows in ascending atom-id order.
        Scaled coordinates are converted to Cartesian.

    Raises
    ------
    ValueError
        If the rows do not all have the header's columns, or if the frame's
        ids are not exactly ``expected_ids``.
    """
    f.seek(frame.data_start)
    tokens = np.array(f.read(frame.data_end - frame.data_start).split())
    if tokens.size != frame.n_atoms * layout.n_columns:
        raise ValueError(
            f"the atom rows do not all have the {layout.n_columns} columns "
            "named in the header"
        )
    table = tokens.reshape(frame.n_atoms, layout.n_columns)
    ids = table[:, layout.id_column].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError(
            "frame atom ids do not match the topology: each topology id "
            "must appear exactly once"
        )
    table = table[order]
    positions = table[:, list(layout.coordinate_columns)].astype(np.float64)
    forces = table[:, list(layout.force_columns)].astype(np.float64)
    if layout.scaled:
        positions = frame.origin + positions @ frame.cell
    return positions, forces


# -----------------------------------------------------------------------------
# LAMMPS Trajectory Class
# -----------------------------------------------------------------------------

class LammpsTrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory obtained from LAMMPS output.

    Dump files are indexed when the trajectory is constructed and frames
    are read from disk as they are needed, so memory use does not grow
    with the length of the trajectory. MDAnalysis provides the topology
    only.

    Frames are returned with atoms in ascending id order, whatever order
    the dump was written in. Several dump files from a run and its restarts
    are joined in timestep order: where a file starts at a timestep an
    earlier file already contains, the earlier frames from that timestep on
    are replaced by the later file's. An incomplete final frame, as left by
    a job killed while writing, is dropped with a warning.

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
        not fully periodic, if the box changes between frames, if a frame's
        atom count differs from the topology's, or if the atom style cannot
        be determined.
    RuntimeError
        If MDAnalysis cannot read the topology, or if the topology atom ids
        are not in ascending order.

    Notes
    -----
    Reading a frame raises ValueError if its atom ids are not exactly the
    topology's atom ids.
    """

    def __init__(
        self,
        trajectory_file: "str | os.PathLike[str] | Sequence[str | os.PathLike[str]]",
        topology_file: "str | os.PathLike[str] | None" = None,
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
        topology = os.fspath(topology_file)
        paths = as_paths(trajectory_file)
        if not paths:
            raise ValueError("At least one LAMMPS dump file is required.")

        columns = resolve_atom_style(atom_style, topology)
        try:
            universe = MD.Universe(topology, topology_format="DATA", atom_style=columns)
        except Exception as e:
            raise RuntimeError(f"Failed to load LAMMPS topology with MDAnalysis: {e}") from e
        self.mdanalysis_universe = universe

        topology_ids = np.asarray(universe.atoms.ids)
        if not np.all(np.diff(topology_ids) > 0):
            raise RuntimeError(
                "Topology atom ids are not in ascending order; positional "
                "indices would not match id-sorted frames."
            )
        self._topology_ids = topology_ids

        self._index = combine_frames([index_dump(path) for path in paths])
        self._layouts: dict[tuple[str, tuple[str, ...]], ColumnLayout] = {}
        first = self._index[0]
        for frame in self._index:
            where = f"{frame.path}, timestep {frame.timestep}"
            if frame.n_atoms != len(topology_ids):
                raise ValueError(
                    f"{where}: {frame.n_atoms} atoms, but the topology has "
                    f"{len(topology_ids)}"
                )
            check_fixed_cell(frame.box, first.box, where)
            key = (frame.path, frame.columns)
            if key not in self._layouts:
                self._layouts[key] = resolve_columns(frame.columns, frame.path)

        self.frames = len(self._index)
        self.cell_matrix = first.cell
        self._validate_cell_matrix(self.cell_matrix)

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
        """
        return self.mdanalysis_universe.select_atoms(f'type {atype}').ix

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
            If charge data is absent from the topology file.
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

    def _read(self, f: BinaryIO, index: int) -> Frame:
        """Read frame ``index`` from its open dump file."""
        frame = self._index[index]
        try:
            positions, forces = read_dump_frame(
                f, frame, self._layouts[(frame.path, frame.columns)], self._topology_ids
            )
        except ValueError as e:
            raise ValueError(
                f"{frame.path}, frame {index} (timestep {frame.timestep}): {e}"
            ) from e
        return Frame(positions, forces)

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """Read the indexed frames in order, keeping each file open while in use."""
        with ExitStack() as stack:
            handles: dict[str, BinaryIO] = {}
            for index in range(start, stop, stride):
                path = self._index[index].path
                if path not in handles:
                    handles[path] = stack.enter_context(open(path, "rb"))
                yield self._read(handles[path], index)

    def get_frame(self, index: int) -> Frame:
        """Return positions and forces for a specific frame by index.

        The frame is read from disk by seeking to it; nothing else is read
        or kept in memory.

        Parameters
        ----------
        index : int
            Frame index to retrieve. Negative indices count from the end.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape ``(n_atoms, 3)``.
        """
        with open(self._index[index].path, "rb") as f:
            return self._read(f, index)
