"""
LAMMPS trajectory backend for RevelsMD.

This module provides the LammpsTrajectory class for reading LAMMPS dump files,
along with helper functions for parsing the LAMMPS dump format.
"""

import mmap
import os
import warnings
from dataclasses import dataclass
from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError


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


# -----------------------------------------------------------------------------
# LAMMPS Dump File Parsing Helpers
# -----------------------------------------------------------------------------

def first_read(dumpFile: str):
    """
    Perform an initial read of a LAMMPS custom dump file to extract header metadata.

    Parameters
    ----------
    dumpFile : str
        Path to the LAMMPS dump file.

    Returns
    -------
    frames : int
        Number of trajectory frames.
    num_ats : int
        Number of atoms per frame.
    dic : list of str
        Column headers extracted from the ``ITEM: ATOMS`` line.
    header_length : int
        Number of header lines per frame (before atom coordinates start).
    dimgrid : numpy.ndarray of shape (3, 2)
        Box boundaries (x, y, z) for the first frame.
    id_column : int
        Index of the ``id`` column within the atom columns, i.e. the
        ``ITEM: ATOMS`` header minus its two leading tokens.

    Raises
    ------
    ValueError
        If the dump has no ``id`` column. Atom ids are required to pair
        dump rows with the topology.

    Notes
    -----
    This function assumes:
    - Box bounds are printed as ``ITEM: BOX BOUNDS pp pp pp``.
    - The number of atoms and header structure remain constant across all frames.
    - If the total line count does not divide evenly into frames, a warning is printed.
    """
    header_length = 0
    dimgrid = np.zeros((3, 2))
    closer = 0
    num_ats = 0

    with open(dumpFile, "r") as f:
        while closer == 0:
            currentString = f.readline()
            if not currentString:
                raise ValueError("Unexpected EOF while parsing header.")
            if currentString.startswith("ITEM: ATOMS"):
                dic = currentString.split()
                closer = 1
            if currentString.strip() == "ITEM: NUMBER OF ATOMS":
                header_length += 1
                currentString = f.readline()
                num_ats = int(currentString)
            header_length += 1
            if currentString.strip().startswith("ITEM: BOX BOUNDS"):
                header_length += 3
                dimgrid[0, :] = np.array(f.readline().split(), dtype=float)
                dimgrid[1, :] = np.array(f.readline().split(), dtype=float)
                dimgrid[2, :] = np.array(f.readline().split(), dtype=float)

    with open(dumpFile) as f:
        numLines = sum(1 for _ in f)
    frames = numLines / float(num_ats + header_length)
    if frames % 1 != 0:
        warnings.warn(
            "Non-integer frame count: incomplete file or inconsistent headers.",
            stacklevel=2,
        )
    columns = dic[2:]
    if "id" not in columns:
        raise ValueError(
            "LAMMPS dump has no 'id' column; it is required to pair dump "
            "rows with the topology."
        )
    id_column = columns.index("id")
    return int(frames), num_ats, dic, header_length, dimgrid, id_column


def get_a_frame(
    f,
    num_ats: int,
    header_length: int,
    strngdex: list[int],
    id_column: int,
    expected_ids: np.ndarray,
) -> np.ndarray:
    """
    Extract a single frame of atomic data from an open LAMMPS dump file.

    Rows are returned in ascending atom-id order regardless of the order
    in which they appear in the file.

    Parameters
    ----------
    f : file object
        Open trajectory file (text mode).
    num_ats : int
        Number of atoms per frame.
    header_length : int
        Number of header lines preceding atomic data.
    strngdex : list of int
        Column indices to extract (relative to the start of atom data).
    id_column : int
        Column index of the atom id.
    expected_ids : numpy.ndarray
        The topology's atom ids in ascending order. The frame must contain
        each of them exactly once.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_ats, len(strngdex))`` containing the requested
        columns, rows sorted by atom id.

    Raises
    ------
    ValueError
        If the sorted ids of the frame are not ``expected_ids``.
    """
    vars_trest = np.zeros((num_ats, len(strngdex)))
    ids = np.zeros(num_ats, dtype=np.int64)
    for _ in range(header_length):
        f.readline()
    for i in range(num_ats):
        currentString = f.readline().split()
        ids[i] = int(currentString[id_column])
        for j, k in enumerate(strngdex):
            vars_trest[i, j] = float(currentString[k])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError(
            "frame atom ids do not match the topology: each topology id "
            "must appear exactly once"
        )
    return vars_trest[order]


def define_strngdex(our_string: list[str], dic: list[str]) -> list[int]:
    """
    Map requested LAMMPS quantities to their corresponding column indices.

    Parameters
    ----------
    our_string : list of str
        Names of quantities to extract (e.g. ``['x', 'y', 'z', 'fx', 'fy', 'fz']``).
    dic : list of str
        Column header list from the LAMMPS dump (``ITEM: ATOMS ...`` line).

    Returns
    -------
    list of int
        Column indices corresponding to the requested quantities.

    Examples
    --------
    >>> dic = ['ITEM:', 'ATOMS', 'id', 'type', 'x', 'y', 'z', 'fx', 'fy', 'fz']
    >>> define_strngdex(['x', 'z'], dic)
    [2, 4]
    """
    return [int(dic.index(ele) - 2) for ele in our_string]


def frame_skip(f, num_ats: int, num_skip: int, header_length: int):
    """
    Skip a specified number of frames in an open trajectory file.

    Parameters
    ----------
    f : file object
        Open trajectory file in text mode.
    num_ats : int
        Number of atoms per frame.
    num_skip : int
        Number of frames to skip.
    header_length : int
        Number of header lines per frame.

    Notes
    -----
    This function simply advances the file pointer by reading and discarding
    the appropriate number of lines, based on the known frame structure.
    """
    for _ in range(num_skip * (num_ats + header_length)):
        f.readline()


# -----------------------------------------------------------------------------
# LAMMPS Trajectory Class
# -----------------------------------------------------------------------------

class LammpsTrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory obtained from LAMMPS output.

    Parses the LAMMPS dump file to obtain metadata (frames, atoms, box size)
    and reads positions and forces with the package's own parser. MDAnalysis
    provides the topology only.

    Frames are returned with atoms in ascending id order, whatever order
    the dump was written in. The dump must contain an ``id`` column, and
    every frame's ids are checked against the topology when it is read.

    Parameters
    ----------
    trajectory_file : str or list of str
        Path(s) to LAMMPS dump files.
    topology_file : str
        Path to corresponding LAMMPS data or topology file.
    temperature : float
        Simulation temperature in Kelvin.
    units : str, optional
        LAMMPS unit system (default: `'real'`).
    atom_style : str, optional
        LAMMPS atom style (default: `'full'`).

    Attributes
    ----------
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.

    Raises
    ------
    ValueError
        If the cell matrix is invalid or box dimensions cannot be parsed.
    RuntimeError
        If the dump header cannot be parsed (including a dump without an
        ``id`` column), if the trajectory cannot be parsed by MDAnalysis, or
        if the topology atom ids are not in ascending order.

    Notes
    -----
    Reading a frame raises ValueError if its atom ids are not exactly the
    topology's atom ids.
    """

    def __init__(
        self,
        trajectory_file: str | list[str],
        topology_file: str | None = None,
        *,
        temperature: float,
        units: str = 'real',
        atom_style: str = 'full',
    ):
        super().__init__(units=units, temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        if topology_file is None:
            raise ValueError("A topology file is required for LAMMPS trajectories.")

        if isinstance(trajectory_file, list):
            first_traj = trajectory_file[0]
            all_trajs = trajectory_file
        else:
            first_traj = trajectory_file
            all_trajs = [trajectory_file]

        try:
            (self.frames, self.num_ats, self.dic, self.header_length,
             self.dimgrid, self._id_column) = first_read(first_traj)
        except (ValueError, OSError) as e:
            raise RuntimeError(f"Failed to parse LAMMPS trajectory header: {e}") from e

        try:
            mdanalysis_universe = MD.Universe(topology_file, *all_trajs, atom_style=atom_style,format="LAMMPSDump")
        except Exception as e:
            raise RuntimeError(f"Failed to load LAMMPS trajectory with MDAnalysis: {e}")

        self.mdanalysis_universe = mdanalysis_universe
        self.frames = len(mdanalysis_universe.trajectory)

        topology_ids = np.asarray(mdanalysis_universe.atoms.ids)
        if not np.all(np.diff(topology_ids) > 0):
            raise RuntimeError(
                "Topology atom ids are not in ascending order; positional "
                "indices would not match id-sorted frames."
            )
        self._topology_ids = topology_ids

        dims = mdanalysis_universe.dimensions
        if len(dims) < 6:
            raise ValueError(f"Invalid LAMMPS box dimensions: {dims}")

        self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
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

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """Parse LAMMPS dump file sequentially for positions and forces."""
        needed_quantities = ["x", "y", "z", "fx", "fy", "fz"]
        strngdex = define_strngdex(needed_quantities, self.dic)

        traj_file = self.trajectory_file
        if isinstance(traj_file, list):
            traj_file = traj_file[0]

        with open(traj_file) as f:
            # Skip to start frame
            if start > 0:
                frame_skip(f, self.num_ats, start, self.header_length)

            frame_idx = start
            while frame_idx < stop:
                try:
                    data = get_a_frame(
                        f, self.num_ats, self.header_length, strngdex,
                        self._id_column, self._topology_ids,
                    )
                except ValueError as e:
                    raise ValueError(f"{traj_file}, frame {frame_idx}: {e}") from e
                positions = data[:, :3]
                forces = data[:, 3:]
                yield Frame(positions, forces)

                # Skip (stride - 1) frames before the next read
                frames_to_skip = stride - 1
                frame_idx += 1
                if frame_idx < stop and frames_to_skip > 0:
                    frame_skip(f, self.num_ats, frames_to_skip, self.header_length)
                    frame_idx += frames_to_skip

    def get_frame(self, index: int) -> Frame:
        """Return positions and forces for a specific frame by index.

        LAMMPS dump files are sequential, so random access requires caching
        all frames in memory on first call. Subsequent calls use the cache.

        This is only needed for interleaved blocking. Contiguous blocking
        (the default) streams frames via ``iter_frames()`` and does not
        trigger the cache.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape ``(n_atoms, 3)``.

        Warning
        -------
        This can cause significant memory usage for large trajectories.
        """
        # Lazy-load cache on first random access
        if not hasattr(self, '_frame_cache'):
            self._frame_cache = list(self.iter_frames())

        return self._frame_cache[index]
