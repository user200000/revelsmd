"""
LAMMPS trajectory backend for RevelsMD.

This module provides the LammpsTrajectory class for reading LAMMPS dump files,
along with helper functions for parsing the LAMMPS dump format.
"""

from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError


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
        Index of the ``id`` column within the atom rows.

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
        print("WARNING: Non-integer frame count - incomplete file or inconsistent headers.")
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
    [3, 5]
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
        If the trajectory cannot be parsed by MDAnalysis, or if the topology
        atom ids are not in ascending order.

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
        except Exception as e:
            raise RuntimeError(f"Failed to parse LAMMPS trajectory header: {e}")

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
