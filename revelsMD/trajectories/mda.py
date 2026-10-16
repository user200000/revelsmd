"""
MDAnalysis trajectory backend for RevelsMD.

This module provides the MDATrajectory class for reading trajectories
via MDAnalysis.
"""

import os
import struct
from collections.abc import Generator

from MDAnalysis.coordinates.H5MD import H5MDReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.LAMMPS import DumpReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.TRR import TRRReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.chain import ChainReader  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
from MDAnalysis.lib.util import NamedStream  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import (
    TrajectoryFiles, TrajectoryPath, Trajectory, as_paths, check_fixed_cell, open_universe,
    locate, reading_with_mdanalysis, topology_values, warn_cut_short,
)


class MDATrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory handled by **MDAnalysis**.

    This class acts as a unified interface for reading, validating, and accessing
    data from MDAnalysis-compatible trajectory and topology files.

    Positions and forces are converted to MDAnalysis units (Angstrom and
    kJ/(mol Angstrom)), so the trajectory format must record its length and
    force units: GROMACS TRR and TNG, AMBER NetCDF, and H5MD files that
    give units. LAMMPS dumps do not, and are read with
    :class:`~revelsMD.trajectories.LammpsTrajectory`.

    Parameters
    ----------
    trajectory_file : str, path, or sequence of these
        Path to a trajectory file in which every frame stores positions and
        forces (e.g. a `.trr` written with ``nstfout`` equal to ``nstxout``
        and ``nstvout`` 0 or a multiple of it), or a list of files, whose
        frames are concatenated in the order given. Frames that files share,
        such as the frame GROMACS repeats at each restart written with
        ``-noappend``, are counted once per file; join such files with
        ``gmx trjcat`` first.
    topology_file : str, bytes or path
        Path to the topology file (e.g., `.tpr`, `.pdb`, `.gro`), or an
        MDAnalysis stream.
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    frames : int
        Number of trajectory frames. ``mdanalysis_universe.trajectory``
        also holds a last TRR frame cut short, which is not counted.
    cell_matrix : np.ndarray
        Cell matrix with rows = lattice vectors, shape ``(3, 3)``.
    units : str
        Unit system identifier (`'mda'`).
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in mol/kJ.

    Raises
    ------
    ValueError
        If no topology file is provided, if no trajectory file is given, if
        MDAnalysis does not know the format's length or force units, if an
        H5MD file saves positions and forces at different steps, if the
        first frame has no box, no positions or no forces, or if the cell
        matrix is invalid.
    RuntimeError
        If MDAnalysis cannot parse the trajectory or topology files (the
        message names the file, and the exception MDAnalysis raised), for
        example an H5MD file when ``h5py`` is not installed, or if the steps
        of an H5MD file cannot be read.
    OSError
        If a file cannot be opened: ``FileNotFoundError``,
        ``PermissionError`` or ``IsADirectoryError``.
    ImportError
        If a TNG file is read without ``pytng`` installed.

    Notes
    -----
    A last TRR frame cut short, as left by a job killed while
    writing, is dropped with a warning. Reading a frame raises ValueError
    if MDAnalysis cannot read it, if it has no positions, no forces or no
    box, if its box differs from the first frame's, or if its positions or
    forces contain NaN or infinity; the message names the file and frame.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        topology_file: "TrajectoryPath | object",
        *,
        temperature: float,
    ):
        if not topology_file:
            raise ValueError("A topology file is required for MDAnalysis trajectories.")
        paths = as_paths(trajectory_file)

        super().__init__(units='mda', temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        # MDAnalysis streams are passed on as they are.
        topology = (
            os.fsdecode(topology_file)
            if isinstance(topology_file, (str, bytes, os.PathLike))
            and not isinstance(topology_file, NamedStream)
            else topology_file
        )
        self._topology = str(topology)
        mdanalysis_universe = open_universe(topology, paths)

        self.mdanalysis_universe = mdanalysis_universe
        trajectory = mdanalysis_universe.trajectory
        readers = _readers(trajectory)
        if len(readers) != len(paths):
            raise RuntimeError(
                f"MDAnalysis opened {len(readers)} readers for {len(paths)} files"
            )
        self._paths = paths

        dims = mdanalysis_universe.dimensions
        if dims is None:
            raise ValueError(
                f"{paths[0]}: the trajectory has no simulation box; "
                "a periodic box is required"
            )
        if not trajectory.ts.has_positions:
            raise ValueError(
                f"{paths[0]}: the trajectory has no positions; "
                "positions are needed in every frame"
            )
        if not trajectory.ts.has_forces:
            raise ValueError(
                f"{paths[0]}: the trajectory has no forces; force "
                "sampling needs forces in every frame"
            )
        complete = []
        for reader, path in zip(readers, paths):
            _check_reader(reader)
            n = len(reader)
            if n and isinstance(reader, TRRReader) and _trr_frame_is_cut(path, reader._xdr.offsets[-1]):
                warn_cut_short(path, f"frame (frame {n - 1})")
                n -= 1
            complete.append(n)
        # Frame i is frame i - _first[k] of file k, which is MDAnalysis frame
        # i - _first[k] + _mdanalysis_first[k]; a cut final frame is skipped.
        self._first = np.concatenate([[0], np.cumsum(complete)])
        self._mdanalysis_first = np.concatenate([[0], np.cumsum([len(r) for r in readers])])
        self.frames = int(self._first[-1])
        self._dimensions = np.array(dims, dtype=np.float64)
        self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
        self._validate_cell_matrix(self.cell_matrix, paths[0])

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return positional indices of atoms matching a given atom name.

        The indices are row positions in the frames yielded by this
        trajectory, independent of the atom numbering in the topology file.

        Parameters
        ----------
        atype : str
            Atom name to select (e.g., `'O'`, `'H'`, `'C'`).

        Returns
        -------
        np.ndarray
            Row indices into the per-frame position and force arrays.

        Raises
        ------
        ValueError
            If no atom has that name.
        """
        indices = np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').ix)
        if len(indices) == 0:
            raise ValueError(f"{self._topology}: no atoms named '{atype}'")
        return indices

    def get_charges(self, atype: str) -> np.ndarray:
        """
        Return atomic charges for atoms of a given name.

        Parameters
        ----------
        atype : str
            Atom name to select.

        Returns
        -------
        np.ndarray
            Array of atomic charges.

        Raises
        ------
        DataUnavailableError
            If the topology has no charges.
        """
        return topology_values(self.mdanalysis_universe.select_atoms(f'name {atype}'), "charges")

    def get_masses(self, atype: str) -> np.ndarray:
        """
        Return atomic masses for atoms of a given name.

        Parameters
        ----------
        atype : str
            Atom name to select.

        Returns
        -------
        np.ndarray
            Array of atomic masses.

        Raises
        ------
        DataUnavailableError
            If the topology has no masses.
        """
        return topology_values(self.mdanalysis_universe.select_atoms(f'name {atype}'), "masses")

    def _where(self, index: int) -> str:
        k, local = locate(self._first, index)
        return f"{self._paths[k]}, frame {local}"

    def _get_frame(self, index: int) -> Frame:
        """Read frame ``index`` with MDAnalysis and check it."""
        where = self._where(index)
        k, local = locate(self._first, index)
        with reading_with_mdanalysis(where, ValueError):
            ts = self.mdanalysis_universe.trajectory[int(self._mdanalysis_first[k]) + local]
        if not ts.has_positions:
            raise ValueError(f"{where}: no positions")
        if not ts.has_forces:
            raise ValueError(f"{where}: no forces; force sampling needs forces in every frame")
        if ts.dimensions is None:
            raise ValueError(f"{where}: no simulation box")
        check_fixed_cell(np.array(ts.dimensions, dtype=np.float64), self._dimensions, where)
        return Frame(ts.positions.copy(), ts.forces.copy())

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Generator[Frame, None, None]:
        """Read the selected frames in order."""
        for index in range(start, stop, stride):
            yield self._get_frame(index)


def _trr_frame_is_cut(path: str, offset: int) -> bool:
    """Whether the TRR frame at ``offset`` runs past the end of the file.

    The frame's size is read from its header: the sizes of its blocks,
    and its precision, which GROMACS infers from the box or coordinate size.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        _, _, length = struct.unpack(">3i", f.read(12))
        f.seek(offset + 12 + (length + 3) // 4 * 4)
        sizes = struct.unpack(">13i", f.read(52))
    box, n_atoms = sizes[2], sizes[10]
    data = max(sizes[7:10])
    double = box == 9 * 8 if box else bool(n_atoms) and data == n_atoms * 3 * 8
    header = 12 + (length + 3) // 4 * 4 + 52 + 2 * (8 if double else 4)
    return offset + header + sum(sizes[:10]) > os.path.getsize(path)


def _check_reader(reader) -> None:
    """Raise unless MDAnalysis converts a reader's positions and forces to its units.

    MDAnalysis converts lengths and forces only for formats that record
    their units; for others it returns the numbers in the file unchanged.
    H5MD files store a step for each saved position and force, and the
    reader pairs them by position in the file, so the steps must agree; the
    check reads the reader's particle group, and raises if it cannot.
    """
    units = reader.units
    if units.get("length") is None or units.get("force") is None:
        hint = "; read LAMMPS dumps with LammpsTrajectory" if isinstance(reader, DumpReader) else ""
        raise ValueError(
            f"{reader.filename}: MDAnalysis does not know the length and force "
            f"units of this format ({type(reader).__name__}), so the forces "
            f"cannot be converted to kJ/(mol Angstrom){hint}"
        )
    if not isinstance(reader, H5MDReader):
        return
    cannot_check = (
        f"{reader.filename}: cannot check that this H5MD file saves positions "
        "and forces at the same steps"
    )
    particles = getattr(reader, "_particle_group", None)
    if particles is None:
        raise RuntimeError(f"{cannot_check} with this version of MDAnalysis")
    if "position" in particles and "force" in particles:
        try:
            position_steps = particles["position/step"][()]
            force_steps = particles["force/step"][()]
        except (KeyError, OSError, TypeError) as e:
            raise RuntimeError(f"{cannot_check}: {e}") from e
        if not np.array_equal(position_steps, force_steps):
            raise ValueError(
                f"{reader.filename}: the forces are saved at different steps "
                "from the positions; every frame needs both"
            )


def _readers(trajectory) -> list:
    """The readers of an MDAnalysis trajectory, one per file."""
    if isinstance(trajectory, ChainReader):
        return list(trajectory.readers)
    return [trajectory]

