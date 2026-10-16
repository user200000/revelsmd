"""
MDAnalysis trajectory backend for RevelsMD.

This module provides the MDATrajectory class for reading trajectories
via MDAnalysis.
"""

import errno
import os
from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.coordinates.H5MD import H5MDReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.LAMMPS import DumpReader  # type: ignore[import-untyped]
from MDAnalysis.coordinates.chain import ChainReader  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import (
    TrajectoryFiles,
    Trajectory, as_paths, check_finite, check_fixed_cell, reading_with_mdanalysis,
    replaced_frames_dropped, warn_user,
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
        and ``nstvout`` 0 or a multiple of it), or a list of the files of a
        run and its restarts, in run order. Each file's frames replace the
        previous file's frames from its own first step on (first time, for
        formats without steps). Independent runs are loaded separately.
    topology_file : str
        Path to the topology file (e.g., `.tpr`, `.pdb`, `.gro`).
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    frames : int
        Number of trajectory frames, after joining restart files.
        ``mdanalysis_universe.trajectory`` still holds every frame of every
        file.
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
        If no topology file is provided, if no trajectory file is given or
        the same file is given twice, if a file starts at or before the
        first step of the file before it, if MDAnalysis does not know the
        format's length or force units, if an H5MD file saves positions and
        forces at different steps, if the first frame has no box, no
        positions or no forces, or if the cell matrix is invalid.
    TypeError
        If ``trajectory_file`` is an unordered collection such as a set.
    RuntimeError
        If MDAnalysis cannot parse the trajectory or topology files (the
        message names them), or if the steps of an H5MD file cannot be read.
    OSError
        If a file does not exist (``FileNotFoundError``) or cannot be opened.
    ImportError
        If MDAnalysis needs a package that is not installed to read the
        format (``pytng`` for TNG, ``h5py`` for H5MD).

    Notes
    -----
    Reading a frame raises ValueError if it has no positions, no forces or
    no box, if its box differs from the first frame's, if its positions or
    forces contain NaN or infinity, or if its step (time, for formats
    without steps) does not increase from the frame of the same file read
    before it.
    """

    def __init__(
        self,
        trajectory_file: TrajectoryFiles,
        topology_file: str,
        *,
        temperature: float,
    ):
        if not topology_file:
            raise ValueError("A topology file is required for MDAnalysis trajectories.")
        paths = as_paths(trajectory_file)

        super().__init__(units='mda', temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        for path in (os.fspath(topology_file), *paths):
            if not os.path.exists(path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        with reading_with_mdanalysis(f"{topology_file} with {', '.join(paths)}"):
            mdanalysis_universe = MD.Universe(
                topology_file, paths[0] if len(paths) == 1 else paths
            )

        self.mdanalysis_universe = mdanalysis_universe
        trajectory = mdanalysis_universe.trajectory
        readers = _readers(trajectory)
        if len(readers) != len(paths):
            raise RuntimeError(
                f"MDAnalysis opened {len(readers)} readers for {len(paths)} files"
            )
        # MDAnalysis frame j is frame j - _first_frames[k] of paths[k].
        self._paths = paths
        self._first_frames = np.cumsum([0] + [len(r) for r in readers[:-1]])

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
        for reader in readers:
            _check_reader(reader)
        self._order = _FrameOrder(readers)
        # Frame i is MDAnalysis frame _kept[i]; None keeps every frame.
        self._kept = _join_restarts(readers, paths, self._order) if len(paths) > 1 else None
        self._last_read: tuple[int, int, float] | None = None
        trajectory.rewind()
        self.frames = len(trajectory) if self._kept is None else len(self._kept)
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
            raise ValueError(f"{self.topology_file}: no atoms named '{atype}'")
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
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').charges)

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
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').masses)

    def _locate(self, frame: int) -> tuple[int, int]:
        """The file and the frame within it of MDAnalysis frame ``frame``."""
        k = int(np.searchsorted(self._first_frames, frame, side="right")) - 1
        return k, frame - int(self._first_frames[k])

    def _frame(self, ts) -> Frame:
        """Check one MDAnalysis timestep and return it as a Frame."""
        k, local = self._locate(ts.frame)
        where = f"{self._paths[k]}, frame {local}"
        key = self._order.key(ts)
        if self._last_read is not None:
            last_k, last_local, last_key = self._last_read
            if last_k == k and last_local != local and (local > last_local) != (key > last_key):
                raise ValueError(
                    f"{where}: {self._order.unit}s do not increase within the file "
                    f"({self._order.unit} {last_key} at frame {last_local}, {key} at "
                    f"frame {local})"
                )
        self._last_read = (k, local, key)
        if not ts.has_positions:
            raise ValueError(f"{where}: no positions")
        if not ts.has_forces:
            raise ValueError(f"{where}: no forces; force sampling needs forces in every frame")
        if ts.dimensions is None:
            raise ValueError(f"{where}: no simulation box")
        check_fixed_cell(np.array(ts.dimensions, dtype=np.float64), self._dimensions, where)
        check_finite(ts.positions, ts.forces, where)
        return Frame(ts.positions.copy(), ts.forces.copy())

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """Iterate using MDAnalysis trajectory slicing."""
        trajectory = self.mdanalysis_universe.trajectory
        selected = (
            trajectory[start:stop:stride] if self._kept is None
            else trajectory[self._kept[start:stop:stride]]
        )
        for ts in selected:
            yield self._frame(ts)

    def _get_frame(self, index: int) -> Frame:
        frame = index if self._kept is None else int(self._kept[index])
        return self._frame(self.mdanalysis_universe.trajectory[frame])


class _FrameOrder:
    """How frames are ordered: by integer step if every file records one.

    TRR, TNG and H5MD files record each frame's step. Times are stored in
    single precision, so late in a long run neighbouring frames can share a
    time; they are used only for formats without steps (NetCDF).
    """

    def __init__(self, readers: list) -> None:
        self.unit = "step" if all("step" in reader[0].data for reader in readers) else "time"

    def key(self, ts) -> float:
        return int(ts.data["step"]) if self.unit == "step" else float(ts.time)

    def at(self, reader, j: int) -> float:
        return self.key(reader[j])


def _join_restarts(readers: list, paths: list[str], order: _FrameOrder) -> np.ndarray:
    """The MDAnalysis frames kept when files from a run and its restarts are joined.

    As for LAMMPS dumps, each file's frames replace the previous file's
    frames from its own first step on; this drops the frame GROMACS repeats
    at the start of each file written with ``-noappend``. Replaced frames
    outside every joined file's range of kept steps are reported with a
    warning. Steps are taken to increase within each file (frames are
    checked when read), so each file is searched by bisection and only a
    few frames of each are read here.

    Raises
    ------
    ValueError
        If a file starts at or before the first step of the file before it.
    """
    unit = order.unit
    firsts = [order.at(reader, 0) for reader in readers]
    kept = [len(reader) for reader in readers]
    replaced: list[tuple[int, int, int]] = []  # (file, first replaced, end of replaced)
    for k in range(1, len(readers)):
        if firsts[k] <= firsts[k - 1]:
            raise ValueError(
                f"{paths[k]} starts at {unit} {firsts[k]}, at or before the start of "
                f"{paths[k - 1]} ({unit} {firsts[k - 1]}); {unit}s must continue from "
                "file to file, as in a run and its restarts. Load independent runs "
                "as separate trajectories"
            )
        n = _bisect(readers[k - 1], 0, kept[k - 1], firsts[k], order)
        if n < kept[k - 1]:
            replaced.append((k - 1, n, kept[k - 1]))
        kept[k - 1] = n
    spans = [(firsts[k], order.at(reader, kept[k] - 1)) for k, reader in enumerate(readers)]
    for k, start, end in replaced:
        reader = readers[k]
        # Frames [start, end) of file k are replaced; mark those a span covers.
        covered = sorted(
            (_bisect(reader, start, end, lo, order), _bisect(reader, start, end, hi, order, right=True))
            for lo, hi in spans
        )
        lost: list[tuple[int, int]] = []
        position = start
        for a, b in covered:
            if a > position:
                lost.append((position, a))
            position = max(position, b)
        if position < end:
            lost.append((position, end))
        if lost:
            warn_user(replaced_frames_dropped(
                paths[k + 1], paths[k], unit, firsts[k + 1], sum(b - a for a, b in lost),
                order.at(reader, lost[0][0]), order.at(reader, lost[-1][1] - 1),
            ))
    offsets = np.cumsum([0] + [len(reader) for reader in readers[:-1]])
    return np.concatenate(
        [offset + np.arange(n, dtype=np.int64) for offset, n in zip(offsets, kept)]
    )


def _bisect(reader, lo: int, hi: int, target: float, order: _FrameOrder, right: bool = False) -> int:
    """The first frame in [lo, hi) whose step is at least (``right``: above) ``target``."""
    while lo < hi:
        middle = (lo + hi) // 2
        key = order.at(reader, middle)
        if key < target or (right and key == target):
            lo = middle + 1
        else:
            hi = middle
    return lo


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

