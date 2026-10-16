"""
Base classes for trajectory handling in RevelsMD.

This module defines the abstract base class and common exceptions used by
all trajectory backends.
"""

import operator
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator, Self

import numpy as np
import scipy.constants as constants

from revelsMD.cell import is_orthorhombic as _is_orthorhombic_cell
from revelsMD.frame_sources import Frame


def normalize_bounds(
    n_frames: int, start: int, stop: int | None, stride: int
) -> tuple[int, int, int]:
    """Normalise the representation of start/stop/stride bounds.

    Converts negative indices to positive and resolves ``None`` stop.
    Out-of-bounds values are clamped, not rejected — callers that need
    strict validation (e.g. user-facing APIs) should check bounds
    *before* calling this function. Only positive strides are supported.

    Parameters
    ----------
    n_frames : int
        Total number of frames.
    start : int
        Start index (can be negative).
    stop : int or None
        Stop index (can be negative or None for end of trajectory).
    stride : int
        Step between frames. Must be >= 1.

    Returns
    -------
    tuple of (int, int, int)
        Normalised (start, stop, stride) suitable for use with ``range()``.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stop is None:
        stop = n_frames
    if start < 0:
        start = max(0, n_frames + start)
    if stop < 0:
        stop = max(0, n_frames + stop)
    start = min(start, n_frames)
    stop = min(stop, n_frames)
    return start, stop, stride


def read_frames_one_file_at_a_time(
    indices: Iterable[int],
    path_of: Callable[[int], str],
    read: Callable[[BinaryIO, int], Frame],
) -> Iterator[Frame]:
    """Yield ``read(handle, i)`` for each index, keeping at most one file open.

    Consecutive indices in the same file share one open handle; the file is
    closed when the path changes and when iteration ends, raises or is
    abandoned.
    """
    handle: BinaryIO | None = None
    current: str | None = None
    try:
        for index in indices:
            path = path_of(index)
            if path != current:
                if handle is not None:
                    handle.close()
                handle = open(path, "rb")
                current = path
            assert handle is not None
            yield read(handle, index)
    finally:
        if handle is not None:
            handle.close()


# Not made absolute: frames' file names are as imported, and a package
# imported through a relative or ".." path must still match.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__)) + os.sep


def warn_user(message: str) -> None:
    """Issue a UserWarning attributed to the first caller outside RevelsMD."""
    warnings.warn(message, UserWarning, skip_file_prefixes=(_PACKAGE_DIR,))


def check_fixed_cell(cell: np.ndarray, reference: np.ndarray, where: str) -> None:
    """Raise unless a frame's cell is exactly the first frame's.

    Parameters
    ----------
    cell : np.ndarray
        The frame's cell, in whatever representation the loader compares
        (a cell matrix, box bounds, or box dimensions).
    reference : np.ndarray
        The first frame's cell in the same representation.
    where : str
        File and frame, for the error message.

    Raises
    ------
    ValueError
        If the two differ in any element.
    """
    if not np.array_equal(cell, reference):
        raise ValueError(
            f"{where}: the cell differs from the first frame's; "
            "trajectories must have a fixed cell"
        )


def check_finite(positions: np.ndarray, forces: np.ndarray, where: str) -> None:
    """Raise if a frame's positions or forces contain NaN or infinity.

    Parameters
    ----------
    positions, forces : np.ndarray
        The frame's arrays.
    where : str
        File and frame, for the error message.

    Raises
    ------
    ValueError
        If any element of either array is not finite.
    """
    for name, values in (("positions", positions), ("forces", forces)):
        if not np.isfinite(values).all():
            raise ValueError(f"{where}: the {name} contain NaN or infinite values")


def as_paths(
    files: "str | os.PathLike[str] | Sequence[str | os.PathLike[str]]",
) -> list[str]:
    """Return a list of path strings from one path or a sequence of paths.

    Raises
    ------
    TypeError
        If ``files`` is neither a path, a sequence nor a one-dimensional
        array (a set, for example, has no order).
    ValueError
        If no paths are given, or if the same file is given twice.
    """
    if isinstance(files, (str, bytes, os.PathLike)):
        return [os.fsdecode(files)]
    if isinstance(files, np.ndarray) and files.ndim == 1:
        files = files.tolist()
    if not isinstance(files, Sequence):
        raise TypeError(
            "trajectory files must be a path or a sequence of paths in run "
            f"order, not {type(files).__name__}"
        )
    paths = [os.fsdecode(path) for path in files]
    if not paths:
        raise ValueError("At least one trajectory file is required.")
    seen: dict[object, str] = {}
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue  # the loader reports the file it cannot open
        # Without inode numbers, fall back to the resolved path.
        key: object = (
            (stat.st_dev, stat.st_ino) if stat.st_ino
            else os.path.normcase(os.path.realpath(path))
        )
        if key in seen:
            raise ValueError(f"{path}: the same file as {seen[key]} is given twice")
        seen[key] = path
    return paths


# Boltzmann constants in different unit systems
_BOLTZMANN_CONSTANTS: dict[str, float] = {
    'lj': 1.0,
    'real': constants.physical_constants["molar gas constant"][0] / constants.calorie / 1000,
    'metal': constants.physical_constants["Boltzmann constant in eV/K"][0],
    'mda': constants.physical_constants["molar gas constant"][0] / 1000,
}


def compute_beta(units: str, temperature: float) -> float:
    """
    Compute beta = 1/(kB*T) for the given unit system and temperature.

    Parameters
    ----------
    units : str
        Unit system ('lj', 'real', 'metal', 'mda').
    temperature : float
        Temperature in Kelvin for 'real', 'metal', and 'mda' units.
        For 'lj' (Lennard-Jones reduced units), this is the dimensionless
        reduced temperature T* = kB*T/epsilon.

    Returns
    -------
    float
        Inverse thermal energy in the appropriate units.

    Raises
    ------
    ValueError
        If the unit system is not recognised, or if temperature is not
        finite and positive.
    """
    units = units.lower().strip()
    if units not in _BOLTZMANN_CONSTANTS:
        raise ValueError(
            f"Unsupported unit system: '{units}'. "
            f"Expected one of {list(_BOLTZMANN_CONSTANTS.keys())}."
        )
    if not np.isfinite(temperature):
        raise ValueError(f"Temperature must be finite. Got: {temperature!r}")
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive. Got: {temperature!r}")
    return 1.0 / (_BOLTZMANN_CONSTANTS[units] * temperature)


# What MDAnalysis raises for a file it can open but not parse. OSError,
# for a missing or unreadable file, is not included and passes through.
MDANALYSIS_PARSE_ERRORS = (ValueError, KeyError, IndexError, TypeError)


class DataUnavailableError(Exception):
    """Raised when requested data (charges, masses) is not available for a trajectory type."""
    pass


class Trajectory(ABC):
    """
    Abstract base class defining the interface for trajectory objects.

    All trajectory backends must implement this interface to ensure consistent
    access patterns across different file formats and data sources.

    Required Attributes
    -------------------
    frames : int
        Number of frames in the trajectory.
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.
    units : str
        Unit system identifier (e.g., 'real', 'metal', 'mda', 'lj').
    temperature : float
        Simulation temperature in Kelvin for 'real', 'metal', and 'mda' units.
        For 'lj' units, this is the dimensionless reduced temperature T*.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.

    Properties
    ----------
    box_x, box_y, box_z : float
        Simulation box dimensions (orthorhombic cells only).
        Raises ``AttributeError`` for non-orthorhombic cells.
    cell_volume : float
        Volume of the simulation cell.
    is_orthorhombic : bool
        Whether the cell is orthorhombic (all off-diagonal elements negligible).
    """

    # Required attributes - subclasses must set these
    frames: int
    cell_matrix: np.ndarray
    units: str
    temperature: float
    beta: float

    def __init__(self, *, units: str, temperature: float) -> None:
        """
        Initialise common trajectory attributes.

        Subclasses should call ``super().__init__(units=..., temperature=...)``
        before setting their own attributes.

        Parameters
        ----------
        units : str
            Unit system identifier (e.g., 'real', 'metal', 'mda', 'lj').
        temperature : float
            Simulation temperature in Kelvin for 'real', 'metal', and 'mda' units.
            For 'lj' units, this is the dimensionless reduced temperature T*.
        """
        self.units = units
        self.temperature = temperature
        self.beta = compute_beta(units, temperature)

    @property
    def cell_volume(self) -> float:
        """Volume of the simulation cell: ``abs(det(cell_matrix))``."""
        return float(abs(np.linalg.det(self.cell_matrix)))

    @property
    def is_orthorhombic(self) -> bool:
        """Whether the cell matrix is orthorhombic (diagonal)."""
        return _is_orthorhombic_cell(self.cell_matrix)

    @property
    def box_x(self) -> float:
        """Box length along x (orthorhombic cells only)."""
        if not self.is_orthorhombic:
            raise AttributeError(
                "box_x is not defined for non-orthorhombic cells. "
                "Use cell_matrix instead."
            )
        return float(self.cell_matrix[0, 0])

    @property
    def box_y(self) -> float:
        """Box length along y (orthorhombic cells only)."""
        if not self.is_orthorhombic:
            raise AttributeError(
                "box_y is not defined for non-orthorhombic cells. "
                "Use cell_matrix instead."
            )
        return float(self.cell_matrix[1, 1])

    @property
    def box_z(self) -> float:
        """Box length along z (orthorhombic cells only)."""
        if not self.is_orthorhombic:
            raise AttributeError(
                "box_z is not defined for non-orthorhombic cells. "
                "Use cell_matrix instead."
            )
        return float(self.cell_matrix[2, 2])

    @staticmethod
    def _cell_matrix_from_dimensions(
        lx: float, ly: float, lz: float
    ) -> np.ndarray:
        """Build a diagonal cell matrix from orthorhombic box dimensions."""
        return np.diag(np.array([lx, ly, lz], dtype=np.float64))

    def _normalize_bounds(
        self, start: int, stop: int | None, stride: int
    ) -> tuple[int, int, int]:
        """Normalise start/stop bounds following Python slice semantics.

        Delegates to the module-level :func:`normalize_bounds` using
        ``self.frames`` as the frame count.
        """
        return normalize_bounds(self.frames, start, stop, stride)

    @staticmethod
    def _validate_cell_matrix(cell_matrix: np.ndarray, where: str | None = None) -> None:
        """
        Validate that a cell matrix is well-formed.

        Parameters
        ----------
        cell_matrix : np.ndarray
            Expected shape (3, 3), all elements finite, positive volume.
        where : str, optional
            File and frame the cell comes from, prefixed to the error message.

        Raises
        ------
        ValueError
            If the cell matrix has wrong shape, non-finite elements, or
            zero/negative volume.
        """
        prefix = f"{where}: " if where is not None else ""
        if cell_matrix.shape != (3, 3):
            raise ValueError(
                f"{prefix}Cell matrix must have shape (3, 3). Got shape: {cell_matrix.shape}"
            )
        if not np.all(np.isfinite(cell_matrix)):
            raise ValueError(
                f"{prefix}Cell matrix elements must all be finite. Got:\n{cell_matrix}"
            )
        volume = abs(np.linalg.det(cell_matrix))
        if volume < 1e-12:
            raise ValueError(
                f"{prefix}Cell matrix must have positive volume. Got volume: {volume}"
            )

    @abstractmethod
    def get_indices(self, atype: str) -> np.ndarray:
        """Return atom indices for a given species or type.

        Parameters
        ----------
        atype : str
            Atom species or type name (format is backend-specific).

        Returns
        -------
        np.ndarray
            Row positions into the per-frame position and force arrays for
            atoms matching the given species or type.

        Raises
        ------
        ValueError
            If no atom matches.
        """
        ...

    def get_charges(self, atype: str) -> np.ndarray:
        """Return atomic charges for atoms of a given species or type.

        Subclasses should override this method if charge data is available.
        The default implementation raises ``DataUnavailableError``.

        Parameters
        ----------
        atype : str
            Atom species or type name (format is backend-specific).

        Returns
        -------
        np.ndarray
            Array of atomic charges for the selected atoms.

        Raises
        ------
        DataUnavailableError
            If charge data is not available for this trajectory backend.
        """
        raise DataUnavailableError("Charge data not available for this trajectory type.")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for atoms of a given species or type.

        Subclasses should override this method if mass data is available.
        The default implementation raises ``DataUnavailableError``.

        Parameters
        ----------
        atype : str
            Atom species or type name (format is backend-specific).

        Returns
        -------
        np.ndarray
            Array of atomic masses for the selected atoms.

        Raises
        ------
        DataUnavailableError
            If mass data is not available for this trajectory backend.
        """
        raise DataUnavailableError("Mass data not available for this trajectory type.")

    def iter_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1
    ) -> Iterator[Frame]:
        """
        Iterate over trajectory frames, yielding Frame instances.

        Parameters
        ----------
        start : int, optional
            First frame index (default: 0). Negative indices count from end.
        stop : int, optional
            Stop iteration before this frame (default: None, meaning all frames).
            Negative indices count from end.
        stride : int, optional
            Step between frames (default: 1).

        Yields
        ------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape (n_atoms, 3).
        """
        start, stop, stride = self._normalize_bounds(start, stop, stride)
        return self._iter_frames_impl(start, stop, stride)

    @abstractmethod
    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[Frame]:
        """
        Internal implementation of frame iteration.

        Subclasses implement this with normalized (non-negative) bounds.
        """
        ...

    def get_frame(self, index: int) -> Frame:
        """
        Return positions and forces for a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index to retrieve. Negative indices count from the end.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape (n_atoms, 3).

        Raises
        ------
        TypeError
            If ``index`` is not an integer.
        IndexError
            If ``index`` is out of range.
        ValueError
            If the frame cannot be read or fails the backend's checks.
        """
        index = operator.index(index)
        if not -self.frames <= index < self.frames:
            raise IndexError(f"frame {index} is out of range for {self.frames} frames")
        return self._get_frame(index % self.frames)

    @abstractmethod
    def _get_frame(self, index: int) -> Frame:
        """Return frame ``index``, where ``0 <= index < frames``."""
        ...


@dataclass(frozen=True, eq=False)
class FileFrames:
    """Where each frame of a trajectory joined from several files is stored.

    Frame ``i`` is bytes ``[starts[i], ends[i])`` of ``paths[file[i]]``.
    ``file`` does not decrease. Tables made by :meth:`join` have read-only
    arrays.
    """

    paths: list[str]
    file: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    @classmethod
    def join(
        cls,
        paths: list[str],
        starts: list[np.ndarray],
        ends: list[np.ndarray],
        **fields: Any,
    ) -> Self:
        """Join per-file byte ranges, in file order, into one table.

        ``fields`` are passed on to a subclass's constructor; array fields
        are made read-only in place, so pass arrays the table can own.
        """
        arrays = {
            "file": np.repeat(
                np.arange(len(paths), dtype=np.int32), [len(s) for s in starts]
            ),
            "starts": np.concatenate(starts),
            "ends": np.concatenate(ends),
        }
        for values in (*arrays.values(), *fields.values()):
            if isinstance(values, np.ndarray):
                values.flags.writeable = False
        return cls(paths=paths, **arrays, **fields)

    def __len__(self) -> int:
        return len(self.file)

    def path(self, i: int) -> str:
        """The path of frame ``i``."""
        return self.paths[self.file[i]]


class IndexedFileTrajectory(Trajectory):
    """A trajectory whose frames are read from indexed byte ranges of files.

    Subclasses set ``_frames`` and implement ``_where`` and ``_read_arrays``.
    Frames are read with one file open at a time; a reader's ValueError is
    raised again naming the file and frame, and frames with NaN or infinite
    values are rejected.
    """

    _frames: FileFrames

    @abstractmethod
    def _where(self, index: int) -> str:
        """The file and frame ``index``, for error messages."""
        ...

    @abstractmethod
    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the positions and forces of frame ``index`` from its open file."""
        ...

    def _read(self, f: BinaryIO, index: int) -> Frame:
        where = self._where(index)
        try:
            positions, forces = self._read_arrays(f, index)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from e
        check_finite(positions, forces, where)
        return Frame(positions, forces)

    def _iter_frames_impl(self, start: int, stop: int, stride: int) -> Iterator[Frame]:
        """Read the indexed frames in order, with one file open at a time."""
        return read_frames_one_file_at_a_time(
            range(start, stop, stride), self._frames.path, self._read
        )

    def _get_frame(self, index: int) -> Frame:
        """Read one frame by seeking to it; nothing else is read or kept."""
        with open(self._frames.path(index), "rb") as f:
            return self._read(f, index)
