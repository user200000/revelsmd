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
from typing import Any, BinaryIO, Iterator, Literal, Self

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
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


# Not made absolute: the file names that stack frames record (co_filename)
# are as imported, and a package imported through a relative or ".." path
# must still match.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__)) + os.sep


def warn_user(message: str) -> None:
    """Issue a UserWarning attributed to the first caller outside RevelsMD."""
    warnings.warn(message, UserWarning, skip_file_prefixes=(_PACKAGE_DIR,))


def warn_cut_short(path: str, what: str) -> None:
    """Warn that the incomplete final ``what`` of ``path``, as a killed job leaves, is dropped."""
    warn_user(
        f"{path}: dropping the incomplete final {what}; the file appears to have "
        "been cut short"
    )


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


TrajectoryPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]
TrajectoryFiles = TrajectoryPath | Iterable[TrajectoryPath]


def as_paths(files: TrajectoryFiles) -> list[str]:
    """Return a list of path strings from one path or several, in the order given.

    Raises
    ------
    ValueError
        If no paths are given.
    """
    if isinstance(files, (str, bytes, os.PathLike)):
        return [os.fsdecode(files)]
    paths = [os.fsdecode(path) for path in files]
    if not paths:
        raise ValueError("At least one trajectory file is required.")
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


# Exceptions that say something other than "MDAnalysis cannot parse this
# file", and so are not relabelled: errors opening a file, a missing
# optional package (pytng for TNG), resource exhaustion, and warnings turned
# into errors.
_NOT_PARSE_ERRORS = (
    FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError,
    ImportError, MemoryError, RecursionError, Warning,
)


class reading_with_mdanalysis:
    """Context manager: raise ``error`` naming ``files`` if MDAnalysis cannot read them.

    MDAnalysis raises many exception types for a file it can open but not
    parse (ValueError, KeyError, OSError, EOFError, StopIteration and
    others), so every Exception is relabelled except those in
    ``_NOT_PARSE_ERRORS``. It is a class rather than a generator-based
    context manager, which would let StopIteration escape unchanged.
    """

    def __init__(self, files: str, error: type[Exception] = RuntimeError) -> None:
        self.files = files
        self.error = error

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        if exc is None or not isinstance(exc, Exception) or isinstance(exc, _NOT_PARSE_ERRORS):
            return False
        raise self.error(
            f"MDAnalysis cannot read {self.files}: {type(exc).__name__}: {exc}"
        ) from exc


def open_universe(topology, trajectories: Sequence[str] = (), **options):
    """Open an MDAnalysis Universe, naming any file that cannot be read.

    Topology and trajectory paths are opened first, so a file that cannot
    be opened raises the operating system's error (FileNotFoundError,
    PermissionError, IsADirectoryError) naming it; a topology that is not a
    path, such as an MDAnalysis stream, is passed on as it is. If MDAnalysis
    cannot read the files, RuntimeError names them and the exception it
    raised; with several trajectory files, each is opened alone to find the
    one it cannot read.
    """
    paths = list(trajectories)
    for path in ([topology] if isinstance(topology, str) else []) + paths:
        open(path, "rb").close()
    files = f"{topology} with {', '.join(paths)}" if paths else str(topology)
    arguments = [topology] + ([paths[0] if len(paths) == 1 else paths] if paths else [])
    try:
        with reading_with_mdanalysis(files):
            return MD.Universe(*arguments, **options)
    except RuntimeError as joined:
        for path in paths if len(paths) > 1 else []:
            try:
                with reading_with_mdanalysis(f"{topology} with {path}"):
                    MD.Universe(topology, path, **options)
            except RuntimeError as alone:
                raise alone from joined.__cause__
        raise


class DataUnavailableError(Exception):
    """Raised when requested data (charges, masses) is not available for a trajectory type."""
    pass


def topology_values(atoms: Any, name: Literal["charges", "masses"]) -> np.ndarray:
    """Return the ``charges`` or ``masses`` of MDAnalysis ``atoms``.

    Raises DataUnavailableError if the topology does not record them.
    """
    try:
        return np.array(getattr(atoms, name))
    except NoDataError:
        what = "Charge" if name == "charges" else "Mass"
        raise DataUnavailableError(f"{what} data not available in this topology.") from None


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
        return self._checked(range(start, stop, stride), self._iter_frames_impl(start, stop, stride))

    def _checked(self, indices: Iterable[int], frames: Iterator[Frame]) -> Iterator[Frame]:
        """Yield each frame after checking its values are finite."""
        for index, frame in zip(indices, frames):
            check_finite(frame.positions, frame.forces, self._where(index))
            yield frame

    def _where(self, index: int) -> str:
        """Name frame ``index`` for error messages; backends add the file."""
        return f"frame {index}"

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
            If the frame cannot be read, fails the backend's checks, or has
            NaN or infinite positions or forces.
        """
        index = operator.index(index)
        if not -self.frames <= index < self.frames:
            raise IndexError(f"frame {index} is out of range for {self.frames} frames")
        index %= self.frames
        frame = self._get_frame(index)
        check_finite(frame.positions, frame.forces, self._where(index))
        return frame

    @abstractmethod
    def _get_frame(self, index: int) -> Frame:
        """Return frame ``index``, where ``0 <= index < frames``."""
        ...


@dataclass(frozen=True, eq=False)
class FileFrames:
    """Where each frame of a trajectory joined from several files is stored.

    The frames of ``paths[k]`` are frames ``first[k]`` to ``first[k + 1] - 1``;
    frame ``i`` is bytes ``[starts[i], ends[i])`` of its file. Tables made
    by :meth:`join` have read-only arrays.
    """

    paths: list[str]
    first: np.ndarray
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
            "first": np.concatenate([[0], np.cumsum([len(s) for s in starts])]).astype(np.int64),
            "starts": np.concatenate(starts),
            "ends": np.concatenate(ends),
        }
        for values in (*arrays.values(), *fields.values()):
            if isinstance(values, np.ndarray):
                values.flags.writeable = False
        return cls(paths=paths, **arrays, **fields)

    def __len__(self) -> int:
        return int(self.first[-1])

    def locate(self, i: int) -> tuple[int, int]:
        """The file of frame ``i`` and the frame's position in it."""
        k = int(np.searchsorted(self.first, i, side="right")) - 1
        return k, i - int(self.first[k])

    def path(self, i: int) -> str:
        """The path of frame ``i``."""
        return self.paths[self.locate(i)[0]]


class IndexedFileTrajectory(Trajectory):
    """A trajectory whose frames are read from indexed byte ranges of files.

    Subclasses set ``_frames`` and implement ``_where`` and ``_read_arrays``.
    Frames are read with one file open at a time; a reader's ValueError is
    raised again naming the file and frame.
    """

    _frames: FileFrames

    @abstractmethod
    def _where(self, index: int) -> str:
        """The file and the position within it of frame ``index``, for error messages."""
        ...

    @abstractmethod
    def _read_arrays(self, f: BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the positions and forces of frame ``index`` from its open file.

        Raise ValueError without the file or frame, which are added.
        """
        ...

    def _read(self, f: BinaryIO, index: int) -> Frame:
        try:
            positions, forces = self._read_arrays(f, index)
        except ValueError as e:
            raise ValueError(f"{self._where(index)}: {e}") from e
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
