"""Base class, shared helpers and exceptions for the trajectory backends."""

import contextlib
import numbers
import operator
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, BinaryIO, Literal, Self

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.exceptions import NoDataError  # type: ignore[import-untyped]
import numpy as np
import scipy.constants as constants

from revelsMD.cell import cells_are_compatible
from revelsMD.cell import is_orthorhombic as _is_orthorhombic_cell
from revelsMD.frame_sources import Frame


def normalize_bounds(
    n_frames: int, start: int, stop: int | None, stride: int
) -> tuple[int, int, int]:
    """Resolve slice bounds to non-negative (start, stop, stride) for ``range()``.

    Out-of-range values are clamped, not rejected (:func:`frame_range`
    rejects them). Raises ValueError if ``stride`` is less than 1.
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
) -> Generator[Frame, None, None]:
    """Yield ``read(handle, i)`` for each index, keeping at most one file open.

    The file is closed when the path changes and when iteration ends, raises
    or is abandoned.
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


def warn_cut_short(path: str, what: str, reason: str | None = None) -> None:
    """Warn that the final ``what`` of ``path``, which cannot be read, is dropped."""
    because = "" if reason is None else f" ({reason})"
    warn_user(
        f"{path}: dropping the incomplete final {what}; the file appears to have "
        f"been cut short{because}"
    )


def check_fixed_cell(cell: np.ndarray, reference: np.ndarray, where: str) -> None:
    """Raise ValueError naming ``where`` unless ``cell`` equals ``reference`` exactly.

    Both may be in any representation the loader compares (cell matrix,
    box bounds or box dimensions), as long as it is the same one.
    """
    if not np.array_equal(cell, reference):
        raise ValueError(
            f"{where}: the cell differs from the first frame's; "
            "trajectories must have a fixed cell"
        )


def check_finite(positions: np.ndarray, forces: np.ndarray, where: str) -> None:
    """Raise ValueError naming ``where`` if the positions or forces contain NaN or infinity."""
    for name, values in (("positions", positions), ("forces", forces)):
        if not np.isfinite(values).all():
            raise ValueError(f"{where}: the {name} contain NaN or infinite values")


TrajectoryPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]
TrajectoryFiles = TrajectoryPath | Iterable[TrajectoryPath]


def as_paths(files: TrajectoryFiles) -> list[str]:
    """Return one path or several as a list of strings, in order; ValueError if none."""
    if isinstance(files, (str, bytes, os.PathLike)):
        return [os.fsdecode(files)]
    paths = [os.fsdecode(path) for path in files]
    if not paths:
        raise ValueError("At least one trajectory file is required.")
    return paths


def check_same_system(trajectory: Any, cell_matrix: np.ndarray, beta: float, owner: str) -> None:
    """Raise ValueError unless ``trajectory`` has the given cell and beta.

    Frames from several trajectories can be accumulated into one ``owner``
    (a density grid or RDF) only if they describe the same system: the same
    cell, and the same temperature and units, which set beta.
    """
    if not cells_are_compatible(trajectory.cell_matrix, cell_matrix):
        raise ValueError(f"the trajectory's cell differs from the {owner}'s")
    if trajectory.beta != beta:
        raise ValueError(
            f"the trajectory's temperature or units (beta = {trajectory.beta}) differ "
            f"from the {owner}'s (beta = {beta})"
        )


def frame_range(n_frames: int, start: int, stop: int | None, period: int) -> range:
    """Return the frames that ``start``, ``stop`` and ``period`` select.

    Negative indices count from the end. Unlike :func:`normalize_bounds`,
    raises ValueError if ``start`` or ``stop`` is outside
    ``[-n_frames, n_frames]``, ``period`` is less than 1, or no frames are
    selected.
    """
    for name, index in (("start", start), ("stop", stop)):
        if index is not None and not -n_frames <= index <= n_frames:
            raise ValueError(
                f"{name}={index} is outside the trajectory's {n_frames} frames"
            )
    frames = range(*normalize_bounds(n_frames, start, stop, period))
    if len(frames) == 0:
        raise ValueError(
            f"No frames selected: start={start}, stop={stop}, "
            f"period={period} yields an empty range."
        )
    return frames


# Boltzmann constants in different unit systems
_BOLTZMANN_CONSTANTS: dict[str, float] = {
    'lj': 1.0,
    'real': constants.physical_constants["molar gas constant"][0] / constants.calorie / 1000,
    'metal': constants.physical_constants["Boltzmann constant in eV/K"][0],
    'mda': constants.physical_constants["molar gas constant"][0] / 1000,
}


def compute_beta(units: str, temperature: float) -> float:
    """Return beta = 1/(kB*T) in the energy units of ``units``.

    ``temperature`` is in Kelvin, or the reduced temperature T* = kB*T/epsilon
    for ``'lj'``. Raises ValueError for an unknown unit system or a
    temperature that is not finite and positive.
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
NOT_PARSE_ERRORS = (
    FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError,
    ImportError, MemoryError, RecursionError, Warning,
)


class reading_with_mdanalysis:
    """Context manager: raise ``error`` naming ``files`` if MDAnalysis cannot read them.

    MDAnalysis raises many exception types for a file it cannot parse, so
    every Exception except ``NOT_PARSE_ERRORS`` is relabelled. A class, since
    a generator-based context manager would let StopIteration escape.
    """

    def __init__(self, files: str, error: type[Exception] = RuntimeError) -> None:
        self.files = files
        self.error = error

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc is None or not isinstance(exc, Exception) or isinstance(exc, NOT_PARSE_ERRORS):
            return False
        raise self.error(
            f"MDAnalysis cannot read {self.files}: {type(exc).__name__}: {exc}"
        ) from exc


def open_universe(topology: Any, trajectories: Sequence[str] = (), **options: Any) -> MD.Universe:
    """Open an MDAnalysis Universe that guesses no atom attributes (types, masses, ...).

    Paths are opened first, so a file that cannot be opened raises the
    operating system's error; a topology that is not a path (an MDAnalysis
    stream) is passed on as it is. RuntimeError names the files if
    MDAnalysis cannot read them.
    """
    paths = list(trajectories)
    for path in ([topology] if isinstance(topology, str) else []) + paths:
        open(path, "rb").close()
    files = f"{topology} with {', '.join(paths)}" if paths else str(topology)
    arguments = [topology] + ([paths[0] if len(paths) == 1 else paths] if paths else [])
    with reading_with_mdanalysis(files):
        return MD.Universe(*arguments, to_guess=(), **options)


class DataUnavailableError(Exception):
    """Raised when requested data (charges, masses) is not available for a trajectory type."""
    pass


def given_values(
    given: Mapping[Any, float] | None, labels: Callable[[], Iterable[str]], name: str, what: str
) -> dict[str, float]:
    """Return the user's per-label charges or masses keyed by label string.

    ``labels()`` is called only if values are given. Raises ValueError if a
    value is not a finite number (or, for masses, is negative), or a label
    is not one of the atoms'.
    """
    values: dict[str, float] = {}
    for label, value in (given or {}).items():
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or (name == "masses" and value < 0)):
            raise ValueError(
                f"{name}= gives {label!r} the value {value!r}; values must be finite "
                f"numbers{', not negative' if name == 'masses' else ''}"
            )
        values[str(label)] = float(value)
    unknown = sorted(set(values) - set(labels())) if values else []
    if unknown:
        raise ValueError(f"{name}= names {what} that no atom has: {', '.join(unknown)}")
    return values


def atom_values(
    labels: Iterable[Any],
    name: Literal["charges", "masses"],
    given: Mapping[str, float],
    recorded: Callable[[], Any],
    source: str,
) -> np.ndarray:
    """Return the charges or masses of atoms with ``labels`` (names, types or species).

    Each atom takes the value given for its label, else the one ``recorded()``
    returns for it (called only if needed; None if nothing is recorded).
    Raises DataUnavailableError naming the labels with neither.
    """
    labels = np.array([str(label) for label in labels])
    is_given = np.isin(labels, list(given))
    values = np.zeros(len(labels)) if is_given.all() else recorded()
    if values is None:
        missing = ", ".join(sorted(set(labels[~is_given])))
        raise DataUnavailableError(
            f"{source} records no {name}, and RevelsMD does not guess them; give "
            f"them for {missing} with {name}="
        )
    values = np.array(values, dtype=float)
    values[is_given] = [given[label] for label in labels[is_given]]
    return values


def topology_values(atoms: Any, name: Literal["charges", "masses"]) -> np.ndarray | None:
    """The charges or masses an MDAnalysis topology records for ``atoms``, or None."""
    try:
        return getattr(atoms, name)
    except NoDataError:
        return None


class Trajectory(ABC):
    """Abstract base class: the interface every trajectory backend implements.

    Attributes
    ----------
    frames : int
        Number of frames.
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix; rows are lattice vectors.
    units : str
        Unit system: ``'real'``, ``'metal'``, ``'mda'`` or ``'lj'``.
    temperature : float
        Temperature in Kelvin, or the reduced temperature T* for ``'lj'``.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.
    """

    # Required attributes - subclasses must set these
    frames: int
    cell_matrix: np.ndarray
    units: str
    temperature: float
    beta: float

    def __init__(self, *, units: str, temperature: float) -> None:
        """Set ``units``, ``temperature`` and ``beta``; subclasses call this first."""
        self.units = units.lower().strip()
        self.temperature = temperature
        self.beta = compute_beta(self.units, temperature)

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
        """:func:`normalize_bounds` for this trajectory's frames."""
        return normalize_bounds(self.frames, start, stop, stride)

    @staticmethod
    def _validate_cell_matrix(cell_matrix: np.ndarray, where: str | None = None) -> None:
        """Raise ValueError naming ``where`` unless the cell is 3x3, finite and of non-zero volume."""
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
        """Return the indices of the atoms of a species or type.

        Parameters
        ----------
        atype : str
            Species or type name, in the backend's convention.

        Returns
        -------
        np.ndarray
            Row positions into the per-frame position and force arrays.

        Raises
        ------
        ValueError
            If no atom matches.
        """
        ...

    def get_charges(self, atype: str) -> np.ndarray:
        """Return the charges of the atoms of a species or type.

        Backends with charges override this; the default raises.

        Parameters
        ----------
        atype : str
            Species or type name, in the backend's convention.

        Raises
        ------
        DataUnavailableError
            If the trajectory has no charges.
        """
        raise DataUnavailableError("Charge data not available for this trajectory type.")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return the masses of the atoms of a species or type.

        Backends with masses override this; the default raises.

        Parameters
        ----------
        atype : str
            Species or type name, in the backend's convention.

        Raises
        ------
        DataUnavailableError
            If the trajectory has no masses.
        """
        raise DataUnavailableError("Mass data not available for this trajectory type.")

    def iter_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1
    ) -> Generator[Frame, None, None]:
        """Iterate over frames, with slice semantics for the bounds.

        Parameters
        ----------
        start : int, optional
            First frame (default 0); negative counts from the end.
        stop : int, optional
            Frame to stop before (default: the end); negative counts from the end.
        stride : int, optional
            Step between frames (default 1).

        Yields
        ------
        Frame
            ``positions`` and ``forces``, each of shape ``(n_atoms, 3)``.

        Raises
        ------
        ValueError
            If ``stride`` is less than 1, or a frame cannot be read or has
            NaN or infinite values.
        """
        start, stop, stride = self._normalize_bounds(start, stop, stride)
        return self._checked(range(start, stop, stride), self._iter_frames_impl(start, stop, stride))

    def _checked(
        self, indices: Iterable[int], frames: Generator[Frame, None, None]
    ) -> Generator[Frame, None, None]:
        """Yield each frame after checking its values are finite.

        ``frames`` is closed when this generator ends, raises or is closed,
        so a backend's open file is not held by a suspended reader.
        """
        with contextlib.closing(frames):
            for index, frame in zip(indices, frames, strict=True):
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
    ) -> Generator[Frame, None, None]:
        """Yield frames ``range(start, stop, stride)``; the bounds are already normalised."""
        ...

    def get_frame(self, index: int) -> Frame:
        """Return one frame.

        Parameters
        ----------
        index : int
            Frame index; negative counts from the end.

        Returns
        -------
        Frame
            ``positions`` and ``forces``, each of shape ``(n_atoms, 3)``.

        Raises
        ------
        TypeError
            If ``index`` is not an integer.
        IndexError
            If ``index`` is out of range.
        ValueError
            If the frame cannot be read, fails the backend's checks, or has
            NaN or infinite values.
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


def locate(first: np.ndarray, i: int) -> tuple[int, int]:
    """The file of frame ``i`` and the frame's position in it.

    The frames of file ``k`` are frames ``first[k]`` to ``first[k + 1] - 1``.
    """
    k = int(np.searchsorted(first, i, side="right")) - 1
    return k, i - int(first[k])


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
        return locate(self.first, i)

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

    def _iter_frames_impl(self, start: int, stop: int, stride: int) -> Generator[Frame, None, None]:
        """Read the indexed frames in order, with one file open at a time."""
        return read_frames_one_file_at_a_time(
            range(start, stop, stride), self._frames.path, self._read
        )

    def _get_frame(self, index: int) -> Frame:
        """Read one frame by seeking to it; nothing else is read or kept."""
        with open(self._frames.path(index), "rb") as f:
            return self._read(f, index)
