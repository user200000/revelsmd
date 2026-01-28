"""
Base classes for trajectory handling in RevelsMD.

This module defines the abstract base class and common exceptions used by
all trajectory backends.
"""

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import scipy.constants as constants


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
        Temperature in Kelvin.

    Returns
    -------
    float
        Inverse thermal energy in the appropriate units.

    Raises
    ------
    ValueError
        If the unit system is not recognised.
    """
    units = units.lower().strip()
    if units not in _BOLTZMANN_CONSTANTS:
        raise ValueError(
            f"Unsupported unit system: '{units}'. "
            f"Expected one of {list(_BOLTZMANN_CONSTANTS.keys())}."
        )
    return 1.0 / (_BOLTZMANN_CONSTANTS[units] * temperature)


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
    box_x, box_y, box_z : float
        Simulation box dimensions in each Cartesian direction.
    units : str
        Unit system identifier (e.g., 'real', 'metal', 'mda').
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.
    """

    # Required attributes - subclasses must set these
    frames: int
    box_x: float
    box_y: float
    box_z: float
    units: str
    temperature: float
    beta: float

    def __init__(self, *, units: str, temperature: float) -> None:
        """
        Initialise common trajectory attributes.

        Subclasses should call ``super().__init__(units=..., temperature=...)``
        after setting their own attributes.

        Parameters
        ----------
        units : str
            Unit system identifier (e.g., 'real', 'metal', 'mda', 'lj').
        temperature : float
            Simulation temperature in Kelvin.
        """
        self.units = units
        self.temperature = temperature
        self.beta = compute_beta(units, temperature)

    def _normalize_bounds(
        self, start: int, stop: int | None, stride: int
    ) -> tuple[int, int, int]:
        """
        Normalize start/stop bounds to handle negative indices Pythonically.

        Parameters
        ----------
        start : int
            Start index (can be negative).
        stop : int or None
            Stop index (can be negative or None for end of trajectory).
        stride : int
            Step between frames.

        Returns
        -------
        tuple of (int, int, int)
            Normalized (start, stop, stride) suitable for use with range().

        Notes
        -----
        Follows Python slice semantics:
        - Negative indices count from the end (e.g., -1 is the last frame)
        - None for stop means iterate to the end
        - Out-of-bounds indices are clamped to valid range
        """
        n = self.frames

        # Handle None stop
        if stop is None:
            stop = n

        # Handle negative start
        if start < 0:
            start = max(0, n + start)

        # Handle negative stop
        if stop < 0:
            stop = max(0, n + stop)

        # Clamp to valid range
        start = min(start, n)
        stop = min(stop, n)

        return start, stop, stride

    @staticmethod
    def _validate_orthorhombic(angles: list[float], atol: float = 1e-3) -> None:
        """
        Validate that cell angles are orthorhombic (all 90 degrees).

        Parameters
        ----------
        angles : list of float
            The three cell angles [alpha, beta, gamma] in degrees.
        atol : float, optional
            Absolute tolerance for comparison to 90 degrees (default: 1e-3).

        Raises
        ------
        ValueError
            If any angle is not within tolerance of 90 degrees.
        """
        if not np.allclose(angles, 90.0, atol=atol):
            raise ValueError(
                "Only orthorhombic or cubic cells are supported. "
                f"Got angles: {angles}"
            )

    @staticmethod
    def _validate_box_dimensions(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        """
        Validate that box dimensions are positive and finite.

        Parameters
        ----------
        lx, ly, lz : float
            Box dimensions in each Cartesian direction.

        Returns
        -------
        tuple of (float, float, float)
            The validated box dimensions.

        Raises
        ------
        ValueError
            If any dimension is not positive or not finite.
        """
        dims = [lx, ly, lz]
        if not all(np.isfinite(dims)):
            raise ValueError(f"Box dimensions must be finite. Got: ({lx}, {ly}, {lz})")
        if not all(d > 0 for d in dims):
            raise ValueError(f"Box dimensions must be positive. Got: ({lx}, {ly}, {lz})")
        return lx, ly, lz

    @abstractmethod
    def get_indices(self, atype: str) -> np.ndarray:
        """Return atom indices for a given species or type."""
        ...

    def get_charges(self, atype: str) -> np.ndarray:
        """Return atomic charges for atoms of a given species or type.

        Subclasses should override this method if charge data is available.
        The default implementation raises DataUnavailableError.
        """
        raise DataUnavailableError("Charge data not available for this trajectory type.")

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for atoms of a given species or type.

        Subclasses should override this method if mass data is available.
        The default implementation raises DataUnavailableError.
        """
        raise DataUnavailableError("Mass data not available for this trajectory type.")

    def iter_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over trajectory frames, yielding positions and forces.

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
        positions : np.ndarray
            Atomic positions for the current frame, shape (n_atoms, 3).
        forces : np.ndarray
            Atomic forces for the current frame, shape (n_atoms, 3).
        """
        start, stop, stride = self._normalize_bounds(start, stop, stride)
        return self._iter_frames_impl(start, stop, stride)

    @abstractmethod
    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Internal implementation of frame iteration.

        Subclasses implement this with normalized (non-negative) bounds.
        """
        ...

    @abstractmethod
    def get_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return positions and forces for a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        positions : np.ndarray
            Atomic positions for the frame, shape (n_atoms, 3).
        forces : np.ndarray
            Atomic forces for the frame, shape (n_atoms, 3).
        """
        ...
