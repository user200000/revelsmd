"""
NumPy array trajectory backend for RevelsMD.

This module provides the NumpyTrajectory class for trajectories stored
directly as NumPy arrays in memory.
"""

from typing import Iterator

import numpy as np

from ._base import Trajectory, DataUnavailableError


class NumpyTrajectory(Trajectory):
    """
    Represents a trajectory stored directly as NumPy arrays.

    Designed for simulation data already resident in memory, or for synthetic
    or analytical trajectories generated numerically.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions of shape ``(frames, atoms, 3)``.
    forces : np.ndarray
        Atomic forces of shape ``(frames, atoms, 3)``.
    box_x, box_y, box_z : float, optional
        Simulation box lengths in each Cartesian direction.
        All three must be provided together.
        Mutually exclusive with ``cell_matrix``.
    species_list : list of str, optional
        Atom names corresponding to each atom index.  If ``None``,
        ``get_indices`` and related methods will not be available.
    temperature : float
        Simulation temperature in Kelvin.
    units : str, optional
        Unit system string (default: `'real'`).
    cell_matrix : np.ndarray, optional
        Full 3x3 cell matrix with rows = lattice vectors.
        Mutually exclusive with ``box_x``, ``box_y``, ``box_z``.
    charge_list : np.ndarray, optional
        Atomic charge array (optional).
    mass_list : np.ndarray, optional
        Atomic mass array (optional).

    Attributes
    ----------
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in the trajectory's unit system.

    Raises
    ------
    ValueError
        If positions and forces are inconsistent, if box dimensions are
        invalid, or if both/neither of ``cell_matrix`` and ``box_x/y/z``
        are provided.

    Notes
    -----
    This class provides a simple in-memory structure compatible with the
    ``Revels3D`` and ``RevelsRDF`` interfaces.
    """

    def __init__(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
        box_x: float | None = None,
        box_y: float | None = None,
        box_z: float | None = None,
        species_list: list[str] | None = None,
        *,
        temperature: float,
        units: str = 'real',
        cell_matrix: np.ndarray | None = None,
        charge_list: np.ndarray | None = None,
        mass_list: np.ndarray | None = None,
    ):
        if positions.shape != forces.shape:
            raise ValueError("Force and position arrays are incommensurate.")

        if species_list is not None and positions.shape[1] != len(species_list):
            raise ValueError("Species list and trajectory arrays are incommensurate.")

        # Determine cell geometry from either cell_matrix or box_x/y/z
        box_args = (box_x, box_y, box_z)
        has_box = any(v is not None for v in box_args)
        has_cell = cell_matrix is not None

        if has_box and has_cell:
            raise ValueError(
                "Cannot specify both cell_matrix and box_x/box_y/box_z."
            )
        if not has_box and not has_cell:
            raise ValueError(
                "Must specify either cell_matrix or box_x/box_y/box_z."
            )

        super().__init__(units=units, temperature=temperature)

        if has_cell:
            cell_matrix = np.array(cell_matrix, dtype=np.float64, copy=True)
            self._validate_cell_matrix(cell_matrix)
            self.cell_matrix = cell_matrix
        else:
            if not all(v is not None for v in box_args):
                raise ValueError(
                    "All three of box_x, box_y, box_z must be provided together."
                )
            assert box_x is not None and box_y is not None and box_z is not None
            if box_x <= 0 or box_y <= 0 or box_z <= 0:
                raise ValueError("Box dimensions must all be positive values.")
            self.cell_matrix = self._cell_matrix_from_dimensions(box_x, box_y, box_z)

        self.positions = positions
        self.forces = forces
        self.species_string = species_list
        self.frames = positions.shape[0]

        if charge_list is not None:
            self.charge_list = charge_list
        if mass_list is not None:
            self.mass_list = mass_list

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return atom indices for a given species.

        Parameters
        ----------
        atype : str
            Atom species name to select (e.g., `'O'`, `'H'`, `'C'`).

        Returns
        -------
        np.ndarray
            Indices of selected atoms.

        Raises
        ------
        ValueError
            If the species name is not present in the provided species list.
        """
        if self.species_string is None:
            raise ValueError("Species list was not provided for this trajectory.")
        inds = np.where(np.array(self.species_string) == atype)[0]
        if len(inds) == 0:
            raise ValueError(f"Species '{atype}' not found in species list.")
        return inds

    get_indicies = get_indices

    def get_charges(self, atype: str) -> np.ndarray:
        """Return atomic charges for atoms of a given species."""
        if not hasattr(self, 'charge_list'):
            raise DataUnavailableError("Charge data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.charge_list[indices]

    def get_masses(self, atype: str) -> np.ndarray:
        """Return atomic masses for atoms of a given species."""
        if not hasattr(self, 'mass_list'):
            raise DataUnavailableError("Mass data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.mass_list[indices]

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over in-memory position/force arrays."""
        for i in range(start, stop, stride):
            yield self.positions[i], self.forces[i]

    def get_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        return self.positions[index], self.forces[index]
