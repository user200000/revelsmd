"""In-memory backend: NumpyTrajectory."""

from collections.abc import Generator

import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, DataUnavailableError


class NumpyTrajectory(Trajectory):
    """A trajectory held in memory as position and force arrays.

    Parameters
    ----------
    positions : np.ndarray
        Positions, shape ``(frames, atoms, 3)``.
    forces : np.ndarray
        Forces, the same shape as ``positions``.
    box_x, box_y, box_z : float, optional
        Orthorhombic box lengths, all three or none; exclusive with
        ``cell_matrix``.
    species_list : list of str, optional
        Label of each atom. Without it, ``get_indices``, ``get_charges`` and
        ``get_masses`` raise.
    temperature : float
        Temperature in Kelvin (reduced temperature for ``'lj'``).
    units : str, optional
        Unit system (default ``'real'``).
    cell_matrix : np.ndarray, optional
        3x3 cell matrix, rows the lattice vectors; exclusive with
        ``box_x``, ``box_y``, ``box_z``.
    charge_list, mass_list : np.ndarray, optional
        Charge and mass of each atom.

    Raises
    ------
    ValueError
        If the arrays' shapes disagree, or the cell is missing, given twice
        or invalid.
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
            self.cell_matrix = np.array(cell_matrix, dtype=np.float64, copy=True)
        else:
            if not all(v is not None for v in box_args):
                raise ValueError(
                    "All three of box_x, box_y, box_z must be provided together."
                )
            assert box_x is not None and box_y is not None and box_z is not None
            if box_x <= 0 or box_y <= 0 or box_z <= 0:
                raise ValueError("Box dimensions must all be positive values.")
            self.cell_matrix = self._cell_matrix_from_dimensions(box_x, box_y, box_z)
        self._validate_cell_matrix(self.cell_matrix)

        self.positions = positions
        self.forces = forces
        self.species_string = species_list
        self.frames = positions.shape[0]

        if charge_list is not None:
            self.charge_list = charge_list
        if mass_list is not None:
            self.mass_list = mass_list

    def get_indices(self, atype: str) -> np.ndarray:
        """Return the indices of the atoms with a given label.

        Parameters
        ----------
        atype : str
            Label from ``species_list`` (e.g. ``'O'``).

        Returns
        -------
        np.ndarray
            Row positions in the frames.

        Raises
        ------
        ValueError
            If there is no ``species_list`` or no atom has that label.
        """
        if self.species_string is None:
            raise ValueError("Species list was not provided for this trajectory.")
        inds = np.where(np.array(self.species_string) == atype)[0]
        if len(inds) == 0:
            raise ValueError(f"Species '{atype}' not found in species list.")
        return inds

    def get_charges(self, atype: str) -> np.ndarray:
        """Return the charges of the atoms with a given label.

        Parameters
        ----------
        atype : str
            Label from ``species_list``.

        Raises
        ------
        DataUnavailableError
            If no ``charge_list`` was given.
        ValueError
            If there is no ``species_list`` or no atom has that label.
        """
        if not hasattr(self, 'charge_list'):
            raise DataUnavailableError("Charge data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.charge_list[indices]

    def get_masses(self, atype: str) -> np.ndarray:
        """Return the masses of the atoms with a given label.

        Parameters
        ----------
        atype : str
            Label from ``species_list``.

        Raises
        ------
        DataUnavailableError
            If no ``mass_list`` was given.
        ValueError
            If there is no ``species_list`` or no atom has that label.
        """
        if not hasattr(self, 'mass_list'):
            raise DataUnavailableError("Mass data not available for this trajectory.")
        indices = self.get_indices(atype)
        return self.mass_list[indices]

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Generator[Frame, None, None]:
        """Yield the selected frames from the arrays."""
        for i in range(start, stop, stride):
            yield self._get_frame(i)

    def _get_frame(self, index: int) -> Frame:
        return Frame(self.positions[index], self.forces[index])
