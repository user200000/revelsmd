"""
MDAnalysis trajectory backend for RevelsMD.

This module provides the MDATrajectory class for reading trajectories
via MDAnalysis.
"""

from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from ._base import Trajectory


class MDATrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory handled by **MDAnalysis**.

    This class acts as a unified interface for reading, validating, and accessing
    data from MDAnalysis-compatible trajectory and topology files.

    Parameters
    ----------
    trajectory_file : str
        Path to the trajectory file (e.g., `.xtc`, `.trr`, `.dcd`, `.lammpstrj`).
    topology_file : str
        Path to the topology file (e.g., `.pdb`, `.gro`, `.data`).
    temperature : float
        Simulation temperature in Kelvin.

    Attributes
    ----------
    frames : int
        Number of trajectory frames.
    cell_matrix : np.ndarray
        Cell matrix with rows = lattice vectors, shape ``(3, 3)``.
    units : str
        Unit system identifier (`'mda'`).
    temperature : float
        Simulation temperature in Kelvin.
    beta : float
        Inverse thermal energy 1/(kB*T) in kJ/mol.

    Raises
    ------
    ValueError
        If no topology file is provided or the cell matrix is invalid.
    RuntimeError
        If MDAnalysis fails to load the trajectory or topology file.
    """

    def __init__(self, trajectory_file: str, topology_file: str, *, temperature: float):
        if not topology_file:
            raise ValueError("A topology file is required for MDAnalysis trajectories.")

        super().__init__(units='mda', temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        try:
            mdanalysis_universe = MD.Universe(topology_file, trajectory_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load MDAnalysis Universe: {e}")

        self.mdanalysis_universe = mdanalysis_universe
        self.frames = len(mdanalysis_universe.trajectory)

        dims = mdanalysis_universe.dimensions
        if len(dims) < 3:
            raise ValueError(f"Invalid simulation box dimensions: {dims}")

        # Build full cell matrix from MDAnalysis dimensions [a, b, c, alpha, beta, gamma]
        # For older trajectories lacking angular information, assume orthorhombic
        if len(dims) < 6:
            lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
            lx, ly, lz = self._validate_box_dimensions(lx, ly, lz)
            self.cell_matrix = self._cell_matrix_from_dimensions(lx, ly, lz)
        else:
            self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
        self._validate_cell_matrix(self.cell_matrix)

    def get_indices(self, atype: str) -> np.ndarray:
        """
        Return indices of atoms matching a given atom name.

        Parameters
        ----------
        atype : str
            Atom name to select (e.g., `'O'`, `'H'`, `'C'`).

        Returns
        -------
        np.ndarray
            Array of atom indices corresponding to the given atom name.
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').ids)

    get_indicies = get_indices  # backward compatibility alias

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

    def _iter_frames_impl(
        self,
        start: int,
        stop: int,
        stride: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate using MDAnalysis trajectory slicing."""
        for ts in self.mdanalysis_universe.trajectory[start:stop:stride]:
            yield ts.positions.copy(), ts.forces.copy()

    def get_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and forces for a specific frame by index."""
        ts = self.mdanalysis_universe.trajectory[index]
        return ts.positions.copy(), ts.forces.copy()
