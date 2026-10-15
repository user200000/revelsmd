"""
MDAnalysis trajectory backend for RevelsMD.

This module provides the MDATrajectory class for reading trajectories
via MDAnalysis.
"""

from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, check_fixed_cell


class MDATrajectory(Trajectory):
    """
    Represents a molecular dynamics trajectory handled by **MDAnalysis**.

    This class acts as a unified interface for reading, validating, and accessing
    data from MDAnalysis-compatible trajectory and topology files.

    Parameters
    ----------
    trajectory_file : str
        Path to a trajectory file that stores forces (e.g. `.trr` written with
        forces at every saved frame).
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
        If no topology file is provided, if the trajectory has no box or no
        forces, or if the cell matrix is invalid.
    RuntimeError
        If MDAnalysis fails to load the trajectory or topology file.

    Notes
    -----
    Reading a frame raises ValueError if it has no forces or if its box
    differs from the first frame's.
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
        if dims is None:
            raise ValueError(
                f"{trajectory_file}: the trajectory has no simulation box; "
                "a periodic box is required"
            )
        if not mdanalysis_universe.trajectory.ts.has_forces:
            raise ValueError(
                f"{trajectory_file}: the trajectory has no forces; force "
                "sampling needs forces in every frame"
            )
        self._dimensions = np.array(dims, dtype=np.float64)
        self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
        self._validate_cell_matrix(self.cell_matrix)

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
        """
        return np.array(self.mdanalysis_universe.select_atoms(f'name {atype}').ix)

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

    def _frame(self, ts) -> Frame:
        """Check one MDAnalysis timestep and return it as a Frame."""
        where = f"{self.trajectory_file}, frame {ts.frame}"
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
    ) -> Iterator[Frame]:
        """Iterate using MDAnalysis trajectory slicing."""
        for ts in self.mdanalysis_universe.trajectory[start:stop:stride]:
            yield self._frame(ts)

    def get_frame(self, index: int) -> Frame:
        """Return positions and forces for a specific frame by index.

        Parameters
        ----------
        index : int
            Frame index to retrieve.

        Returns
        -------
        Frame
            A dataclass with ``positions`` and ``forces`` arrays,
            each of shape ``(n_atoms, 3)``.
        """
        return self._frame(self.mdanalysis_universe.trajectory[index])
