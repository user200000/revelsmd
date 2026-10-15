"""
MDAnalysis trajectory backend for RevelsMD.

This module provides the MDATrajectory class for reading trajectories
via MDAnalysis.
"""

import os
from collections.abc import Sequence
from typing import Iterator

import MDAnalysis as MD  # type: ignore[import-untyped]
from MDAnalysis.coordinates.H5MD import H5MDReader  # type: ignore[import-untyped]
from MDAnalysis.lib.mdamath import triclinic_vectors  # type: ignore[import-untyped]
import numpy as np

from revelsMD.frame_sources import Frame
from ._base import Trajectory, as_paths, check_finite, check_fixed_cell


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
        forces (e.g. a `.trr` written with both at the same steps), or a
        list of such files in run order, which MDAnalysis joins.
    topology_file : str
        Path to the topology file (e.g., `.tpr`, `.pdb`, `.gro`).
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
        Inverse thermal energy 1/(kB*T) in mol/kJ.

    Raises
    ------
    ValueError
        If no topology file is provided, if the same file is given twice,
        if MDAnalysis does not know the
        format's length or force units, if an H5MD file saves positions and
        forces at different steps, if the first frame has no box, no
        positions or no forces, or if the cell matrix is invalid.
    TypeError
        If ``trajectory_file`` is an unordered collection such as a set.
    RuntimeError
        If MDAnalysis fails to load the trajectory or topology file.

    Notes
    -----
    Reading a frame raises ValueError if it has no positions, no forces or
    no box, if its box differs from the first frame's, or if its positions
    or forces contain NaN or infinity.
    """

    def __init__(
        self,
        trajectory_file: "str | os.PathLike[str] | Sequence[str | os.PathLike[str]]",
        topology_file: str,
        *,
        temperature: float,
    ):
        if not topology_file:
            raise ValueError("A topology file is required for MDAnalysis trajectories.")
        if not isinstance(trajectory_file, (str, os.PathLike)):
            trajectory_file = as_paths(trajectory_file)

        super().__init__(units='mda', temperature=temperature)

        self.trajectory_file = trajectory_file
        self.topology_file = topology_file

        try:
            mdanalysis_universe = MD.Universe(topology_file, trajectory_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load MDAnalysis Universe: {e}") from e

        self.mdanalysis_universe = mdanalysis_universe
        self.frames = len(mdanalysis_universe.trajectory)

        dims = mdanalysis_universe.dimensions
        if dims is None:
            raise ValueError(
                f"{trajectory_file}: the trajectory has no simulation box; "
                "a periodic box is required"
            )
        if not mdanalysis_universe.trajectory.ts.has_positions:
            raise ValueError(
                f"{trajectory_file}: the trajectory has no positions; "
                "positions are needed in every frame"
            )
        if not mdanalysis_universe.trajectory.ts.has_forces:
            raise ValueError(
                f"{trajectory_file}: the trajectory has no forces; force "
                "sampling needs forces in every frame"
            )
        trajectory = mdanalysis_universe.trajectory
        for reader in getattr(trajectory, "readers", [trajectory]):
            _check_reader(reader)
        self._dimensions = np.array(dims, dtype=np.float64)
        self.cell_matrix = np.array(triclinic_vectors(dims), dtype=np.float64)
        try:
            self._validate_cell_matrix(self.cell_matrix)
        except ValueError as e:
            raise ValueError(f"{trajectory_file}: {e}") from e

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

    def _frame(self, ts) -> Frame:
        """Check one MDAnalysis timestep and return it as a Frame."""
        where = f"{self.trajectory_file}, frame {ts.frame}"
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
        raise ValueError(
            f"{reader.filename}: MDAnalysis does not know the length and force "
            f"units of this format ({type(reader).__name__}), so the forces "
            "cannot be converted to kJ/(mol Angstrom); read LAMMPS dumps with "
            "LammpsTrajectory"
        )
    if not isinstance(reader, H5MDReader):
        return
    particles = getattr(reader, "_particle_group", None)
    if particles is None:
        raise RuntimeError(
            f"{reader.filename}: cannot check that this H5MD file saves positions "
            "and forces at the same steps with this version of MDAnalysis"
        )
    if "position" in particles and "force" in particles:
        position_steps = particles["position/step"][()]
        force_steps = particles["force/step"][()]
        if not np.array_equal(position_steps, force_steps):
            raise ValueError(
                f"{reader.filename}: the forces are saved at different steps "
                "from the positions; every frame needs both"
            )

