"""HelperFunctions class for grid allocation and COM calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from revelsMD.trajectories._base import Trajectory
from revelsMD.density.grid_helpers import get_backend_functions as _get_grid_backend_functions
from revelsMD.density.selection_state import SelectionState

if TYPE_CHECKING:
    from revelsMD.density.grid_state import GridState

# Module-level backend functions (loaded once at import)
_triangular_allocation, _box_allocation = _get_grid_backend_functions()


class HelperFunctions:
    """
    Helper numerics for per-frame deposition, COMs, dipoles, and rigid sums.
    """

    @staticmethod
    def process_frame(trajectory: Trajectory, GS: GridState, positions: np.ndarray, forces: np.ndarray, a: float = 1.0, kernel: str = "triangular") -> None:
        """
        Deposit a frame's positions/forces to the grid using a given kernel.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory state (box lengths used here).
        GS : GridState
            Grid accumulators and bin geometry.
        positions : (N, 3) np.ndarray
            Positions in Cartesian coordinates.
        forces : (N, 3) np.ndarray
            Forces corresponding to `positions`.
        a : float or np.ndarray, optional
            Scalar or per-particle weight (number/charge/polarisation projection).
        kernel : {'triangular','box'}, optional
            Assignment kernel.

        Notes
        -----
        - Positions are reduced into a primary image by component-wise remainders.
        - `np.digitize` is used to map positions to voxel indices; subsequent kernels
          deposit weighted contributions to neighbors (triangular) or the host voxel (box).
        """
        GS.count += 1

        # Bring positions to the primary image (periodic remainder)
        homeZ = np.remainder(positions[:, 2], GS.box_z)
        homeY = np.remainder(positions[:, 1], GS.box_y)
        homeX = np.remainder(positions[:, 0], GS.box_x)

        # Component forces (scalar arrays for vectorized deposition)
        fox = forces[:, 0]
        foy = forces[:, 1]
        foz = forces[:, 2]

        # Map to voxel indices (np.digitize returns 1..len(bins)-1)
        x = np.digitize(homeX, GS.binsx)
        y = np.digitize(homeY, GS.binsy)
        z = np.digitize(homeZ, GS.binsz)

        if kernel.lower() == "triangular":
            _triangular_allocation(
                GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, a,
                GS.lx, GS.ly, GS.lz,
                GS.nbinsx, GS.nbinsy, GS.nbinsz,
            )
        elif kernel.lower() == "box":
            # Convert to 0-based indices for box allocation
            _box_allocation(
                GS.forceX, GS.forceY, GS.forceZ, GS.counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, a,
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel!r}")

    @staticmethod
    def box_allocation(GS: GridState, x: np.ndarray, y: np.ndarray, z: np.ndarray, fox: np.ndarray, foy: np.ndarray, foz: np.ndarray, a: float | np.ndarray) -> None:
        """
        Deposit contributions to the host voxel (no neighbour spreading).

        This method delegates to the backend allocation function selected
        at module import time. Uses np.add.at() or Numba JIT to correctly
        handle overlapping particles.

        Parameters
        ----------
        GS : GridState
            Grid state object with forceX, forceY, forceZ, counter arrays.
        x, y, z : np.ndarray
            Voxel indices (1-based from np.digitize).
        fox, foy, foz : np.ndarray
            Force components for each particle.
        a : float or np.ndarray
            Weight factor (scalar or per-particle array).
        """
        # Convert to 0-based voxel indices
        _box_allocation(
            GS.forceX, GS.forceY, GS.forceZ, GS.counter,
            x - 1, y - 1, z - 1,
            fox, foy, foz, a,
        )

    @staticmethod
    def triangular_allocation(
        GS: GridState,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        homeX: np.ndarray,
        homeY: np.ndarray,
        homeZ: np.ndarray,
        fox: np.ndarray,
        foy: np.ndarray,
        foz: np.ndarray,
        a: float | np.ndarray,
    ) -> None:
        """
        Deposit contributions to the 8 neighbouring voxel vertices (CIC/triangular).

        This method delegates to the backend allocation function selected
        at module import time. Uses np.add.at() or Numba JIT to correctly
        handle overlapping particles.

        Parameters
        ----------
        GS : GridState
            Grid state object with forceX, forceY, forceZ, counter arrays
            and grid parameters (lx, ly, lz, nbinsx, nbinsy, nbinsz).
        x, y, z : np.ndarray
            Voxel indices (1-based from np.digitize).
        homeX, homeY, homeZ : np.ndarray
            Actual particle positions.
        fox, foy, foz : np.ndarray
            Force components for each particle.
        a : float or np.ndarray
            Weight factor (scalar or per-particle array).
        """
        _triangular_allocation(
            GS.forceX, GS.forceY, GS.forceZ, GS.counter,
            x, y, z, homeX, homeY, homeZ,
            fox, foy, foz, a,
            GS.lx, GS.ly, GS.lz,
            GS.nbinsx, GS.nbinsy, GS.nbinsz,
        )


    @staticmethod
    def find_coms(positions: np.ndarray, trajectory: Trajectory, GS: GridState, SS: SelectionState, calc_dipoles: bool = False):
        """
        Compute centers-of-mass (and optionally molecular dipoles) for a rigid set.

        Parameters
        ----------
        positions : (N, 3) np.ndarray
            Cartesian coordinates.
        trajectory : Trajectory
            Trajectory state containing box lengths.
        GS : GridState
            Unused numerically here; preserved for signature compatibility.
        SS : SelectionState
            Provides `indices`, `masses`, and `charges` (if available).
        calc_dipoles : bool, optional
            If True, also compute per-molecule dipole moments relative to COMs.

        Returns
        -------
        coms : (M, 3) np.ndarray
            Center-of-mass coordinates for M molecules.
        molecular_dipole : (M, 3) np.ndarray, optional
            Returned if `calc_dipoles=True`.

        Notes
        -----
        Enforces minimum-image displacements when aligning species to a reference
        for COM and dipole accumulation.
        """
        mass_tot = SS.masses[0]
        mass_cumulant = positions[SS.indices[0]] * SS.masses[0][:, np.newaxis]
        for species_index in range(1, len(SS.indices)):
            diffs = positions[SS.indices[0]] - positions[SS.indices[species_index]]
            logical_diffs = np.transpose(
                np.array(
                    [
                        trajectory.box_x * (diffs[:, 0] < -trajectory.box_x / 2) - trajectory.box_x * (diffs[:, 0] > trajectory.box_x / 2),
                        trajectory.box_y * (diffs[:, 1] < -trajectory.box_y / 2) - trajectory.box_y * (diffs[:, 1] > trajectory.box_y / 2),
                        trajectory.box_z * (diffs[:, 2] < -trajectory.box_z / 2) - trajectory.box_z * (diffs[:, 2] > trajectory.box_z / 2),
                    ]
                )
            )
            diffs += logical_diffs
            mass_tot += SS.masses[species_index]
            mass_cumulant += positions[SS.indices[species_index]] * SS.masses[species_index][:, np.newaxis]
        coms = mass_cumulant / mass_tot[:, np.newaxis]

        if calc_dipoles:
            charges_cumulant = GS.SS.charges[0][:, np.newaxis] * (positions[SS.indices[0]] - coms)
            for species_index in range(1, len(SS.indices)):
                separation = (positions[SS.indices[species_index]] - coms)
                # Minimum-image correction component-wise
                separation[:, 0] -= (np.ceil((np.abs(separation[:, 0]) - trajectory.box_x / 2) / trajectory.box_x)) * (trajectory.box_x) * np.sign(separation[:, 0])
                separation[:, 1] -= (np.ceil((np.abs(separation[:, 1]) - trajectory.box_y / 2) / trajectory.box_y)) * (trajectory.box_y) * np.sign(separation[:, 1])
                separation[:, 2] -= (np.ceil((np.abs(separation[:, 2]) - trajectory.box_z / 2) / trajectory.box_z)) * (trajectory.box_z) * np.sign(separation[:, 2])
                charges_cumulant += GS.SS.charges[species_index][:, np.newaxis] * separation
            molecular_dipole = charges_cumulant
            return coms, molecular_dipole
        else:
            return coms

    @staticmethod
    def sum_forces(SS: SelectionState, forces: np.ndarray) -> np.ndarray:
        """
        Sum forces across a rigid group (species lists).

        Parameters
        ----------
        SS : SelectionState
            Selection containing per-species atom index arrays.
        forces : (N, 3) np.ndarray
            Force array for all atoms.

        Returns
        -------
        (M, 3) np.ndarray
            Per-molecule net force for the rigid group (M = multiplicity).
        """
        rigid_forces = forces[SS.indices[0], :]
        for rigid_body_component in SS.indices[1:]:
            rigid_forces += forces[rigid_body_component, :]
        return rigid_forces
