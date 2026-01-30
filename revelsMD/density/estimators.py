"""Estimators class for per-frame density estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from revelsMD.density.helper_functions import HelperFunctions
from revelsMD.density.selection_state import SelectionState

if TYPE_CHECKING:
    from revelsMD.density.grid_state import GridState
    from revelsMD.trajectories._base import Trajectory
    import numpy as np


class Estimators:
    """
    Per-frame estimators that compute positions/weights and deposit to the grid.

    Notes
    -----
    All estimators call `HelperFunctions.process_frame(...)` which:
    - Applies periodic imaging to positions
    - Assigns per-atom contributions using a chosen kernel
    - Updates force accumulators (X/Y/Z) and the counting `counter` field
    """

    @staticmethod
    def single_frame_rigid_number_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: number density at COM position; forces summed over rigid members."""
        coms = HelperFunctions.find_coms(positions, trajectory, GS, SS)
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(trajectory, GS, coms, rigid_forces, kernel=kernel)

    @staticmethod
    def single_frame_rigid_number_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: number density at a specific atom's position; forces summed over rigid members."""
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(trajectory, GS, positions[SS.indices[SS.centre_location], :], rigid_forces, kernel=kernel)

    @staticmethod
    def single_frame_number_many_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Non-rigid: number density for each species list; deposit per-entry."""
        for count in range(len(SS.indices)):
            HelperFunctions.process_frame(
                trajectory, GS, positions[SS.indices[count], :], forces[SS.indices[count], :], kernel=kernel
            )

    @staticmethod
    def single_frame_number_single_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Single species: number density at that species' positions."""
        HelperFunctions.process_frame(trajectory, GS, positions[SS.indices, :], forces[SS.indices, :], kernel=kernel)

    @staticmethod
    def single_frame_rigid_charge_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: charge-weighted density at a specific atom's position."""
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(
            trajectory, GS, positions[SS.indices[SS.centre_location], :], rigid_forces, a=SS.charges[SS.centre_location], kernel=kernel
        )

    @staticmethod
    def single_frame_charge_many_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Non-rigid: charge-weighted density for each entry."""
        for count in range(len(SS.indices)):
            HelperFunctions.process_frame(
                trajectory, GS, positions[SS.indices[count], :], forces[SS.indices[count], :], a=SS.charges[count], kernel=kernel
            )

    @staticmethod
    def single_frame_charge_single_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Single species: charge-weighted density."""
        HelperFunctions.process_frame(
            trajectory, GS, positions[SS.indices, :], forces[SS.indices, :], a=SS.charges, kernel=kernel
        )

    @staticmethod
    def single_frame_rigid_charge_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: charge-weighted density at COM."""
        coms = HelperFunctions.find_coms(positions, trajectory, GS, SS)
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(trajectory, GS, coms, rigid_forces, kernel=kernel, a=SS.charges)

    @staticmethod
    def single_frame_rigid_polarisation_com_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: polarisation density projected along `SS.polarisation_axis` at COM."""
        coms, molecular_dipole = HelperFunctions.find_coms(positions, trajectory, GS, SS, calc_dipoles=True)
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(
            trajectory, GS, coms, rigid_forces, a=molecular_dipole[:, GS.SS.polarisation_axis], kernel=kernel
        )

    @staticmethod
    def single_frame_rigid_polarisation_atom_grid(positions, forces, trajectory, GS, SS, kernel="triangular"):
        """Rigid molecule: polarisation density projected along `SS.polarisation_axis` at COM (per original code)."""
        coms, molecular_dipole = HelperFunctions.find_coms(positions, trajectory, GS, SS, calc_dipoles=True)
        rigid_forces = HelperFunctions.sum_forces(SS, forces)
        HelperFunctions.process_frame(
            trajectory, GS, coms, rigid_forces, a=molecular_dipole[:, GS.SS.polarisation_axis], kernel=kernel
        )
