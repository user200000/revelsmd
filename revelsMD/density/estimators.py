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
    def single_frame_rigid_number_com_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: number density at COM position; forces summed over rigid members."""
        coms = HelperFunctions.find_coms(positions, trajectory, grid_state, selection_state)
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(trajectory, grid_state, coms, rigid_forces, kernel=kernel)

    @staticmethod
    def single_frame_rigid_number_atom_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: number density at a specific atom's position; forces summed over rigid members."""
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(trajectory, grid_state, positions[selection_state.indices[selection_state.centre_location], :], rigid_forces, kernel=kernel)

    @staticmethod
    def single_frame_number_many_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Non-rigid: number density for each species list; deposit per-entry."""
        for count in range(len(selection_state.indices)):
            HelperFunctions.process_frame(
                trajectory, grid_state, positions[selection_state.indices[count], :], forces[selection_state.indices[count], :], kernel=kernel
            )

    @staticmethod
    def single_frame_number_single_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Single species: number density at that species' positions."""
        HelperFunctions.process_frame(trajectory, grid_state, positions[selection_state.indices, :], forces[selection_state.indices, :], kernel=kernel)

    @staticmethod
    def single_frame_rigid_charge_atom_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: charge-weighted density at a specific atom's position."""
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(
            trajectory, grid_state, positions[selection_state.indices[selection_state.centre_location], :], rigid_forces, a=selection_state.charges[selection_state.centre_location], kernel=kernel
        )

    @staticmethod
    def single_frame_charge_many_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Non-rigid: charge-weighted density for each entry."""
        for count in range(len(selection_state.indices)):
            HelperFunctions.process_frame(
                trajectory, grid_state, positions[selection_state.indices[count], :], forces[selection_state.indices[count], :], a=selection_state.charges[count], kernel=kernel
            )

    @staticmethod
    def single_frame_charge_single_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Single species: charge-weighted density."""
        HelperFunctions.process_frame(
            trajectory, grid_state, positions[selection_state.indices, :], forces[selection_state.indices, :], a=selection_state.charges, kernel=kernel
        )

    @staticmethod
    def single_frame_rigid_charge_com_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: charge-weighted density at COM."""
        coms = HelperFunctions.find_coms(positions, trajectory, grid_state, selection_state)
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(trajectory, grid_state, coms, rigid_forces, kernel=kernel, a=selection_state.charges)

    @staticmethod
    def single_frame_rigid_polarisation_com_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: polarisation density projected along `selection_state.polarisation_axis` at COM."""
        coms, molecular_dipole = HelperFunctions.find_coms(positions, trajectory, grid_state, selection_state, calc_dipoles=True)
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(
            trajectory, grid_state, coms, rigid_forces, a=molecular_dipole[:, grid_state.selection_state.polarisation_axis], kernel=kernel
        )

    @staticmethod
    def single_frame_rigid_polarisation_atom_grid(positions, forces, trajectory, grid_state, selection_state, kernel="triangular"):
        """Rigid molecule: polarisation density projected along `selection_state.polarisation_axis` at COM (per original code)."""
        coms, molecular_dipole = HelperFunctions.find_coms(positions, trajectory, grid_state, selection_state, calc_dipoles=True)
        rigid_forces = HelperFunctions.sum_forces(selection_state, forces)
        HelperFunctions.process_frame(
            trajectory, grid_state, coms, rigid_forces, a=molecular_dipole[:, grid_state.selection_state.polarisation_axis], kernel=kernel
        )
