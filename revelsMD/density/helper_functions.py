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
    def process_frame(
        trajectory: Trajectory,
        grid_state: GridState,
        positions: np.ndarray,
        forces: np.ndarray,
        a: float = 1.0,
        kernel: str = "triangular"
    ) -> None:
        """
        Deposit a frame's positions/forces to the grid using a given kernel.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory state (box lengths used here).
        grid_state : GridState
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
        grid_state.count += 1

        # Bring positions to the primary image (periodic remainder)
        homeZ = np.remainder(positions[:, 2], grid_state.box_z)
        homeY = np.remainder(positions[:, 1], grid_state.box_y)
        homeX = np.remainder(positions[:, 0], grid_state.box_x)

        # Component forces (scalar arrays for vectorized deposition)
        fox = forces[:, 0]
        foy = forces[:, 1]
        foz = forces[:, 2]

        # Map to voxel indices (np.digitize returns 1..len(bins)-1)
        x = np.digitize(homeX, grid_state.binsx)
        y = np.digitize(homeY, grid_state.binsy)
        z = np.digitize(homeZ, grid_state.binsz)

        if kernel.lower() == "triangular":
            _triangular_allocation(
                grid_state.forceX, grid_state.forceY, grid_state.forceZ, grid_state.counter,
                x, y, z, homeX, homeY, homeZ,
                fox, foy, foz, a,
                grid_state.lx, grid_state.ly, grid_state.lz,
                grid_state.nbinsx, grid_state.nbinsy, grid_state.nbinsz,
            )
        elif kernel.lower() == "box":
            # Convert to 0-based indices for box allocation
            _box_allocation(
                grid_state.forceX, grid_state.forceY, grid_state.forceZ, grid_state.counter,
                x - 1, y - 1, z - 1,
                fox, foy, foz, a,
            )
        else:
            raise ValueError(f"Unsupported kernel: {kernel!r}")
