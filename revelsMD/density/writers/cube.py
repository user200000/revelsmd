"""Write volumetric data to Gaussian cube format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.constants import angstrom, physical_constants

# 1 Angstrom in Bohr
ANGSTROM_TO_BOHR = angstrom / physical_constants["Bohr radius"][0]


def write_cube(
    filename: str | Path,
    grid: np.ndarray,
    cell_matrix: np.ndarray,
    *,
    comment: str = "",
) -> None:
    """Write a 3D grid to a Gaussian ``.cube`` file.

    The file is written with zero atoms — this is intended for density
    or other volumetric data, not atomic structure.

    Parameters
    ----------
    filename : str or Path
        Output file path.
    grid : numpy.ndarray
        3D array of volumetric data (shape: nx, ny, nz).
    cell_matrix : numpy.ndarray
        3x3 matrix whose rows are the lattice vectors (a, b, c) in
        Angstroms.
    comment : str, optional
        Comment for the first line of the cube file.
    """
    grid = np.asarray(grid)
    cell_matrix = np.asarray(cell_matrix, dtype=np.float64)

    if grid.ndim != 3:
        raise ValueError(f"grid must be a 3D array, got shape {grid.shape}")
    if cell_matrix.shape != (3, 3):
        raise ValueError(
            f"cell_matrix must have shape (3, 3), got {cell_matrix.shape}"
        )

    with open(filename, "w") as f:
        # Two comment lines
        f.write(f"{comment}\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")

        # Number of atoms and origin (all in Bohr)
        f.write(f"{0:5d}{0:12.6f}{0:12.6f}{0:12.6f}\n")

        # Voxel vectors: cell_vector[i] / n_i, converted to Bohr
        for i in range(3):
            n = grid.shape[i]
            voxel = cell_matrix[i] / n * ANGSTROM_TO_BOHR
            f.write(f"{n:5d}{voxel[0]:12.6f}{voxel[1]:12.6f}{voxel[2]:12.6f}\n")

        # No atom lines (natoms = 0)

        # Volumetric data
        grid.tofile(f, sep="\n", format="%e")
