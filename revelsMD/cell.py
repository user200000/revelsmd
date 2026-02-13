"""
Coordinate transforms and cell geometry for periodic simulation cells.

All functions use the convention that rows of the cell matrix are lattice
vectors: ``M[0] = a``, ``M[1] = b``, ``M[2] = c`` (matching pymatgen and ASE).

Key operations:

- Cartesian from fractional: ``r = s @ M``
- Fractional from Cartesian: ``s = r @ inv(M)``
- Periodic wrapping (fractional): ``s_wrapped = s - floor(s)``
- Minimum image convention: ``ds = r @ inv(M)``, ``ds -= round(ds)``,
  ``dr = ds @ M``
- Cell volume: ``abs(det(M))``
"""

import numpy as np

#: Absolute tolerance for deciding whether off-diagonal cell matrix elements
#: are zero, i.e. whether the cell is orthorhombic.
ORTHORHOMBIC_TOLERANCE: float = 1e-6


def is_orthorhombic(
    cell_matrix: np.ndarray, atol: float = ORTHORHOMBIC_TOLERANCE
) -> bool:
    """
    Return ``True`` if all off-diagonal elements of *cell_matrix* are below
    *atol*, i.e. the cell is orthorhombic (or cubic).

    Parameters
    ----------
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.
    atol : float, optional
        Absolute tolerance for off-diagonal elements (default:
        ``ORTHORHOMBIC_TOLERANCE``).
    """
    off_diagonal = cell_matrix[~np.eye(3, dtype=bool)]
    return bool(np.all(np.abs(off_diagonal) < atol))


def cartesian_to_fractional(
    positions: np.ndarray, cell_inverse: np.ndarray
) -> np.ndarray:
    """
    Convert Cartesian positions to fractional coordinates.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian positions.
    cell_inverse : np.ndarray, shape (3, 3)
        Inverse of the cell matrix (``np.linalg.inv(cell_matrix)``).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Fractional coordinates.
    """
    return positions @ cell_inverse


def fractional_to_cartesian(
    fractional: np.ndarray, cell_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert fractional coordinates to Cartesian positions.

    Parameters
    ----------
    fractional : np.ndarray, shape (N, 3)
        Fractional coordinates.
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Cartesian positions.
    """
    return fractional @ cell_matrix


def wrap_fractional(fractional: np.ndarray) -> np.ndarray:
    """
    Wrap fractional coordinates into [0, 1).

    Parameters
    ----------
    fractional : np.ndarray, shape (N, 3)
        Fractional coordinates (may be outside [0, 1)).

    Returns
    -------
    np.ndarray, shape (N, 3)
        Wrapped fractional coordinates in [0, 1).
    """
    return fractional - np.floor(fractional)
