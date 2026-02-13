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


def apply_minimum_image(
    displacement: np.ndarray,
    cell_matrix: np.ndarray,
    cell_inverse: np.ndarray,
) -> np.ndarray:
    """
    Apply the minimum image convention to displacement vectors.

    Works for arbitrary triclinic cells.  The displacement is converted to
    fractional coordinates, each component is rounded to the nearest integer
    and subtracted, then the result is converted back to Cartesian.

    Parameters
    ----------
    displacement : np.ndarray, shape (..., 3)
        Displacement vectors in Cartesian coordinates.
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.
    cell_inverse : np.ndarray, shape (3, 3)
        Inverse of the cell matrix.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Minimum-image displacement vectors in Cartesian coordinates.
    """
    fractional = displacement @ cell_inverse
    fractional -= np.round(fractional)
    return fractional @ cell_matrix


def apply_minimum_image_orthorhombic(
    displacement: np.ndarray, box: np.ndarray
) -> np.ndarray:
    """
    Apply the minimum image convention for an orthorhombic cell.

    Uses the per-axis formula:
    ``r -= ceil((abs(r) - box/2) / box) * box * sign(r)``

    Parameters
    ----------
    displacement : np.ndarray, shape (..., 3)
        Displacement vectors in Cartesian coordinates.
    box : np.ndarray, shape (3,)
        Box dimensions ``[box_x, box_y, box_z]``.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Minimum-image displacement vectors.
    """
    result = displacement.copy()
    for i in range(3):
        result[..., i] -= (
            np.ceil((np.abs(result[..., i]) - box[i] / 2) / box[i])
            * box[i]
            * np.sign(result[..., i])
        )
    return result


def inscribed_sphere_radius(cell_matrix: np.ndarray) -> float:
    """
    Compute the inscribed sphere radius of the parallelepiped defined by
    the cell matrix.

    This is the maximum valid cutoff for the minimum image convention:
    half the smallest perpendicular height of the cell.

    For orthorhombic cells this reduces to ``min(Lx, Ly, Lz) / 2``.

    Parameters
    ----------
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.

    Returns
    -------
    float
        Inscribed sphere radius.
    """
    a, b, c = cell_matrix[0], cell_matrix[1], cell_matrix[2]
    volume = abs(np.linalg.det(cell_matrix))

    cross_bc = np.cross(b, c)
    cross_ca = np.cross(c, a)
    cross_ab = np.cross(a, b)

    h_bc = volume / np.linalg.norm(cross_bc)
    h_ca = volume / np.linalg.norm(cross_ca)
    h_ab = volume / np.linalg.norm(cross_ab)

    return min(h_bc, h_ca, h_ab) / 2


def cells_are_compatible(
    cell_a: np.ndarray, cell_b: np.ndarray, atol: float = 1e-6
) -> bool:
    """
    Return ``True`` if two cell matrices match element-wise within tolerance.

    Parameters
    ----------
    cell_a, cell_b : np.ndarray, shape (3, 3)
        Cell matrices to compare.
    atol : float, optional
        Absolute tolerance for element-wise comparison (default: 1e-6).
    """
    return bool(np.allclose(cell_a, cell_b, atol=atol, rtol=0))
