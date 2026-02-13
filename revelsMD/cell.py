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
