"""
Vectorised helper functions for RDF calculations.

This module provides optimised implementations of the core computational
routines used in force-based RDF calculations. Two backends are available:

- 'numba' (default): Uses Numba JIT compilation with parallel execution.
  ~30x speedup over original Python loops.

- 'numpy': Uses NumPy broadcasting and vectorised operations.
  ~2x speedup over original Python loops. Fallback if Numba unavailable.

The backend can be selected via:
- Environment variable: REVELSMD_RDF_BACKEND=numpy
- Function parameter: get_backend_functions(backend='numpy')
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_AVAILABLE_BACKENDS = {'numpy', 'numba'}
_DEFAULT_BACKEND = 'numba'


def _get_numba_functions() -> tuple[Callable, Callable, Callable]:
    """Import and return Numba backend functions."""
    try:
        from revelsMD.rdf_helpers_numba import (
            compute_pairwise_contributions_like_numba,
            compute_pairwise_contributions_unlike_numba,
            accumulate_binned_contributions_numba,
        )
        return (
            compute_pairwise_contributions_like_numba,
            compute_pairwise_contributions_unlike_numba,
            accumulate_binned_contributions_numba,
        )
    except ImportError as e:
        raise ImportError(
            "Numba backend requested but numba is not installed. "
            "Install with: pip install numba"
        ) from e


def get_backend_functions(
    backend: str | None = None,
) -> tuple[Callable, Callable, Callable]:
    """
    Get the RDF helper functions for the specified backend.

    Parameters
    ----------
    backend : str or None
        Backend to use: 'numpy' or 'numba'. If None, uses the
        REVELSMD_RDF_BACKEND environment variable, defaulting to 'numba'.

    Returns
    -------
    tuple of (compute_like, compute_unlike, accumulate)
        The three helper functions for the selected backend.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    ImportError
        If numba backend is requested but numba is not installed.
    """
    if backend is None:
        backend = os.environ.get('REVELSMD_RDF_BACKEND', _DEFAULT_BACKEND).lower()

    if backend not in _AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown RDF backend: {backend!r}. "
            f"Available backends: {sorted(_AVAILABLE_BACKENDS)}"
        )

    if backend == 'numba':
        return _get_numba_functions()

    # Default: numpy backend
    return (
        compute_pairwise_contributions_like,
        compute_pairwise_contributions_unlike,
        accumulate_binned_contributions,
    )


# ---------------------------------------------------------------------------
# NumPy backend implementation
# ---------------------------------------------------------------------------


def apply_minimum_image(
    displacement: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    Apply minimum image convention to displacement vectors.

    Parameters
    ----------
    displacement : np.ndarray, shape (..., 3)
        Displacement vectors (can be any shape with last dimension = 3).
    box : np.ndarray, shape (3,)
        Box dimensions [box_x, box_y, box_z].

    Returns
    -------
    np.ndarray
        Corrected displacements with same shape as input.

    Notes
    -----
    Uses the original formula from revels_rdf.py for bit-identical results:
        r -= ceil((abs(r) - box/2) / box) * box * sign(r)
    """
    result = displacement.copy()
    for i in range(3):
        result[..., i] -= (
            np.ceil((np.abs(result[..., i]) - box[i] / 2) / box[i])
            * box[i]
            * np.sign(result[..., i])
        )
    return result


def compute_pairwise_contributions_like(
    pos: np.ndarray,
    forces: np.ndarray,
    box: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and force projections for like species.

    Parameters
    ----------
    pos : np.ndarray, shape (n, 3)
        Positions of atoms in the species.
    forces : np.ndarray, shape (n, 3)
        Forces on atoms in the species.
    box : tuple of (box_x, box_y, box_z)
        Orthorhombic box dimensions.

    Returns
    -------
    r_flat : np.ndarray, shape (n*n,), dtype=np.longdouble
        Flattened pairwise distances after MIC.
    dot_prod_flat : np.ndarray, shape (n*n,), dtype=np.longdouble
        Flattened force projections F[j] . r_ij / |r|^3.

    Notes
    -----
    The force projection uses the force on atom j (the second index).
    This matches the original implementation where F[x, :] = force_total[:, :]
    means F[i, j] = force[j].
    """
    ns = pos.shape[0]
    box_arr = np.array(box)

    # Pairwise displacements: r[i, j] = pos[j] - pos[i]
    # Shape: (ns, ns, 3)
    r_vec = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]

    # Apply minimum image convention
    r_vec = apply_minimum_image(r_vec, box_arr)

    # Distance magnitudes: shape (ns, ns)
    r_mag = np.sqrt(np.sum(r_vec ** 2, axis=2))

    # Forces broadcast to (ns, ns, 3) where F[i, j, :] = forces[j, :]
    # This matches the original: Fx[x, :] = force_total[:, 0]
    F_vec = np.broadcast_to(forces[np.newaxis, :, :], (ns, ns, 3))

    # Force projection: F[j] . r_ij / |r|^3
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_prod = np.sum(F_vec * r_vec, axis=2) / r_mag ** 3

    # Sanity filter matching legacy behaviour
    half_box = box_arr / 2
    outside_mask = (
        (np.abs(r_vec[..., 0]) > half_box[0]) |
        (np.abs(r_vec[..., 1]) > half_box[1]) |
        (np.abs(r_vec[..., 2]) > half_box[2])
    )
    dot_prod[outside_mask] = 0

    return (
        r_mag.ravel().astype(np.longdouble),
        np.nan_to_num(dot_prod.ravel(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.longdouble),
    )


def compute_pairwise_contributions_unlike(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    forces_a: np.ndarray,
    forces_b: np.ndarray,
    box: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and force projections for unlike species.

    Parameters
    ----------
    pos_a : np.ndarray, shape (n1, 3)
        Positions of atoms in species A.
    pos_b : np.ndarray, shape (n2, 3)
        Positions of atoms in species B.
    forces_a : np.ndarray, shape (n1, 3)
        Forces on atoms in species A.
    forces_b : np.ndarray, shape (n2, 3)
        Forces on atoms in species B.
    box : tuple of (box_x, box_y, box_z)
        Orthorhombic box dimensions.

    Returns
    -------
    r_flat : np.ndarray, shape (n1*n2,), dtype=np.longdouble
        Flattened pairwise distances after MIC.
    dot_prod_flat : np.ndarray, shape (n1*n2,), dtype=np.longdouble
        Flattened force difference projections (F_A - F_B) . r_AB / |r|^3.

    Notes
    -----
    The output shape matches the original implementation where the matrix
    has shape (n2, n1), i.e. iterating over B atoms in the outer dimension.
    """
    n1 = pos_a.shape[0]
    n2 = pos_b.shape[0]
    box_arr = np.array(box)

    # Pairwise displacements: r[i, j] = pos_a[j] - pos_b[i]
    # Shape: (n2, n1, 3) - matches original loop structure
    r_vec = pos_a[np.newaxis, :, :] - pos_b[:, np.newaxis, :]

    # Apply minimum image convention
    r_vec = apply_minimum_image(r_vec, box_arr)

    # Distance magnitudes: shape (n2, n1)
    r_mag = np.sqrt(np.sum(r_vec ** 2, axis=2))

    # Force differences: F[i, j] = forces_a[j] - forces_b[i]
    # Shape: (n2, n1, 3)
    F_diff = forces_a[np.newaxis, :, :] - forces_b[:, np.newaxis, :]

    # Force projection: (F_A - F_B) . r_AB / |r|^3
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_prod = np.sum(F_diff * r_vec, axis=2) / r_mag ** 3

    # Sanity filter matching legacy behaviour
    half_box = box_arr / 2
    outside_mask = (
        (np.abs(r_vec[..., 0]) > half_box[0]) |
        (np.abs(r_vec[..., 1]) > half_box[1]) |
        (np.abs(r_vec[..., 2]) > half_box[2])
    )
    dot_prod[outside_mask] = 0

    return (
        r_mag.ravel().astype(np.longdouble),
        np.nan_to_num(dot_prod.ravel(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.longdouble),
    )


def accumulate_binned_contributions(
    values: np.ndarray,
    distances: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Accumulate values into bins based on distances.

    Parameters
    ----------
    values : np.ndarray, shape (m,)
        Values to accumulate (e.g., force projections).
    distances : np.ndarray, shape (m,)
        Distances determining bin assignment.
    bins : np.ndarray, shape (n_bins,)
        Bin edges (right edge of each bin).

    Returns
    -------
    np.ndarray, shape (n_bins,), dtype=np.longdouble
        Accumulated values per bin.

    Notes
    -----
    - Uses np.digitize for bin assignment (same as original).
    - Uses np.bincount for O(m) accumulation instead of O(n_bins) loop.
    - Values in the last bin are set to zero (matching original behaviour).
    - Handles nan/inf via nan_to_num at the end.
    """
    n_bins = len(bins)

    if n_bins == 0:
        return np.zeros(1, dtype=np.longdouble)

    # Bin assignment (same as original)
    bin_indices = np.digitize(distances, bins) - 1

    # Zero out last bin contributions (matching original behaviour)
    values = np.asarray(values, dtype=np.longdouble).copy()
    values[bin_indices == n_bins - 1] = 0

    # Clip indices to valid range for bincount
    # Negative indices (before first bin) go to bin 0
    bin_indices_clipped = np.clip(bin_indices, 0, n_bins - 1)

    # Single-pass accumulation: O(m) instead of O(m * n_bins)
    storage = np.bincount(
        bin_indices_clipped,
        weights=values,
        minlength=n_bins,
    ).astype(np.longdouble)

    return np.nan_to_num(storage, nan=0.0, posinf=0.0, neginf=0.0)
