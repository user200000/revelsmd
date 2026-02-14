"""
Vectorised helper functions for RDF calculations.

This module provides optimised implementations of the core computational
routines used in force-based RDF calculations. Two backends are available:

- 'numba' (default): Uses Numba JIT compilation with parallel execution.
  ~30x speedup over original Python loops.

- 'numpy': Uses NumPy broadcasting and vectorised operations.
  ~2x speedup over original Python loops. Fallback if Numba unavailable.

Backend selection is controlled by the REVELSMD_BACKEND environment variable.
See `revelsMD.backends` for configuration details.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from revelsMD.backends import get_backend, AVAILABLE_BACKENDS
from revelsMD.cell import apply_minimum_image as _cell_apply_minimum_image


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _get_numba_functions() -> tuple[Callable, Callable, Callable]:
    """Import and return Numba backend functions."""
    try:
        from revelsMD.rdf.rdf_helpers_numba import (
            compute_pairwise_contributions_numba,
            accumulate_binned_contributions_numba,
            accumulate_triangular_counts_numba,
        )
        return (
            compute_pairwise_contributions_numba,
            accumulate_binned_contributions_numba,
            accumulate_triangular_counts_numba,
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
        REVELSMD_BACKEND environment variable, defaulting to 'numba'.

    Returns
    -------
    tuple of (compute_pairwise, accumulate_binned, accumulate_triangular)
        The three helper functions for the selected backend.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    ImportError
        If numba backend is requested but numba is not installed.
    """
    if backend is None:
        backend = get_backend()

    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown RDF backend: {backend!r}. "
            f"Available backends: {sorted(AVAILABLE_BACKENDS)}"
        )

    if backend == 'numba':
        return _get_numba_functions()

    # Default: numpy backend
    return (
        compute_pairwise_contributions,
        accumulate_binned_contributions,
        accumulate_triangular_counts,
    )


# ---------------------------------------------------------------------------
# NumPy backend implementation
# ---------------------------------------------------------------------------


def apply_minimum_image(
    displacement: np.ndarray,
    cell_matrix: np.ndarray,
    cell_inverse: np.ndarray,
) -> np.ndarray:
    """
    Apply minimum image convention to displacement vectors.

    Works for arbitrary cell geometries (orthorhombic and triclinic).

    Parameters
    ----------
    displacement : np.ndarray, shape (..., 3)
        Displacement vectors (can be any shape with last dimension = 3).
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.
    cell_inverse : np.ndarray, shape (3, 3)
        Inverse of the cell matrix.

    Returns
    -------
    np.ndarray
        Corrected displacements with same shape as input.
    """
    return _cell_apply_minimum_image(displacement, cell_matrix, cell_inverse)


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
    np.ndarray, shape (n_bins,), dtype=np.float64
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
        return np.zeros(1, dtype=np.float64)

    # Bin assignment (same as original)
    bin_indices = np.digitize(distances, bins) - 1

    # Zero out last bin contributions (matching original behaviour)
    values = np.asarray(values, dtype=np.float64).copy()
    values[bin_indices == n_bins - 1] = 0

    # Clip indices to valid range for bincount
    # Negative indices (before first bin) go to bin 0
    bin_indices_clipped = np.clip(bin_indices, 0, n_bins - 1)

    # Single-pass accumulation: O(m) instead of O(m * n_bins)
    storage = np.bincount(
        bin_indices_clipped,
        weights=values,
        minlength=n_bins,
    ).astype(np.float64)

    return np.nan_to_num(storage, nan=0.0, posinf=0.0, neginf=0.0)


def accumulate_triangular_counts(
    distances: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Accumulate pair counts using triangular (CIC) deposition.

    Each pair's contribution is distributed between the two nearest bin edges
    using linear interpolation weights, analogous to Cloud-in-Cell deposition
    used in DensityGrid.

    Parameters
    ----------
    distances : np.ndarray, shape (m,)
        Pairwise distances.
    bins : np.ndarray, shape (n_bins,)
        Bin edges (radial positions where g(r) is evaluated).

    Returns
    -------
    np.ndarray, shape (n_bins,), dtype=np.float64
        Accumulated counts per bin edge.

    Notes
    -----
    For a pair at distance r between bins[i] and bins[i+1]:
    - Weight to bins[i]: (bins[i+1] - r) / delr
    - Weight to bins[i+1]: (r - bins[i]) / delr

    Pairs exactly at a bin edge contribute 1.0 to that bin.
    Pairs beyond the last bin edge are excluded (matching force accumulation).
    """
    n_bins = len(bins)
    if n_bins == 0:
        return np.zeros(1, dtype=np.float64)

    counts = np.zeros(n_bins, dtype=np.float64)
    delr = bins[1] - bins[0] if n_bins > 1 else 1.0

    for r in distances:
        # Find bin index (left edge)
        bin_idx = np.searchsorted(bins, r, side='right') - 1

        # Skip if beyond last bin edge (matching force accumulation behaviour)
        if bin_idx >= n_bins - 1:
            continue

        # Handle before first bin
        if bin_idx < 0:
            # Contribute to first bin only
            counts[0] += 1.0
            continue

        # Triangular (CIC) weights
        r_lower = bins[bin_idx]
        weight_upper = (r - r_lower) / delr
        weight_lower = 1.0 - weight_upper

        counts[bin_idx] += weight_lower
        counts[bin_idx + 1] += weight_upper

    return counts


def compute_pairwise_contributions(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    forces_a: np.ndarray,
    forces_b: np.ndarray,
    cell_matrix: np.ndarray,
    cell_inverse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and force projections for any species combination.

    This unified function handles both like-species (A-A) and unlike-species (A-B)
    cases using the same formula: (F_a - F_b) . r_ab / |r|^3.

    For like-species (detected via np.array_equal), only the upper triangle
    (j > i) is computed to avoid double-counting.

    For unlike-species, all n_a * n_b pairs are computed.

    Parameters
    ----------
    pos_a : np.ndarray, shape (n_a, 3)
        Positions of atoms in species A.
    pos_b : np.ndarray, shape (n_b, 3)
        Positions of atoms in species B.
    forces_a : np.ndarray, shape (n_a, 3)
        Forces on atoms in species A.
    forces_b : np.ndarray, shape (n_b, 3)
        Forces on atoms in species B.
    cell_matrix : np.ndarray, shape (3, 3)
        Cell matrix with rows = lattice vectors.
    cell_inverse : np.ndarray, shape (3, 3)
        Inverse of the cell matrix.

    Returns
    -------
    r_flat : np.ndarray, dtype=np.float64
        Flattened pairwise distances after MIC.
        Shape is (n_a * (n_a - 1) // 2,) for like-species,
        or (n_a * n_b,) for unlike-species.
    dot_prod_flat : np.ndarray, dtype=np.float64
        Flattened force projections (F_a - F_b) . r_ab / |r|^3.
        Same shape as r_flat.
    """
    same_species = np.array_equal(pos_a, pos_b)

    # Build displacement and force-difference arrays.
    # Like-species uses upper triangle indexing (n*(n-1)/2 pairs).
    # Unlike-species uses broadcasting (n_a * n_b pairs).
    # Both result in arrays where the last axis is the 3D vector component.
    if same_species:
        n = pos_a.shape[0]
        i_idx, j_idx = np.triu_indices(n, k=1)
        r_vec = pos_a[j_idx] - pos_a[i_idx]
        F_diff = forces_a[j_idx] - forces_a[i_idx]
    else:
        r_vec = pos_a[np.newaxis, :, :] - pos_b[:, np.newaxis, :]
        F_diff = forces_a[np.newaxis, :, :] - forces_b[:, np.newaxis, :]

    r_vec = apply_minimum_image(r_vec, cell_matrix, cell_inverse)

    # Compute |r| and (F_diff . r) / |r|^3. Using axis=-1 works for both
    # 2D (n_pairs, 3) and 3D (n_b, n_a, 3) arrays.
    r_mag = np.sqrt(np.sum(r_vec ** 2, axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_prod = np.sum(F_diff * r_vec, axis=-1) / r_mag ** 3

    return (
        r_mag.ravel().astype(np.float64),
        np.nan_to_num(dot_prod.ravel(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
    )
