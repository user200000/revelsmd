"""
Numba-accelerated helper functions for RDF calculations.

This module provides JIT-compiled implementations of the core computational
routines used in force-based RDF calculations, using Numba for performance.

Performance: ~8-12x speedup over original Python loops, ~3-4x over NumPy.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange  # type: ignore[import-untyped]

from revelsMD.cell import is_orthorhombic as _cell_is_orthorhombic


# ---------------------------------------------------------------------------
# Internal JIT-compiled functions
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _apply_minimum_image_numba(
    displacement: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    Apply minimum image convention to displacement vectors.

    Parameters
    ----------
    displacement : np.ndarray, shape (n, 3)
        Displacement vectors.
    box : np.ndarray, shape (3,)
        Box dimensions [box_x, box_y, box_z].

    Returns
    -------
    np.ndarray
        Corrected displacements with same shape as input.
    """
    result = displacement.copy()
    n = result.shape[0]
    for i in range(n):
        for j in range(3):
            r = result[i, j]
            half_box = box[j] / 2
            if abs(r) > half_box:
                result[i, j] -= np.ceil((abs(r) - half_box) / box[j]) * box[j] * np.sign(r)
    return result


@jit(nopython=True, parallel=True, cache=True)
def _compute_pairwise_contributions_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    forces_a: np.ndarray,
    forces_b: np.ndarray,
    box_x: float,
    box_y: float,
    box_z: float,
    same_species: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances and force projections for any species combination.

    For like-species (same_species=True), computes upper triangle only (j > i).
    For unlike-species, computes all n_a * n_b pairs.

    Both use the formula: (F_a - F_b) . r_ab / |r|^3
    """
    n_a = pos_a.shape[0]
    n_b = pos_b.shape[0]

    half_box_x = box_x / 2
    half_box_y = box_y / 2
    half_box_z = box_z / 2

    if same_species:
        # Upper triangle: n*(n-1)/2 pairs
        n_pairs = n_a * (n_a - 1) // 2
        r_flat = np.zeros(n_pairs, dtype=np.float64)
        dot_prod_flat = np.zeros(n_pairs, dtype=np.float64)

        for i in prange(n_a):
            for j in range(i + 1, n_a):
                # Displacement r_ij = pos[j] - pos[i]
                rx = pos_a[j, 0] - pos_a[i, 0]
                ry = pos_a[j, 1] - pos_a[i, 1]
                rz = pos_a[j, 2] - pos_a[i, 2]

                # Minimum image convention
                if abs(rx) > half_box_x:
                    rx -= np.ceil((abs(rx) - half_box_x) / box_x) * box_x * np.sign(rx)
                if abs(ry) > half_box_y:
                    ry -= np.ceil((abs(ry) - half_box_y) / box_y) * box_y * np.sign(ry)
                if abs(rz) > half_box_z:
                    rz -= np.ceil((abs(rz) - half_box_z) / box_z) * box_z * np.sign(rz)

                r_mag = np.sqrt(rx * rx + ry * ry + rz * rz)

                # Force difference: F[j] - F[i]
                if r_mag > 0:
                    fx = forces_a[j, 0] - forces_a[i, 0]
                    fy = forces_a[j, 1] - forces_a[i, 1]
                    fz = forces_a[j, 2] - forces_a[i, 2]
                    dot_prod = (fx * rx + fy * ry + fz * rz) / (r_mag * r_mag * r_mag)
                else:
                    dot_prod = 0.0

                # Upper triangle linear index: sum of (n-1) + (n-2) + ... + (n-i) + (j-i-1)
                # = i*n - i*(i+1)/2 + (j-i-1)
                idx = i * n_a - (i * (i + 1)) // 2 + (j - i - 1)
                r_flat[idx] = r_mag
                dot_prod_flat[idx] = dot_prod
    else:
        # Unlike species: all n_b * n_a pairs
        n_pairs = n_b * n_a
        r_flat = np.zeros(n_pairs, dtype=np.float64)
        dot_prod_flat = np.zeros(n_pairs, dtype=np.float64)

        for i in prange(n_b):
            for j in range(n_a):
                # Displacement r = pos_a[j] - pos_b[i]
                rx = pos_a[j, 0] - pos_b[i, 0]
                ry = pos_a[j, 1] - pos_b[i, 1]
                rz = pos_a[j, 2] - pos_b[i, 2]

                # Minimum image convention
                if abs(rx) > half_box_x:
                    rx -= np.ceil((abs(rx) - half_box_x) / box_x) * box_x * np.sign(rx)
                if abs(ry) > half_box_y:
                    ry -= np.ceil((abs(ry) - half_box_y) / box_y) * box_y * np.sign(ry)
                if abs(rz) > half_box_z:
                    rz -= np.ceil((abs(rz) - half_box_z) / box_z) * box_z * np.sign(rz)

                r_mag = np.sqrt(rx * rx + ry * ry + rz * rz)

                # Force difference: F_a[j] - F_b[i]
                if r_mag > 0:
                    fx = forces_a[j, 0] - forces_b[i, 0]
                    fy = forces_a[j, 1] - forces_b[i, 1]
                    fz = forces_a[j, 2] - forces_b[i, 2]
                    dot_prod = (fx * rx + fy * ry + fz * rz) / (r_mag * r_mag * r_mag)
                else:
                    dot_prod = 0.0

                idx = i * n_a + j
                r_flat[idx] = r_mag
                dot_prod_flat[idx] = dot_prod

    return r_flat, dot_prod_flat


@jit(nopython=True, cache=True)
def _accumulate_binned_contributions_numba(
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
    np.ndarray, shape (n_bins,)
        Accumulated values per bin.

    Notes
    -----
    Uses np.searchsorted to match np.digitize behaviour exactly.
    """
    n_bins = len(bins)
    if n_bins == 0:
        return np.zeros(1, dtype=np.float64)

    storage = np.zeros(n_bins, dtype=np.float64)
    m = len(values)

    for k in range(m):
        d = distances[k]
        v = values[k]

        # Skip NaN/inf values
        if not np.isfinite(v):
            continue

        # np.digitize(d, bins) - 1: find rightmost bin edge <= d
        # This is equivalent to searchsorted with side='right' minus 1
        bin_idx = np.searchsorted(bins, d, side='right') - 1

        # Clip to valid range
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= n_bins:
            bin_idx = n_bins - 1

        # Zero out last bin contributions (matching original behaviour)
        if bin_idx == n_bins - 1:
            continue

        storage[bin_idx] += v

    return storage


# ---------------------------------------------------------------------------
# Public wrapper functions with same signature as NumPy backend
# ---------------------------------------------------------------------------

def compute_pairwise_contributions_numba(
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
    if not _cell_is_orthorhombic(cell_matrix):
        raise NotImplementedError(
            "Numba RDF backend does not yet support triclinic cells. "
            "Use the numpy backend (REVELSMD_BACKEND=numpy) for triclinic cells."
        )
    same_species = np.array_equal(pos_a, pos_b)
    pos_a = np.ascontiguousarray(pos_a, dtype=np.float64)
    pos_b = np.ascontiguousarray(pos_b, dtype=np.float64)
    forces_a = np.ascontiguousarray(forces_a, dtype=np.float64)
    forces_b = np.ascontiguousarray(forces_b, dtype=np.float64)
    return _compute_pairwise_contributions_numba(
        pos_a, pos_b, forces_a, forces_b,
        cell_matrix[0, 0], cell_matrix[1, 1], cell_matrix[2, 2],
        same_species,
    )


def accumulate_binned_contributions_numba(
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
    np.ndarray, shape (n_bins,)
        Accumulated values per bin.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    bins = np.ascontiguousarray(bins, dtype=np.float64)
    return _accumulate_binned_contributions_numba(values, distances, bins)


@jit(nopython=True, cache=True)
def _accumulate_triangular_counts_numba(
    distances: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Numba implementation of triangular count accumulation.

    Each pair's contribution is distributed between the two nearest bin edges
    using linear interpolation weights (Cloud-in-Cell deposition).

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
    """
    n_bins = len(bins)
    if n_bins == 0:
        return np.zeros(1, dtype=np.float64)

    counts = np.zeros(n_bins, dtype=np.float64)
    delr = bins[1] - bins[0] if n_bins > 1 else 1.0
    m = len(distances)

    for k in range(m):
        r = distances[k]

        # Find bin index (left edge) using searchsorted
        bin_idx = np.searchsorted(bins, r, side='right') - 1

        # Skip if beyond last bin edge (matching force accumulation behaviour)
        if bin_idx >= n_bins - 1:
            continue

        # Handle before first bin
        if bin_idx < 0:
            counts[0] += 1.0
            continue

        # Triangular (CIC) weights
        r_lower = bins[bin_idx]
        weight_upper = (r - r_lower) / delr
        weight_lower = 1.0 - weight_upper

        counts[bin_idx] += weight_lower
        counts[bin_idx + 1] += weight_upper

    return counts


def accumulate_triangular_counts_numba(
    distances: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    Public wrapper for Numba triangular count accumulation.

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
    """
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    bins = np.ascontiguousarray(bins, dtype=np.float64)
    return _accumulate_triangular_counts_numba(distances, bins)
